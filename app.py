"""🎬 PRIYA — AI Movie & TV Assistant (Streamlit entry point).

Run with:
    streamlit run app.py

Pages (top navigation): Home (chat), Discover, My Picks, About.
Backed by the hybrid recommendation engine (FAISS + embeddings + ranking),
Claude for conversational understanding, TMDB for metadata, and the Priya
voice assistant.
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings
from src.chatbot.conversation import ConversationManager
from src.models.preferences import UserPreferences
from src.recommender.recommender import Recommender
from src.services.llm_service import LLMService
from src.services.tmdb_client import TMDBClient
from src.services.voice_assistant import VoiceAssistant
from src.ui import components, pages, session, theme
from src.utils.logging_config import get_logger

try:
    from streamlit_mic_recorder import mic_recorder

    MIC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    mic_recorder = None
    MIC_AVAILABLE = False

logger = get_logger("app")

st.set_page_config(
    page_title="Priya — AI Movie & TV Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_VOICE_OPTIONS = {
    "Priya (en-IN) 🇮🇳": "en-IN-PriyaNeural",
    "Jenny (en-US) 🇺🇸": "en-US-JennyNeural",
    "Aria (en-US) 🇺🇸": "en-US-AriaNeural",
    "Sonia (en-GB) 🇬🇧": "en-GB-SoniaNeural",
    "Natasha (en-AU) 🇦🇺": "en-AU-NatashaNeural",
}


# --------------------------------------------------------------------- #
# Global resources (cached across reruns)
# --------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_recommender() -> Recommender:
    return Recommender()


@st.cache_resource(show_spinner=False)
def load_llm() -> LLMService:
    return LLMService()


@st.cache_resource(show_spinner=False)
def load_tmdb() -> TMDBClient:
    return TMDBClient()


@st.cache_resource(show_spinner=False)
def load_voice() -> VoiceAssistant:
    return VoiceAssistant()


# --------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------- #
def render_sidebar(conversation: ConversationManager) -> None:
    settings = get_settings()
    with st.sidebar:
        st.header("🎛️ Filters")
        st.caption("These act as persistent, hard filters.")

        content_type = st.selectbox(
            "Content Type", options=["both", "movie", "tv"], index=0
        )
        min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5)
        current_year = 2026
        release_min, release_max = st.slider(
            "Release Year Range",
            min_value=1900,
            max_value=current_year,
            value=(1900, current_year),
        )
        runtime_max = st.slider(
            "Maximum Runtime (min)", 0, 300, 0, step=15,
            help="0 means no limit",
        )

        genre_options = sorted(
            {
                "Action", "Adventure", "Animation", "Comedy", "Crime",
                "Documentary", "Drama", "Family", "Fantasy", "History",
                "Horror", "Music", "Mystery", "Romance", "Science Fiction",
                "Thriller", "War", "Western",
            }
        )
        selected_genres = st.multiselect("Genres", genre_options)

        language = st.selectbox(
            "Language",
            options=["", "English", "Korean", "Spanish", "French", "Japanese", "Hindi"],
            format_func=lambda x: "Any" if x == "" else x,
        )

        if st.button("🧹 Reset Conversation", use_container_width=True):
            conversation.reset()
            st.rerun()

        st.divider()
        st.markdown("**Current preferences**")
        st.caption(conversation.preferences.summary())

        sidebar_prefs = UserPreferences()
        sidebar_prefs.content_type = content_type
        if min_rating:
            sidebar_prefs.minimum_rating = min_rating
        if release_min != 1900 or release_max != current_year:
            sidebar_prefs.release_year_min = release_min if release_min != 1900 else None
            sidebar_prefs.release_year_max = release_max if release_max != current_year else None
        if runtime_max:
            sidebar_prefs.runtime_max = runtime_max
        if selected_genres:
            sidebar_prefs.genres = selected_genres
        if language:
            sidebar_prefs.languages = [language]
        st.session_state["sidebar_prefs"] = sidebar_prefs


def render_voice_sidebar(voice: VoiceAssistant) -> None:
    """Voice controls: enable toggle, voice picker and mic button."""
    settings = get_settings()
    with st.sidebar:
        st.divider()
        st.header("🎙️ Voice · Priya")
        enabled = st.checkbox(
            f"Enable voice assistant ({settings.voice_assistant_name})",
            value=st.session_state.get("voice_enabled", settings.voice_enabled_default),
            key="voice_enabled",
        )
        if not enabled:
            st.caption("Voice mode is off. Turn it on and Priya will speak replies.")
            return

        voice_labels = list(_VOICE_OPTIONS)
        default_label = next(
            (label for label, vid in _VOICE_OPTIONS.items() if vid == settings.tts_voice),
            voice_labels[0],
        )
        selected = st.selectbox(
            "Priya's voice", voice_labels, index=voice_labels.index(default_label),
            key="priya_voice_label",
        )
        voice.tts_voice = _VOICE_OPTIONS[selected]

        if MIC_AVAILABLE and mic_recorder is not None:
            recording = mic_recorder(
                key="priya_mic",
                start_prompt="🎤 Click to talk to Priya",
                stop_prompt="⏹ Stop recording",
                just_once=True,
                use_container_width=True,
                format="wav",
            )
            if recording is not None:
                audio_bytes = recording.get("bytes")
                sample_rate = recording.get("sample_rate", 44100)
                if audio_bytes:
                    with st.spinner("Listening…"):
                        transcript = voice.stt(audio_bytes, sample_rate)
                    if transcript:
                        st.session_state["priya_voice_input"] = transcript
                    else:
                        st.warning("I couldn't hear that. Please try again.")
        else:
            st.caption("Mic recorder not installed — run: `pip install streamlit-mic-recorder`")


# --------------------------------------------------------------------- #
# Page routing
# --------------------------------------------------------------------- #
def current_page() -> str:
    """Resolve the active page from query params (or session fallback)."""
    page = st.query_params.get("page")
    if isinstance(page, list):
        page = page[0]
    if page in components.PAGES:
        return page
    return st.session_state.get("page", "home")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main() -> None:
    theme.inject()
    settings = get_settings()

    conversation = ConversationManager(st.session_state)
    recommender = load_recommender()
    llm = load_llm()
    tmdb = load_tmdb()
    voice = load_voice()

    render_sidebar(conversation)
    render_voice_sidebar(voice)

    page = current_page()
    components.render_navbar(
        page,
        llm_ready=llm.available,
        tmdb_ready=bool(tmdb.api_key),
    )

    if page == "discover":
        pages.render_discover(recommender, tmdb)
    elif page == "picks":
        pages.render_picks(recommender, tmdb)
    elif page == "about":
        pages.render_about()
    else:
        if not settings.has_anthropic_key or not settings.has_tmdb_key:
            st.info(
                "Running in **local demo mode**. Add `ANTHROPIC_API_KEY` and "
                "`TMDB_API_KEY` to a `.env` file (see `.env.example`) for Claude "
                "explanations and live posters/trailers."
            )
        pages.render_home(conversation, recommender, llm, tmdb, voice)

    components.render_footer()

    # Greet Priya once when the user switches voice mode on (user gesture).
    prev_enabled = st.session_state.get("voice_was_enabled")
    now_enabled = st.session_state.get("voice_enabled", settings.voice_enabled_default)
    if prev_enabled is False and now_enabled is True:
        session.speak(voice.intro(), voice)
    st.session_state["voice_was_enabled"] = now_enabled


if __name__ == "__main__":
    main()
