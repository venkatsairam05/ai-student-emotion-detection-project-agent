"""PRIYA pages: Home (chat), Discover, My Picks, About + movie details dialog.

This module renders everything; the actual recommendation engine, TMDB and
Claude services are passed in so the UI stays decoupled from the back end.
"""

from __future__ import annotations

from html import escape
from typing import Optional

import streamlit as st

from src.chatbot.preference_extractor import extract_preferences_rule_based
from src.ui import components, session, theme
from src.utils.logging_config import get_logger

logger = get_logger("ui.pages")

AVATAR = theme.priya_avatar_svg(64)

QUICK_PROMPTS = [
    ("✨ Surprise me", "Surprise me with a great movie."),
    ("🍿 Movie for tonight", "Recommend a fun movie for tonight."),
    ("😂 Make me laugh", "I want something funny, make me laugh."),
    ("🧠 Something mind-bending", "I want a mind-bending movie."),
    ("❤️ Romantic movie", "Recommend a romantic movie."),
    ("🔥 Best thriller", "Give me the best thriller."),
    ("📺 New TV series", "Recommend a new TV series."),
    ("🎬 Like my favorite movie", "Recommend movies like my favorite films."),
]
QUICK_MAP = dict(QUICK_PROMPTS)

MOODS = [
    ("😊 Happy", "I'm feeling happy and upbeat — recommend something cheerful."),
    ("😌 Relaxed", "I'm feeling relaxed — suggest something calm and easy to watch."),
    ("🤯 Mind-blown", "I want something mind-blowing and unexpected."),
    ("😱 Thrilled", "I'm in the mood for something thrilling and suspenseful."),
    ("❤️ Romantic", "I'm feeling romantic — recommend a love story."),
    ("😂 Need a laugh", "I need a laugh — something funny."),
    ("😭 Emotional", "I want something emotional and moving."),
    ("🔥 Excited", "I'm excited — give me something epic and action-packed."),
    ("🧠 Thoughtful", "I want something thoughtful and intelligent."),
    ("👨‍👩‍👧 Family night", "It's family movie night — something family-friendly."),
]
MOOD_MAP = dict(MOODS)


# --------------------------------------------------------------------- #
# Home — chat
# --------------------------------------------------------------------- #
def render_home(conversation, recommender, llm, tmdb, voice) -> None:
    components.render_hero()

    user_input: Optional[str] = None
    pressed = components.chip_buttons([label for label, _ in QUICK_PROMPTS], "qp", per_row=4)
    if pressed:
        user_input = QUICK_MAP[pressed]

    components.section_title("🌈 How are you feeling today?")
    mood = components.chip_buttons([label for label, _ in MOODS], "mood", per_row=5)
    if mood:
        user_input = MOOD_MAP[mood]

    # Voice transcript (from the sidebar mic) and cross-page prompts.
    if user_input is None:
        user_input = session.consume_voice_input()
    if user_input is None:
        user_input = session.pop_prompt()

    # Conversation history.
    for message in conversation.history:
        with st.chat_message(
            message["role"],
            avatar=AVATAR if message["role"] == "assistant" else None,
        ):
            st.markdown(message["content"])

    # Main chat input.
    typed = st.chat_input("✨ Tell Priya what you want to watch...")
    if typed:
        user_input = typed

    if user_input:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        process_message(user_input, conversation, recommender, llm, tmdb, voice)
    else:
        _maybe_render_last_cards(tmdb)

    _maybe_open_details(tmdb, recommender)


def _maybe_render_last_cards(tmdb) -> None:
    """Re-render the previous recommendation cards so their buttons stay live
    across reruns (a chat_input value is consumed after one run)."""
    cards = session.get_last_cards()
    if not cards:
        return
    result = components.card_grid(cards, tmdb, "rec")
    if result:
        _handle_card_action(result["action"], result["card"])


def process_message(user_input: str, conversation, recommender, llm, tmdb, voice) -> None:
    conversation.add_message("user", user_input)

    if conversation.is_reset_request(user_input):
        conversation.reset()
        session.set_last_cards([])
        reply = "I've cleared your preferences and history. How can I help you find something new? 🍿"
        with st.chat_message("assistant", avatar=AVATAR):
            st.markdown(reply)
        session.speak(reply, voice)
        return

    with st.spinner("Reading your preferences…"):
        if llm.available:
            extracted = llm.extract_preferences(user_input, conversation.history_text())
        else:
            extracted = extract_preferences_rule_based(user_input)
    conversation.update_preferences(extracted)

    prefs = conversation.preferences
    sidebar = st.session_state.get("sidebar_prefs")
    if sidebar is not None:
        prefs = prefs.merge(sidebar)

    if not recommender.ready:
        components.empty_state(
            "🎬",
            "Priya is still learning her movie library.",
            "Run `python scripts/prepare_data.py` and `python scripts/build_index.py`, then refresh.",
        )
        session.speak("I'm still learning my movie library. Run the setup scripts and I'll be ready.", voice)
        return

    with st.spinner("Priya is searching the movie universe... ✨"):
        movies = recommender.recommend(prefs, query_text=user_input)
        for movie in movies:
            tmdb.enrich(movie)
    conversation.recommendations = movies

    if not movies:
        reply = llm.generate_followup_response(
            user_input, conversation.preferences, conversation.history
        )
        with st.chat_message("assistant", avatar=AVATAR):
            st.markdown(reply)
        session.speak(reply, voice)
        return

    explanation = ""
    if llm.available:
        with st.spinner("Crafting your explanation…"):
            explanation = llm.generate_recommendation_explanation(
                [movie.to_candidate_dict() for movie in movies],
                prefs,
            )

    with st.chat_message("assistant", avatar=AVATAR):
        if explanation:
            st.markdown(explanation)

    cards = [components.movie_to_card(movie, tmdb) for movie in movies]
    session.set_last_cards(cards)
    components.section_title("✨ Priya's Picks")
    result = components.card_grid(cards, tmdb, "rec")
    if result:
        _handle_card_action(result["action"], result["card"])

    if not explanation:
        titles = ", ".join(movie.title for movie in movies)
        explanation = f"Here are your top {len(movies)} picks: {titles}."
    session.speak(explanation, voice)


def _handle_card_action(action: str, card: dict) -> None:
    if action == "save":
        ok = session.save_pick(card)
        st.toast("Saved to My Picks ❤️" if ok else "Already in My Picks ❤️")
    elif action == "view":
        st.session_state["priya_detail_card"] = card


def _maybe_open_details(tmdb, recommender) -> None:
    card = st.session_state.pop("priya_detail_card", None)
    if card:
        movie_details_dialog(card, tmdb, recommender)


# --------------------------------------------------------------------- #
# Movie details dialog
# --------------------------------------------------------------------- #
@st.dialog("Movie details", width="large")
def movie_details_dialog(card: dict, tmdb, recommender) -> None:
    card = _enrich_card_from_tmdb(card, tmdb)

    backdrop = card.get("backdrop") or ""
    if backdrop:
        st.markdown(
            f'<div class="priya-detail-backdrop"><img src="{backdrop}" alt="{card["title"]} backdrop"/></div>',
            unsafe_allow_html=True,
        )

    st.markdown(f'<div class="priya-detail-title">{card["title"]}</div>', unsafe_allow_html=True)
    meta = [f'⭐ **{card["rating"]:.1f}**' if card.get("rating") else "",
            card.get("year") or "N/A"]
    if card.get("runtime"):
        h, m = divmod(int(card["runtime"]), 60)
        meta.append(f"{h}h {m}m" if h else f"{m}m")
    st.markdown('<div class="priya-detail-meta">' + " • ".join(x for x in meta if x) + "</div>", unsafe_allow_html=True)
    if card.get("genres"):
        st.markdown(components.genre_badges(card["genres"]), unsafe_allow_html=True)

    if card.get("overview"):
        st.markdown(card["overview"])

    if card.get("match_pct") is not None:
        st.markdown(
            f'<div class="priya-reason">✨ <b>AI Match: {card["match_pct"]}%</b> from Priya\'s recommendation engine.</div>',
            unsafe_allow_html=True,
        )

    if card.get("reasons"):
        st.markdown("**💜 Why Priya picked this**")
        for reason in card["reasons"]:
            st.markdown(f"- ✓ {reason}")

    c1, c2 = st.columns([1, 1])
    if card.get("trailer_key"):
        c1.link_button("▶ Watch Trailer", f"https://www.youtube.com/watch?v={card['trailer_key']}", use_container_width=True)
    tmdb_url = card.get("tmdb_url")
    if not tmdb_url and card.get("tmdb_id"):
        media = card.get("media") or card.get("content_type") or "movie"
        section = "tv" if media == "tv" else "movie"
        tmdb_url = f"https://www.themoviedb.org/{section}/{card['tmdb_id']}"
    if tmdb_url:
        c2.link_button("🔗 TMDB", tmdb_url, use_container_width=True)

    if card.get("director"):
        st.markdown(f"**Director:** {card['director']}")
    if card.get("cast"):
        st.markdown("**Cast:** " + ", ".join(card["cast"][:6]))

    providers = card.get("providers") or []
    if providers:
        st.markdown("**Where to Watch:** " + ", ".join(providers[:6]))

    st.divider()
    col_a, col_b = st.columns(2)
    if col_a.button("❤️ Save to My Picks", use_container_width=True, type="primary"):
        ok = session.save_pick(card)
        st.toast("Saved to My Picks ❤️" if ok else "Already in My Picks ❤️")
    if col_b.button("✨ Ask Priya about this", use_container_width=True):
        st.query_params["page"] = "home"
        session.set_prompt(f"I'd love more recommendations like {card['title']}. Give me similar ones.")
        st.rerun()

    similar = _similar_cards(card, recommender, tmdb)
    if similar:
        st.markdown("**More like this**")
        result = components.card_grid(similar, tmdb, "similar")
        if result:
            _handle_card_action(result["action"], result["card"])


def _enrich_card_from_tmdb(card: dict, tmdb) -> dict:
    """Best-effort: fill cast/director/backdrop/trailer/providers from TMDB."""
    out = dict(card)
    out.setdefault("providers", [])
    media = card.get("media") or "movie"
    mid = card.get("tmdb_id")
    if not tmdb or not getattr(tmdb, "api_key", None) or not mid:
        return out
    try:
        details = tmdb.get_movie_details(mid) if media == "movie" else tmdb.get_tv_details(mid)
        credits = tmdb.get_movie_credits(mid) if media == "movie" else tmdb.get_tv_credits(mid)
        videos = tmdb.get_movie_videos(mid) if media == "movie" else tmdb.get_tv_videos(mid)
        providers = tmdb.watch_providers(media, mid)
        if details:
            out["overview"] = details.get("overview") or out.get("overview") or ""
            out["rating"] = float(details.get("vote_average") or out.get("rating") or 0.0)
            if details.get("backdrop_path"):
                out["backdrop"] = tmdb.backdrop_url(details["backdrop_path"])
            if details.get("runtime"):
                out["runtime"] = details["runtime"]
            genre_names = [g.get("name") for g in (details.get("genres") or []) if g.get("name")]
            if genre_names:
                out["genres"] = genre_names
            year = (details.get("release_date") or details.get("first_air_date") or "")[:4]
            if year:
                out["year"] = year
        if credits:
            for crew in credits.get("crew") or []:
                if crew.get("job") == "Director" and crew.get("name"):
                    out["director"] = crew["name"]
                    break
            cast = [c.get("name") for c in (credits.get("cast") or [])[:8] if c.get("name")]
            if cast:
                out["cast"] = cast
        if videos:
            for v in sorted(
                videos.get("results") or [],
                key=lambda r: (r.get("site") == "YouTube", r.get("type") == "Trailer", bool(r.get("official"))),
                reverse=True,
            ):
                if v.get("site") == "YouTube" and v.get("key"):
                    out["trailer_key"] = v["key"]
                    break
        if providers:
            res = providers.get("results") or {}
            market = res.get("US") or (next(iter(res.values())) if res else None)
            names = set()
            for kind in ("flatrate", "rent", "buy", "ads", "free"):
                for p in (market.get(kind) or []) if market else []:
                    if p.get("provider_name"):
                        names.add(p["provider_name"])
            if names:
                out["providers"] = sorted(names)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("TMDB detail enrichment failed for %s: %s", card.get("title"), exc)
    return out


def _similar_cards(card: dict, recommender, tmdb) -> list[dict]:
    try:
        if recommender is not None and recommender.ready:
            similar = recommender.similar(card["title"], top_k=4)
            return [components.movie_to_card(movie, tmdb) for movie in similar]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Similar lookup failed: %s", exc)
    return []


# --------------------------------------------------------------------- #
# Discover
# --------------------------------------------------------------------- #
SECTIONS = [
    ("🔥 Trending Movies", "trending_movies"),
    ("⭐ Popular Movies", "popular_movies"),
    ("🏆 Top Rated Movies", "top_rated_movies"),
    ("📺 Popular TV Shows", "popular_tv"),
    ("🏆 Top Rated TV Shows", "top_rated_tv"),
    ("🔥 Trending TV Shows", "trending_tv"),
    ("🚀 Sci-Fi", ("genre", "Science Fiction")),
    ("😂 Comedy", ("genre", "Comedy")),
    ("😱 Thriller", ("genre", "Thriller")),
    ("❤️ Romance", ("genre", "Romance")),
    ("👨‍👩‍👧 Family", ("genre", "Family")),
    ("🇰🇷 Korean", "korean"),
    ("🧠 Mind-Bending", "mind_bending"),
]

_MIND_BEND_KEYWORDS = (
    "mind-bend", "time travel", "twist", "psychological", "surreal",
    "identity", "dream", "memory", "parallel", "loop",
)

SEARCH_SUGGESTIONS = [
    "Inception", "Interstellar", "Breaking Bad", "Stranger Things",
    "The Dark Knight", "Parasite", "Tom Hanks", "Christopher Nolan",
]


def render_discover(recommender, tmdb) -> None:
    components.section_title("🔍 Discover")
    st.markdown(
        '<div class="priya-subtext">Search titles, or browse trending, top-rated and mood-driven rows curated by Priya.</div>',
        unsafe_allow_html=True,
    )

    if not (tmdb and getattr(tmdb, "api_key", None)):
        st.warning(
            "⚠️ TMDB API key is not configured. Add your TMDB API key to the `.env` file and restart Priya."
        )

    suggested = components.chip_buttons(SEARCH_SUGGESTIONS, "ss", per_row=4)
    if suggested:
        st.session_state["discover_search"] = suggested
        st.rerun()
    search = st.text_input("🔍 Search movies, TV shows, actors…", key="discover_search", placeholder="e.g. Inception, Breaking Bad, Tom Hanks")
    if search.strip():
        st.markdown(f'<div class="priya-section-title">🎬 Search results for "{escape(search.strip())}"</div>', unsafe_allow_html=True)
        with st.spinner("✨ Priya is searching the movie universe…"):
            cards, api_error = _search_cards(search.strip(), recommender, tmdb)
        if api_error:
            components.empty_state(
                "🎬",
                "Priya couldn't connect to the movie database right now.",
                "Please try again in a moment.",
            )
        elif cards:
            result = components.card_grid(cards, tmdb, "search")
            if result:
                _handle_card_action(result["action"], result["card"])
        else:
            components.empty_state(
                "🔍",
                f'Priya couldn\'t find anything for "{search.strip()}".',
                "Try searching for a movie, TV show, actor, or director.",
            )
    else:
        for title, kind in SECTIONS:
            cards = _section_cards(kind, recommender, tmdb)
            if not cards:
                continue
            st.markdown(f'<div class="priya-section-title">{title}</div>', unsafe_allow_html=True)
            namespace = f"sec_{kind}" if isinstance(kind, str) else f"sec_{kind[0]}_{kind[1]}"
            result = components.card_grid(cards, tmdb, namespace)
            if result:
                _handle_card_action(result["action"], result["card"])

    _maybe_open_details(tmdb, recommender)


def _search_cards(query: str, recommender, tmdb) -> tuple[list[dict], bool]:
    """Search for ``query`` and normalize to card dicts.

    Returns ``(cards, api_error)`` — ``api_error=True`` means TMDB is
    configured but the primary call failed (shown as an error, not "no
    results"). Without a TMDB key we fall back to the local dataset.
    """
    query = query.strip()
    if tmdb and getattr(tmdb, "api_key", None):
        movies, ok = tmdb.search(query, limit=20)
        if not ok:
            return [], True
        cards = [components.movie_to_card(movie, tmdb) for movie in movies]
        return cards, False
    cards = _local_search_cards(query, recommender, tmdb)
    return cards, False


def _local_search_cards(query: str, recommender, tmdb) -> list[dict]:
    """Local-dataset fallback: substring match on title/overview/cast/director."""
    cards: list[dict] = []
    if recommender is None or not recommender.ready:
        return cards
    q = query.lower()
    df = recommender.movies
    for _, row in df.head(4000).iterrows():
        haystack = f"{row.get('title', '')} {row.get('overview', '')} {row.get('cast', '')} {row.get('director', '')}".lower()
        if q in haystack:
            movie = recommender._to_movie(row.to_dict())
            cards.append(components.movie_to_card(movie, tmdb))
            if len(cards) >= 20:
                break
    return cards


def _section_cards(kind, recommender, tmdb) -> list[dict]:
    # TMDB-backed rows when available.
    if tmdb and getattr(tmdb, "api_key", None):
        tmdb_map = {
            "trending_movies": lambda: (tmdb.get_trending_movies() or {}).get("results") or [],
            "trending_tv": lambda: (tmdb.get_trending_tv() or {}).get("results") or [],
            "popular_movies": lambda: (tmdb.get_popular_movies() or {}).get("results") or [],
            "popular_tv": lambda: (tmdb.get_popular_tv() or {}).get("results") or [],
            "top_rated_movies": lambda: (tmdb.get_top_rated_movies() or {}).get("results") or [],
            "top_rated_tv": lambda: (tmdb.get_top_rated_tv() or {}).get("results") or [],
        }
        fetcher = tmdb_map.get(kind if isinstance(kind, str) else "")
        if fetcher:
            try:
                items = fetcher()[:10]
                media = "tv" if kind in ("popular_tv", "top_rated_tv", "trending_tv") else "movie"
                return [components.tmdb_item_to_card(item, tmdb, media) for item in items]
            except Exception as exc:  # pragma: no cover - network
                logger.debug("TMDB section %s failed: %s", kind, exc)

    # Local dataset fallback (always available after setup scripts).
    if recommender is None or not recommender.ready:
        return []
    try:
        movies = _local_movies(recommender, kind)
        return [components.movie_to_card(movie, tmdb) for movie in movies]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Local section %s failed: %s", kind, exc)
        return []


def _local_movies(recommender, kind, limit: int = 10) -> list:
    df = recommender.movies
    rows = []
    for _, row in df.head(4000).iterrows():
        movie = recommender._to_movie(row.to_dict())
        if _section_filter(movie, kind):
            rows.append(movie)
    sort_keys = {
        "trending_movies": lambda m: m.popularity,
        "trending_tv": lambda m: m.popularity,
        "popular_movies": lambda m: m.popularity,
        "popular_tv": lambda m: m.popularity,
        "top_rated_movies": lambda m: m.rating,
        "top_rated_tv": lambda m: m.rating,
    }
    key = sort_keys.get(kind if isinstance(kind, str) else "")
    if key:
        rows.sort(key=key, reverse=True)
    return rows[:limit]


def _section_filter(movie, kind) -> bool:
    if kind in ("trending_movies", "trending_tv"):
        return True
    if kind == "popular_movies":
        return movie.content_type == "movie"
    if kind == "popular_tv":
        return movie.content_type == "tv"
    if kind == "top_rated_movies":
        return movie.content_type == "movie" and movie.rating >= 7.5
    if kind == "top_rated_tv":
        return movie.content_type == "tv" and movie.rating >= 7.5
    if isinstance(kind, tuple) and kind[0] == "genre":
        return kind[1] in (movie.genres or [])
    if kind == "korean":
        return (movie.language or "").lower() == "ko"
    if kind == "mind_bending":
        blob = " ".join((movie.genres or []) + (movie.keywords or [])) + f" {movie.overview}".lower()
        return any(k in blob.lower() for k in _MIND_BEND_KEYWORDS)
    return False


# --------------------------------------------------------------------- #
# My Picks
# --------------------------------------------------------------------- #
def render_picks(recommender, tmdb) -> None:
    picks = session.get_picks()
    if not picks:
        components.empty_state(
            "❤️",
            "No picks yet.",
            "Browse Discover or ask Priya for recommendations, then tap ❤️ to save movies and shows here.",
        )
        _maybe_open_details(tmdb, recommender)
        return

    components.section_title("❤️ My Picks")
    st.markdown(
        '<div class="priya-subtext">Your saved movies and shows (stored in this session only).</div>',
        unsafe_allow_html=True,
    )
    result = components.manage_grid(picks, tmdb, "picks")
    if result:
        if result["action"] == "remove":
            session.remove_pick(result["card"])
            st.rerun()
        else:
            _handle_card_action("view", result["card"])
    _maybe_open_details(tmdb, recommender)


# --------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------- #
def render_about() -> None:
    avatar = theme.priya_avatar_svg(110)
    st.markdown(
        f"""
        <div class="priya-hero" style="padding-top:1.5rem">
          <div class="priya-hero-avatar"><img src="{avatar}" width="110" height="110" alt="Priya"/></div>
          <div class="priya-hero-eyebrow">About Priya</div>
          <div class="priya-hero-title">Your AI companion for<br/>what to watch next</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Priya is an AI-powered movie and TV recommendation assistant designed to
        understand what you're really in the mood for.

        Instead of simply matching genres, Priya combines conversational AI,
        semantic search, movie metadata, and personalized ranking to find
        recommendations that fit your preferences.

        **How it works**
        - 🧠 **Conversational AI (Claude)** extracts your preferences from natural language
        - 🔎 **Semantic search (Sentence Transformers + FAISS)** finds similar titles
        - ⚖️ **Hybrid ranking** blends similarity, preferences, rating and popularity
        - 🎬 **TMDB** powers posters, trailers, cast and watch providers
        - 💬 **Memory** — Priya remembers your conversation and refines as you talk

        **Voice mode** — click the mic in the sidebar and talk to Priya directly.
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Powered by Anthropic Claude • Sentence Transformers • FAISS • TMDB • Streamlit")
