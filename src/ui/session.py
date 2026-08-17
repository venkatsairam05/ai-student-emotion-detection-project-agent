"""Session-state helpers for PRIYA: My Picks, speech playback, prompt queue.

Everything is stored in Streamlit session state (in-memory only).
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from src.services.voice_assistant import audio_player_html
from src.utils.logging_config import get_logger

logger = get_logger("ui.session")

PICKS_KEY = "priya_picks"
PROMPT_KEY = "priya_prompt"
LAST_CARDS_KEY = "priya_last_cards"


# --------------------------------------------------------------------- #
# My Picks
# --------------------------------------------------------------------- #
def get_picks() -> list[dict]:
    if PICKS_KEY not in st.session_state:
        st.session_state[PICKS_KEY] = []
    return st.session_state[PICKS_KEY]


def save_pick(card: dict) -> bool:
    """Add a card to My Picks; returns False if it was already saved."""
    picks = get_picks()
    if any(p.get("title") == card.get("title") and p.get("media") == card.get("media") for p in picks):
        return False
    picks.append(card)
    return True


def remove_pick(card: dict) -> None:
    picks = get_picks()
    st.session_state[PICKS_KEY] = [
        p for p in picks
        if not (p.get("title") == card.get("title") and p.get("media") == card.get("media"))
    ]


# --------------------------------------------------------------------- #
# Last recommendation cards (kept across reruns so Save/View stay live)
# --------------------------------------------------------------------- #
def set_last_cards(cards: list[dict]) -> None:
    st.session_state[LAST_CARDS_KEY] = cards


def get_last_cards() -> list[dict]:
    return st.session_state.get(LAST_CARDS_KEY, [])


# --------------------------------------------------------------------- #
# Prompt queue (chips, moods, "Ask Priya" from details)
# --------------------------------------------------------------------- #
def set_prompt(text: str) -> None:
    st.session_state[PROMPT_KEY] = text


def pop_prompt() -> Optional[str]:
    return st.session_state.pop(PROMPT_KEY, None)


# --------------------------------------------------------------------- #
# Voice
# --------------------------------------------------------------------- #
def voice_enabled() -> bool:
    return bool(st.session_state.get("voice_enabled", False))


def speak(text: str, voice) -> None:
    """Synthesize and autoplay ``text`` through Priya (no-op when disabled)."""
    if not voice_enabled() or not text or not text.strip():
        return
    audio = voice.tts(text)
    if audio:
        mime = "audio/wav" if audio[:4] == b"RIFF" else "audio/mp3"
        st.components.v1.html(audio_player_html(audio, mime), height=70)


def consume_voice_input() -> Optional[str]:
    transcript = st.session_state.pop("priya_voice_input", None)
    if transcript and transcript.strip():
        return transcript.strip()
    return None
