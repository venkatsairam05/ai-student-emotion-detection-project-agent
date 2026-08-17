"""Tests for the session-state helpers (My Picks, prompt queue, last cards)."""

from __future__ import annotations

import streamlit as st

from src.ui import session


class _FakeSessionState(dict):
    def get(self, key, default=None):
        return super().get(key, default)

    def pop(self, key, default=None):
        return super().pop(key, default)


class _FakeComponents:
    def __init__(self):
        self.html_calls = []

    def html(self, markup, height=None):
        self.html_calls.append((markup, height))


class _FakeVoice:
    def __init__(self, audio):
        self._audio = audio

    def tts(self, text):
        return self._audio


def _install_fakes(monkeypatch):
    state = _FakeSessionState()
    components = _FakeComponents()
    fake_st = st
    monkeypatch.setattr(fake_st, "session_state", state)
    monkeypatch.setattr(fake_st, "components", type("C", (), {"v1": components})())
    return state, components


def _card(title="Groundhog Day", media="movie"):
    return {"title": title, "media": media, "year": "1993"}


def test_save_pick_adds_card(monkeypatch):
    state, _ = _install_fakes(monkeypatch)
    assert session.save_pick(_card()) is True
    assert len(session.get_picks()) == 1


def test_save_pick_is_idempotent(monkeypatch):
    state, _ = _install_fakes(monkeypatch)
    assert session.save_pick(_card()) is True
    assert session.save_pick(_card()) is False
    assert len(session.get_picks()) == 1


def test_same_title_different_media_can_both_be_saved(monkeypatch):
    state, _ = _install_fakes(monkeypatch)
    assert session.save_pick(_card("Twilight", "movie")) is True
    assert session.save_pick(_card("Twilight", "tv")) is True
    assert len(session.get_picks()) == 2


def test_remove_pick_only_removes_matching(monkeypatch):
    state, _ = _install_fakes(monkeypatch)
    session.save_pick(_card("A"))
    session.save_pick(_card("B"))
    session.remove_pick(_card("A"))
    titles = [p["title"] for p in session.get_picks()]
    assert titles == ["B"]


def test_get_picks_initializes_empty(monkeypatch):
    state, _ = _install_fakes(monkeypatch)
    assert session.get_picks() == []


def test_prompt_queue_set_and_pop(monkeypatch):
    state, _ = _install_fakes(monkeypatch)
    assert session.pop_prompt() is None
    session.set_prompt("hello")
    assert session.pop_prompt() == "hello"
    assert session.pop_prompt() is None


def test_last_cards_round_trip(monkeypatch):
    state, _ = _install_fakes(monkeypatch)
    assert session.get_last_cards() == []
    cards = [_card("A"), _card("B")]
    session.set_last_cards(cards)
    assert session.get_last_cards() == cards
    session.set_last_cards([])
    assert session.get_last_cards() == []


def test_voice_enabled_defaults_off(monkeypatch):
    state, _ = _install_fakes(monkeypatch)
    assert session.voice_enabled() is False
    state["voice_enabled"] = True
    assert session.voice_enabled() is True


def test_speak_synthesizes_and_renders_audio(monkeypatch):
    state, components = _install_fakes(monkeypatch)
    state["voice_enabled"] = True
    voice = _FakeVoice(b"RIFF fake wav data")
    session.speak("Hello Priya", voice)
    assert len(components.html_calls) == 1
    markup, height = components.html_calls[0]
    assert "audio/wav" in markup
    assert height == 70


def test_speak_noop_when_disabled(monkeypatch):
    state, components = _install_fakes(monkeypatch)
    session.speak("Hi", _FakeVoice(b"RIFF"))
    assert components.html_calls == []
