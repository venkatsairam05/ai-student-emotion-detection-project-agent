"""Tests for the conversation manager (session memory)."""

from __future__ import annotations

from src.chatbot.conversation import ConversationManager
from src.models.preferences import UserPreferences


class _FakeSessionState(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def test_init_state():
    cm = ConversationManager(_FakeSessionState())
    assert cm.history == []
    assert cm.preferences.is_empty()
    assert cm.recommendations == []


def test_add_message_and_history_text():
    cm = ConversationManager(_FakeSessionState())
    cm.add_message("user", "I like Nolan films")
    cm.add_message("assistant", "Got it!")
    text = cm.history_text()
    assert "USER: I like Nolan films" in text
    assert "ASSISTANT: Got it!" in text


def test_update_preferences_merges():
    cm = ConversationManager(_FakeSessionState())
    changed = cm.update_preferences(UserPreferences(genres=["Drama"]))
    assert changed is True
    assert cm.preferences.genres == ["Drama"]

    changed_again = cm.update_preferences(
        UserPreferences(genres=["Drama"], excluded_genres=["Horror"])
    )
    assert changed_again is True
    assert cm.preferences.genres == ["Drama"]
    assert cm.preferences.excluded_genres == ["Horror"]


def test_update_empty_preferences_no_change():
    cm = ConversationManager(_FakeSessionState())
    assert cm.update_preferences(UserPreferences()) is False


def test_reset():
    cm = ConversationManager(_FakeSessionState())
    cm.add_message("user", "hi")
    cm.update_preferences(UserPreferences(genres=["Comedy"]))
    cm.recommendations = [1, 2, 3]
    cm.reset()
    assert cm.history == []
    assert cm.preferences.is_empty()
    assert cm.recommendations == []


def test_reset_request_detection():
    cm = ConversationManager(_FakeSessionState())
    assert cm.is_reset_request("reset")
    assert cm.is_reset_request("please clear my preferences")
    assert cm.is_reset_request("start over")
    assert not cm.is_reset_request("recommend a movie")


def test_history_is_capped():
    cm = ConversationManager(_FakeSessionState())
    for i in range(70):
        cm.add_message("user", f"msg {i}")
    assert len(cm.history) <= 60
