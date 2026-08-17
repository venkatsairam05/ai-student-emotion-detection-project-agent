"""Tests for the Claude LLM service (mocked Anthropic responses)."""

from __future__ import annotations

from unittest import mock

from src.services import llm_service
from src.services.llm_service import LLMService, _extract_json_from_text


class _DummyClient:
    def __init__(self, **kwargs):
        pass


class _DummyAnthropicModule:
    Anthropic = _DummyClient


def _service_with_fake_anthropic() -> LLMService:
    with mock.patch.object(
        llm_service, "_try_import_anthropic", return_value=_DummyAnthropicModule
    ):
        return LLMService(api_key="fake-key")


def test_extract_preferences_parses_json():
    service = _service_with_fake_anthropic()
    payload = (
        '{"genres": ["Comedy"], "excluded_genres": ["Horror"], '
        '"release_year_min": 2020, "content_type": "movie", '
        '"runtime_max": 120}'
    )
    with mock.patch.object(service, "_complete", return_value=payload):
        prefs = service.extract_preferences("funny movie after 2020, no horror")
    assert "Comedy" in prefs.genres
    assert "Horror" in prefs.excluded_genres
    assert prefs.release_year_min == 2020
    assert prefs.runtime_max == 120
    assert prefs.content_type == "movie"


def test_extract_preferences_fenced_json():
    service = _service_with_fake_anthropic()
    payload = 'Here you go:\n```json\n{"genres": ["Thriller"]}\n```'
    with mock.patch.object(service, "_complete", return_value=payload):
        prefs = service.extract_preferences("thriller please")
    assert "Thriller" in prefs.genres


def test_extract_preferences_invalid_json_falls_back():
    service = _service_with_fake_anthropic()
    with mock.patch.object(service, "_complete", return_value="not json at all"):
        prefs = service.extract_preferences("whatever")
    assert prefs.is_empty()


def test_extract_preferences_no_key():
    service = LLMService(api_key="")
    assert service.available is False
    prefs = service.extract_preferences("comedy")
    assert prefs.is_empty()


def test_explanation_unavailable_uses_rule_based():
    service = LLMService(api_key="")
    text = service.generate_recommendation_explanation(
        [{"title": "Inception", "match_reasons": ["Genre: Sci-Fi"]}],
        llm_service.UserPreferences(genres=["Science Fiction"]),
    )
    assert "Inception" in text


def test_explanation_available_calls_complete():
    service = _service_with_fake_anthropic()
    with mock.patch.object(
        service, "_complete", return_value="These match your taste for sci-fi."
    ) as complete:
        text = service.generate_recommendation_explanation(
            [{"title": "Inception", "match_reasons": ["Genre: Sci-Fi"]}],
            llm_service.UserPreferences(genres=["Science Fiction"]),
        )
    assert text == "These match your taste for sci-fi."
    complete.assert_called_once()


def test_followup_unavailable_returns_helpful_text():
    service = LLMService(api_key="")
    text = service.generate_followup_response("nothing found", llm_service.UserPreferences(), [])
    assert len(text) > 0


def test_extract_json_helper():
    assert _extract_json_from_text('{"a": 1}') == {"a": 1}
    assert _extract_json_from_text('prefix {"a": 1} suffix') == {"a": 1}
    assert _extract_json_from_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json_from_text("no json here") is None
    assert _extract_json_from_text("") is None
