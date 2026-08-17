"""Claude (Anthropic) LLM service wrapper.

Three responsibilities:

1. ``extract_preferences`` - turn free-text chat into ``UserPreferences``.
2. ``generate_recommendation_explanation`` - explain engine-generated
   candidates (never inventing facts).
3. ``generate_followup_response`` - short conversational reply when the
   engine produces no candidates.

Every method degrades gracefully when the API key is missing or the API call
fails, so the app keeps working in offline/demo mode.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import ValidationError

from config.settings import get_settings
from src.models.preferences import UserPreferences
from src.utils.logging_config import get_logger

logger = get_logger("llm")


class LLMServiceError(Exception):
    """Raised when the LLM layer fails in an unrecoverable way."""


def _try_import_anthropic():
    """Import the anthropic SDK lazily so the app runs without it installed."""
    try:
        import anthropic  # type: ignore

        return anthropic
    except ImportError:
        return None


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from model output.

    Handles fences (```json ... ```), leading prose, trailing prose and stray
    trailing commas.
    """
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = None
        for i in range(start, len(text)):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return None
        text = text[start:end]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON, will fall back to defaults")
        return None


class LLMService:
    """Wrapper around the Anthropic Messages API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.anthropic_model
        self.timeout = settings.claude_timeout_seconds
        self.max_tokens = settings.claude_max_tokens
        self._anthropic = _try_import_anthropic()
        self._client: Any = None
        if self.api_key and self._anthropic:
            self._client = self._anthropic.Anthropic(
                api_key=self.api_key, timeout=self.timeout
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------ #
    # Low-level completion
    # ------------------------------------------------------------------ #
    def _complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        if not self.available:
            raise LLMServiceError("Claude client is not configured (missing API key)")
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    # ------------------------------------------------------------------ #
    # Preference extraction
    # ------------------------------------------------------------------ #
    def extract_preferences(
        self,
        user_message: str,
        conversation_summary: str = "",
    ) -> UserPreferences:
        """Extract structured preferences from a chat message.

        Falls back to an empty ``UserPreferences`` on any failure (invalid
        JSON, missing key, network error). The caller merges the result into
        the conversation state.
        """
        if not self.available:
            logger.info("LLM unavailable; using empty preferences")
            return UserPreferences()

        system_prompt = self._preference_extraction_system_prompt()
        user_prompt = (
            "Conversation summary so far:\n"
            f"{conversation_summary or '(none)'}\n\n"
            "Latest user message:\n"
            f"{user_message}\n\n"
            "Respond with ONLY a JSON object."
        )
        try:
            text = self._complete(system_prompt, user_prompt)
            payload = _extract_json_from_text(text) or {}
            prefs = UserPreferences.model_validate(payload)
            _sanitize_preferences(prefs)
            return prefs
        except (ValidationError, LLMServiceError, ValueError, TypeError) as exc:
            logger.warning("Preference extraction failed (%s); returning empty prefs", exc)
            return UserPreferences()

    # ------------------------------------------------------------------ #
    # Recommendation explanation
    # ------------------------------------------------------------------ #
    def generate_recommendation_explanation(
        self,
        candidates: list[dict],
        preferences: UserPreferences,
    ) -> str:
        """Explain engine-generated candidates.

        The model is explicitly told only to use the supplied candidate data,
        never to invent ratings/cast/facts.
        """
        if not self.available:
            return _rule_based_explanation(candidates, preferences)
        system_prompt = (
            "You are the explanation engine of a movie/TV recommendation system. "
            "You explain the *system's* choices. You NEVER invent facts such as "
            "ratings, actors, directors, plots, awards or release dates - use ONLY "
            "the candidate data supplied in the prompt. Be concise and warm."
        )
        user_prompt = (
            "User preferences:\n"
            f"{json.dumps(preferences.model_dump(exclude_none=True), indent=2)}\n\n"
            "Candidate titles from the recommendation engine:\n"
            f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            "Write 2-4 sentences explaining why these picks match the user. "
            "Reference the match reasons when available."
        )
        try:
            return self._complete(system_prompt, user_prompt)
        except LLMServiceError as exc:
            logger.warning("Explanation failed (%s); using rule-based fallback", exc)
            return _rule_based_explanation(candidates, preferences)

    # ------------------------------------------------------------------ #
    # Follow-up / chit-chat
    # ------------------------------------------------------------------ #
    def generate_followup_response(
        self,
        user_message: str,
        preferences: UserPreferences,
        conversation_history: list[dict],
    ) -> str:
        """Short conversational reply for messages that yield no candidates."""
        if not self.available:
            return (
                "I couldn't find titles matching that description. "
                "Try adding a genre, mood, year range or a 'similar to' title. "
                "You can also reset your preferences with 'reset'."
            )
        system_prompt = (
            "You are CineMatch, a helpful movie/TV recommendation assistant. "
            "Keep responses short (max 3 sentences) and never invent movie facts."
        )
        recent = conversation_history[-6:]
        user_prompt = (
            "Known user preferences:\n"
            f"{preferences.summary()}\n\n"
            "Conversation:\n"
            f"{json.dumps(recent, ensure_ascii=False, default=str)}\n\n"
            "Latest message:\n"
            f"{user_message}\n\n"
            "The recommendation engine found no matches. Respond helpfully."
        )
        try:
            return self._complete(system_prompt, user_prompt)
        except LLMServiceError as exc:
            logger.warning("Follow-up response failed (%s)", exc)
            return (
                "I couldn't find titles matching that. Please try a different "
                "description, genre, or movie to compare against."
            )

    # ------------------------------------------------------------------ #
    # Prompts
    # ------------------------------------------------------------------ #
    @staticmethod
    def _preference_extraction_system_prompt() -> str:
        return (
            "You extract structured movie/TV preferences from conversational text. "
            "Return ONLY a valid JSON object matching this exact schema (all fields "
            "optional, use empty arrays/null when not mentioned):\n"
            "{\n"
            '  "genres": ["Comedy", "Thriller"],\n'
            '  "excluded_genres": ["Horror"],\n'
            '  "languages": ["English"],\n'
            '  "mood": "funny | dark | uplifting | ...",\n'
            '  "content_type": "movie" | "tv" | "both",\n'
            '  "release_year_min": 2020,\n'
            '  "release_year_max": 2024,\n'
            '  "minimum_rating": 7.5,\n'
            '  "runtime_min": 90,\n'
            '  "runtime_max": 120,\n'
            '  "similar_to": ["Inception"],\n'
            '  "actors": ["Emma Stone"],\n'
            '  "directors": ["Christopher Nolan"],\n'
            '  "country": "USA",\n'
            '  "family_friendly": true,\n'
            '  "keywords": ["time travel"],\n'
            '  "sort_by": "relevance" | "rating" | "popularity"\n'
            "}\n"
            "Rules:\n"
            "- 'I don't want horror' -> excluded_genres: [\"Horror\"].\n"
            "- 'under 2 hours' -> runtime_max: 120.\n"
            "- 'like Inception' / 'similar to Inception' -> similar_to: [\"Inception\"].\n"
            "- Use canonical genre names (Action, Adventure, Animation, Comedy, Crime, "
            "Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, "
            "Romance, Science Fiction, Thriller, War, Western).\n"
            "- Negations must go into excluded_genres, never into genres.\n"
            "- movie -> content_type \"movie\"; series/show -> \"tv\"; unspecified -> \"both\"."
        )


# --------------------------------------------------------------------- #
# Rule-based fallbacks (used when Claude is unavailable)
# --------------------------------------------------------------------- #
def _rule_based_explanation(candidates: list[dict], preferences: UserPreferences) -> str:
    lines = ["Here's why these picks match you:"]
    for cand in candidates[:3]:
        reasons = cand.get("match_reasons") or []
        why = "; ".join(reasons) if reasons else (
            f"{cand.get('title', 'This title')} scored well on semantic match."
        )
        lines.append(f"- **{cand.get('title', 'Title')}** - {why}")
    return "\n".join(lines)


def _sanitize_preferences(prefs: UserPreferences) -> None:
    """Normalize LLM-extracted values (drop nonsense, fix content type)."""
    if prefs.content_type not in ("movie", "tv", "both"):
        prefs.content_type = "both"
    prefs.minimum_rating = max(0.0, min(10.0, float(prefs.minimum_rating or 0.0)))
    for attr in (
        "genres", "excluded_genres", "languages", "similar_to",
        "actors", "directors", "keywords",
    ):
        clean = [str(x).strip() for x in getattr(prefs, attr) if str(x).strip()]
        setattr(prefs, attr, list(dict.fromkeys(clean)))
