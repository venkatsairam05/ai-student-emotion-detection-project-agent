"""Conversation manager: owns session state, memory and history.

Stores only conversation preferences and history in-memory (Streamlit session
state) - no sensitive user information is persisted anywhere.
"""

from __future__ import annotations

from typing import Any

from src.models.preferences import UserPreferences

RESET_WORDS = {"reset", "clear", "start over", "start over", "restart"}
RESET_PHRASES = (
    "reset", "clear my preferences", "start over", "start fresh",
    "forget everything", "clear preferences",
)


class ConversationManager:
    """Wraps the ``st.session_state`` keys used by the app.

    Keys (all in-memory only):

    * ``conversation_history`` - list of {"role", "content"} messages
    * ``user_preferences`` - :class:`UserPreferences` accumulated state
    * ``recommended_movies`` - last list of :class:`Movie` recommendations
    * ``preference_summary`` - cached human-readable summary
    """

    _KEYS = (
        "conversation_history",
        "user_preferences",
        "recommended_movies",
        "preference_summary",
    )

    def __init__(self, session_state: Any) -> None:
        self._state = session_state
        self._init_state()

    def _init_state(self) -> None:
        if "conversation_history" not in self._state:
            self._state.conversation_history = []
        if "user_preferences" not in self._state:
            self._state.user_preferences = UserPreferences()
        if "recommended_movies" not in self._state:
            self._state.recommended_movies = []
        if "preference_summary" not in self._state:
            self._state.preference_summary = ""

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def preferences(self) -> UserPreferences:
        return self._state.user_preferences

    @preferences.setter
    def preferences(self, value: UserPreferences) -> None:
        self._state.user_preferences = value
        self._state.preference_summary = value.summary()

    @property
    def history(self) -> list[dict]:
        return self._state.conversation_history

    @property
    def recommendations(self) -> list:
        return self._state.recommended_movies

    @recommendations.setter
    def recommendations(self, value: list) -> None:
        self._state.recommended_movies = value

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def add_message(self, role: str, content: str) -> None:
        """Append a chat message, capping history length."""
        self._state.conversation_history.append(
            {"role": role, "content": content}
        )
        if len(self._state.conversation_history) > 60:
            self._state.conversation_history = self._state.conversation_history[-60:]

    def update_preferences(self, incoming: UserPreferences) -> bool:
        """Merge new extracted preferences into the conversation state.

        Returns True when something actually changed.
        """
        if incoming.is_empty():
            return False
        merged = self.preferences.merge(incoming)
        changed = merged.model_dump() != self.preferences.model_dump()
        self.preferences = merged
        return changed

    def reset(self) -> None:
        """Clear preferences, history and recommendations."""
        self._state.user_preferences = UserPreferences()
        self._state.conversation_history = []
        self._state.recommended_movies = []
        self._state.preference_summary = ""

    def is_reset_request(self, message: str) -> bool:
        lowered = message.strip().lower()
        return any(phrase in lowered for phrase in RESET_PHRASES)

    def history_text(self, limit: int = 6) -> str:
        """Flatten recent history into a compact summary for the LLM."""
        lines = []
        for entry in self.history[-limit:]:
            lines.append(f"{entry['role'].upper()}: {entry['content']}")
        return "\n".join(lines)
