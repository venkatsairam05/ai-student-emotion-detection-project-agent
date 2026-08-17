"""Voice assistant service ("Priya").

Provides:
    * Text-to-speech  - edge-tts (natural female voices) with a pyttsx3
                        (offline SAPI5) fallback. Returns audio bytes.
    * Speech-to-text  - SpeechRecognition (Google Web Speech by default)
                        over browser-recorded WAV bytes.
    * ``audio_player_html`` - base64 <audio autoplay> snippet for Streamlit.

Every public method degrades gracefully: ``None`` is returned when a backend
is missing or a call fails, so the rest of the app is never blocked.
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
import time
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from src.utils.logging_config import get_logger

logger = get_logger("voice")

_INTRO = (
    "Hi, I'm Priya, your movie and TV assistant! "
    "Tell me what you feel like watching and I'll find it for you."
)
_NAME_REPLY = "My name is Priya. I'm your personal movie and TV recommendation assistant."


class VoiceAssistant:
    """TTS + STT wrapper. Never raises; returns ``None`` on failure."""

    def __init__(self, name: Optional[str] = None) -> None:
        settings = get_settings()
        self.name: str = (name or settings.voice_assistant_name).strip() or "Priya"
        self.tts_voice: str = settings.tts_voice
        self.tts_voice_alt: str = settings.tts_voice_alt
        self.tts_speed: float = settings.tts_speed
        self.tts_timeout: float = settings.tts_timeout
        self.stt_language: str = settings.stt_language
        self.stt_api: str = settings.stt_google_api
        self._edge_tts = self._import_optional("edge_tts")
        self._speech_recognition = self._import_optional("speech_recognition")
        self._pyttsx3 = self._import_optional("pyttsx3")

    @staticmethod
    def _import_optional(module_name: str):
        try:
            return __import__(module_name)
        except ImportError:
            return None

    # ------------------------------------------------------------------ #
    # Capability flags
    # ------------------------------------------------------------------ #
    @property
    def has_online_tts(self) -> bool:
        return self._edge_tts is not None

    @property
    def has_offline_tts(self) -> bool:
        return self._pyttsx3 is not None

    @property
    def has_stt(self) -> bool:
        return self._speech_recognition is not None

    # ------------------------------------------------------------------ #
    # Text-to-speech
    # ------------------------------------------------------------------ #
    def tts(self, text: str) -> Optional[bytes]:
        """Synthesize speech and return audio bytes (mp3 or wav).

        Priority: edge-tts (primary voice, then alt voice) -> pyttsx3.
        """
        if not text or not text.strip():
            return None
        for voice in (self.tts_voice, self.tts_voice_alt):
            audio = self._edge_tts_speak(text, voice)
            if audio:
                return audio
        return self._pyttsx3_speak(text)

    def _edge_tts_speak(self, text: str, voice: str) -> Optional[bytes]:
        if not self.has_online_tts:
            return None
        try:
            tmp = Path(tempfile.gettempdir()) / f"priya_tts_{int(time.time() * 1000)}.mp3"
            communicate = self._edge_tts.Communicate(text, voice=voice, rate=f"+{int((self.tts_speed - 1) * 100)}%" if self.tts_speed else "+0%")

            async def _save() -> None:
                await communicate.save(str(tmp))

            asyncio.run(asyncio.wait_for(_save(), timeout=self.tts_timeout))
            if not tmp.exists():
                return None
            data = tmp.read_bytes()
            tmp.unlink(missing_ok=True)
            if data:
                logger.info("edge-tts synthesized %d bytes (voice=%s)", len(data), voice)
                return data
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("edge-tts failed with voice %s: %s", voice, exc)
        return None

    def _pyttsx3_speak(self, text: str) -> Optional[bytes]:
        if not self.has_offline_tts:
            return None
        try:
            tmp = Path(tempfile.gettempdir()) / f"priya_tts_{int(time.time() * 1000)}.wav"
            engine = self._pyttsx3.init()
            engine.setProperty("rate", int(170 * self.tts_speed))
            self._select_female_voice(engine)
            engine.save_to_file(text, str(tmp))
            engine.runAndWait()
            engine.stop()
            if not tmp.exists():
                return None
            data = tmp.read_bytes()
            tmp.unlink(missing_ok=True)
            if data:
                logger.info("pyttsx3 synthesized %d bytes (offline)", len(data))
                return data
        except Exception as exc:  # pragma: no cover - engine dependent
            logger.warning("pyttsx3 failed: %s", exc)
        return None

    def _select_female_voice(self, engine) -> None:
        """Try to pick a female SAPI5 voice (fallback = default)."""
        try:
            voices = engine.getProperty("voices") or []
            settings = get_settings()
            keywords = settings.tts_fallback_voice_keywords
            for voice in voices:
                name = str(getattr(voice, "name", "") or "").lower()
                if any(k in name for k in keywords):
                    engine.setProperty("voice", voice.id)
                    logger.info("Selected fallback female voice: %s", name)
                    return
        except Exception as exc:  # pragma: no cover - engine dependent
            logger.debug("Could not select female voice: %s", exc)

    # ------------------------------------------------------------------ #
    # Speech-to-text
    # ------------------------------------------------------------------ #
    def stt(self, audio_bytes: bytes, sample_rate: int = 44100) -> Optional[str]:
        """Transcribe WAV audio bytes; returns text or ``None``."""
        if not self.has_stt or not audio_bytes:
            return None
        audio_data = self._to_audio_data(audio_bytes, sample_rate)
        if audio_data is None:
            return None
        recognizer = self._speech_recognition.Recognizer()
        try:
            text = recognizer.recognize_google(audio_data, language=self.stt_language)
            return text.strip() or None
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Speech-to-text failed: %s", exc)
            return None

    def _to_audio_data(self, audio_bytes: bytes, sample_rate: int):
        """Convert WAV bytes (or raw PCM) into an ``AudioData`` object."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            with self._speech_recognition.AudioFile(tmp_path) as source:
                recognizer = self._speech_recognition.Recognizer()
                data = recognizer.record(source)
            Path(tmp_path).unlink(missing_ok=True)
            return data
        except Exception as exc:  # pragma: no cover - corrupt audio
            logger.debug("AudioFile path failed (%s); trying raw PCM", exc)
        try:
            # Assume raw 16-bit PCM if the WAV header path didn't work.
            return self._speech_recognition.AudioData(
                audio_bytes, sample_rate=sample_rate, sample_width=2
            )
        except Exception:  # pragma: no cover - defensive
            return None

    # ------------------------------------------------------------------ #
    # Persona helpers
    # ------------------------------------------------------------------ #
    def intro(self) -> str:
        return _INTRO.replace("your movie and TV assistant", f"your {self.name.lower()}")

    @property
    def reply_on_name(self) -> str:
        return _NAME_REPLY.replace("My name is Priya", f"My name is {self.name}")


def audio_player_html(audio_bytes: bytes, mime: str = "audio/mp3") -> str:
    """Base64-encoded autoplaying HTML5 audio player for Streamlit."""
    if not audio_bytes:
        return ""
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return (
        f'<audio autoplay controls src="data:{mime};base64,{encoded}" '
        f'style="width:100%;margin-top:8px"></audio>'
    )
