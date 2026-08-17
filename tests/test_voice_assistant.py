"""Tests for the Priya voice assistant (TTS + STT), all backends mocked."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from src.services.voice_assistant import VoiceAssistant, audio_player_html


class _FakeEdgeTTS:
    """Stand-in for edge_tts: save() writes an mp3 file."""

    def __init__(self, text, voice, rate=""):
        self.text = text
        self.voice = voice

    async def save(self, path):
        Path(path).write_bytes(b"fake-mp3-bytes")


def _assistant_with_fake_backends(tts=True, stt=True, pyttsx3=True):
    fake_edge = type("M", (), {"Communicate": _FakeEdgeTTS}) if tts else None
    fake_stt = mock.MagicMock() if stt else None
    fake_pyttsx3 = mock.MagicMock() if pyttsx3 else None
    with mock.patch.object(VoiceAssistant, "_import_optional", side_effect=[
        fake_edge, fake_stt, fake_pyttsx3,
    ]):
        return VoiceAssistant(name="Priya")


def test_name_and_persona():
    assistant = VoiceAssistant(name="Priya")
    assert assistant.name == "Priya"
    assert "Priya" in assistant.intro()
    assert "Priya" in assistant.reply_on_name


def test_tts_with_edge_returns_mp3():
    assistant = _assistant_with_fake_backends()
    audio = assistant.tts("Hello there")
    assert audio == b"fake-mp3-bytes"
    # Primary voice used.
    assert assistant.tts_voice == "en-IN-PriyaNeural"


def test_tts_empty_text_returns_none():
    assistant = _assistant_with_fake_backends()
    assert assistant.tts("   ") is None


def test_tts_falls_back_to_pyttsx3_when_edge_unavailable():
    fake_edge = mock.MagicMock()
    fake_edge.Communicate = mock.MagicMock(
        side_effect=RuntimeError("network down")
    )
    fake_stt = mock.MagicMock()
    fake_pyttsx3 = mock.MagicMock()

    class _Engine:
        def __init__(self):
            self.calls = []

        def setProperty(self, *a):
            self.calls.append(("set", a))

        def getProperty(self, name):
            if name == "voices":
                return [type("V", (), {"id": "v0", "name": "Microsoft Zira Desktop"})()]
            return 170

        def save_to_file(self, text, path):
            Path(path).write_bytes(b"fake-wav-bytes")

        def runAndWait(self):
            pass

        def stop(self):
            pass

    fake_pyttsx3.init = mock.MagicMock(return_value=_Engine())
    with mock.patch.object(VoiceAssistant, "_import_optional", side_effect=[
        fake_edge, fake_stt, fake_pyttsx3,
    ]):
        assistant = VoiceAssistant(name="Priya")
    audio = assistant.tts("offline hello")
    assert audio == b"fake-wav-bytes"


def test_tts_returns_none_when_no_backends():
    assistant = _assistant_with_fake_backends(tts=False, pyttsx3=False)
    assert assistant.tts("hello") is None
    assert assistant.has_online_tts is False
    assert assistant.has_offline_tts is False


def test_stt_returns_transcript():
    assistant = _assistant_with_fake_backends()
    fake_stt = assistant._speech_recognition
    fake_stt.Recognizer.return_value.recognize_google.return_value = "I want a sci-fi movie"
    fake_stt.AudioFile.return_value.__enter__.return_value = object()

    class _AudioData:
        pass

    fake_stt.AudioFile.return_value.__enter__.return_value = _AudioData()
    with mock.patch("tempfile.NamedTemporaryFile") as ntf:
        ntf.return_value.__enter__.return_value = mock.MagicMock(name="tmp")
        fake_stt.Recognizer.return_value.record.return_value = _AudioData()
        text = assistant.stt(b"wav-data", 44100)
    assert text == "I want a sci-fi movie"


def test_stt_returns_none_without_backend():
    assistant = _assistant_with_fake_backends(stt=False)
    assert assistant.stt(b"data", 44100) is None


def test_stt_returns_none_on_failure():
    assistant = _assistant_with_fake_backends()
    fake_stt = assistant._speech_recognition
    fake_stt.Recognizer.return_value.recognize_google.side_effect = RuntimeError("no speech")
    fake_stt.AudioFile.return_value.__enter__.return_value = object()
    with mock.patch("tempfile.NamedTemporaryFile") as ntf:
        ntf.return_value.__enter__.return_value = mock.MagicMock(name="tmp")
        fake_stt.Recognizer.return_value.record.return_value = mock.MagicMock()
        assert assistant.stt(b"wav-data", 44100) is None


def test_audio_player_html():
    assert audio_player_html(b"") == ""
    html = audio_player_html(b"\x00\x01\x02")
    assert "audio/mp3" in html
    assert "autoplay" in html
    assert "base64" in html
