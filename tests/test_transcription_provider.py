import sys
import types
import unittest
from unittest.mock import patch

from asr_viz.providers.transcription import FasterWhisperTranscriptionProvider


class FasterWhisperTranscriptionProviderTests(unittest.TestCase):
    def test_transcribe_passes_language_override_when_requested(self) -> None:
        recorded_calls: list[dict] = []

        class FakeWhisperModel:
            def __init__(self, model_size: str) -> None:
                self.model_size = model_size

            def transcribe(self, source_uri: str, **kwargs):
                recorded_calls.append({"source_uri": source_uri, **kwargs})
                info = types.SimpleNamespace(language="ko", duration=1.0)
                return [], info

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)

        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            provider = FasterWhisperTranscriptionProvider("small")
            provider.transcribe("/tmp/sample.wav", preferred_language="ko")

        self.assertEqual(recorded_calls[0]["language"], "ko")
        self.assertTrue(recorded_calls[0]["word_timestamps"])

    def test_transcribe_omits_language_override_for_auto(self) -> None:
        recorded_calls: list[dict] = []

        class FakeWhisperModel:
            def __init__(self, model_size: str) -> None:
                self.model_size = model_size

            def transcribe(self, source_uri: str, **kwargs):
                recorded_calls.append({"source_uri": source_uri, **kwargs})
                info = types.SimpleNamespace(language="en", duration=1.0)
                return [], info

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)

        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            provider = FasterWhisperTranscriptionProvider("small")
            provider.transcribe("/tmp/sample.wav", preferred_language="auto")

        self.assertNotIn("language", recorded_calls[0])
        self.assertTrue(recorded_calls[0]["word_timestamps"])
