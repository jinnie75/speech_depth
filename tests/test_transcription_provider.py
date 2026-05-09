import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from asr_viz.providers.transcription import FasterWhisperTranscriptionProvider


class FasterWhisperTranscriptionProviderTests(unittest.TestCase):
    def test_transcribe_passes_language_override_when_requested(self) -> None:
        recorded_calls: list[dict] = []

        class FakeWhisperModel:
            def __init__(self, model_size: str, **kwargs) -> None:
                self.model_size = model_size
                self.init_kwargs = kwargs

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
            def __init__(self, model_size: str, **kwargs) -> None:
                self.model_size = model_size
                self.init_kwargs = kwargs

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

    def test_transcribe_retries_same_audio_source_for_container_open_failures(self) -> None:
        recorded_calls: list[dict] = []
        sample_attempts = 0

        class FakeWhisperModel:
            def __init__(self, model_size: str, **kwargs) -> None:
                self.model_size = model_size
                self.init_kwargs = kwargs

            def transcribe(self, source_uri: str, **kwargs):
                nonlocal sample_attempts
                recorded_calls.append({"source_uri": source_uri, **kwargs})
                if source_uri.endswith("sample.wav"):
                    sample_attempts += 1
                    if sample_attempts == 1:
                        raise RuntimeError(f"Error opening '{source_uri}': Format not recognised.")
                info = types.SimpleNamespace(language="en", duration=1.0)
                return [], info

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)

        @contextmanager
        def fake_fallback(_source_uri: str):
            yield "/tmp/extracted-audio.wav"

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = f"{tmp_dir}/sample.wav"
            with open(sample_path, "wb") as handle:
                handle.write(b"not-a-real-wav")

            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                provider = FasterWhisperTranscriptionProvider("small")
                with patch.object(provider, "_fallback_transcription_source", side_effect=fake_fallback) as fallback_mock:
                    provider.transcribe(sample_path, preferred_language="auto")

        self.assertEqual(
            [call["source_uri"] for call in recorded_calls],
            [str(Path(sample_path).resolve()), str(Path(sample_path).resolve())],
        )
        fallback_mock.assert_not_called()

    def test_transcribe_extracts_audio_first_for_video_sources(self) -> None:
        recorded_calls: list[dict] = []

        class FakeWhisperModel:
            def __init__(self, model_size: str, **kwargs) -> None:
                self.model_size = model_size
                self.init_kwargs = kwargs

            def transcribe(self, source_uri: str, **kwargs):
                recorded_calls.append({"source_uri": source_uri, **kwargs})
                info = types.SimpleNamespace(language="en", duration=1.0)
                return [], info

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)

        @contextmanager
        def fake_fallback(_source_uri: str):
            yield "/tmp/extracted-audio.wav"

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = f"{tmp_dir}/sample.mov"
            with open(sample_path, "wb") as handle:
                handle.write(b"not-a-real-mov")

            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                provider = FasterWhisperTranscriptionProvider("small")
                with patch.object(provider, "_fallback_transcription_source", side_effect=fake_fallback) as fallback_mock:
                    provider.transcribe(sample_path, preferred_language="auto")

        self.assertEqual(
            [call["source_uri"] for call in recorded_calls],
            ["/tmp/extracted-audio.wav"],
        )
        fallback_mock.assert_called_once_with(str(Path(sample_path).resolve()))

    def test_transcribe_retries_transient_video_open_failures_before_succeeding(self) -> None:
        recorded_calls: list[str] = []
        extracted_attempts = 0

        class FakeWhisperModel:
            def __init__(self, model_size: str, **kwargs) -> None:
                self.model_size = model_size
                self.init_kwargs = kwargs

            def transcribe(self, source_uri: str, **kwargs):
                nonlocal extracted_attempts
                recorded_calls.append(source_uri)
                if source_uri == "/tmp/extracted-audio.wav":
                    extracted_attempts += 1
                    if extracted_attempts == 1:
                        raise RuntimeError(f"Error opening '{source_uri}': Format not recognised.")
                info = types.SimpleNamespace(language="en", duration=1.0)
                return [], info

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)

        @contextmanager
        def fake_fallback(_source_uri: str):
            yield "/tmp/extracted-audio.wav"

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = f"{tmp_dir}/sample.mov"
            with open(sample_path, "wb") as handle:
                handle.write(b"not-a-real-mov")

            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                provider = FasterWhisperTranscriptionProvider("small")
                with patch.object(provider, "_fallback_transcription_source", side_effect=fake_fallback) as fallback_mock:
                    with patch("asr_viz.providers.transcription.time.sleep") as sleep_mock:
                        provider.transcribe(sample_path, preferred_language="auto")

        self.assertEqual(recorded_calls, ["/tmp/extracted-audio.wav", "/tmp/extracted-audio.wav"])
        self.assertEqual(fallback_mock.call_count, 2)
        sleep_mock.assert_called_once()

    def test_transcribe_retries_same_audio_source_when_container_error_is_raised_during_segment_iteration(self) -> None:
        recorded_calls: list[dict] = []
        sample_attempts = 0

        class FakeWhisperModel:
            def __init__(self, model_size: str, **kwargs) -> None:
                self.model_size = model_size
                self.init_kwargs = kwargs

            def transcribe(self, source_uri: str, **kwargs):
                nonlocal sample_attempts
                recorded_calls.append({"source_uri": source_uri, **kwargs})

                def iter_segments():
                    nonlocal sample_attempts
                    if source_uri.endswith("sample.wav"):
                        sample_attempts += 1
                        if sample_attempts == 1:
                            raise RuntimeError(f"Error opening '{source_uri}': Format not recognised.")
                    return
                    yield

                info = types.SimpleNamespace(language="en", duration=1.0)
                return iter_segments(), info

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)

        @contextmanager
        def fake_fallback(_source_uri: str):
            yield "/tmp/extracted-audio.wav"

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = f"{tmp_dir}/sample.wav"
            with open(sample_path, "wb") as handle:
                handle.write(b"not-a-real-wav")

            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                provider = FasterWhisperTranscriptionProvider("small")
                with patch.object(provider, "_fallback_transcription_source", side_effect=fake_fallback) as fallback_mock:
                    provider.transcribe(sample_path, preferred_language="auto")

        self.assertEqual(
            [call["source_uri"] for call in recorded_calls],
            [str(Path(sample_path).resolve()), str(Path(sample_path).resolve())],
        )
        fallback_mock.assert_not_called()

    def test_transcribe_does_not_retry_for_non_container_errors(self) -> None:
        class FakeWhisperModel:
            def __init__(self, model_size: str, **kwargs) -> None:
                self.model_size = model_size
                self.init_kwargs = kwargs

            def transcribe(self, source_uri: str, **kwargs):
                raise RuntimeError("model weights unavailable")

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)

        @contextmanager
        def fake_fallback(_source_uri: str):
            yield "/tmp/extracted-audio.wav"

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = f"{tmp_dir}/sample.wav"
            with open(sample_path, "wb") as handle:
                handle.write(b"not-a-real-wav")

            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                provider = FasterWhisperTranscriptionProvider("small")
                with patch.object(provider, "_fallback_transcription_source", side_effect=fake_fallback) as fallback_mock:
                    with self.assertRaisesRegex(RuntimeError, "model weights unavailable"):
                        provider.transcribe(sample_path, preferred_language="auto")

        fallback_mock.assert_not_called()
