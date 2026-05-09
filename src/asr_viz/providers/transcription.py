from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import gc
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Literal

from asr_viz.pipeline.types import ASRSegment, ASRWord, TranscriptResult

PreferredLanguage = Literal["auto", "en", "ko"]


class TranscriptionProvider(ABC):
    model_version: str = "unknown"

    @abstractmethod
    def transcribe(
        self,
        source_uri: str,
        preferred_language: PreferredLanguage | None = None,
    ) -> TranscriptResult:
        raise NotImplementedError

    def release_resources(self) -> None:
        """Allow implementations to drop heavyweight state between jobs."""


class FasterWhisperTranscriptionProvider(TranscriptionProvider):
    _VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
    _TRANSIENT_MEDIA_OPEN_RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)

    def __init__(
        self,
        model_size: str,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int | None = None,
    ) -> None:
        self.model_version = f"faster-whisper:{model_size}:{device}:{compute_type}"
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed") from exc

        init_kwargs: dict[str, str | int] = {
            "device": self._device,
            "compute_type": self._compute_type,
        }
        if self._cpu_threads is not None:
            init_kwargs["cpu_threads"] = self._cpu_threads
        self._model = WhisperModel(self._model_size, **init_kwargs)
        return self._model

    def transcribe(
        self,
        source_uri: str,
        preferred_language: PreferredLanguage | None = None,
    ) -> TranscriptResult:
        normalized_source_uri = self._normalize_source_uri(source_uri)
        transcribe_kwargs = {"word_timestamps": True}
        if preferred_language and preferred_language != "auto":
            transcribe_kwargs["language"] = preferred_language

        segments, info = self._transcribe_with_targeted_retries(normalized_source_uri, transcribe_kwargs)

        parsed_segments: list[ASRSegment] = []
        text_parts: list[str] = []
        for index, segment in enumerate(segments):
            text_parts.append(segment.text.strip())
            words = [
                ASRWord(
                    word=word.word.strip(),
                    start_ms=int(word.start * 1000),
                    end_ms=int(word.end * 1000),
                    probability=word.probability,
                )
                for word in (segment.words or [])
            ]
            parsed_segments.append(
                ASRSegment(
                    segment_index=index,
                    start_ms=int(segment.start * 1000),
                    end_ms=int(segment.end * 1000),
                    text=segment.text.strip(),
                    avg_logprob=segment.avg_logprob,
                    no_speech_prob=segment.no_speech_prob,
                    words=words,
                    raw_payload={
                        "temperature": getattr(segment, "temperature", None),
                    },
                )
            )

        return TranscriptResult(
            language_code=getattr(info, "language", None),
            full_text=" ".join(part for part in text_parts if part).strip(),
            segments=parsed_segments,
            metadata={"duration": getattr(info, "duration", None)},
        )

    def _normalize_source_uri(self, source_uri: str) -> str:
        path = Path(source_uri).expanduser()
        if path.exists():
            return str(path.resolve())
        return source_uri

    def _transcribe_with_model(self, model, source_uri: str, transcribe_kwargs: dict) -> tuple[list, object]:
        segments, info = model.transcribe(source_uri, **transcribe_kwargs)
        return list(segments), info

    def _transcribe_with_targeted_retries(self, source_uri: str, transcribe_kwargs: dict) -> tuple[list, object]:
        attempts_remaining = len(self._TRANSIENT_MEDIA_OPEN_RETRY_DELAYS_SECONDS)

        while True:
            model = self._get_model()
            try:
                if self._should_extract_audio_first(source_uri):
                    with self._fallback_transcription_source(source_uri) as extracted_source_uri:
                        if extracted_source_uri is None:
                            raise RuntimeError(f"unable to extract audio from {source_uri}")
                        return self._transcribe_with_model(model, extracted_source_uri, transcribe_kwargs)
                return self._transcribe_with_model(model, source_uri, transcribe_kwargs)
            except Exception as exc:
                if not self._should_retry_with_extracted_audio(source_uri, exc) or attempts_remaining <= 0:
                    raise
                retry_index = len(self._TRANSIENT_MEDIA_OPEN_RETRY_DELAYS_SECONDS) - attempts_remaining
                retry_delay = self._TRANSIENT_MEDIA_OPEN_RETRY_DELAYS_SECONDS[retry_index]
                attempts_remaining -= 1
                self.release_resources()
                time.sleep(retry_delay)

    def _should_extract_audio_first(self, source_uri: str) -> bool:
        path = Path(source_uri)
        return path.exists() and path.is_file() and path.suffix.lower() in self._VIDEO_SUFFIXES

    def _should_retry_with_extracted_audio(self, source_uri: str, exc: Exception) -> bool:
        path = Path(source_uri)
        if not path.exists() or not path.is_file():
            return False

        normalized_message = str(exc).lower()
        retry_markers = (
            "error opening",
            "format not recognised",
            "format not recognized",
            "invalid data found",
            "moov atom not found",
        )
        return any(marker in normalized_message for marker in retry_markers)

    @contextmanager
    def _fallback_transcription_source(self, source_uri: str):
        source_path = Path(source_uri)
        if not source_path.exists() or not source_path.is_file():
            yield None
            return

        with tempfile.TemporaryDirectory(prefix="asr-viz-transcription-") as temp_dir:
            extracted_audio_path = Path(temp_dir) / f"{source_path.stem}.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(extracted_audio_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            yield str(extracted_audio_path)

    def release_resources(self) -> None:
        self._model = None
        gc.collect()


class MockTranscriptionProvider(TranscriptionProvider):
    model_version = "mock-transcriber:v1"

    def transcribe(
        self,
        source_uri: str,
        preferred_language: PreferredLanguage | None = None,
    ) -> TranscriptResult:
        path = Path(source_uri)
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "mock transcription only supports UTF-8 text sources. "
                "Set ENABLE_MOCK_TRANSCRIPTION=false and install faster-whisper to process audio/video media."
            ) from exc
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            lines = ["No speech content detected."]

        segments: list[ASRSegment] = []
        start_ms = 0
        for index, line in enumerate(lines):
            end_ms = start_ms + max(len(line.split()), 1) * 500
            words = []
            cursor = start_ms
            for token in line.split():
                token_end = cursor + 450
                words.append(ASRWord(word=token, start_ms=cursor, end_ms=token_end, probability=0.9))
                cursor = token_end
            segments.append(
                ASRSegment(
                    segment_index=index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=line,
                    avg_logprob=-0.2,
                    no_speech_prob=0.01,
                    words=words,
                    raw_payload={"mock": True},
                )
            )
            start_ms = end_ms

        return TranscriptResult(
            language_code=preferred_language if preferred_language in {"en", "ko"} else "en",
            full_text=" ".join(lines),
            segments=segments,
            metadata={"mock_source": True},
        )
