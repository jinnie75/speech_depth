from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
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


class FasterWhisperTranscriptionProvider(TranscriptionProvider):
    def __init__(self, model_size: str) -> None:
        self.model_version = f"faster-whisper:{model_size}"
        self._model_size = model_size

    def transcribe(
        self,
        source_uri: str,
        preferred_language: PreferredLanguage | None = None,
    ) -> TranscriptResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed") from exc

        model = WhisperModel(self._model_size)
        transcribe_kwargs = {"word_timestamps": True}
        if preferred_language and preferred_language != "auto":
            transcribe_kwargs["language"] = preferred_language
        segments, info = model.transcribe(source_uri, **transcribe_kwargs)

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
