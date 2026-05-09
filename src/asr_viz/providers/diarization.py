from __future__ import annotations

from abc import ABC
from contextlib import contextmanager
import gc
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from asr_viz.pipeline.types import SentenceCandidate


class DiarizationProvider(ABC):
    model_version: str = "unknown"

    @property
    def is_available(self) -> bool:
        return True

    def assign_speakers(
        self,
        sentences: list[SentenceCandidate],
        source_uri: str,
        *,
        num_speakers_override: int | None = None,
    ) -> list[SentenceCandidate]:
        return sentences

    def release_resources(self) -> None:
        """Allow implementations to drop heavyweight state between jobs."""


class NoOpDiarizationProvider(DiarizationProvider):
    model_version = "noop-diarizer:v1"

    @property
    def is_available(self) -> bool:
        return False


class DiarizationUnavailableError(RuntimeError):
    """Raised when diarization cannot run because the provider is unavailable."""

    def __init__(self, message: str, *, original_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.original_exception_type = type(original_exception).__name__ if original_exception is not None else None
        self.original_exception_message = str(original_exception) if original_exception is not None else None


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_id: str
    start_ms: int
    end_ms: int


class PyannoteDiarizationProvider(DiarizationProvider):
    _VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}

    def __init__(
        self,
        *,
        model_name: str,
        token: str,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        self.model_version = model_name
        self._model_name = model_name
        self._token = token
        self._num_speakers = num_speakers
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._pipeline = None

    def assign_speakers(
        self,
        sentences: list[SentenceCandidate],
        source_uri: str,
        *,
        num_speakers_override: int | None = None,
    ) -> list[SentenceCandidate]:
        if not sentences:
            return sentences

        turns = self._extract_turns(source_uri, num_speakers_override=num_speakers_override)
        return assign_speakers_by_overlap(sentences, turns)

    def _extract_turns(self, source_uri: str, *, num_speakers_override: int | None = None) -> list[SpeakerTurn]:
        if self._pipeline is None:
            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise DiarizationUnavailableError(
                    "pyannote.audio is not installed. Install with `pip install '.[diarization]'`."
                ) from exc
            except Exception as exc:
                raise _normalize_diarization_import_error(exc, model_name=self._model_name) from exc
            try:
                self._pipeline = Pipeline.from_pretrained(self._model_name, token=self._token)
            except TypeError:
                try:
                    self._pipeline = Pipeline.from_pretrained(self._model_name, use_auth_token=self._token)
                except Exception as exc:  # pragma: no cover - depends on optional dependency behavior
                    raise _normalize_diarization_error(exc, model_name=self._model_name) from exc
            except Exception as exc:  # pragma: no cover - depends on optional dependency behavior
                raise _normalize_diarization_error(exc, model_name=self._model_name) from exc

        kwargs = {}
        if num_speakers_override is not None:
            kwargs["num_speakers"] = num_speakers_override
        elif self._num_speakers is not None:
            kwargs["num_speakers"] = self._num_speakers
        else:
            if self._min_speakers is not None:
                kwargs["min_speakers"] = self._min_speakers
            if self._max_speakers is not None:
                kwargs["max_speakers"] = self._max_speakers

        try:
            with self._pipeline_source(source_uri) as pipeline_source_uri:
                diarization = self._pipeline(pipeline_source_uri, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on optional dependency behavior
            raise _normalize_diarization_error(exc, model_name=self._model_name) from exc
        turns: list[SpeakerTurn] = []
        for segment, _, speaker in _iter_diarization_tracks(diarization):
            turns.append(
                SpeakerTurn(
                    speaker_id=str(speaker),
                    start_ms=int(segment.start * 1000),
                    end_ms=int(segment.end * 1000),
                )
            )
        return turns

    def release_resources(self) -> None:
        self._pipeline = None
        gc.collect()

    def _pipeline_source(self, source_uri: str):
        if not self._should_extract_audio_first(source_uri):
            return _passthrough_context(source_uri)
        normalized_source_uri = self._normalize_source_uri(source_uri)
        return self._fallback_diarization_source(normalized_source_uri)

    def _normalize_source_uri(self, source_uri: str) -> str:
        path = Path(source_uri).expanduser()
        if path.exists():
            return str(path.resolve())
        return source_uri

    def _should_extract_audio_first(self, source_uri: str) -> bool:
        path = Path(source_uri)
        return path.exists() and path.is_file() and path.suffix.lower() in self._VIDEO_SUFFIXES

    @contextmanager
    def _fallback_diarization_source(self, source_uri: str):
        source_path = Path(source_uri)
        if not source_path.exists() or not source_path.is_file():
            yield source_uri
            return

        with tempfile.TemporaryDirectory(prefix="asr-viz-diarization-") as temp_dir:
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


@contextmanager
def _passthrough_context(value: str):
    yield value


def assign_speakers_by_overlap(
    sentences: list[SentenceCandidate],
    speaker_turns: list[SpeakerTurn],
) -> list[SentenceCandidate]:
    assigned: list[SentenceCandidate] = []

    for sentence in sentences:
        overlaps: dict[str, int] = {}
        for turn in speaker_turns:
            overlap_ms = _overlap_ms(sentence.start_ms, sentence.end_ms, turn.start_ms, turn.end_ms)
            if overlap_ms <= 0:
                continue
            overlaps[turn.speaker_id] = overlaps.get(turn.speaker_id, 0) + overlap_ms

        if not overlaps:
            assigned.append(sentence)
            continue

        sorted_overlaps = sorted(overlaps.items(), key=lambda item: (-item[1], item[0]))
        best_speaker, best_ms = sorted_overlaps[0]
        total_ms = max(sentence.end_ms - sentence.start_ms, 1)
        confidence = min(best_ms / total_ms, 1.0)
        assigned.append(
            sentence.model_copy(
                update={
                    "speaker_id": best_speaker,
                    "speaker_confidence": round(confidence, 4),
                    "sentence_metadata": {
                        **sentence.sentence_metadata,
                        "speaker_overlap_ms": overlaps,
                    },
                }
            )
        )

    return assigned


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _iter_diarization_tracks(diarization):
    if hasattr(diarization, "itertracks"):
        return diarization.itertracks(yield_label=True)

    for attr in ("speaker_diarization", "annotation", "diarization", "output"):
        candidate = getattr(diarization, attr, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate.itertracks(yield_label=True)

    raise RuntimeError(
        f"unsupported diarization output type: {type(diarization).__name__}. "
        "Expected an object exposing itertracks() or a wrapped annotation."
    )


def _normalize_diarization_error(exc: Exception, *, model_name: str) -> Exception:
    message = str(exc)
    lowered = message.lower()
    access_error_markers = (
        "401",
        "403",
        "gated repo",
        "restricted",
        "access to model",
        "cannot access gated repo",
        "must have access",
        "please log in",
        "use_auth_token",
        "token",
        "authentication",
        "authorized",
        "unauthorized",
        "forbidden",
    )
    if any(marker in lowered for marker in access_error_markers):
        return DiarizationUnavailableError(
            f"Diarization model `{model_name}` is unavailable. "
            "Confirm that the Hugging Face token is set and that the account has accepted access to the gated model.",
            original_exception=exc,
        )
    return exc


def _normalize_diarization_import_error(exc: Exception, *, model_name: str) -> DiarizationUnavailableError:
    message = str(exc)
    lowered = message.lower()
    if "numpy" in lowered or "_array_api" in lowered:
        return DiarizationUnavailableError(
            f"Diarization model `{model_name}` is unavailable because the pyannote/torch runtime is not "
            "compatible with the installed NumPy version. Pin `numpy<2` for the worker environment.",
            original_exception=exc,
        )
    return DiarizationUnavailableError(
        f"Diarization model `{model_name}` is unavailable because pyannote.audio failed to import or initialize.",
        original_exception=exc,
    )
