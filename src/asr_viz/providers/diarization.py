from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from asr_viz.pipeline.types import SentenceCandidate


class DiarizationProvider(ABC):
    model_version: str = "unknown"

    def assign_speakers(
        self,
        sentences: list[SentenceCandidate],
        source_uri: str,
        *,
        num_speakers_override: int | None = None,
    ) -> list[SentenceCandidate]:
        return sentences


class NoOpDiarizationProvider(DiarizationProvider):
    model_version = "noop-diarizer:v1"


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_id: str
    start_ms: int
    end_ms: int


class PyannoteDiarizationProvider(DiarizationProvider):
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
                raise RuntimeError(
                    "pyannote.audio is not installed. Install with `pip install '.[diarization]'`."
                ) from exc
            try:
                self._pipeline = Pipeline.from_pretrained(self._model_name, token=self._token)
            except TypeError:
                self._pipeline = Pipeline.from_pretrained(self._model_name, use_auth_token=self._token)

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

        diarization = self._pipeline(source_uri, **kwargs)
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
