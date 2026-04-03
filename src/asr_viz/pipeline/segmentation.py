from __future__ import annotations

import re

from asr_viz.pipeline.types import ASRSegment, SentenceCandidate


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def segment_transcript(segments: list[ASRSegment]) -> list[SentenceCandidate]:
    sentences: list[SentenceCandidate] = []
    buffer_segments: list[ASRSegment] = []
    utterance_index = 0

    for segment in segments:
        buffer_segments.append(segment)
        combined_text = " ".join(item.text.strip() for item in buffer_segments if item.text.strip()).strip()
        if combined_text and _ends_sentence(combined_text):
            sentences.append(_build_sentence_candidate(buffer_segments, utterance_index))
            utterance_index += 1
            buffer_segments = []

    if buffer_segments:
        sentences.append(_build_sentence_candidate(buffer_segments, utterance_index))

    return sentences


def _ends_sentence(text: str) -> bool:
    return bool(text and text[-1] in ".!?")


def _build_sentence_candidate(buffer_segments: list[ASRSegment], utterance_index: int) -> SentenceCandidate:
    merged_text = " ".join(segment.text.strip() for segment in buffer_segments if segment.text.strip()).strip()
    pieces = [piece.strip() for piece in _SENTENCE_END_RE.split(merged_text) if piece.strip()]
    text = pieces[0] if len(pieces) == 1 else " ".join(pieces)
    avg_logprobs = [segment.avg_logprob for segment in buffer_segments if segment.avg_logprob is not None]
    no_speech_probs = [segment.no_speech_prob for segment in buffer_segments if segment.no_speech_prob is not None]
    word_probabilities = [
        word.probability
        for segment in buffer_segments
        for word in segment.words
        if word.probability is not None
    ]
    return SentenceCandidate(
        utterance_index=utterance_index,
        start_ms=buffer_segments[0].start_ms,
        end_ms=buffer_segments[-1].end_ms,
        text=text,
        source_segment_ids=[segment.segment_index for segment in buffer_segments],
        sentence_metadata={
            "source_segment_count": len(buffer_segments),
            "avg_segment_logprob": round(sum(avg_logprobs) / len(avg_logprobs), 4) if avg_logprobs else None,
            "max_no_speech_prob": round(max(no_speech_probs), 4) if no_speech_probs else None,
            "mean_word_probability": round(sum(word_probabilities) / len(word_probabilities), 4)
            if word_probabilities
            else None,
            "low_confidence_word_ratio": round(
                sum(1 for probability in word_probabilities if probability < 0.6) / len(word_probabilities),
                4,
            )
            if word_probabilities
            else None,
        },
    )
