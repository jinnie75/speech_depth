from __future__ import annotations

from dataclasses import dataclass
import re

from asr_viz.pipeline.types import ASRSegment, ASRWord, SentenceCandidate


_TOKEN_RE = re.compile(r"\S+")
_CAPITALIZED_STARTER_MIN_TOKENS = 4
_CAPITALIZED_STARTER_GAP_MS = 250
_NON_TERMINAL_ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "mt.",
    "rev.",
    "fr.",
    "pres.",
    "gov.",
    "sen.",
    "rep.",
    "gen.",
    "lt.",
    "col.",
    "capt.",
    "cmdr.",
    "adm.",
    "sgt.",
    "cpl.",
    "maj.",
    "etc.",
    "e.g.",
    "i.e.",
    "vs.",
}
_SENTENCE_STARTER_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "because",
    "but",
    "can",
    "could",
    "did",
    "didn't",
    "do",
    "does",
    "don't",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "here's",
    "how",
    "i",
    "if",
    "is",
    "it",
    "it's",
    "let's",
    "maybe",
    "my",
    "no",
    "our",
    "please",
    "she",
    "should",
    "so",
    "that",
    "that's",
    "the",
    "their",
    "there",
    "there's",
    "these",
    "they",
    "this",
    "those",
    "was",
    "we",
    "well",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "would",
    "yes",
    "you",
    "your",
}


@dataclass(frozen=True)
class _WordEntry:
    word: ASRWord
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _TokenEntry:
    text: str
    char_start: int
    char_end: int
    word: _WordEntry | None


@dataclass(frozen=True)
class _SegmentEntry:
    segment: ASRSegment
    text: str
    char_start: int
    char_end: int
    words: list[_WordEntry]


def segment_transcript(segments: list[ASRSegment]) -> list[SentenceCandidate]:
    timeline = _build_timeline(segments)
    if timeline is None:
        return []

    combined_text, segment_entries = timeline
    spans = _sentence_spans(combined_text, _tokenize(combined_text, segment_entries))
    return [
        _build_sentence_candidate(
            combined_text,
            segment_entries,
            sentence_start,
            sentence_end,
            utterance_index,
        )
        for utterance_index, (sentence_start, sentence_end) in enumerate(spans)
    ]


def _build_sentence_candidate(
    combined_text: str,
    segment_entries: list[_SegmentEntry],
    sentence_start: int,
    sentence_end: int,
    utterance_index: int,
) -> SentenceCandidate:
    text = combined_text[sentence_start:sentence_end].strip()
    overlapping_segments = [
        entry
        for entry in segment_entries
        if _ranges_overlap(sentence_start, sentence_end, entry.char_start, entry.char_end)
    ]
    overlapping_words = [
        word_entry
        for entry in overlapping_segments
        for word_entry in entry.words
        if _ranges_overlap(sentence_start, sentence_end, word_entry.char_start, word_entry.char_end)
    ]
    avg_logprobs = [
        entry.segment.avg_logprob for entry in overlapping_segments if entry.segment.avg_logprob is not None
    ]
    no_speech_probs = [
        entry.segment.no_speech_prob for entry in overlapping_segments if entry.segment.no_speech_prob is not None
    ]
    word_probabilities = [
        word_entry.word.probability
        for word_entry in overlapping_words
        if word_entry.word.probability is not None
    ]
    return SentenceCandidate(
        utterance_index=utterance_index,
        start_ms=_resolve_sentence_start_ms(sentence_start, overlapping_segments, overlapping_words),
        end_ms=_resolve_sentence_end_ms(sentence_end, overlapping_segments, overlapping_words),
        text=text,
        source_segment_ids=[entry.segment.segment_index for entry in overlapping_segments],
        sentence_metadata={
            "source_segment_count": len(overlapping_segments),
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


def _build_timeline(segments: list[ASRSegment]) -> tuple[str, list[_SegmentEntry]] | None:
    entries: list[_SegmentEntry] = []
    cursor = 0

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        char_start = cursor
        char_end = char_start + len(text)
        entries.append(
            _SegmentEntry(
                segment=segment,
                text=text,
                char_start=char_start,
                char_end=char_end,
                words=_map_words_to_global_ranges(text, segment.words, char_start),
            )
        )
        cursor = char_end + 1

    if not entries:
        return None

    combined_text = " ".join(entry.text for entry in entries)
    return combined_text, entries


def _tokenize(text: str, segment_entries: list[_SegmentEntry]) -> list[_TokenEntry]:
    word_map = {
        (word_entry.char_start, word_entry.char_end): word_entry
        for segment_entry in segment_entries
        for word_entry in segment_entry.words
    }
    return [
        _TokenEntry(
            text=match.group(0),
            char_start=match.start(),
            char_end=match.end(),
            word=word_map.get((match.start(), match.end())),
        )
        for match in _TOKEN_RE.finditer(text)
    ]


def _sentence_spans(text: str, tokens: list[_TokenEntry]) -> list[tuple[int, int]]:
    if not tokens:
        return []

    spans: list[tuple[int, int]] = []
    start = 0

    for token_index, token in enumerate(tokens):
        next_token = tokens[token_index + 1] if token_index + 1 < len(tokens) else None
        if _should_end_sentence(token.text, next_token):
            spans.append((start, token.char_end))
            start = _skip_whitespace(text, token.char_end)
            continue

        if token_index == 0:
            continue

        if _should_split_before_capitalized_starter(tokens, token_index):
            previous_token = tokens[token_index - 1]
            spans.append((start, previous_token.char_end))
            start = token.char_start

    if start < len(text):
        spans.append((start, len(text)))

    return [(span_start, span_end) for span_start, span_end in spans if text[span_start:span_end].strip()]


def _map_words_to_global_ranges(text: str, words: list[ASRWord], global_offset: int) -> list[_WordEntry]:
    if not words:
        return []

    tokens = text.split()
    if len(tokens) != len(words):
        return []

    entries: list[_WordEntry] = []
    cursor = 0
    for token, word in zip(tokens, words, strict=True):
        token_start = text.find(token, cursor)
        if token_start < 0:
            return []
        token_end = token_start + len(token)
        entries.append(
            _WordEntry(
                word=word,
                char_start=global_offset + token_start,
                char_end=global_offset + token_end,
            )
        )
        cursor = token_end

    return entries


def _should_split_before_capitalized_starter(tokens: list[_TokenEntry], token_index: int) -> bool:
    token = tokens[token_index]
    previous_token = tokens[token_index - 1]
    normalized = _normalize_token(token.text)
    if normalized not in _SENTENCE_STARTER_WORDS:
        return False
    if not _starts_with_capital(token.text):
        return False
    if previous_token.text.endswith("..."):
        return False

    tokens_since_boundary = 0
    for lookback_index in range(token_index - 1, -1, -1):
        next_token = tokens[lookback_index + 1] if lookback_index + 1 < len(tokens) else None
        if _should_end_sentence(tokens[lookback_index].text, next_token):
            break
        tokens_since_boundary += 1

    if tokens_since_boundary < _CAPITALIZED_STARTER_MIN_TOKENS:
        return False

    if previous_token.word is not None and token.word is not None:
        gap_ms = token.word.word.start_ms - previous_token.word.word.end_ms
        return gap_ms >= _CAPITALIZED_STARTER_GAP_MS

    return True


def _starts_with_capital(text: str) -> bool:
    stripped = text.lstrip("\"'([{")
    return bool(stripped) and stripped[0].isupper()


def _normalize_token(text: str) -> str:
    return text.strip("\"'()[]{}").rstrip(",;:").lower()


def _skip_whitespace(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    return start


def _should_end_sentence(token_text: str, next_token: _TokenEntry | None) -> bool:
    stripped = _strip_terminal_wrapping(token_text)
    if not stripped or stripped.endswith("..."):
        return False
    if not stripped.endswith(("!", "?", ".")):
        return False
    if stripped.lower() in _NON_TERMINAL_ABBREVIATIONS and next_token is not None:
        return False
    return True


def _strip_terminal_wrapping(token_text: str) -> str:
    return token_text.rstrip("\"')]}").rstrip()


def _resolve_sentence_start_ms(
    sentence_start: int,
    overlapping_segments: list[_SegmentEntry],
    overlapping_words: list[_WordEntry],
) -> int:
    if overlapping_words:
        return overlapping_words[0].word.start_ms
    first_segment = overlapping_segments[0]
    return _interpolate_ms(first_segment, sentence_start)


def _resolve_sentence_end_ms(
    sentence_end: int,
    overlapping_segments: list[_SegmentEntry],
    overlapping_words: list[_WordEntry],
) -> int:
    if overlapping_words:
        return overlapping_words[-1].word.end_ms
    last_segment = overlapping_segments[-1]
    return _interpolate_ms(last_segment, sentence_end)


def _interpolate_ms(segment_entry: _SegmentEntry, char_offset: int) -> int:
    segment_length = max(segment_entry.char_end - segment_entry.char_start, 1)
    bounded_offset = min(max(char_offset, segment_entry.char_start), segment_entry.char_end)
    ratio = (bounded_offset - segment_entry.char_start) / segment_length
    duration_ms = segment_entry.segment.end_ms - segment_entry.segment.start_ms
    return int(round(segment_entry.segment.start_ms + duration_ms * ratio))


def _ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)
