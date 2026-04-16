from __future__ import annotations

from typing import Any

from asr_viz.models.transcript import SentenceUnit, Transcript

FALLBACK_SPEAKER_ID = "UNKNOWN_SPEAKER"
FALLBACK_SPEAKER_LABEL = "Speaker"
MARGIN_NOTE_SETTLE_DURATION_MS = 3000
MARGIN_NOTE_DELAY_MS = 120


def update_transcript_archive_preview(transcript: Transcript) -> dict[str, Any] | None:
    preview = build_archive_preview(transcript)
    transcript.transcript_metadata = {
        **(transcript.transcript_metadata or {}),
        "archive_preview": preview,
    }
    return preview


def build_archive_preview(transcript: Transcript) -> dict[str, Any] | None:
    utterances = _build_utterances(transcript)
    if not utterances:
        return None

    speakers = _build_speaker_summaries(utterances, transcript.speaker_labels)
    if not speakers:
        return None

    margin_notes = _build_margin_notes(utterances)
    final_snapshot_time_ms = _get_conversation_final_snapshot_time_ms(utterances, margin_notes)
    final_active_utterance = _find_active_utterance(utterances, final_snapshot_time_ms)
    final_active_playback = _get_utterance_playback_state(final_active_utterance, final_snapshot_time_ms)
    active_speaker_id = final_active_utterance["speaker_id"] if final_active_utterance else None

    return {
        "speakers": [
            {
                "id": speaker["id"],
                "label": speaker["label"],
                "side": "left" if index % 2 == 0 else "right",
                "opacity": 1 if active_speaker_id == speaker["id"] else 0.5,
                "total_duration_ms": speaker["total_duration_ms"],
                "average_politeness": speaker["average_politeness"],
            }
            for index, speaker in enumerate(speakers)
        ],
        "utterances": [
            {
                "id": utterance["id"],
                "speaker_id": utterance["speaker_id"],
                "politeness_score": utterance["politeness_score"],
                "contour_signal_count": float(len(utterance["contour_signals"])),
                "progress": 1.0,
                "order": index,
            }
            for index, utterance in enumerate(utterances)
        ],
        "active_speaker_id": active_speaker_id,
        "active_transcript": _build_active_transcript(
            final_active_utterance,
            final_active_playback["visible_word_count"] if final_active_playback else 0,
            final_snapshot_time_ms,
        ),
        "margin_notes": margin_notes,
        "current_time_ms": final_snapshot_time_ms,
    }


def _build_utterances(transcript: Transcript) -> list[dict[str, Any]]:
    utterances: list[dict[str, Any]] = []

    for sentence in sorted(
        transcript.sentence_units,
        key=lambda item: (item.start_ms, item.end_ms, item.utterance_index),
    ):
        text = sentence.display_text or ""
        word_ranges = _extract_word_ranges(text)
        word_timings = _build_fallback_word_timings(word_ranges, sentence.start_ms, sentence.end_ms)
        analysis_payload = sentence.analysis_result.analysis_payload if sentence.analysis_result is not None else {}
        contour_ranges = _find_match_ranges(text, _extract_analysis_matches(analysis_payload.get("hedging")))
        contour_signals = [
            _resolve_range_timing(match_range, text, word_ranges, word_timings, sentence.start_ms, sentence.end_ms)
            for match_range in contour_ranges
        ]

        utterances.append(
            {
                "id": sentence.id,
                "speaker_id": sentence.display_speaker_id or FALLBACK_SPEAKER_ID,
                "text": text,
                "start_ms": sentence.start_ms,
                "end_ms": sentence.end_ms,
                "duration_ms": max(1, sentence.end_ms - sentence.start_ms),
                "politeness_score": (
                    float(sentence.analysis_result.politeness_score) if sentence.analysis_result is not None else 0.5
                ),
                "analysis_payload": analysis_payload,
                "word_ranges": word_ranges,
                "word_timings": word_timings,
                "contour_signals": contour_signals,
            }
        )

    return utterances


def _build_speaker_summaries(
    utterances: list[dict[str, Any]],
    speaker_labels: dict[str, str],
) -> list[dict[str, Any]]:
    first_seen: dict[str, int] = {}
    total_duration_ms: dict[str, int] = {}
    politeness_totals: dict[str, float] = {}
    politeness_counts: dict[str, int] = {}

    for utterance in utterances:
        speaker_id = utterance["speaker_id"]
        first_seen[speaker_id] = min(first_seen.get(speaker_id, utterance["start_ms"]), utterance["start_ms"])
        total_duration_ms[speaker_id] = total_duration_ms.get(speaker_id, 0) + utterance["duration_ms"]
        politeness_totals[speaker_id] = politeness_totals.get(speaker_id, 0) + utterance["politeness_score"] * 10
        politeness_counts[speaker_id] = politeness_counts.get(speaker_id, 0) + 1

    ordered = sorted(first_seen.items(), key=lambda item: (item[1], item[0]))
    return [
        {
            "id": speaker_id,
            "label": _format_speaker_label(speaker_id, index, speaker_labels),
            "first_seen_ms": first_seen_ms,
            "total_duration_ms": total_duration_ms.get(speaker_id, 0),
            "average_politeness": politeness_totals.get(speaker_id, 0) / max(politeness_counts.get(speaker_id, 1), 1),
        }
        for index, (speaker_id, first_seen_ms) in enumerate(ordered)
    ]


def _build_margin_notes(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    margin_notes: list[dict[str, Any]] = []

    for utterance in utterances:
        clause_ranges = _build_clause_ranges(
            utterance["text"],
            _find_match_ranges(utterance["text"], _extract_analysis_matches(utterance["analysis_payload"].get("substance"))),
        )
        for index, clause_range in enumerate(clause_ranges):
            timing = _resolve_range_timing(
                clause_range,
                utterance["text"],
                utterance["word_ranges"],
                utterance["word_timings"],
                utterance["start_ms"],
                utterance["end_ms"],
            )
            note_text = utterance["text"][clause_range["start"]:clause_range["end"]].strip()
            if not note_text:
                continue
            margin_notes.append(
                {
                    "id": f'{utterance["id"]}-substance-clause-{index}-{clause_range["start"]}-{clause_range["end"]}',
                    "text": note_text,
                    "speaker_id": utterance["speaker_id"],
                    "utterance_id": utterance["id"],
                    "appear_at_ms": timing["end_ms"] + MARGIN_NOTE_DELAY_MS,
                    "source_start": clause_range["start"],
                    "source_end": clause_range["end"],
                    "settle_duration_ms": MARGIN_NOTE_SETTLE_DURATION_MS,
                }
            )

    return margin_notes


def _build_active_transcript(
    utterance: dict[str, Any] | None,
    visible_word_count: int,
    current_time_ms: int,
) -> dict[str, Any] | None:
    if utterance is None or visible_word_count <= 0 or not utterance["text"]:
        return None

    transcript_text = utterance["text"]
    hedge_ranges = _find_match_ranges(
        transcript_text,
        _extract_analysis_matches(utterance["analysis_payload"].get("hedging")),
    )
    substance_ranges = _find_match_ranges(
        transcript_text,
        _extract_analysis_matches(utterance["analysis_payload"].get("substance")),
    )
    tokens: list[dict[str, Any]] = []
    visible_ranges = utterance["word_ranges"][:visible_word_count]
    cursor = 0

    for index, word_range in enumerate(visible_ranges):
        if cursor < word_range["start"]:
            gap_text = transcript_text[cursor:word_range["start"]]
            if gap_text:
                tokens.append(
                    {
                        "id": f'{utterance["id"]}-gap-{cursor}',
                        "kind": "text",
                        "text": gap_text,
                    }
                )

        word_timing = utterance["word_timings"][index] if index < len(utterance["word_timings"]) else None
        word_start_ms = word_timing["start_ms"] if word_timing else utterance["start_ms"]
        word_end_ms = word_timing["end_ms"] if word_timing else utterance["end_ms"]
        word_duration_ms = max(word_end_ms - word_start_ms, 1)
        token_text = transcript_text[word_range["start"]:word_range["end"]]
        if token_text:
            tokens.append(
                {
                    "id": f'{utterance["id"]}-word-{index}-{word_range["start"]}-{word_range["end"]}',
                    "kind": "word",
                    "text": token_text,
                    "start": word_range["start"],
                    "end": word_range["end"],
                    "is_hedge": any(_ranges_overlap(range_item, word_range) for range_item in hedge_ranges),
                    "is_substance": any(_ranges_overlap(range_item, word_range) for range_item in substance_ranges),
                    "drop_start_ms": word_start_ms + min(int(word_duration_ms * 0.16), 90),
                    "drop_duration_ms": 1600 + min(int(word_duration_ms * 2.5), 420),
                }
            )
        cursor = word_range["end"]

    return {
        "utterance_id": utterance["id"],
        "current_time_ms": current_time_ms,
        "tokens": tokens,
    }


def _get_conversation_final_snapshot_time_ms(
    utterances: list[dict[str, Any]],
    margin_notes: list[dict[str, Any]],
) -> int:
    latest_utterance_end_ms = max((utterance["end_ms"] for utterance in utterances), default=0)
    latest_drop_end_ms = max((_get_utterance_drop_end_ms(utterance) for utterance in utterances), default=0)
    latest_note_end_ms = max(
        (note["appear_at_ms"] + note["settle_duration_ms"] for note in margin_notes),
        default=0,
    )
    return max(latest_utterance_end_ms, latest_drop_end_ms, latest_note_end_ms, 0)


def _get_utterance_drop_end_ms(utterance: dict[str, Any]) -> int:
    hedge_ranges = _find_match_ranges(
        utterance["text"],
        _extract_analysis_matches(utterance["analysis_payload"].get("hedging")),
    )
    latest_drop_end_ms = utterance["end_ms"]

    for index, word_range in enumerate(utterance["word_ranges"]):
        if not any(_ranges_overlap(hedge_range, word_range) for hedge_range in hedge_ranges):
            continue
        word_timing = utterance["word_timings"][index] if index < len(utterance["word_timings"]) else None
        word_start_ms = word_timing["start_ms"] if word_timing else utterance["start_ms"]
        word_end_ms = word_timing["end_ms"] if word_timing else utterance["end_ms"]
        word_duration_ms = max(word_end_ms - word_start_ms, 1)
        drop_start_ms = word_start_ms + min(int(word_duration_ms * 0.16), 90)
        drop_duration_ms = 1600 + min(int(word_duration_ms * 2.5), 420)
        latest_drop_end_ms = max(latest_drop_end_ms, drop_start_ms + drop_duration_ms)

    return latest_drop_end_ms


def _find_active_utterance(utterances: list[dict[str, Any]], current_time_ms: int) -> dict[str, Any] | None:
    latest_started: dict[str, Any] | None = None
    for utterance in utterances:
        if utterance["start_ms"] <= current_time_ms <= utterance["end_ms"]:
            return utterance
        if utterance["start_ms"] <= current_time_ms:
            latest_started = utterance
        else:
            break
    return latest_started


def _get_utterance_playback_state(utterance: dict[str, Any] | None, current_time_ms: int) -> dict[str, Any] | None:
    if utterance is None:
        return None

    total_words = len(utterance["word_timings"])
    if total_words == 0:
        return {
            "progress": 1.0 if current_time_ms >= utterance["start_ms"] else 0.0,
            "visible_word_count": len(utterance["word_ranges"]) if current_time_ms >= utterance["start_ms"] else 0,
        }

    if current_time_ms < utterance["word_timings"][0]["start_ms"]:
        return {"progress": 0.0, "visible_word_count": 0}

    completed_words = 0
    progress = 1.0
    for index, word_timing in enumerate(utterance["word_timings"]):
        if current_time_ms < word_timing["start_ms"]:
            progress = completed_words / max(total_words, 1)
            break
        if current_time_ms < word_timing["end_ms"]:
            word_duration_ms = max(word_timing["end_ms"] - word_timing["start_ms"], 1)
            partial_progress = max(0.0, min((current_time_ms - word_timing["start_ms"]) / word_duration_ms, 1.0))
            progress = (index + partial_progress) / max(total_words, 1)
            completed_words = index + 1
            break
        completed_words = index + 1

    return {
        "progress": max(0.0, min(progress, 1.0)),
        "visible_word_count": completed_words,
    }


def _format_speaker_label(speaker_id: str, index: int, speaker_labels: dict[str, str]) -> str:
    named_label = str(speaker_labels.get(speaker_id, "")).strip()
    if named_label:
        return named_label
    if speaker_id == FALLBACK_SPEAKER_ID:
        return FALLBACK_SPEAKER_LABEL
    if speaker_id.startswith("SPEAKER_"):
        try:
            suffix = int(speaker_id.replace("SPEAKER_", ""))
        except ValueError:
            suffix = None
        if suffix is not None:
            return f"Speaker {suffix + 1}"
    return f"Speaker {index + 1}" if index >= 0 else speaker_id


def _extract_analysis_matches(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    matches = value.get("matches")
    if not isinstance(matches, list):
        return []
    unique_matches: list[str] = []
    seen: set[str] = set()
    for match in matches:
        if not isinstance(match, str):
            continue
        normalized = match.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_matches.append(normalized)
    return unique_matches


def _build_match_variants(match: str) -> list[str]:
    normalized = match.lower()
    variants = {
        normalized,
        normalized.replace("'", "’"),
        normalized.replace("’", "'"),
    }
    contraction_pairs = [
        ("i am", "i'm"),
        ("i do not", "i don't"),
        ("do not", "don't"),
        ("cannot", "can't"),
    ]
    for expanded, contracted in contraction_pairs:
        if expanded in normalized:
            variants.add(normalized.replace(expanded, contracted))
            variants.add(normalized.replace(expanded, contracted.replace("'", "’")))
    return [variant for variant in variants if variant]


def _find_match_ranges(text: str, matches: list[str]) -> list[dict[str, int]]:
    lowered_text = text.lower()
    seen: set[tuple[int, int]] = set()
    ranges: list[dict[str, int]] = []

    for match in matches:
        for variant in _build_match_variants(match):
            search_index = 0
            while search_index < len(lowered_text):
                match_index = lowered_text.find(variant, search_index)
                if match_index == -1:
                    break
                key = (match_index, match_index + len(variant))
                if key not in seen:
                    seen.add(key)
                    ranges.append({"start": key[0], "end": key[1]})
                search_index = match_index + max(len(variant), 1)

    return sorted(ranges, key=lambda item: (item["start"], item["end"]))


def _build_clause_ranges(text: str, ranges: list[dict[str, int]]) -> list[dict[str, int]]:
    clause_ranges: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()

    for range_item in ranges:
        expanded = _expand_range_to_clause(text, range_item)
        key = (expanded["start"], expanded["end"])
        if expanded["end"] <= expanded["start"] or key in seen:
            continue
        seen.add(key)
        clause_ranges.append(expanded)

    return clause_ranges


def _expand_range_to_clause(text: str, range_item: dict[str, int]) -> dict[str, int]:
    start = max(0, min(range_item["start"], len(text)))
    end = max(0, min(range_item["end"], len(text)))

    while start > 0 and text[start - 1] not in ",:;.!?\n":
        start -= 1
    while start < len(text) and text[start] in " \t\r\n\"'([{":
        start += 1

    while end < len(text) and text[end] not in ",:;.!?\n":
        end += 1
    while end > start and text[end - 1] in " \t\r\n)\"']":
        end -= 1

    return {"start": start, "end": end}


def _extract_word_ranges(text: str) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    in_word = False
    word_start = 0

    for index, character in enumerate(text):
        if character.isspace():
            if in_word:
                ranges.append({"start": word_start, "end": index})
                in_word = False
            continue
        if not in_word:
            word_start = index
            in_word = True

    if in_word:
        ranges.append({"start": word_start, "end": len(text)})

    return ranges


def _build_fallback_word_timings(
    word_ranges: list[dict[str, int]],
    start_ms: int,
    end_ms: int,
) -> list[dict[str, int]]:
    word_count = len(word_ranges)
    if word_count == 0:
        return []
    duration_ms = max(end_ms - start_ms, word_count)
    timings: list[dict[str, int]] = []
    for index in range(word_count):
        word_start_ms = start_ms + int((duration_ms * index) / word_count)
        word_end_ms = start_ms + int((duration_ms * (index + 1) + word_count - 1) / word_count)
        timings.append(
            {
                "start_ms": word_start_ms,
                "end_ms": max(word_start_ms + 1, word_end_ms),
            }
        )
    return timings


def _resolve_range_timing(
    range_item: dict[str, int],
    text: str,
    word_ranges: list[dict[str, int]],
    word_timings: list[dict[str, int]],
    utterance_start_ms: int,
    utterance_end_ms: int,
) -> dict[str, int]:
    overlapping_indexes = [
        index
        for index, word_range in enumerate(word_ranges)
        if range_item["start"] < word_range["end"] and word_range["start"] < range_item["end"]
    ]
    if not overlapping_indexes:
        return _fallback_signal_timing(range_item, len(text), utterance_start_ms, utterance_end_ms)

    first_timing = word_timings[overlapping_indexes[0]] if overlapping_indexes[0] < len(word_timings) else None
    last_timing = word_timings[overlapping_indexes[-1]] if overlapping_indexes[-1] < len(word_timings) else None
    if first_timing is None or last_timing is None:
        return _fallback_signal_timing(range_item, len(text), utterance_start_ms, utterance_end_ms)

    return {
        "start_ms": first_timing["start_ms"],
        "end_ms": max(first_timing["start_ms"] + 1, last_timing["end_ms"]),
    }


def _fallback_signal_timing(
    range_item: dict[str, int],
    text_length: int,
    utterance_start_ms: int,
    utterance_end_ms: int,
) -> dict[str, int]:
    duration_ms = max(utterance_end_ms - utterance_start_ms, 1)
    start_ratio = 0 if text_length <= 0 else range_item["start"] / text_length
    end_ratio = 1 if text_length <= 0 else range_item["end"] / text_length
    start_ms = utterance_start_ms + int(duration_ms * start_ratio)
    end_ms = utterance_start_ms + int(duration_ms * end_ratio)
    return {"start_ms": start_ms, "end_ms": max(start_ms + 1, end_ms)}


def _ranges_overlap(left: dict[str, int], right: dict[str, int]) -> bool:
    return left["start"] < right["end"] and right["start"] < left["end"]
