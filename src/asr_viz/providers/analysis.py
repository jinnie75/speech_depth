from __future__ import annotations

from abc import ABC, abstractmethod
import re

from openai import OpenAI

from asr_viz.pipeline.types import SentenceAnalysis, SentenceAnalysisBatch, SentenceCandidate

_WORD_RE = re.compile(r"[a-z']+")
_POSITIVE_POLITENESS_MARKERS = (
    "please",
    "thank you",
    "thanks",
    "appreciate",
    "sorry",
    "excuse me",
    "if you can",
    "when you can",
    "if that's okay",
)
_SOFTENER_PHRASES = (
    "could you",
    "would you",
    "can you",
    "do you mind",
    "would it be possible",
    "i was wondering",
    "i think",
    "i feel",
    "perhaps",
    "maybe",
    "a bit",
    "kind of",
    "sort of",
)
_DIRECTIVE_PHRASES = (
    "do this",
    "do that",
    "give me",
    "send me",
    "tell me",
    "stop",
    "just",
    "obviously",
    "clearly",
)
_HARSH_MARKERS = ("shut up", "whatever", "no.", "no!", "wrong", "ridiculous")
_FILLER_TOKENS = {
    "uh",
    "um",
    "erm",
    "hmm",
    "like",
    "basically",
    "literally",
    "actually",
}
_FILLER_PHRASES = ("you know", "i mean")
_UNCERTAINTY_MARKERS = (
    "maybe",
    "perhaps",
    "i guess",
    "i think",
    "sort of",
    "kind of",
    "probably",
    "possibly",
    "maybe like",
)
_VAGUE_REFERENCES = (
    "something",
    "stuff",
    "things",
    "whatever",
    "somehow",
    "that thing",
)
_COMMON_VERBS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "go",
    "goes",
    "went",
    "make",
    "makes",
    "made",
    "need",
    "needs",
    "want",
    "wants",
    "think",
    "feel",
    "know",
    "say",
    "said",
    "review",
    "fix",
    "send",
    "tell",
    "look",
    "looking",
}


class AnalysisProvider(ABC):
    model_version: str = "unknown"

    @abstractmethod
    def analyze(self, sentences: list[SentenceCandidate], transcript_text: str) -> list[SentenceAnalysis]:
        raise NotImplementedError


class OpenAIAnalysisProvider(AnalysisProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.model_version = model
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def analyze(self, sentences: list[SentenceCandidate], transcript_text: str) -> list[SentenceAnalysis]:
        prompt = _build_prompt(sentences, transcript_text)
        completion = self._client.beta.chat.completions.parse(
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze conversation transcripts sentence by sentence. "
                        "Return scores between 0 and 1 and valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format=SentenceAnalysisBatch,
        )

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("analysis provider returned no parsed output")
        if len(parsed.items) != len(sentences):
            raise RuntimeError("analysis provider returned an unexpected number of results")
        return parsed.items


class HeuristicAnalysisProvider(AnalysisProvider):
    model_version = "heuristic-analyzer:v2"

    def analyze(self, sentences: list[SentenceCandidate], transcript_text: str) -> list[SentenceAnalysis]:
        results: list[SentenceAnalysis] = []
        for sentence in sentences:
            normalized = _normalize_text(sentence.text)
            politeness_score, politeness_payload = _heuristic_politeness(normalized)
            semantic_confidence_score, semantic_payload = _heuristic_semantic_confidence(
                normalized,
                sentence.sentence_metadata,
                sentence.speaker_confidence,
            )
            results.append(
                SentenceAnalysis(
                    politeness_score=politeness_score,
                    semantic_confidence_score=semantic_confidence_score,
                    main_message_likelihood=_heuristic_main_message(sentence, transcript_text),
                    analysis_payload={
                        "provider": "heuristic",
                        "heuristic_version": 2,
                        "politeness_features": politeness_payload,
                        "semantic_confidence_features": semantic_payload,
                    },
                )
            )
        return results


def _build_prompt(sentences: list[SentenceCandidate], transcript_text: str) -> str:
    serialized_sentences = "\n".join(
        f"{item.utterance_index}. [{item.start_ms}-{item.end_ms}] {item.text}" for item in sentences
    )
    return (
        "Analyze each sentence independently but in conversation context.\n"
        "Fields:\n"
        "- politeness_score: politeness/considerateness of phrasing\n"
        "- semantic_confidence_score: confidence that sentence meaning is coherent and unambiguous in context\n"
        "- main_message_likelihood: likelihood that this sentence expresses a central message of the conversation\n\n"
        "Conversation:\n"
        f"{transcript_text}\n\n"
        "Sentences:\n"
        f"{serialized_sentences}"
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _phrase_count(text: str, phrases: tuple[str, ...]) -> int:
    return sum(text.count(phrase) for phrase in phrases)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _heuristic_politeness(text: str) -> tuple[float, dict]:
    tokens = _tokenize(text)
    token_count = len(tokens)
    positive_markers = _phrase_count(text, _POSITIVE_POLITENESS_MARKERS)
    softener_markers = _phrase_count(text, _SOFTENER_PHRASES)
    directive_markers = _phrase_count(text, _DIRECTIVE_PHRASES)
    harsh_markers = _phrase_count(text, _HARSH_MARKERS)
    imperative_open = 1 if tokens and tokens[0] in {"do", "give", "send", "stop", "tell", "look"} else 0
    question_softening = 1 if text.endswith("?") and (softener_markers > 0 or "can you" in text or "could you" in text) else 0
    exclamation_count = text.count("!")

    score = 0.48
    score += min(positive_markers, 3) * 0.12
    score += min(softener_markers, 3) * 0.08
    score += question_softening * 0.06
    score -= min(directive_markers, 3) * 0.08
    score -= harsh_markers * 0.14
    score -= imperative_open * 0.08
    score -= min(exclamation_count, 2) * 0.04

    if token_count <= 2 and positive_markers == 0:
        score -= 0.05

    score = round(_clamp(score, 0.05, 0.98), 4)
    payload = {
        "positive_markers": positive_markers,
        "softener_markers": softener_markers,
        "directive_markers": directive_markers,
        "harsh_markers": harsh_markers,
        "imperative_open": imperative_open,
        "question_softening": question_softening,
        "exclamation_count": exclamation_count,
        "token_count": token_count,
    }
    return score, payload


def _heuristic_semantic_confidence(
    text: str,
    sentence_metadata: dict | None,
    speaker_confidence: float | None,
) -> tuple[float, dict]:
    metadata = sentence_metadata or {}
    tokens = _tokenize(text)
    token_count = len(tokens)
    filler_hits = sum(1 for token in tokens if token in _FILLER_TOKENS) + _phrase_count(text, _FILLER_PHRASES)
    filler_ratio = filler_hits / token_count if token_count else 0.0
    uncertainty_markers = _phrase_count(text, _UNCERTAINTY_MARKERS)
    vague_reference_hits = _phrase_count(text, _VAGUE_REFERENCES)
    repeated_word_pairs = sum(1 for left, right in zip(tokens, tokens[1:]) if left == right)
    has_terminal_punctuation = 1 if text.endswith((".", "?", "!")) else 0
    has_common_verb = 1 if any(token in _COMMON_VERBS or token.endswith("ed") or token.endswith("ing") for token in tokens) else 0
    fragment_penalty = 1 if token_count <= 2 else 0
    trailing_incomplete = 1 if text.endswith(("...", "-", ",")) else 0

    mean_word_probability = _safe_float(metadata.get("mean_word_probability"))
    low_confidence_word_ratio = _safe_float(metadata.get("low_confidence_word_ratio"))
    avg_segment_logprob = _safe_float(metadata.get("avg_segment_logprob"))
    max_no_speech_prob = _safe_float(metadata.get("max_no_speech_prob"))
    source_segment_count = int(metadata.get("source_segment_count") or 0)

    score = 0.5
    if 4 <= token_count <= 22:
        score += 0.12
    elif token_count >= 3:
        score += 0.04

    score += has_common_verb * 0.1
    score += has_terminal_punctuation * 0.06
    score -= fragment_penalty * 0.2
    score -= trailing_incomplete * 0.08
    score -= min(filler_ratio, 0.4) * 0.28
    score -= min(uncertainty_markers, 3) * 0.05
    score -= min(vague_reference_hits, 3) * 0.05
    score -= min(repeated_word_pairs, 2) * 0.06

    if mean_word_probability is not None:
        score += (mean_word_probability - 0.6) * 0.25
    if low_confidence_word_ratio is not None:
        score -= low_confidence_word_ratio * 0.22
    if avg_segment_logprob is not None:
        score += _clamp((avg_segment_logprob + 1.5) / 2.5, 0.0, 1.0) * 0.08 - 0.02
    if max_no_speech_prob is not None:
        score -= max_no_speech_prob * 0.08
    if speaker_confidence is not None:
        score += (speaker_confidence - 0.5) * 0.05
    if source_segment_count >= 3:
        score -= min(source_segment_count - 2, 3) * 0.015

    score = round(_clamp(score, 0.05, 0.98), 4)
    payload = {
        "token_count": token_count,
        "filler_hits": filler_hits,
        "filler_ratio": round(filler_ratio, 4),
        "uncertainty_markers": uncertainty_markers,
        "vague_reference_hits": vague_reference_hits,
        "repeated_word_pairs": repeated_word_pairs,
        "has_terminal_punctuation": has_terminal_punctuation,
        "has_common_verb": has_common_verb,
        "fragment_penalty": fragment_penalty,
        "trailing_incomplete": trailing_incomplete,
        "mean_word_probability": mean_word_probability,
        "low_confidence_word_ratio": low_confidence_word_ratio,
        "avg_segment_logprob": avg_segment_logprob,
        "max_no_speech_prob": max_no_speech_prob,
        "speaker_confidence": speaker_confidence,
        "source_segment_count": source_segment_count,
    }
    return score, payload


def _safe_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _heuristic_main_message(sentence: SentenceCandidate, transcript_text: str) -> float:
    sentence_length = len(sentence.text.split())
    full_length = max(len(transcript_text.split()), 1)
    ratio = min(sentence_length / full_length * 4, 1.0)
    if sentence.utterance_index == 0:
        return min(0.4 + ratio, 1.0)
    return min(0.25 + ratio, 1.0)
