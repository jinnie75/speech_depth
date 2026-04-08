from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
import os
from pathlib import Path
import re

from asr_viz.pipeline.types import SentenceAnalysis, SentenceCandidate

_WORD_RE = re.compile(r"[a-z']+")
_HEDGE_PHRASES = (
    "maybe",
    "probably",
    "perhaps",
    "sort of",
    "kind of",
    "a bit",
    "a little",
    "in a way",
    "i'm not sure",
    "i don't know for sure",
    "i can't say for sure",
    "it seems",
    "it sounds like",
    "it looks like",
    "it feels like",
    "i guess",
    "i think",
    "i feel like",
    "it could be",
    "it might be",
    "i was wondering",
    "i wanted to ask",
    "would you mind",
    "do you think",
    "a quick question",
    "if you don't mind",
    "if it's not too much trouble",
    "if that's not too much trouble",
    "if it's okay",
    "if that's okay",
    "if that works",
    "no rush",
    "no pressure",
    "a small thing",
    "i may be wrong",
    "i might be wrong",
    "i could be wrong",
    "stupid question",
    "silly idea",
    "bother you",
    "bothering you",
    "i just",
    "it's just that",
    "if you don't mind",
    "if you dont mind",
    "if it's not too much trouble",
    "if its not too much trouble",
    "if that's okay",
    "if thats okay",
    "if that works",
    "no rush",
    "no pressure",
)
_HEDGE_APOLOGY_PHRASES = (
    "sorry",
)
_HEDGE_REASSURANCE_PHRASES = (
    "it's okay",
    "its okay",
    "it's fine",
    "its fine",
    "it's alright",
    "its alright",
    "that's okay",
    "i'm okay",
    "im okay",
    "i'm fine",
    "im fine",
    "i'm alright",
    "im alright",
    "i'm totally fine",
    "im totally fine",
    "i'm totally alright",
    "im totally alright",
    "everything's fine",
    "everythings fine",
    "everything's going to be fine",
    "everythings going to be fine",
    "everything's gonna be fine",
    "everythings gonna be fine",
    "everything's okay",
    "everythings okay",
    "everything's going to be okay",
    "everythings going to be okay",
    "everything's gonna be okay",
    "everythings gonna be okay",
    "everything's alright",
    "everything's going to be alright",
    "everything's gonna be alright",
    "never mind",
    "it's nothing",
    "it's no big deal",
    "it's not a big deal",
    "it was nothing",
    "it was no big deal",
    "it was not a big deal",
    "it doesn't matter",
    "it doesnt matter",
    "it really doesn't matter",
    "it really doesnt matter",
)
_HEDGE_TRANSITIONS = (
    "actually",
    "well",
    "honestly",
    "to be honest",
)
_HEDGE_TAG_QUESTIONS = (
    "you know",
    "you see",
    "i mean",
    "right?",
    "know what i mean",
    "don't you think",
    "dont you think",
    "know what i'm saying",
    "know what im saying",
)
_SUBSTANCE_FALLBACK_SELF_ADJECTIVES = {
    "afraid",
    "alone",
    "angry",
    "anxious",
    "ashamed",
    "bad",
    "confused",
    "depressed",
    "different",
    "embarrassed",
    "empty",
    "fine",
    "frustrated",
    "grateful",
    "guilty",
    "happy",
    "hurt",
    "lonely",
    "lost",
    "mad",
    "miserable",
    "nervous",
    "numb",
    "okay",
    "ok",
    "overwhelmed",
    "sad",
    "scared",
    "sensitive",
    "stressed",
    "stuck",
    "tired",
    "upset",
    "unsafe",
    "unwell",
    "weak",
    "worried",
    "worthless",
}
_SUBSTANCE_EMOTION_LABELS = {
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
}
_SUBSTANCE_I_AM_INTENSIFIERS = {
    "extremely",
    "kind",
    "pretty",
    "quite",
    "really",
    "so",
    "sort",
    "too",
    "very",
}
_NRC_EMOLEX_ENV_VAR = "NRC_EMOLEX_PATH"
_NRC_EMOLEX_FILENAME = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
_SUBSTANCE_LIFE_TREATMENT_PHRASES = (
    "my life",
    "my whole life",
    "my entire life",
    "my whole entire life",
    "treat me like",
    "treats me like",
    "treated me like",
    "thinks i'm",
    "thinks im",
    "thinks i am",
    "think i'm",
    "think im",
    "think i am",
    "thought i'm",
    "thought im",
    "thought i am",
    "thinks that i'm",
    "thinks that im",
    "thinks that i am",
    "think that i'm",
    "think that im",
    "think that i am",
    "thought that i'm",
    "thought that im",
    "thought that i am",
)
_CLAUSE_SPLIT_RE = re.compile(r"\s*[,;:]\s*")

class AnalysisProvider(ABC):
    model_version: str = "unknown"

    @abstractmethod
    def analyze(self, sentences: list[SentenceCandidate], transcript_text: str) -> list[SentenceAnalysis]:
        raise NotImplementedError


class HeuristicAnalysisProvider(AnalysisProvider):
    model_version = "heuristic-analyzer:v6"

    def analyze(self, sentences: list[SentenceCandidate], transcript_text: str) -> list[SentenceAnalysis]:
        results: list[SentenceAnalysis] = []
        for sentence in sentences:
            normalized = _normalize_text(sentence.text)
            hedging_payload = _heuristic_hedging(sentence.text, normalized)
            substance_payload = _heuristic_substance(sentence.text, normalized)
            results.append(
                SentenceAnalysis(
                    politeness_score=0.5,
                    semantic_confidence_score=0.5,
                    main_message_likelihood=0.5,
                    analysis_payload={
                        "provider": "heuristic",
                        "heuristic_version": 6,
                        "hedging": hedging_payload,
                        "substance": substance_payload,
                    },
                )
            )
        return results


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _phrase_count(text: str, phrases: tuple[str, ...]) -> int:
    return sum(
        len(re.findall(rf"(?<![a-z']){re.escape(phrase)}(?![a-z'])", text))
        for phrase in phrases
    )


def _matched_phrases(text: str, phrases: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for phrase in phrases:
        pattern = rf"(?<![a-z']){re.escape(phrase)}(?![a-z'])"
        if re.search(pattern, text):
            matches.append(phrase)
    return matches


def _phrase_present(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z']){re.escape(phrase)}(?![a-z'])"
    return re.search(pattern, text) is not None


def _clause_segments(text: str) -> list[str]:
    clauses = [segment.strip() for segment in _CLAUSE_SPLIT_RE.split(text) if segment.strip()]
    return clauses or [text.strip()]


def _matching_clause(original_text: str, match: str) -> str:
    for clause in _clause_segments(original_text):
        if _phrase_present(_normalize_text(clause), match):
            return clause
    return original_text.strip()


def _repeated_word_pair_matches(tokens: list[str]) -> list[str]:
    matches: list[str] = []
    for left, right in zip(tokens, tokens[1:]):
        if left == right:
            pair = f"{left} {right}"
            if pair not in matches:
                matches.append(pair)
    return matches


@lru_cache(maxsize=1)
def _load_nrc_emolex() -> dict[str, set[str]]:
    for path in _nrc_emolex_candidate_paths():
        if not path.exists():
            continue

        lexicon: dict[str, set[str]] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                continue

            term, label, association = parts
            normalized_term = term.strip().lower()
            normalized_label = label.strip().lower()
            normalized_association = association.strip()
            if (
                not normalized_term
                or normalized_label not in _SUBSTANCE_EMOTION_LABELS
                or normalized_association != "1"
            ):
                continue

            lexicon.setdefault(normalized_term, set()).add(normalized_label)

        if lexicon:
            return lexicon

    return {}


def _nrc_emolex_candidate_paths() -> list[Path]:
    env_path = os.getenv(_NRC_EMOLEX_ENV_VAR)
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / ".lexicons" / "nrc_emolex" / _NRC_EMOLEX_FILENAME)
    return candidates


def _emotion_word_matches(normalized_text: str) -> list[str]:
    lexicon = _load_nrc_emolex()
    matches: list[str] = []
    for token in _tokenize(normalized_text):
        if token in lexicon and _is_emotion_modifier_word(token):
            matches.append(token)
    return _dedupe_preserve_order(matches)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _clause_for_match(text: str, matched_fragment: str) -> str:
    start_index = text.find(matched_fragment)
    if start_index < 0:
        return matched_fragment.strip(" ,;:.!?")

    clause_start = 0
    for match in re.finditer(r"[,:;.!?]", text):
        if match.start() >= start_index:
            break
        clause_start = match.end()

    clause_end = len(text)
    fragment_end = start_index + len(matched_fragment)
    for match in re.finditer(r"[,:;.!?]", text):
        if match.start() < fragment_end:
            continue
        clause_end = match.start()
        break

    return text[clause_start:clause_end].strip(" ,;:.!?")


def _heuristic_substance(text: str, normalized_text: str) -> dict:
    categories: list[str] = []
    match_fragments: list[str] = []
    excerpt_fragments: list[str] = []
    rationale_parts: list[str] = []

    #  "I feel/felt..."
    i_feel_felt_match = re.search(r"\bi(?:\s+just)?\s+(feel|felt)(?:\s+like)?\s+([^.!?,;]+)", normalized_text)
    if i_feel_felt_match:
        verb = i_feel_felt_match.group(1).strip()
        matched_text = i_feel_felt_match.group(0).strip()
        matched_clause = _clause_for_match(normalized_text, matched_text)
        categories.append("i_felt" if verb == "felt" else "i_feel")
        match_fragments.append(matched_text)
        excerpt_fragments.append(matched_clause)
        rationale_parts.append(f"Uses 'I feel' phrasing")

    # "I am" + descriptor
    i_am_match = _extract_i_am_substance_match(normalized_text)
    if i_am_match:
        matched_clause = _clause_for_match(normalized_text, i_am_match)
        categories.append("i_am")
        match_fragments.append(i_am_match)
        excerpt_fragments.append(matched_clause)
        rationale_parts.append("Uses 'I am' self-description")

    # "I want ..."
    i_want_match = re.search(r"\bi want(?:ed)?\s+([^.!?,;]+)", normalized_text)
    if i_want_match:
        matched_text = i_want_match.group(0).strip()
        matched_clause = _clause_for_match(normalized_text, matched_text)
        categories.append("i_want")
        match_fragments.append(matched_text)
        excerpt_fragments.append(matched_clause)
        rationale_parts.append("Uses 'I want' self-expression")

    # "I don't know ..."
    i_dont_know_match = re.search(r"\bi do(?:n't|nt) know(?:\s+([^.!?,;]+))?", normalized_text)
    if i_dont_know_match:
        matched_text = i_dont_know_match.group(0).strip()
        matched_clause = _clause_for_match(normalized_text, matched_text)
        categories.append("i_dont_know")
        match_fragments.append(matched_text)
        excerpt_fragments.append(matched_clause)
        rationale_parts.append("Uses 'I don't know' self-expression")

    # uses any emotion expression (NRC adj, adv, v)
    emotion_matches = _emotion_word_matches(normalized_text)
    if emotion_matches:
        categories.append("emotion_word")
        emotion_clauses = [_clause_for_match(normalized_text, item) for item in emotion_matches]
        match_fragments.extend(emotion_matches)
        excerpt_fragments.extend(emotion_clauses[:2])
        rationale_parts.append("Includes emotion language")

    # people treat me / comments about life
    life_treatment_matches = _matched_phrases(normalized_text, _SUBSTANCE_LIFE_TREATMENT_PHRASES)
    if life_treatment_matches:
        categories.append("life_or_treatment_comment")
        life_treatment_clauses = [_clause_for_match(normalized_text, item) for item in life_treatment_matches]
        match_fragments.extend(life_treatment_matches)
        excerpt_fragments.extend(life_treatment_clauses[:1])
        rationale_parts.append("Comments on life or how people treat the speaker")

    excerpt = ", ".join(_dedupe_preserve_order(excerpt_fragments)[:2])
    rationale = ". ".join(rationale_parts)

    return {
        "categories": categories,
        "matches": _dedupe_preserve_order(match_fragments),
        "excerpt": excerpt,
        "rationale": f"{rationale}." if rationale else "",
    }


def _extract_i_am_substance_match(normalized_text: str) -> str | None:
    match = re.search(r"\b(?:i am|i'm|i was)\s+([a-z']+(?:\s+[a-z']+){0,4})", normalized_text)
    if not match:
        return None

    phrase = match.group(0).strip()
    descriptor = match.group(1).strip()
    words = descriptor.split()
    if not words:
        return None

    descriptor_index = 0
    while descriptor_index < len(words) and words[descriptor_index] in _SUBSTANCE_I_AM_INTENSIFIERS:
        descriptor_index += 1

    if descriptor_index >= len(words):
        return None

    descriptor_word = words[descriptor_index]

    if descriptor_word.endswith("ing") and descriptor_index + 1 < len(words) and words[descriptor_index + 1] == "to":
        return None

    if descriptor_word == "not":
        next_index = descriptor_index + 1
        if next_index >= len(words):
            return None
        negative_target = words[next_index]
        if _is_substance_i_am_descriptor(negative_target):
            return " ".join(["i am", *words[: next_index + 1]])
        return None

    if _is_substance_i_am_descriptor(descriptor_word):
        return " ".join(["i am", *words[: descriptor_index + 1]])

    return None


@lru_cache(maxsize=1024)
def _is_substance_i_am_descriptor(word: str) -> bool:
    normalized_word = re.sub(r"[^a-z']", "", word.lower()).strip("'")
    if not normalized_word:
        return False

    lookup = _lookup_nltk_adjective_sentiment(normalized_word)
    if lookup is not None:
        has_adjective_sense, positive_score, negative_score = lookup
        if not has_adjective_sense:
            return False
        if positive_score > 0.05 or negative_score > 0.05:
            return True
        return normalized_word in _SUBSTANCE_FALLBACK_SELF_ADJECTIVES

    return normalized_word in _SUBSTANCE_FALLBACK_SELF_ADJECTIVES


@lru_cache(maxsize=1024)
def _is_emotion_modifier_word(word: str) -> bool:
    pos_lookup = _lookup_nltk_emotion_modifier_pos(word)
    if pos_lookup is not None:
        has_adjective_sense, has_adverb_sense, has_verb_sense = pos_lookup
        return has_adjective_sense or has_adverb_sense or has_verb_sense

    return False


@lru_cache(maxsize=1024)
def _lookup_nltk_emotion_modifier_pos(word: str) -> tuple[bool, bool, bool] | None:
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None

    repo_nltk_data = Path(__file__).resolve().parents[3] / ".nltk_data"
    if repo_nltk_data.exists():
        repo_nltk_data_str = str(repo_nltk_data)
        if repo_nltk_data_str not in nltk.data.path:
            nltk.data.path.insert(0, repo_nltk_data_str)

    try:
        adjective_synsets = list(wn.synsets(word, pos=wn.ADJ))
        adverb_synsets = list(wn.synsets(word, pos=wn.ADV))
        verb_synsets = list(wn.synsets(word, pos=wn.VERB))
    except LookupError:
        return None

    return (bool(adjective_synsets), bool(adverb_synsets), bool(verb_synsets))


@lru_cache(maxsize=1024)
def _lookup_nltk_adjective_sentiment(word: str) -> tuple[bool, float, float] | None:
    try:
        import nltk
        from nltk.corpus import sentiwordnet as swn
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None

    repo_nltk_data = Path(__file__).resolve().parents[3] / ".nltk_data"
    if repo_nltk_data.exists():
        repo_nltk_data_str = str(repo_nltk_data)
        if repo_nltk_data_str not in nltk.data.path:
            nltk.data.path.insert(0, repo_nltk_data_str)

    try:
        adjective_synsets = list(wn.synsets(word, pos=wn.ADJ))
    except LookupError:
        return None

    if not adjective_synsets:
        return (False, 0.0, 0.0)

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for synset in adjective_synsets:
        try:
            senti_synset = swn.senti_synset(synset.name())
        except (LookupError, ValueError):
            continue
        positive_scores.append(senti_synset.pos_score())
        negative_scores.append(senti_synset.neg_score())

    if not positive_scores and not negative_scores:
        return (True, 0.0, 0.0)

    return (
        True,
        sum(positive_scores) / len(positive_scores),
        sum(negative_scores) / len(negative_scores),
    )


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _heuristic_hedging(text: str, normalized_text: str) -> dict:
    tokens = _tokenize(normalized_text)
    hedge_phrase_matches = _matched_phrases(normalized_text, _HEDGE_PHRASES)
    apology_matches = _matched_phrases(normalized_text, _HEDGE_APOLOGY_PHRASES)
    reassurance_matches = _matched_phrases(normalized_text, _HEDGE_REASSURANCE_PHRASES)
    transition_matches = _matched_phrases(normalized_text, _HEDGE_TRANSITIONS)
    tag_question_matches = _matched_phrases(normalized_text, _HEDGE_TAG_QUESTIONS)
    trailing_incomplete = 1 if "..." in normalized_text else 0
    rules: list[dict[str, str]] = []
    categories: list[str] = []
    match_fragments: list[str] = []
    excerpt_fragments: list[str] = []
    rationale_parts: list[str] = []

    def add_phrase_rules(rule: str, matches: list[str]) -> None:
        for match in matches:
            if not match.strip():
                continue
            clause = _matching_clause(text, match)
            rules.append({"rule": rule, "match": match, "clause": clause})
            if rule not in categories:
                categories.append(rule)
            match_fragments.append(match)
            excerpt_fragments.append(clause)
            rationale_parts.append(f'{rule} matched "{match}" in clause "{clause}"')

    add_phrase_rules("hedge_phrase", hedge_phrase_matches)
    add_phrase_rules("apology", apology_matches)
    add_phrase_rules("reassurance", reassurance_matches)
    add_phrase_rules("transition", transition_matches)
    add_phrase_rules("tag_question", tag_question_matches)

    if trailing_incomplete:
        clause = text.strip()
        rules.append({"rule": "trailing_incomplete", "match": "...", "clause": clause})
        categories.append("trailing_incomplete")
        match_fragments.append("...")
        excerpt_fragments.append(clause)
        rationale_parts.append('trailing_incomplete matched "..." in clause "{0}"'.format(clause))

    for match in _repeated_word_pair_matches(tokens):
        clause = _matching_clause(text, match)
        rules.append({"rule": "repeated_word_pair", "match": match, "clause": clause})
        if "repeated_word_pair" not in categories:
            categories.append("repeated_word_pair")
        match_fragments.append(match)
        excerpt_fragments.append(clause)
        rationale_parts.append(f'repeated_word_pair matched "{match}" in clause "{clause}"')

    if not rules:
        return {
            "categories": [],
            "matches": [],
            "excerpt": "",
            "rationale": "",
            "rules": [],
        }

    return {
        "categories": categories,
        "matches": _dedupe_preserve_order(match_fragments),
        "excerpt": " | ".join(_dedupe_preserve_order(excerpt_fragments)),
        "rationale": ". ".join(rationale_parts) + ".",
        "rules": rules,
    }
