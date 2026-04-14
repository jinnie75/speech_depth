import unittest

from asr_viz.pipeline.segmentation import segment_transcript
from asr_viz.pipeline.types import ASRSegment, ASRWord, SentenceCandidate
from asr_viz.providers.analysis_v2 import HeuristicAnalysisProvider, _emotion_word_matches


class HeuristicAnalysisProviderTests(unittest.TestCase):
    def test_hedging_payload_captures_softening_language(self) -> None:
        provider = HeuristicAnalysisProvider()
        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text="Maybe I was wondering if that's okay.",
                    sentence_metadata={"source_segment_count": 1, "mean_word_probability": 0.95},
                ),
                SentenceCandidate(
                    utterance_index=1,
                    start_ms=1000,
                    end_ms=2000,
                    text="Send me that now.",
                    sentence_metadata={"source_segment_count": 1, "mean_word_probability": 0.95},
                ),
            ],
            transcript_text="Maybe I was wondering if that's okay. Send me that now.",
        )

        self.assertIn("hedging", results[0].analysis_payload)
        self.assertTrue(results[0].analysis_payload["hedging"]["matches"])
        self.assertEqual(results[0].politeness_score, 0.5)

    def test_substance_payload_detects_self_expression(self) -> None:
        provider = HeuristicAnalysisProvider()
        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text="I feel overwhelmed and I want to leave.",
                    sentence_metadata={
                        "source_segment_count": 1,
                        "mean_word_probability": 0.94,
                        "low_confidence_word_ratio": 0.0,
                        "avg_segment_logprob": -0.2,
                        "max_no_speech_prob": 0.01,
                    },
                    speaker_confidence=0.95,
                ),
                SentenceCandidate(
                    utterance_index=1,
                    start_ms=1000,
                    end_ms=1600,
                    text="We should review the design tomorrow.",
                    sentence_metadata={
                        "source_segment_count": 2,
                        "mean_word_probability": 0.58,
                        "low_confidence_word_ratio": 0.5,
                        "avg_segment_logprob": -1.2,
                        "max_no_speech_prob": 0.18,
                    },
                    speaker_confidence=0.6,
                ),
            ],
            transcript_text="I feel overwhelmed and I want to leave. We should review the design tomorrow.",
        )

        self.assertIn("substance", results[0].analysis_payload)
        self.assertTrue(results[0].analysis_payload["substance"]["matches"])
        self.assertEqual(results[0].semantic_confidence_score, 0.5)

    def test_i_am_substance_requires_emotion_word_in_same_sentence(self) -> None:
        provider = HeuristicAnalysisProvider()
        sentence = "I am tired and overwhelmed."

        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text=sentence,
                    sentence_metadata={},
                )
            ],
            transcript_text=sentence,
        )

        substance = results[0].analysis_payload["substance"]
        self.assertIn("i_am", substance["categories"])
        self.assertIn("i am tired", substance["matches"])
        self.assertIn("emotion_word", substance["categories"])
        self.assertIn("overwhelmed", substance["matches"])

    def test_i_am_substance_allows_non_adjective_descriptor(self) -> None:
        provider = HeuristicAnalysisProvider()
        sentence = "I am a mess and overwhelmed."

        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text=sentence,
                    sentence_metadata={},
                )
            ],
            transcript_text=sentence,
        )

        substance = results[0].analysis_payload["substance"]
        self.assertIn("i_am", substance["categories"])
        self.assertIn("i am a mess", substance["matches"])

    def test_i_am_substance_uses_nrc_membership_without_pos_requirement(self) -> None:
        provider = HeuristicAnalysisProvider()
        sentence = "I'm a failure."

        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text=sentence,
                    sentence_metadata={},
                )
            ],
            transcript_text=sentence,
        )

        substance = results[0].analysis_payload["substance"]
        self.assertIn("i_am", substance["categories"])
        self.assertIn("i am a failure", substance["matches"])
        self.assertNotIn("emotion_word", substance["categories"])

    def test_i_am_substance_skips_sentences_without_emotion_word(self) -> None:
        provider = HeuristicAnalysisProvider()
        sentence = "I am tall today."

        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text=sentence,
                    sentence_metadata={},
                )
            ],
            transcript_text=sentence,
        )

        substance = results[0].analysis_payload["substance"]
        self.assertNotIn("i_am", substance["categories"])
        self.assertNotIn("i am tall", substance["matches"])

    def test_emotion_word_matches_are_backed_by_vendored_lexicon_and_nltk_wordnet(self) -> None:
        provider = HeuristicAnalysisProvider()
        sentence = "Overwhelmed and angry."

        matches = _emotion_word_matches(sentence.lower())
        self.assertIn("overwhelmed", matches)

        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text=sentence,
                    sentence_metadata={},
                )
            ],
            transcript_text=sentence,
        )

        substance = results[0].analysis_payload["substance"]
        self.assertIn("emotion_word", substance["categories"])
        self.assertIn("overwhelmed", substance["matches"])

    def test_segmentation_emits_asr_quality_metadata_for_sentence_scoring(self) -> None:
        segments = [
            ASRSegment(
                segment_index=0,
                start_ms=0,
                end_ms=1000,
                text="Please review the design.",
                avg_logprob=-0.2,
                no_speech_prob=0.02,
                words=[
                    ASRWord(word="Please", start_ms=0, end_ms=200, probability=0.95),
                    ASRWord(word="review", start_ms=200, end_ms=450, probability=0.92),
                    ASRWord(word="the", start_ms=450, end_ms=650, probability=0.9),
                    ASRWord(word="design.", start_ms=650, end_ms=950, probability=0.88),
                ],
            )
        ]

        sentences = segment_transcript(segments)

        self.assertEqual(len(sentences), 1)
        self.assertAlmostEqual(sentences[0].sentence_metadata["mean_word_probability"], 0.9125, places=4)
        self.assertEqual(sentences[0].sentence_metadata["low_confidence_word_ratio"], 0.0)
        self.assertEqual(sentences[0].sentence_metadata["avg_segment_logprob"], -0.2)

    def test_segmentation_scopes_word_quality_metadata_to_each_split_sentence(self) -> None:
        segments = [
            ASRSegment(
                segment_index=0,
                start_ms=0,
                end_ms=2000,
                text="Please review. Maybe later?",
                avg_logprob=-0.4,
                no_speech_prob=0.03,
                words=[
                    ASRWord(word="Please", start_ms=0, end_ms=300, probability=0.98),
                    ASRWord(word="review.", start_ms=300, end_ms=800, probability=0.96),
                    ASRWord(word="Maybe", start_ms=800, end_ms=1200, probability=0.55),
                    ASRWord(word="later?", start_ms=1200, end_ms=2000, probability=0.5),
                ],
            )
        ]

        sentences = segment_transcript(segments)

        self.assertEqual(len(sentences), 2)
        self.assertAlmostEqual(sentences[0].sentence_metadata["mean_word_probability"], 0.97, places=4)
        self.assertEqual(sentences[0].sentence_metadata["low_confidence_word_ratio"], 0.0)
        self.assertAlmostEqual(sentences[1].sentence_metadata["mean_word_probability"], 0.525, places=4)
        self.assertEqual(sentences[1].sentence_metadata["low_confidence_word_ratio"], 1.0)

    def test_non_english_analysis_returns_safe_empty_payload(self) -> None:
        provider = HeuristicAnalysisProvider()
        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text="아마 괜찮다면 오늘 이야기해도 될까요?",
                    sentence_metadata={},
                )
            ],
            transcript_text="아마 괜찮다면 오늘 이야기해도 될까요?",
            language_code="ko",
        )

        payload = results[0].analysis_payload
        self.assertTrue(payload["language_supported"])
        self.assertIn("아마", payload["hedging"]["matches"])
        self.assertIn("괜찮다면", payload["hedging"]["matches"])

    def test_korean_substance_detects_self_expression_and_emotion_language(self) -> None:
        provider = HeuristicAnalysisProvider()
        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text="저는 너무 불안하고 마음이 답답해요.",
                    sentence_metadata={},
                )
            ],
            transcript_text="저는 너무 불안하고 마음이 답답해요.",
            language_code="ko",
        )

        substance = results[0].analysis_payload["substance"]
        self.assertIn("self_expression", substance["categories"])
        self.assertIn("emotion_word", substance["categories"])
        self.assertTrue(any("불안" in match or "답답" in match for match in substance["matches"]))

    def test_unsupported_language_analysis_returns_safe_empty_payload(self) -> None:
        provider = HeuristicAnalysisProvider()
        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text="こんにちは、少し不安です。",
                    sentence_metadata={},
                )
            ],
            transcript_text="こんにちは、少し不安です。",
            language_code="ja",
        )

        payload = results[0].analysis_payload
        self.assertFalse(payload["language_supported"])
        self.assertEqual(payload["unsupported_language"], "ja")
        self.assertEqual(payload["hedging"]["matches"], [])
        self.assertEqual(payload["substance"]["matches"], [])
