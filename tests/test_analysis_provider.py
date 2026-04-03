import unittest

from asr_viz.pipeline.segmentation import segment_transcript
from asr_viz.pipeline.types import ASRSegment, ASRWord, SentenceCandidate
from asr_viz.providers.analysis import HeuristicAnalysisProvider


class HeuristicAnalysisProviderTests(unittest.TestCase):
    def test_politeness_scores_reward_softened_requests_over_blunt_commands(self) -> None:
        provider = HeuristicAnalysisProvider()
        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text="Could you please send that when you can?",
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
            transcript_text="Could you please send that when you can? Send me that now.",
        )

        self.assertGreater(results[0].politeness_score, results[1].politeness_score)
        self.assertIn("politeness_features", results[0].analysis_payload)

    def test_semantic_confidence_prefers_complete_sentences_over_hesitant_fragments(self) -> None:
        provider = HeuristicAnalysisProvider()
        results = provider.analyze(
            [
                SentenceCandidate(
                    utterance_index=0,
                    start_ms=0,
                    end_ms=1000,
                    text="We should review the design tomorrow.",
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
                    text="Uh maybe... something.",
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
            transcript_text="We should review the design tomorrow. Uh maybe... something.",
        )

        self.assertGreater(results[0].semantic_confidence_score, results[1].semantic_confidence_score)
        self.assertIn("semantic_confidence_features", results[0].analysis_payload)

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

