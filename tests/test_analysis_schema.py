import unittest
from pydantic import ValidationError

from asr_viz.pipeline.types import SentenceAnalysis


class SentenceAnalysisSchemaTests(unittest.TestCase):
    def test_sentence_analysis_rejects_out_of_range_scores(self) -> None:
        with self.assertRaises(ValidationError):
            SentenceAnalysis(
                politeness_score=1.2,
                semantic_confidence_score=0.8,
                main_message_likelihood=0.3,
            )
