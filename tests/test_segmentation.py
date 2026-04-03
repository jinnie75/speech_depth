import unittest

from asr_viz.pipeline.segmentation import segment_transcript
from asr_viz.pipeline.types import ASRSegment


class SegmentationTests(unittest.TestCase):
    def test_segment_transcript_preserves_monotonic_ranges(self) -> None:
        segments = [
            ASRSegment(segment_index=0, start_ms=0, end_ms=1000, text="Hello there"),
            ASRSegment(segment_index=1, start_ms=1000, end_ms=2200, text="How are you?"),
            ASRSegment(segment_index=2, start_ms=2200, end_ms=3000, text="Let's review the budget."),
        ]

        sentences = segment_transcript(segments)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].start_ms, 0)
        self.assertEqual(sentences[0].end_ms, 2200)
        self.assertEqual(sentences[0].text, "Hello there How are you?")
        self.assertEqual(sentences[1].start_ms, 2200)
        self.assertEqual(sentences[1].end_ms, 3000)
        self.assertEqual(sentences[1].source_segment_ids, [2])
