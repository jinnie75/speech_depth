import unittest

from asr_viz.pipeline.segmentation import segment_transcript
from asr_viz.pipeline.types import ASRSegment, ASRWord


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

    def test_segment_transcript_splits_multiple_sentences_within_single_segment(self) -> None:
        segments = [
            ASRSegment(
                segment_index=0,
                start_ms=0,
                end_ms=2200,
                text="Hello there. How are you?",
                words=[
                    ASRWord(word="Hello", start_ms=0, end_ms=350, probability=0.97),
                    ASRWord(word="there.", start_ms=350, end_ms=900, probability=0.95),
                    ASRWord(word="How", start_ms=900, end_ms=1200, probability=0.94),
                    ASRWord(word="are", start_ms=1200, end_ms=1500, probability=0.93),
                    ASRWord(word="you?", start_ms=1500, end_ms=2200, probability=0.92),
                ],
            )
        ]

        sentences = segment_transcript(segments)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].text, "Hello there.")
        self.assertEqual(sentences[0].start_ms, 0)
        self.assertEqual(sentences[0].end_ms, 900)
        self.assertEqual(sentences[0].source_segment_ids, [0])
        self.assertEqual(sentences[1].text, "How are you?")
        self.assertEqual(sentences[1].start_ms, 900)
        self.assertEqual(sentences[1].end_ms, 2200)
        self.assertEqual(sentences[1].source_segment_ids, [0])

    def test_segment_transcript_splits_on_capitalized_sentence_starter_without_punctuation(self) -> None:
        segments = [
            ASRSegment(
                segment_index=0,
                start_ms=0,
                end_ms=3400,
                text="Can we please just talk about this Don't do that",
                words=[
                    ASRWord(word="Can", start_ms=0, end_ms=250, probability=0.97),
                    ASRWord(word="we", start_ms=250, end_ms=450, probability=0.96),
                    ASRWord(word="please", start_ms=450, end_ms=850, probability=0.96),
                    ASRWord(word="just", start_ms=850, end_ms=1150, probability=0.95),
                    ASRWord(word="talk", start_ms=1150, end_ms=1450, probability=0.95),
                    ASRWord(word="about", start_ms=1450, end_ms=1750, probability=0.94),
                    ASRWord(word="this", start_ms=1750, end_ms=2050, probability=0.94),
                    ASRWord(word="Don't", start_ms=2450, end_ms=2850, probability=0.95),
                    ASRWord(word="do", start_ms=2850, end_ms=3050, probability=0.94),
                    ASRWord(word="that", start_ms=3050, end_ms=3400, probability=0.94),
                ],
            )
        ]

        sentences = segment_transcript(segments)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].text, "Can we please just talk about this")
        self.assertEqual(sentences[0].end_ms, 2050)
        self.assertEqual(sentences[1].text, "Don't do that")
        self.assertEqual(sentences[1].start_ms, 2450)

    def test_segment_transcript_does_not_split_on_ellipsis(self) -> None:
        segments = [
            ASRSegment(
                segment_index=0,
                start_ms=0,
                end_ms=3900,
                text="I was really deeply thinking... Maybe later",
                words=[
                    ASRWord(word="I", start_ms=0, end_ms=200, probability=0.97),
                    ASRWord(word="was", start_ms=200, end_ms=450, probability=0.97),
                    ASRWord(word="really", start_ms=450, end_ms=900, probability=0.96),
                    ASRWord(word="deeply", start_ms=900, end_ms=1400, probability=0.96),
                    ASRWord(word="thinking...", start_ms=1400, end_ms=2100, probability=0.95),
                    ASRWord(word="Maybe", start_ms=2600, end_ms=3100, probability=0.95),
                    ASRWord(word="later", start_ms=3100, end_ms=3900, probability=0.95),
                ],
            )
        ]

        sentences = segment_transcript(segments)

        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0].text, "I was really deeply thinking... Maybe later")

    def test_segment_transcript_does_not_split_on_common_titles(self) -> None:
        segments = [
            ASRSegment(
                segment_index=0,
                start_ms=0,
                end_ms=4200,
                text="I spoke to Dr. Smith yesterday. He agreed right away.",
                words=[
                    ASRWord(word="I", start_ms=0, end_ms=150, probability=0.97),
                    ASRWord(word="spoke", start_ms=150, end_ms=500, probability=0.97),
                    ASRWord(word="to", start_ms=500, end_ms=700, probability=0.97),
                    ASRWord(word="Dr.", start_ms=700, end_ms=950, probability=0.97),
                    ASRWord(word="Smith", start_ms=950, end_ms=1350, probability=0.97),
                    ASRWord(word="yesterday.", start_ms=1350, end_ms=2200, probability=0.97),
                    ASRWord(word="He", start_ms=2200, end_ms=2450, probability=0.97),
                    ASRWord(word="agreed", start_ms=2450, end_ms=2950, probability=0.97),
                    ASRWord(word="right", start_ms=2950, end_ms=3350, probability=0.97),
                    ASRWord(word="away.", start_ms=3350, end_ms=4200, probability=0.97),
                ],
            )
        ]

        sentences = segment_transcript(segments)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].text, "I spoke to Dr. Smith yesterday.")
        self.assertEqual(sentences[1].text, "He agreed right away.")

    def test_segment_transcript_splits_korean_sentences_on_terminal_punctuation(self) -> None:
        segments = [
            ASRSegment(
                segment_index=0,
                start_ms=0,
                end_ms=2400,
                text="안녕하세요. 오늘 일정 괜찮아요?",
                words=[
                    ASRWord(word="안녕하세요.", start_ms=0, end_ms=900, probability=0.97),
                    ASRWord(word="오늘", start_ms=900, end_ms=1300, probability=0.96),
                    ASRWord(word="일정", start_ms=1300, end_ms=1700, probability=0.95),
                    ASRWord(word="괜찮아요?", start_ms=1700, end_ms=2400, probability=0.94),
                ],
            )
        ]

        sentences = segment_transcript(segments, language_code="ko")

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].text, "안녕하세요.")
        self.assertEqual(sentences[0].end_ms, 900)
        self.assertEqual(sentences[1].text, "오늘 일정 괜찮아요?")
        self.assertEqual(sentences[1].start_ms, 900)

    def test_segment_transcript_keeps_korean_segment_boundaries_without_punctuation(self) -> None:
        segments = [
            ASRSegment(segment_index=0, start_ms=0, end_ms=1000, text="오늘은 회의 준비"),
            ASRSegment(segment_index=1, start_ms=1000, end_ms=2100, text="내일은 발표 연습"),
        ]

        sentences = segment_transcript(segments, language_code="ko")

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].text, "오늘은 회의 준비")
        self.assertEqual(sentences[0].source_segment_ids, [0])
        self.assertEqual(sentences[1].text, "내일은 발표 연습")
        self.assertEqual(sentences[1].source_segment_ids, [1])
