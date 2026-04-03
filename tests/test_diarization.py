import unittest

from asr_viz.pipeline.types import SentenceCandidate
from asr_viz.providers.diarization import SpeakerTurn, assign_speakers_by_overlap


class DiarizationAssignmentTests(unittest.TestCase):
    def test_assigns_speaker_by_greatest_overlap(self) -> None:
        sentences = [
            SentenceCandidate(utterance_index=0, start_ms=0, end_ms=1000, text="Hi there."),
            SentenceCandidate(utterance_index=1, start_ms=1000, end_ms=2200, text="How are you?"),
        ]
        turns = [
            SpeakerTurn(speaker_id="SPEAKER_00", start_ms=0, end_ms=700),
            SpeakerTurn(speaker_id="SPEAKER_01", start_ms=700, end_ms=2200),
        ]

        assigned = assign_speakers_by_overlap(sentences, turns)

        self.assertEqual(assigned[0].speaker_id, "SPEAKER_00")
        self.assertEqual(assigned[1].speaker_id, "SPEAKER_01")
        self.assertGreater(assigned[0].speaker_confidence, 0.69)
        self.assertGreater(assigned[1].speaker_confidence, 0.99)

    def test_leaves_sentence_unassigned_when_no_overlap_exists(self) -> None:
        sentences = [
            SentenceCandidate(utterance_index=0, start_ms=0, end_ms=500, text="Hello."),
        ]
        turns = [
            SpeakerTurn(speaker_id="SPEAKER_00", start_ms=1000, end_ms=1500),
        ]

        assigned = assign_speakers_by_overlap(sentences, turns)

        self.assertIsNone(assigned[0].speaker_id)
        self.assertIsNone(assigned[0].speaker_confidence)
