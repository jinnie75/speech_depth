import unittest

from asr_viz.providers.diarization import PyannoteDiarizationProvider, SpeakerTurn


class _FakeSegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeDiarizationResult:
    def itertracks(self, yield_label: bool = False):
        yield _FakeSegment(0.0, 1.0), None, "SPEAKER_00"


class _FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, file, **kwargs):
        self.calls.append((file, kwargs))
        return _FakeDiarizationResult()


class _WrappedDiarizationResult:
    def __init__(self) -> None:
        self.speaker_diarization = _FakeDiarizationResult()


class _WrappedPipeline:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, file, **kwargs):
        self.calls.append((file, kwargs))
        return _WrappedDiarizationResult()


class PyannoteProviderTests(unittest.TestCase):
    def test_extract_turns_calls_pipeline_with_file_positional_arg(self) -> None:
        provider = PyannoteDiarizationProvider(model_name="test-model", token="token", num_speakers=2)
        fake_pipeline = _FakePipeline()
        provider._pipeline = fake_pipeline

        turns = provider._extract_turns("/tmp/example.wav")

        self.assertEqual(fake_pipeline.calls, [("/tmp/example.wav", {"num_speakers": 2})])
        self.assertEqual(turns, [SpeakerTurn(speaker_id="SPEAKER_00", start_ms=0, end_ms=1000)])

    def test_extract_turns_uses_override_exact_count_when_provided(self) -> None:
        provider = PyannoteDiarizationProvider(
            model_name="test-model",
            token="token",
            num_speakers=None,
            min_speakers=1,
            max_speakers=2,
        )
        fake_pipeline = _FakePipeline()
        provider._pipeline = fake_pipeline

        turns = provider._extract_turns("/tmp/example.wav", num_speakers_override=1)

        self.assertEqual(fake_pipeline.calls, [("/tmp/example.wav", {"num_speakers": 1})])
        self.assertEqual(turns, [SpeakerTurn(speaker_id="SPEAKER_00", start_ms=0, end_ms=1000)])

    def test_extract_turns_uses_min_max_speaker_range_when_exact_count_is_not_forced(self) -> None:
        provider = PyannoteDiarizationProvider(
            model_name="test-model",
            token="token",
            num_speakers=None,
            min_speakers=1,
            max_speakers=2,
        )
        fake_pipeline = _WrappedPipeline()
        provider._pipeline = fake_pipeline

        turns = provider._extract_turns("/tmp/example.wav")

        self.assertEqual(fake_pipeline.calls, [("/tmp/example.wav", {"min_speakers": 1, "max_speakers": 2})])
        self.assertEqual(turns, [SpeakerTurn(speaker_id="SPEAKER_00", start_ms=0, end_ms=1000)])
