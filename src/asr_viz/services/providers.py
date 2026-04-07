from asr_viz.core.settings import settings
from asr_viz.providers.analysis_v2 import AnalysisProvider, HeuristicAnalysisProvider
from asr_viz.providers.diarization import DiarizationProvider, NoOpDiarizationProvider, PyannoteDiarizationProvider
from asr_viz.providers.transcription import (
    FasterWhisperTranscriptionProvider,
    MockTranscriptionProvider,
    TranscriptionProvider,
)


def build_transcription_provider() -> TranscriptionProvider:
    if settings.enable_mock_transcription:
        return MockTranscriptionProvider()
    return FasterWhisperTranscriptionProvider(settings.asr_model)


def build_diarization_provider() -> DiarizationProvider:
    if settings.huggingface_token:
        return PyannoteDiarizationProvider(
            model_name=settings.diarization_model,
            token=settings.huggingface_token,
            num_speakers=settings.diarization_num_speakers,
            min_speakers=settings.diarization_min_speakers,
            max_speakers=settings.diarization_max_speakers,
        )
    return NoOpDiarizationProvider()


def build_analysis_provider() -> AnalysisProvider:
    return HeuristicAnalysisProvider()
