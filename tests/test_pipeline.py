import shutil
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from asr_viz.db.base import Base
from asr_viz.models.analysis import AnalysisResult
from asr_viz.models.job import ProcessingJob
from asr_viz.models.transcript import SentenceUnit, Transcript
from asr_viz.providers.analysis_v2 import HeuristicAnalysisProvider
from asr_viz.providers.diarization import DiarizationProvider, DiarizationUnavailableError, NoOpDiarizationProvider
from asr_viz.providers.transcription import MockTranscriptionProvider
from asr_viz.services.jobs import create_job
from asr_viz.services.pipeline import ProcessingPipeline, _speaker_count_override


class RecordingDiarizationProvider(DiarizationProvider):
    model_version = "recording-diarizer:v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.release_calls = 0

    def assign_speakers(
        self,
        sentences,
        source_uri: str,
        *,
        num_speakers_override: int | None = None,
    ):
        self.calls.append((source_uri, num_speakers_override))
        return sentences

    def release_resources(self) -> None:
        self.release_calls += 1


class FailingDiarizationProvider(DiarizationProvider):
    model_version = "failing-diarizer:v1"

    def __init__(self) -> None:
        self.release_calls = 0

    def assign_speakers(
        self,
        sentences,
        source_uri: str,
        *,
        num_speakers_override: int | None = None,
    ):
        raise DiarizationUnavailableError("Diarization model is unavailable for this environment.")

    def release_resources(self) -> None:
        self.release_calls += 1


class RecordingTranscriptionProvider(MockTranscriptionProvider):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls = 0

    def release_resources(self) -> None:
        self.release_calls += 1


class PipelineTests(unittest.TestCase):
    def test_speaker_count_override_prefers_explicit_ingest_metadata_value(self) -> None:
        self.assertEqual(_speaker_count_override({"diarization_num_speakers": 1}), 1)
        self.assertEqual(_speaker_count_override({"diarization_num_speakers": "2"}), 2)
        self.assertEqual(_speaker_count_override({"diarization_num_speakers": 3}), 3)
        self.assertIsNone(_speaker_count_override({"diarization_num_speakers": 4}))

    def test_pipeline_processes_job_end_to_end(self) -> None:
        tmp_path = Path(self._testMethodName)
        tmp_path.mkdir(exist_ok=True)
        source = tmp_path / "conversation.txt"
        source.write_text("Please review the design.\nWe should fix the bug tomorrow.", encoding="utf-8")

        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))

        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        pipeline = ProcessingPipeline(
            transcription_provider=MockTranscriptionProvider(),
            analysis_provider=HeuristicAnalysisProvider(),
            diarization_provider=NoOpDiarizationProvider(),
        )

        with session_factory() as session:
            create_job(
                session,
                source_uri=str(source),
                source_type="file",
                mime_type="text/plain",
                checksum=None,
                ingest_metadata={"test": True},
            )

        with session_factory() as session:
            claimed_job = pipeline.claim_next_job(session)
            self.assertIsNotNone(claimed_job)
            result_job = pipeline.process_job(session, claimed_job.id)
            self.assertEqual(result_job.status, "completed")
            self.assertIsNotNone(result_job.transcript_id)

        with session_factory() as session:
            transcript = session.scalar(select(Transcript))
            analysis_results = session.scalars(select(AnalysisResult)).all()
            sentence_units = session.scalars(select(SentenceUnit).order_by(SentenceUnit.utterance_index.asc())).all()
            job = session.scalar(select(ProcessingJob))

            self.assertIsNotNone(transcript)
            self.assertIsNotNone(job)
            self.assertEqual(len(sentence_units), 2)
            self.assertEqual(len(analysis_results), 2)
            self.assertTrue(all(sentence.speaker_id == "SPEAKER_00" for sentence in sentence_units))
            self.assertGreaterEqual(analysis_results[0].politeness_score, 0.0)
            self.assertLessEqual(analysis_results[0].semantic_confidence_score, 1.0)
            self.assertIn("hedging", analysis_results[0].analysis_payload)
            self.assertIn("substance", analysis_results[0].analysis_payload)
            self.assertIn("sentence_count", job.stage_details)
            self.assertEqual(job.asr_model_version, "mock-transcriber:v1")
            self.assertEqual(job.stage_details["diarization_skip_reason"], "provider_disabled")

    def test_pipeline_uses_preferred_language_for_korean_jobs(self) -> None:
        tmp_path = Path(self._testMethodName)
        tmp_path.mkdir(exist_ok=True)
        source = tmp_path / "conversation_ko.txt"
        source.write_text("안녕하세요.\n오늘 일정 이야기해요.", encoding="utf-8")

        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))

        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        pipeline = ProcessingPipeline(
            transcription_provider=MockTranscriptionProvider(),
            analysis_provider=HeuristicAnalysisProvider(),
            diarization_provider=NoOpDiarizationProvider(),
        )

        with session_factory() as session:
            create_job(
                session,
                source_uri=str(source),
                source_type="file",
                mime_type="text/plain",
                checksum=None,
                ingest_metadata={"preferred_language": "ko"},
            )

        with session_factory() as session:
            claimed_job = pipeline.claim_next_job(session)
            self.assertIsNotNone(claimed_job)
            result_job = pipeline.process_job(session, claimed_job.id)
            self.assertEqual(result_job.status, "completed")

        with session_factory() as session:
            transcript = session.scalar(select(Transcript))
            analysis_results = session.scalars(select(AnalysisResult)).all()
            job = session.scalar(select(ProcessingJob))

            self.assertIsNotNone(transcript)
            self.assertEqual(transcript.language_code, "ko")
            self.assertEqual(job.stage_details["preferred_language"], "ko")
            self.assertTrue(analysis_results[0].analysis_payload["language_supported"])
            self.assertIn("hedging", analysis_results[0].analysis_payload)
            self.assertIn("substance", analysis_results[0].analysis_payload)

    def test_pipeline_skips_diarization_for_single_speaker_jobs(self) -> None:
        tmp_path = Path(self._testMethodName)
        tmp_path.mkdir(exist_ok=True)
        source = tmp_path / "monologue.txt"
        source.write_text("I have one thing to say.\nThis is still the same speaker.", encoding="utf-8")

        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))

        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        diarization_provider = RecordingDiarizationProvider()
        pipeline = ProcessingPipeline(
            transcription_provider=MockTranscriptionProvider(),
            analysis_provider=HeuristicAnalysisProvider(),
            diarization_provider=diarization_provider,
        )

        with session_factory() as session:
            create_job(
                session,
                source_uri=str(source),
                source_type="file",
                mime_type="text/plain",
                checksum=None,
                ingest_metadata={"speaker_mode": "monologue"},
            )

        with session_factory() as session:
            claimed_job = pipeline.claim_next_job(session)
            self.assertIsNotNone(claimed_job)
            result_job = pipeline.process_job(session, claimed_job.id)
            self.assertEqual(result_job.status, "completed")

        with session_factory() as session:
            sentence_units = session.scalars(select(SentenceUnit).order_by(SentenceUnit.utterance_index.asc())).all()
            job = session.scalar(select(ProcessingJob))

            self.assertEqual(diarization_provider.calls, [])
            self.assertIsNotNone(job)
            self.assertEqual(job.diarization_model_version, None)
            self.assertEqual(job.stage_details["requested_num_speakers"], 1)
            self.assertEqual(job.stage_details["diarization_skipped"], True)
            self.assertEqual(job.stage_details["diarization_skip_reason"], "single_speaker")
            self.assertEqual(job.stage_details["diarization_enabled"], False)
            self.assertTrue(all(sentence.speaker_id == "SPEAKER_00" for sentence in sentence_units))

    def test_pipeline_skips_diarization_when_provider_is_disabled(self) -> None:
        tmp_path = Path(self._testMethodName)
        tmp_path.mkdir(exist_ok=True)
        source = tmp_path / "dialogue.txt"
        source.write_text("Please review the design.\nWe should fix the bug tomorrow.", encoding="utf-8")

        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))

        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        pipeline = ProcessingPipeline(
            transcription_provider=MockTranscriptionProvider(),
            analysis_provider=HeuristicAnalysisProvider(),
            diarization_provider=NoOpDiarizationProvider(),
        )

        with session_factory() as session:
            create_job(
                session,
                source_uri=str(source),
                source_type="file",
                mime_type="text/plain",
                checksum=None,
                ingest_metadata={"speaker_mode": "dialogue"},
            )

        with session_factory() as session:
            claimed_job = pipeline.claim_next_job(session)
            self.assertIsNotNone(claimed_job)
            result_job = pipeline.process_job(session, claimed_job.id)
            self.assertEqual(result_job.status, "completed")

        with session_factory() as session:
            sentence_units = session.scalars(select(SentenceUnit).order_by(SentenceUnit.utterance_index.asc())).all()
            job = session.scalar(select(ProcessingJob))

            self.assertIsNotNone(job)
            self.assertTrue(all(sentence.speaker_id == "SPEAKER_00" for sentence in sentence_units))
            self.assertIsNone(job.diarization_model_version)
            self.assertEqual(job.stage_details["requested_num_speakers"], 2)
            self.assertEqual(job.stage_details["diarization_enabled"], False)
            self.assertEqual(job.stage_details["diarization_skipped"], True)
            self.assertEqual(job.stage_details["diarization_skip_reason"], "provider_disabled")

    def test_pipeline_completes_when_diarization_provider_is_unavailable(self) -> None:
        tmp_path = Path(self._testMethodName)
        tmp_path.mkdir(exist_ok=True)
        source = tmp_path / "dialogue.txt"
        source.write_text("Please review the design.\nWe should fix the bug tomorrow.", encoding="utf-8")

        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))

        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        pipeline = ProcessingPipeline(
            transcription_provider=MockTranscriptionProvider(),
            analysis_provider=HeuristicAnalysisProvider(),
            diarization_provider=FailingDiarizationProvider(),
        )

        with session_factory() as session:
            create_job(
                session,
                source_uri=str(source),
                source_type="file",
                mime_type="text/plain",
                checksum=None,
                ingest_metadata={"speaker_mode": "dialogue"},
            )

        with session_factory() as session:
            claimed_job = pipeline.claim_next_job(session)
            self.assertIsNotNone(claimed_job)
            result_job = pipeline.process_job(session, claimed_job.id)
            self.assertEqual(result_job.status, "completed")
            self.assertIsNotNone(result_job.transcript_id)
            self.assertIsNone(result_job.error_message)

        with session_factory() as session:
            sentence_units = session.scalars(select(SentenceUnit).order_by(SentenceUnit.utterance_index.asc())).all()
            analysis_results = session.scalars(select(AnalysisResult)).all()
            job = session.scalar(select(ProcessingJob))

            self.assertEqual(len(sentence_units), 2)
            self.assertEqual(len(analysis_results), 2)
            self.assertIsNotNone(job)
            self.assertTrue(all(sentence.speaker_id == "SPEAKER_00" for sentence in sentence_units))
            self.assertIsNone(job.diarization_model_version)
            self.assertEqual(job.stage_details["requested_num_speakers"], 2)
            self.assertEqual(job.stage_details["diarization_enabled"], False)
            self.assertEqual(job.stage_details["diarization_failed"], True)
            self.assertEqual(job.stage_details["diarization_failure_reason"], "provider_unavailable")
            self.assertIn("Diarization model is unavailable", job.stage_details["diarization_error"])

    def test_pipeline_releases_heavy_provider_resources_between_stages(self) -> None:
        tmp_path = Path(self._testMethodName)
        tmp_path.mkdir(exist_ok=True)
        source = tmp_path / "dialogue.txt"
        source.write_text("Please review the design.\nWe should fix the bug tomorrow.", encoding="utf-8")

        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))

        engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        transcription_provider = RecordingTranscriptionProvider()
        diarization_provider = RecordingDiarizationProvider()
        pipeline = ProcessingPipeline(
            transcription_provider=transcription_provider,
            analysis_provider=HeuristicAnalysisProvider(),
            diarization_provider=diarization_provider,
        )

        with session_factory() as session:
            create_job(
                session,
                source_uri=str(source),
                source_type="file",
                mime_type="text/plain",
                checksum=None,
                ingest_metadata={"speaker_mode": "dialogue"},
            )

        with session_factory() as session:
            claimed_job = pipeline.claim_next_job(session)
            self.assertIsNotNone(claimed_job)
            result_job = pipeline.process_job(session, claimed_job.id)
            self.assertEqual(result_job.status, "completed")

        self.assertEqual(diarization_provider.calls, [(str(source), 2)])
        self.assertEqual(transcription_provider.release_calls, 1)
        self.assertEqual(diarization_provider.release_calls, 1)
