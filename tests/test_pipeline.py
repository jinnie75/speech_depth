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
from asr_viz.providers.diarization import NoOpDiarizationProvider
from asr_viz.providers.transcription import MockTranscriptionProvider
from asr_viz.services.jobs import create_job
from asr_viz.services.pipeline import ProcessingPipeline, _speaker_count_override


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
                diarization_enabled=False,
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
            self.assertIsNone(sentence_units[0].speaker_id)
            self.assertGreaterEqual(analysis_results[0].politeness_score, 0.0)
            self.assertLessEqual(analysis_results[0].semantic_confidence_score, 1.0)
            self.assertIn("hedging", analysis_results[0].analysis_payload)
            self.assertIn("substance", analysis_results[0].analysis_payload)
            self.assertIn("sentence_count", job.stage_details)
            self.assertEqual(job.asr_model_version, "mock-transcriber:v1")

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
                diarization_enabled=False,
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
