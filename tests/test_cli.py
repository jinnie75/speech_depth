import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import asr_viz.api.main as api_main
import asr_viz.db.session as db_session_module
from asr_viz.db.base import Base
from asr_viz.services import bootstrap as bootstrap_module
from asr_viz.providers.analysis_v2 import HeuristicAnalysisProvider
from asr_viz.providers.diarization import NoOpDiarizationProvider
from asr_viz.providers.transcription import MockTranscriptionProvider
from asr_viz.services.pipeline import ProcessingPipeline
from asr_viz import cli


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test_cli.db"
        self.engine = create_engine(f"sqlite:///{database_path}", future=True)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(bind=self.engine)
        self.original_session_local = db_session_module.SessionLocal
        self.original_cli_session_local = cli.SessionLocal
        self.original_init_db = bootstrap_module.init_db
        self.original_cli_init_db = cli.init_db
        self.original_app_get_session = api_main.app.dependency_overrides.copy()
        db_session_module.SessionLocal = self.session_factory
        cli.SessionLocal = self.session_factory
        bootstrap_module.init_db = lambda: None
        cli.init_db = lambda: None
        api_main.app.dependency_overrides[api_main.get_session] = self._override_get_session

    def tearDown(self) -> None:
        db_session_module.SessionLocal = self.original_session_local
        cli.SessionLocal = self.original_cli_session_local
        bootstrap_module.init_db = self.original_init_db
        cli.init_db = self.original_cli_init_db
        api_main.app.dependency_overrides = self.original_app_get_session
        self.temp_dir.cleanup()

    def _override_get_session(self):
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    def test_submit_status_and_transcript_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "conversation.txt"
            source.write_text("Please review the design.\nWe should fix the bug tomorrow.", encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                cli.main.__wrapped__ if hasattr(cli.main, "__wrapped__") else None
                cli._submit(
                    type(
                        "Args",
                        (),
                        {
                            "source_uri": str(source),
                            "source_type": None,
                            "mime_type": "text/plain",
                            "language": "ko",
                            "diarization": False,
                            "metadata": "{\"source\":\"test\"}",
                        },
                    )()
                )
            submit_output = output.getvalue()
            self.assertIn("job_id=", submit_output)

            with self.session_factory() as session:
                pipeline = ProcessingPipeline(
                    transcription_provider=MockTranscriptionProvider(),
                    analysis_provider=HeuristicAnalysisProvider(),
                    diarization_provider=NoOpDiarizationProvider(),
                )
                claimed = pipeline.claim_next_job(session)
                self.assertIsNotNone(claimed)
                self.assertEqual(claimed.media_asset.ingest_metadata["preferred_language"], "ko")
                pipeline.process_job(session, claimed.id)

            with self.session_factory() as session:
                job = session.query(cli.ProcessingJob).order_by(cli.ProcessingJob.created_at.desc()).first()
                transcript_id = job.transcript_id

            status_output = io.StringIO()
            with redirect_stdout(status_output):
                cli._status(type("Args", (), {"job_id": None})())
            self.assertIn("status=completed", status_output.getvalue())

            transcript_output = io.StringIO()
            with redirect_stdout(transcript_output):
                cli._transcript(
                    type("Args", (), {"transcript_id": transcript_id, "sentences": 2, "preview_chars": 200})()
                )
            text = transcript_output.getvalue()
            self.assertIn("transcript_id=", text)
            self.assertIn("sentence_count=2", text)

    def test_api_lists_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "conversation.txt"
            source.write_text("Hello there.", encoding="utf-8")

            with self.session_factory() as session:
                cli.create_job(
                    session,
                    source_uri=str(source),
                    source_type="file",
                    diarization_enabled=False,
                    mime_type="text/plain",
                    checksum=None,
                    ingest_metadata={},
                )

            client = TestClient(api_main.app)
            response = client.get("/jobs")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(len(payload["jobs"]), 1)
            self.assertEqual(payload["jobs"][0]["status"], "queued")
