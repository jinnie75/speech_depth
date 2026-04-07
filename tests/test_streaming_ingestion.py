import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import asr_viz.api.main as api_main
import asr_viz.db.session as db_session_module
import asr_viz.services.streaming as streaming_module
from asr_viz.db.base import Base
from asr_viz.providers.analysis_v2 import HeuristicAnalysisProvider
from asr_viz.providers.diarization import NoOpDiarizationProvider
from asr_viz.providers.transcription import MockTranscriptionProvider
from asr_viz.services import bootstrap as bootstrap_module
from asr_viz.services.pipeline import ProcessingPipeline


class StreamingIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test_streaming.db"
        self.engine = create_engine(f"sqlite:///{database_path}", future=True)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(bind=self.engine)
        self.original_session_local = db_session_module.SessionLocal
        self.original_init_db = bootstrap_module.init_db
        self.original_app_get_session = api_main.app.dependency_overrides.copy()
        self.original_stream_settings = streaming_module.settings

        db_session_module.SessionLocal = self.session_factory
        bootstrap_module.init_db = lambda: None
        api_main.app.dependency_overrides[api_main.get_session] = self._override_get_session
        streaming_module.settings = type("Settings", (), {"media_storage_dir": self.temp_dir.name})()

    def tearDown(self) -> None:
        db_session_module.SessionLocal = self.original_session_local
        bootstrap_module.init_db = self.original_init_db
        api_main.app.dependency_overrides = self.original_app_get_session
        streaming_module.settings = self.original_stream_settings
        self.temp_dir.cleanup()

    def _override_get_session(self):
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    def test_stream_session_upload_finalize_and_process(self) -> None:
        client = TestClient(api_main.app)

        create_response = client.post(
            "/stream-sessions",
            json={
                "mime_type": "text/plain",
                "original_filename": "conversation.txt",
                "diarization_enabled": False,
                "ingest_metadata": {"source": "test"},
            },
        )
        self.assertEqual(create_response.status_code, 201)
        session_payload = create_response.json()
        session_id = session_payload["id"]
        storage_path = Path(session_payload["storage_path"])

        first_chunk = b"Please review the design.\n"
        second_chunk = b"We should fix the bug tomorrow.\n"

        upload_one = client.put(
            f"/stream-sessions/{session_id}/chunks",
            content=first_chunk,
            headers={"content-type": "application/octet-stream"},
        )
        upload_two = client.put(
            f"/stream-sessions/{session_id}/chunks",
            content=second_chunk,
            headers={"content-type": "application/octet-stream"},
        )
        self.assertEqual(upload_one.status_code, 202)
        self.assertEqual(upload_two.status_code, 202)
        self.assertTrue(storage_path.exists())
        self.assertEqual(storage_path.read_bytes(), first_chunk + second_chunk)

        status_response = client.get(f"/stream-sessions/{session_id}")
        self.assertEqual(status_response.status_code, 200)
        status_payload = status_response.json()
        self.assertEqual(status_payload["status"], "open")
        self.assertEqual(status_payload["received_chunks"], 2)
        self.assertEqual(status_payload["total_bytes"], len(first_chunk) + len(second_chunk))

        finalize_response = client.post(f"/stream-sessions/{session_id}/finalize")
        self.assertEqual(finalize_response.status_code, 200)
        finalize_payload = finalize_response.json()
        self.assertEqual(finalize_payload["status"], "queued")
        self.assertIsNotNone(finalize_payload["processing_job_id"])

        with self.session_factory() as session:
            pipeline = ProcessingPipeline(
                transcription_provider=MockTranscriptionProvider(),
                analysis_provider=HeuristicAnalysisProvider(),
                diarization_provider=NoOpDiarizationProvider(),
            )
            claimed = pipeline.claim_next_job(session)
            self.assertIsNotNone(claimed)
            pipeline.process_job(session, claimed.id)

        completed_response = client.get(f"/stream-sessions/{session_id}")
        self.assertEqual(completed_response.status_code, 200)
        completed_payload = completed_response.json()
        self.assertEqual(completed_payload["status"], "completed")
        self.assertEqual(completed_payload["processing_job_status"], "completed")
        self.assertIsNotNone(completed_payload["transcript_id"])
