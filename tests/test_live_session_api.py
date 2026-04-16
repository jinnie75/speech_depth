import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import asr_viz.api.main as api_main
import asr_viz.db.session as db_session_module
import asr_viz.services.live_sessions as live_sessions_module
from asr_viz.db.base import Base
from asr_viz.models.analysis import AnalysisResult
from asr_viz.models.transcript import Transcript
from asr_viz.services import bootstrap as bootstrap_module


class LiveSessionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test_live_sessions.db"
        self.engine = create_engine(f"sqlite:///{database_path}", future=True)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(bind=self.engine)
        self.original_session_local = db_session_module.SessionLocal
        self.original_init_db = bootstrap_module.init_db
        self.original_app_get_session = api_main.app.dependency_overrides.copy()
        self.original_live_settings = live_sessions_module.settings

        db_session_module.SessionLocal = self.session_factory
        bootstrap_module.init_db = lambda: None
        api_main.app.dependency_overrides[api_main.get_session] = self._override_get_session
        live_sessions_module.settings = type("Settings", (), {"media_storage_dir": self.temp_dir.name})()

    def tearDown(self) -> None:
        db_session_module.SessionLocal = self.original_session_local
        bootstrap_module.init_db = self.original_init_db
        api_main.app.dependency_overrides = self.original_app_get_session
        live_sessions_module.settings = self.original_live_settings
        self.temp_dir.cleanup()

    def _override_get_session(self):
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    def test_live_session_lifecycle_creates_provisional_transcript_and_events(self) -> None:
        client = TestClient(api_main.app)

        create_response = client.post(
            "/live-sessions",
            json={
                "mime_type": "audio/webm",
                "original_filename": "live-note.webm",
                "sample_rate_hz": 48000,
                "channel_count": 1,
                "preferred_language": "en",
                "session_metadata": {"source": "browser-mic"},
            },
        )
        self.assertEqual(create_response.status_code, 201)
        session_payload = create_response.json()
        session_id = session_payload["id"]
        transcript_id = session_payload["transcript_id"]
        storage_path = Path(session_payload["storage_path"])

        self.assertTrue(storage_path.exists())
        self.assertEqual(session_payload["status"], "open")
        self.assertIsNotNone(session_payload["media_asset_id"])
        self.assertIsNotNone(transcript_id)

        upload_response = client.put(
            f"/live-sessions/{session_id}/chunks?chunk_index=0",
            content=b"fake-audio-frame",
            headers={"content-type": "application/octet-stream"},
        )
        self.assertEqual(upload_response.status_code, 202)

        transcript_event_response = client.post(
            f"/live-sessions/{session_id}/events",
            json={
                "event_type": "transcript.final",
                "utterance_key": "utt-1",
                "start_ms": 0,
                "end_ms": 1200,
                "text": "We should test this live.",
                "speaker_id": "SPEAKER_00",
                "is_final": True,
                "payload": {"confidence": 0.91},
            },
        )
        self.assertEqual(transcript_event_response.status_code, 201)

        analysis_event_response = client.post(
            f"/live-sessions/{session_id}/events",
            json={
                "event_type": "analysis.delta",
                "utterance_key": "utt-1",
                "is_final": True,
                "analysis": {
                    "politeness_score": 0.71,
                    "semantic_confidence_score": 0.66,
                    "main_message_likelihood": 0.88,
                    "analysis_model": "live-heuristic:test",
                    "analysis_payload": {"hedging": {"matches": ["should"]}},
                },
            },
        )
        self.assertEqual(analysis_event_response.status_code, 201)

        events_response = client.get(f"/live-sessions/{session_id}/events")
        self.assertEqual(events_response.status_code, 200)
        events_payload = events_response.json()
        self.assertEqual(len(events_payload["events"]), 2)
        self.assertEqual(events_payload["events"][0]["event_type"], "transcript.final")
        self.assertEqual(events_payload["events"][1]["event_type"], "analysis.delta")

        transcript_response = client.get(f"/transcripts/{transcript_id}")
        self.assertEqual(transcript_response.status_code, 200)
        transcript_payload = transcript_response.json()
        self.assertEqual(transcript_payload["full_text"], "We should test this live.")
        self.assertEqual(len(transcript_payload["sentence_units"]), 1)
        self.assertEqual(transcript_payload["sentence_units"][0]["text"], "We should test this live.")
        self.assertEqual(
            transcript_payload["sentence_units"][0]["sentence_metadata"]["live_utterance_key"],
            "utt-1",
        )
        self.assertEqual(
            transcript_payload["sentence_units"][0]["analysis_result"]["analysis_model"],
            "live-heuristic:test",
        )

        stop_response = client.post(f"/live-sessions/{session_id}/stop")
        self.assertEqual(stop_response.status_code, 200)
        self.assertEqual(stop_response.json()["status"], "stopped")

        finalize_response = client.post(f"/live-sessions/{session_id}/finalize")
        self.assertEqual(finalize_response.status_code, 200)
        finalize_payload = finalize_response.json()
        self.assertEqual(finalize_payload["status"], "completed")
        self.assertIsNotNone(finalize_payload["finalized_at"])

        with self.session_factory() as session:
            transcript = session.get(Transcript, transcript_id)
            self.assertIsNotNone(transcript)
            self.assertEqual(transcript.transcript_metadata["live_session_status"], "completed")
            analysis_results = session.query(AnalysisResult).all()
            self.assertEqual(len(analysis_results), 1)

    def test_chunk_indexes_must_be_sequential(self) -> None:
        client = TestClient(api_main.app)

        create_response = client.post(
            "/live-sessions",
            json={
                "mime_type": "audio/webm",
                "original_filename": "live-note.webm",
            },
        )
        self.assertEqual(create_response.status_code, 201)
        session_id = create_response.json()["id"]

        first_response = client.put(
            f"/live-sessions/{session_id}/chunks?chunk_index=0",
            content=b"chunk-0",
            headers={"content-type": "application/octet-stream"},
        )
        self.assertEqual(first_response.status_code, 202)

        skipped_response = client.put(
            f"/live-sessions/{session_id}/chunks?chunk_index=2",
            content=b"chunk-2",
            headers={"content-type": "application/octet-stream"},
        )
        self.assertEqual(skipped_response.status_code, 409)
        self.assertIn("sequential", skipped_response.json()["detail"])
