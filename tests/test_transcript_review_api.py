import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import asr_viz.api.main as api_main
import asr_viz.db.session as db_session_module
from asr_viz.db.base import Base
from asr_viz.models.common import JobStage, JobStatus
from asr_viz.models.job import ProcessingJob
from asr_viz.models.media import MediaAsset
from asr_viz.models.transcript import SentenceUnit, Transcript
from asr_viz.services import bootstrap as bootstrap_module


class TranscriptReviewApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test_review.db"
        self.engine = create_engine(f"sqlite:///{database_path}", future=True)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(bind=self.engine)
        self.original_session_local = db_session_module.SessionLocal
        self.original_init_db = bootstrap_module.init_db
        self.original_app_get_session = api_main.app.dependency_overrides.copy()

        db_session_module.SessionLocal = self.session_factory
        bootstrap_module.init_db = lambda: None
        api_main.app.dependency_overrides[api_main.get_session] = self._override_get_session

        self.transcript_id, self.sentence_ids = self._seed_transcript()

    def tearDown(self) -> None:
        db_session_module.SessionLocal = self.original_session_local
        bootstrap_module.init_db = self.original_init_db
        api_main.app.dependency_overrides = self.original_app_get_session
        self.temp_dir.cleanup()

    def _override_get_session(self):
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    def _seed_transcript(self) -> tuple[str, list[str]]:
        with self.session_factory() as session:
            media_asset = MediaAsset(
                source_type="file",
                source_uri=str(Path(self.temp_dir.name) / "conversation.mp4"),
                mime_type="video/mp4",
                checksum="checksum",
                ingest_metadata={"original_filename": "conversation.mp4"},
            )
            session.add(media_asset)
            session.flush()

            transcript = Transcript(
                media_asset_id=media_asset.id,
                language_code="en",
                full_text="Hello there. General Kenobi.",
                transcript_metadata={},
            )
            session.add(transcript)
            session.flush()

            first_sentence = SentenceUnit(
                transcript_id=transcript.id,
                utterance_index=0,
                start_ms=0,
                end_ms=1000,
                text="Hello there.",
                speaker_id="SPEAKER_00",
                speaker_confidence=0.95,
                source_segment_ids=[0],
                sentence_metadata={},
            )
            second_sentence = SentenceUnit(
                transcript_id=transcript.id,
                utterance_index=1,
                start_ms=1000,
                end_ms=2200,
                text="General Kenobi.",
                speaker_id="SPEAKER_01",
                speaker_confidence=0.97,
                source_segment_ids=[1],
                sentence_metadata={},
            )
            session.add_all([first_sentence, second_sentence])
            session.flush()

            job = ProcessingJob(
                media_asset_id=media_asset.id,
                transcript_id=transcript.id,
                status=JobStatus.COMPLETED.value,
                current_stage=JobStage.COMPLETE.value,
                diarization_enabled=True,
                stage_details={"status": "completed"},
            )
            session.add(job)
            session.flush()

            transcript.job_id = job.id
            session.commit()

            return transcript.id, [first_sentence.id, second_sentence.id]

    def test_get_transcript_defaults_to_unreviewed_fields(self) -> None:
        client = TestClient(api_main.app)

        response = client.get(f"/transcripts/{self.transcript_id}")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["conversation_title"])
        self.assertEqual(payload["review_status"], "not_started")
        self.assertEqual(payload["speaker_labels"], {})
        self.assertEqual(payload["sentence_units"][0]["display_text"], "Hello there.")
        self.assertEqual(payload["sentence_units"][0]["display_speaker_id"], "SPEAKER_00")
        self.assertFalse(payload["sentence_units"][0]["is_edited"])

    def test_patch_review_saves_transcript_and_sentence_overrides(self) -> None:
        client = TestClient(api_main.app)

        response = client.patch(
            f"/transcripts/{self.transcript_id}/review",
            json={
                "conversation_title": "Design Interview",
                "speaker_labels": {
                    "SPEAKER_00": "Jinhee",
                    "SPEAKER_01": "Guest",
                },
                "review_status": "in_progress",
                "sentence_overrides": [
                    {
                        "sentence_unit_id": self.sentence_ids[0],
                        "manual_text": "Hello there, team.",
                        "manual_speaker_id": "SPEAKER_01",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["conversation_title"], "Design Interview")
        self.assertEqual(payload["speaker_labels"]["SPEAKER_00"], "Jinhee")
        self.assertEqual(payload["review_status"], "in_progress")
        self.assertIsNotNone(payload["reviewed_at"])
        self.assertEqual(payload["sentence_units"][0]["display_text"], "Hello there, team.")
        self.assertEqual(payload["sentence_units"][0]["display_speaker_id"], "SPEAKER_01")
        self.assertEqual(payload["sentence_units"][0]["manual_text"], "Hello there, team.")
        self.assertEqual(payload["sentence_units"][0]["manual_speaker_id"], "SPEAKER_01")
        self.assertTrue(payload["sentence_units"][0]["is_edited"])

        jobs_response = client.get("/jobs")
        self.assertEqual(jobs_response.status_code, 200)
        jobs_payload = jobs_response.json()
        self.assertEqual(jobs_payload["jobs"][0]["conversation_title"], "Design Interview")
        self.assertEqual(jobs_payload["jobs"][0]["review_status"], "in_progress")
        self.assertIsNotNone(jobs_payload["jobs"][0]["archive_preview"])
        self.assertEqual(jobs_payload["jobs"][0]["archive_preview"]["speakers"][0]["label"], "Guest")
        self.assertEqual(jobs_payload["jobs"][0]["archive_preview"]["current_time_ms"], 2200)

    def test_jobs_backfill_archive_preview_for_legacy_transcripts(self) -> None:
        client = TestClient(api_main.app)

        jobs_response = client.get("/jobs")

        self.assertEqual(jobs_response.status_code, 200)
        jobs_payload = jobs_response.json()
        self.assertEqual(len(jobs_payload["jobs"]), 1)
        preview = jobs_payload["jobs"][0]["archive_preview"]
        self.assertIsNotNone(preview)
        self.assertEqual(preview["speakers"][0]["label"], "Speaker 1")
        self.assertEqual(preview["utterances"][0]["progress"], 1.0)
        self.assertEqual(preview["active_transcript"]["utterance_id"], self.sentence_ids[1])

        with self.session_factory() as session:
            transcript = session.get(Transcript, self.transcript_id)
            self.assertIsNotNone(transcript)
            self.assertIn("archive_preview", transcript.transcript_metadata)

    def test_jobs_completed_only_supports_limit_offset_and_total(self) -> None:
        self._seed_transcript()
        client = TestClient(api_main.app)

        response = client.get("/jobs?completed_only=true&limit=1&offset=1")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(len(payload["jobs"]), 1)
        self.assertIsNotNone(payload["jobs"][0]["transcript_id"])

    def test_patch_review_rejects_invalid_speaker_and_sentence_ids(self) -> None:
        client = TestClient(api_main.app)

        invalid_speaker_response = client.patch(
            f"/transcripts/{self.transcript_id}/review",
            json={
                "conversation_title": "Review",
                "speaker_labels": {"UNKNOWN": "Ghost"},
                "review_status": "in_progress",
                "sentence_overrides": [],
            },
        )
        self.assertEqual(invalid_speaker_response.status_code, 400)
        self.assertIn("invalid speaker id", invalid_speaker_response.json()["detail"])

        invalid_sentence_response = client.patch(
            f"/transcripts/{self.transcript_id}/review",
            json={
                "conversation_title": "Review",
                "speaker_labels": {},
                "review_status": "in_progress",
                "sentence_overrides": [
                    {
                        "sentence_unit_id": "missing-sentence",
                        "manual_text": "Edited",
                        "manual_speaker_id": "SPEAKER_00",
                    }
                ],
            },
        )
        self.assertEqual(invalid_sentence_response.status_code, 400)
        self.assertIn("invalid sentence id", invalid_sentence_response.json()["detail"])

    def test_delete_transcript_removes_review_target_and_clears_job_link(self) -> None:
        client = TestClient(api_main.app)

        delete_response = client.delete(f"/transcripts/{self.transcript_id}")
        self.assertEqual(delete_response.status_code, 204)

        transcript_response = client.get(f"/transcripts/{self.transcript_id}")
        self.assertEqual(transcript_response.status_code, 404)

        jobs_response = client.get("/jobs")
        self.assertEqual(jobs_response.status_code, 200)
        jobs_payload = jobs_response.json()
        self.assertEqual(len(jobs_payload["jobs"]), 1)
        self.assertIsNone(jobs_payload["jobs"][0]["transcript_id"])

    def test_delete_transcript_via_post_alias_removes_transcript(self) -> None:
        transcript_id, _ = self._seed_transcript()
        client = TestClient(api_main.app)

        delete_response = client.post(f"/transcripts/{transcript_id}/delete")
        self.assertEqual(delete_response.status_code, 204)

        transcript_response = client.get(f"/transcripts/{transcript_id}")
        self.assertEqual(transcript_response.status_code, 404)

    def test_patch_review_allows_naming_single_default_speaker(self) -> None:
        with self.session_factory() as session:
            media_asset = MediaAsset(
                source_type="file",
                source_uri=str(Path(self.temp_dir.name) / "monologue.mp4"),
                mime_type="video/mp4",
                checksum="mono-checksum",
                ingest_metadata={"speaker_mode": "monologue"},
            )
            session.add(media_asset)
            session.flush()

            transcript = Transcript(
                media_asset_id=media_asset.id,
                language_code="en",
                full_text="Only one speaker here.",
                transcript_metadata={},
            )
            session.add(transcript)
            session.flush()

            sentence = SentenceUnit(
                transcript_id=transcript.id,
                utterance_index=0,
                start_ms=0,
                end_ms=1000,
                text="Only one speaker here.",
                speaker_id="SPEAKER_00",
                speaker_confidence=None,
                source_segment_ids=[0],
                sentence_metadata={},
            )
            session.add(sentence)
            session.commit()
            single_speaker_transcript_id = transcript.id

        client = TestClient(api_main.app)

        response = client.patch(
            f"/transcripts/{single_speaker_transcript_id}/review",
            json={
                "conversation_title": "Monologue",
                "speaker_labels": {
                    "SPEAKER_00": "Jinhee",
                },
                "review_status": "in_progress",
                "sentence_overrides": [],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["speaker_labels"]["SPEAKER_00"], "Jinhee")
        self.assertEqual(payload["sentence_units"][0]["display_speaker_id"], "SPEAKER_00")
