import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from asr_viz.db.base import Base
from asr_viz.models.job import ProcessingJob
import asr_viz.services.jobs as jobs_module
from asr_viz.services.jobs import create_job


class JobCreationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test_jobs.db"
        self.engine = create_engine(f"sqlite:///{database_path}", future=True)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(bind=self.engine)
        self.original_settings = jobs_module.settings
        jobs_module.settings = type(
            "Settings",
            (),
            {"auth_dev_user_id": "local-dev-user", "enable_diarization": True},
        )()

    def tearDown(self) -> None:
        jobs_module.settings = self.original_settings
        self.temp_dir.cleanup()

    def test_create_job_persists_diarization_flag_from_speaker_count(self) -> None:
        media_path = Path(self.temp_dir.name) / "source.wav"
        media_path.write_bytes(b"wav-data")

        with self.session_factory() as session:
            single_speaker_job = create_job(
                session,
                source_uri=str(media_path),
                source_type="file",
                mime_type="audio/wav",
                checksum=None,
                ingest_metadata={"diarization_num_speakers": 1},
            )
            multi_speaker_job = create_job(
                session,
                source_uri=str(media_path),
                source_type="file",
                mime_type="audio/wav",
                checksum=None,
                ingest_metadata={"diarization_num_speakers": 2},
            )

            persisted_single_speaker_job = session.get(ProcessingJob, single_speaker_job.id)
            persisted_multi_speaker_job = session.get(ProcessingJob, multi_speaker_job.id)

            self.assertIsNotNone(persisted_single_speaker_job)
            self.assertIsNotNone(persisted_multi_speaker_job)
            self.assertFalse(persisted_single_speaker_job.diarization_enabled)
            self.assertTrue(persisted_multi_speaker_job.diarization_enabled)
