import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from asr_viz.services import bootstrap as bootstrap_module


class BootstrapSchemaTests(unittest.TestCase):
    def test_init_db_adds_missing_stream_and_job_diarization_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "bootstrap.db"
            engine = create_engine(f"sqlite:///{database_path}", future=True)

            with engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        CREATE TABLE stream_ingestion_sessions (
                            id VARCHAR(36) NOT NULL PRIMARY KEY,
                            owner_user_id VARCHAR(255) NOT NULL,
                            status VARCHAR(32) NOT NULL,
                            storage_path VARCHAR(2048) NOT NULL,
                            total_bytes INTEGER NOT NULL DEFAULT 0,
                            received_chunks INTEGER NOT NULL DEFAULT 0,
                            ingest_metadata JSON NOT NULL DEFAULT '{}',
                            created_at DATETIME NOT NULL,
                            updated_at DATETIME NOT NULL
                        )
                        """
                    )
                )
                connection.execute(
                    text(
                        """
                        CREATE TABLE processing_jobs (
                            id VARCHAR(36) NOT NULL PRIMARY KEY,
                            owner_user_id VARCHAR(255) NOT NULL,
                            media_asset_id VARCHAR(36) NOT NULL,
                            status VARCHAR(32) NOT NULL,
                            current_stage VARCHAR(32) NOT NULL,
                            retry_count INTEGER NOT NULL DEFAULT 0,
                            stage_details JSON NOT NULL DEFAULT '{}',
                            created_at DATETIME NOT NULL,
                            updated_at DATETIME NOT NULL
                        )
                        """
                    )
                )

            original_engine = bootstrap_module.engine
            original_settings = bootstrap_module.settings
            bootstrap_module.engine = engine
            bootstrap_module.settings = type("Settings", (), {"auto_create_schema": True})()
            try:
                bootstrap_module.init_db()
            finally:
                bootstrap_module.engine = original_engine
                bootstrap_module.settings = original_settings

            stream_columns = {column["name"] for column in inspect(engine).get_columns("stream_ingestion_sessions")}
            job_columns = {column["name"] for column in inspect(engine).get_columns("processing_jobs")}
            self.assertIn("diarization_enabled", stream_columns)
            self.assertIn("diarization_enabled", job_columns)
