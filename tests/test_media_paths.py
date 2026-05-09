import tempfile
import unittest
from pathlib import Path

import asr_viz.services.media as media_module
from asr_viz.services.media import resolve_local_media_path, upload_local_file_to_configured_storage


class MediaPathResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name) / "project"
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.original_project_root_dir = media_module.project_root_dir
        self.original_settings = media_module.settings
        media_module.project_root_dir = lambda: self.project_root
        media_module.settings = type("Settings", (), {"media_storage_dir": "./.media", "storage_backend": "local"})()

    def tearDown(self) -> None:
        media_module.project_root_dir = self.original_project_root_dir
        media_module.settings = self.original_settings
        self.temp_dir.cleanup()

    def test_resolve_local_media_path_prefers_project_relative_media_files(self) -> None:
        target_path = self.project_root / ".media" / "stream_ingestion" / "example.mov"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"mov-data")

        resolved = resolve_local_media_path(".media/stream_ingestion/example.mov")

        self.assertEqual(resolved, target_path.resolve())

    def test_local_storage_backend_returns_absolute_file_path(self) -> None:
        target_path = self.project_root / ".media" / "stream_ingestion" / "example.mov"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"mov-data")

        source_type, source_uri = upload_local_file_to_configured_storage(
            local_path=".media/stream_ingestion/example.mov",
            owner_user_id="test-user",
            media_category="stream_ingestion",
            media_id="example",
            original_filename="example.mov",
            mime_type="video/quicktime",
        )

        self.assertEqual(source_type, "file")
        self.assertEqual(source_uri, str(target_path.resolve()))
