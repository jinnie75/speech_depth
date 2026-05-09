from types import SimpleNamespace
import unittest

from asr_viz.models.common import JobStatus
from asr_viz.services.streaming import refresh_stream_session_status


class RefreshStreamSessionStatusTests(unittest.TestCase):
    def test_clears_stale_error_when_job_completes(self) -> None:
        stream_session = SimpleNamespace(
            status="failed",
            error_message="old failure",
            processing_job=SimpleNamespace(
                status=JobStatus.COMPLETED.value,
                transcript_id="transcript-123",
                error_message=None,
            ),
        )

        refresh_stream_session_status(stream_session)

        self.assertEqual(stream_session.status, "completed")
        self.assertIsNone(stream_session.error_message)

    def test_keeps_failed_error_when_job_is_failed(self) -> None:
        stream_session = SimpleNamespace(
            status="processing",
            error_message=None,
            processing_job=SimpleNamespace(
                status=JobStatus.FAILED.value,
                transcript_id=None,
                error_message="format error",
            ),
        )

        refresh_stream_session_status(stream_session)

        self.assertEqual(stream_session.status, "failed")
        self.assertEqual(stream_session.error_message, "format error")
