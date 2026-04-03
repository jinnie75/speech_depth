from __future__ import annotations

import argparse
import time

from asr_viz.core.settings import settings
from asr_viz.db.session import SessionLocal
from asr_viz.services.bootstrap import init_db
from asr_viz.services.pipeline import ProcessingPipeline
from asr_viz.services.providers import (
    build_analysis_provider,
    build_diarization_provider,
    build_transcription_provider,
)


def process_next_job() -> bool:
    pipeline = ProcessingPipeline(
        transcription_provider=build_transcription_provider(),
        analysis_provider=build_analysis_provider(),
        diarization_provider=build_diarization_provider(),
    )
    with SessionLocal() as session:
        job = pipeline.claim_next_job(session)
        if job is None:
            return False
        pipeline.process_job(session, job.id)
        return True


def run_worker(*, once: bool = False) -> None:
    init_db()
    while True:
        processed = process_next_job()
        if once:
            return
        if not processed:
            time.sleep(settings.job_poll_interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ASR Viz background worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit.")
    args = parser.parse_args()
    run_worker(once=args.once)


if __name__ == "__main__":
    main()
