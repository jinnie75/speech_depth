from __future__ import annotations

import argparse
import os
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


def _mask_token(token: str | None) -> str:
    if token is None:
        return "missing"
    normalized = token.strip()
    if not normalized:
        return "blank"
    if len(normalized) <= 10:
        return f"len={len(normalized)} value={normalized}"
    return f"len={len(normalized)} prefix={normalized[:6]} suffix={normalized[-4:]}"


def _configure_huggingface_environment() -> None:
    if settings.huggingface_token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = settings.huggingface_token


def _log_worker_startup_configuration() -> None:
    diarization_provider_name = "pyannote" if settings.huggingface_token else "noop"
    print(
        "worker_startup "
        f"app_env={settings.app_env} "
        f"worker_name={settings.worker_name} "
        f"diarization_provider={diarization_provider_name} "
        f"diarization_model={settings.diarization_model} "
        f"huggingface_token={_mask_token(settings.huggingface_token)} "
        f"hf_token_env={_mask_token(os.getenv('HF_TOKEN'))}"
    )


def build_processing_pipeline() -> ProcessingPipeline:
    return ProcessingPipeline(
        transcription_provider=build_transcription_provider(),
        analysis_provider=build_analysis_provider(),
        diarization_provider=build_diarization_provider(),
    )


def process_next_job(pipeline: ProcessingPipeline) -> bool:
    with SessionLocal() as session:
        job = pipeline.claim_next_job(session)
        if job is None:
            return False
        pipeline.process_job(session, job.id)
        return True


def run_worker(*, once: bool = False) -> None:
    init_db()
    _configure_huggingface_environment()
    _log_worker_startup_configuration()
    pipeline = build_processing_pipeline()
    while True:
        processed = process_next_job(pipeline)
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
