from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from asr_viz.db.session import SessionLocal
from asr_viz.models.job import ProcessingJob
from asr_viz.models.transcript import SentenceUnit, Transcript
from asr_viz.services.bootstrap import init_db
from asr_viz.services.jobs import create_job, infer_source_type
from asr_viz.worker import run_worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI helpers for the ASR Viz backend.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Submit a media file or URL for processing.")
    submit.add_argument("source_uri", help="Absolute local path or URL to process.")
    submit.add_argument("--source-type", choices=["file", "url", "opaque"], default=None)
    submit.add_argument("--mime-type", default=None)
    submit.add_argument("--diarization", action="store_true", help="Enable speaker diarization when available.")
    submit.add_argument(
        "--metadata",
        default="{}",
        help="JSON object stored as ingest metadata, for example '{\"project\":\"amy\"}'.",
    )

    status = subparsers.add_parser("status", help="Show job status.")
    status.add_argument("job_id", nargs="?", default=None, help="Job id. Defaults to the latest job.")

    transcript = subparsers.add_parser("transcript", help="Show transcript details.")
    transcript.add_argument("transcript_id", nargs="?", default=None, help="Transcript id. Defaults to the latest one.")
    transcript.add_argument("--sentences", type=int, default=5, help="Number of sentence rows to print.")
    transcript.add_argument("--preview-chars", type=int, default=1200, help="Transcript preview length.")

    worker = subparsers.add_parser("worker", help="Run the background worker.")
    worker.add_argument("--once", action="store_true", help="Process at most one queued job and exit.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    init_db()

    if args.command == "submit":
        _submit(args)
        return
    if args.command == "status":
        _status(args)
        return
    if args.command == "transcript":
        _transcript(args)
        return
    if args.command == "worker":
        run_worker(once=args.once)
        return
    parser.error("unknown command")


def _submit(args: argparse.Namespace) -> None:
    metadata = json.loads(args.metadata)
    source_type = args.source_type or infer_source_type(args.source_uri)
    if source_type == "file":
        path = Path(args.source_uri)
        if not path.exists():
            raise SystemExit(f"local file not found: {args.source_uri}")

    with SessionLocal() as session:
        job = create_job(
            session,
            source_uri=args.source_uri,
            source_type=source_type,
            diarization_enabled=args.diarization,
            mime_type=args.mime_type,
            checksum=None,
            ingest_metadata=metadata,
        )
        print(f"job_id={job.id}")
        print(f"status={job.status}")
        print(f"source_type={source_type}")
        print(f"diarization_enabled={job.diarization_enabled}")


def _status(args: argparse.Namespace) -> None:
    with SessionLocal() as session:
        job = _resolve_job(session, args.job_id)
        if job is None:
            raise SystemExit("no jobs found")
        print(f"job_id={job.id}")
        print(f"status={job.status}")
        print(f"stage={job.current_stage}")
        print(f"transcript_id={job.transcript_id or ''}")
        print(f"error={job.error_message or ''}")


def _transcript(args: argparse.Namespace) -> None:
    with SessionLocal() as session:
        transcript = _resolve_transcript(session, args.transcript_id)
        if transcript is None:
            raise SystemExit("no transcripts found")

        sentences = (
            session.scalars(
                select(SentenceUnit)
                .where(SentenceUnit.transcript_id == transcript.id)
                .order_by(SentenceUnit.utterance_index.asc())
            )
            .all()
        )
        print(f"transcript_id={transcript.id}")
        print(f"language={transcript.language_code or ''}")
        print(f"sentence_count={len(sentences)}")
        print("preview:")
        print(transcript.full_text[: args.preview_chars])
        print("sentences:")
        for sentence in sentences[: args.sentences]:
            print(f"{sentence.utterance_index}\t{sentence.start_ms}\t{sentence.end_ms}\t{sentence.text}")


def _resolve_job(session, job_id: str | None) -> ProcessingJob | None:
    if job_id:
        return session.get(ProcessingJob, job_id)
    return session.scalar(select(ProcessingJob).order_by(ProcessingJob.created_at.desc()))


def _resolve_transcript(session, transcript_id: str | None) -> Transcript | None:
    statement = (
        select(Transcript)
        .options(
            selectinload(Transcript.segments),
            selectinload(Transcript.sentence_units).selectinload(SentenceUnit.analysis_result),
        )
        .order_by(Transcript.created_at.desc())
    )
    if transcript_id:
        statement = statement.where(Transcript.id == transcript_id)
    return session.scalar(statement)


if __name__ == "__main__":
    main()
