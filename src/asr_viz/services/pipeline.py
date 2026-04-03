from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from asr_viz.models.analysis import AnalysisResult
from asr_viz.models.common import JobStage, JobStatus, utcnow
from asr_viz.models.job import ProcessingJob
from asr_viz.models.transcript import SentenceUnit, Transcript, TranscriptSegment
from asr_viz.pipeline.segmentation import segment_transcript
from asr_viz.providers.analysis import AnalysisProvider
from asr_viz.providers.diarization import DiarizationProvider
from asr_viz.providers.transcription import TranscriptionProvider
from asr_viz.services.media import resolve_media_source


class ProcessingPipeline:
    def __init__(
        self,
        transcription_provider: TranscriptionProvider,
        analysis_provider: AnalysisProvider,
        diarization_provider: DiarizationProvider,
    ) -> None:
        self._transcription_provider = transcription_provider
        self._analysis_provider = analysis_provider
        self._diarization_provider = diarization_provider

    def claim_next_job(self, session: Session) -> ProcessingJob | None:
        job = session.scalar(
            select(ProcessingJob)
            .where(ProcessingJob.status == JobStatus.QUEUED.value)
            .order_by(ProcessingJob.created_at.asc())
        )
        if job is None:
            return None

        job.status = JobStatus.PROCESSING.value
        job.current_stage = JobStage.TRANSCRIPTION.value
        job.started_at = utcnow()
        session.commit()
        session.refresh(job)
        return job

    def process_job(self, session: Session, job_id: str) -> ProcessingJob:
        job = session.scalar(
            select(ProcessingJob)
            .where(ProcessingJob.id == job_id)
            .options(selectinload(ProcessingJob.media_asset))
        )
        if job is None:
            raise ValueError(f"job {job_id} not found")

        try:
            source_uri = resolve_media_source(job, job.media_asset)
            transcript_result = self._transcription_provider.transcribe(source_uri)
            job.asr_model_version = self._transcription_provider.model_version
            job.current_stage = JobStage.DIARIZATION.value if job.diarization_enabled else JobStage.ANALYSIS.value
            job.stage_details = {
                **job.stage_details,
                "segments": len(transcript_result.segments),
                "resolved_source_uri": source_uri,
            }

            transcript = Transcript(
                media_asset_id=job.media_asset_id,
                job_id=job.id,
                language_code=transcript_result.language_code,
                full_text=transcript_result.full_text,
                transcript_metadata=transcript_result.metadata,
            )
            session.add(transcript)
            session.flush()

            for segment in transcript_result.segments:
                session.add(
                    TranscriptSegment(
                        transcript_id=transcript.id,
                        segment_index=segment.segment_index,
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        text=segment.text,
                        avg_logprob=segment.avg_logprob,
                        no_speech_prob=segment.no_speech_prob,
                        words_json=[word.model_dump() for word in segment.words],
                        raw_payload=segment.raw_payload,
                    )
                )

            sentences = segment_transcript(transcript_result.segments)
            if job.diarization_enabled:
                speaker_count_override = _speaker_count_override(job.media_asset.ingest_metadata)
                sentences = self._diarization_provider.assign_speakers(
                    sentences,
                    source_uri,
                    num_speakers_override=speaker_count_override,
                )
                job.diarization_model_version = self._diarization_provider.model_version
                job.stage_details = {
                    **job.stage_details,
                    "diarization_enabled": True,
                    "speaker_count": len({sentence.speaker_id for sentence in sentences if sentence.speaker_id}),
                    "requested_num_speakers": speaker_count_override,
                }

            sentence_units: list[SentenceUnit] = []
            for sentence in sentences:
                unit = SentenceUnit(
                    transcript_id=transcript.id,
                    utterance_index=sentence.utterance_index,
                    start_ms=sentence.start_ms,
                    end_ms=sentence.end_ms,
                    text=sentence.text,
                    speaker_id=sentence.speaker_id,
                    speaker_confidence=sentence.speaker_confidence,
                    source_segment_ids=sentence.source_segment_ids,
                    sentence_metadata=sentence.sentence_metadata,
                )
                session.add(unit)
                sentence_units.append(unit)

            session.flush()

            analysis_results = self._analysis_provider.analyze(sentences, transcript_result.full_text)
            job.analysis_model_version = self._analysis_provider.model_version

            for unit, analysis in zip(sentence_units, analysis_results, strict=True):
                session.add(
                    AnalysisResult(
                        sentence_unit_id=unit.id,
                        politeness_score=analysis.politeness_score,
                        semantic_confidence_score=analysis.semantic_confidence_score,
                        main_message_likelihood=analysis.main_message_likelihood,
                        analysis_model=self._analysis_provider.model_version,
                        analysis_payload=analysis.analysis_payload,
                    )
                )

            session.flush()
            job.status = JobStatus.COMPLETED.value
            job.current_stage = JobStage.COMPLETE.value
            job.completed_at = utcnow()
            job.error_message = None
            job.transcript_id = transcript.id
            job.stage_details = {
                **job.stage_details,
                "sentence_count": len(sentence_units),
            }
            session.commit()
            session.refresh(job)
            return job
        except Exception as exc:
            session.rollback()
            failed_job = session.get(ProcessingJob, job_id)
            if failed_job is None:
                raise
            failed_job.status = JobStatus.FAILED.value
            failed_job.current_stage = JobStage.FAILED.value
            failed_job.error_message = str(exc)
            failed_job.retry_count += 1
            session.commit()
            session.refresh(failed_job)
            return failed_job


def _speaker_count_override(ingest_metadata: dict | None) -> int | None:
    metadata = ingest_metadata or {}
    explicit_value = metadata.get("diarization_num_speakers")
    if explicit_value is not None:
        try:
            parsed = int(explicit_value)
        except (TypeError, ValueError):
            parsed = None
        if parsed in {1, 2}:
            return parsed

    speaker_mode = metadata.get("speaker_mode")
    if speaker_mode == "monologue":
        return 1
    if speaker_mode == "dialogue":
        return 2
    return None
