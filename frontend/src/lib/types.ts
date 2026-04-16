export interface ArchivePreviewSpeakerResponse {
  id: string;
  label: string;
  side: "left" | "right";
  opacity: number;
  total_duration_ms: number;
  average_politeness: number;
}

export interface ArchivePreviewUtteranceResponse {
  id: string;
  speaker_id: string;
  politeness_score: number;
  contour_signal_count: number;
  progress: number;
  order: number;
}

export interface ArchivePreviewTranscriptTokenResponse {
  id: string;
  kind: "text" | "word";
  text: string;
  start?: number | null;
  end?: number | null;
  is_hedge?: boolean | null;
  is_substance?: boolean | null;
  drop_start_ms?: number | null;
  drop_duration_ms?: number | null;
}

export interface ArchivePreviewTranscriptResponse {
  utterance_id: string;
  current_time_ms: number;
  tokens: ArchivePreviewTranscriptTokenResponse[];
}

export interface ArchivePreviewMarginNoteResponse {
  id: string;
  text: string;
  speaker_id: string;
  utterance_id: string;
  appear_at_ms: number;
  source_start: number;
  source_end: number;
  settle_duration_ms: number;
}

export interface ArchivePreviewResponse {
  speakers: ArchivePreviewSpeakerResponse[];
  utterances: ArchivePreviewUtteranceResponse[];
  active_speaker_id: string | null;
  active_transcript: ArchivePreviewTranscriptResponse | null;
  margin_notes: ArchivePreviewMarginNoteResponse[];
  current_time_ms: number;
}

export interface JobSummary {
  id: string;
  status: string;
  transcript_id: string | null;
  created_at: string;
  media_source_uri: string | null;
  media_mime_type: string | null;
  media_ingest_metadata: Record<string, unknown>;
  media_display_name: string | null;
  conversation_title: string | null;
  review_status: ReviewStatus;
  archive_preview: ArchivePreviewResponse | null;
}

export interface JobListPage {
  jobs: JobSummary[];
  total: number;
}

export type ReviewStatus = "not_started" | "in_progress" | "completed";

export interface AnalysisResultResponse {
  politeness_score: number;
  semantic_confidence_score: number;
  main_message_likelihood: number;
  analysis_model: string;
  analysis_payload: Record<string, unknown>;
}

export interface TranscriptWordResponse {
  word: string;
  start_ms: number;
  end_ms: number;
  probability: number | null;
}

export interface TranscriptSegmentResponse {
  id: string;
  segment_index: number;
  start_ms: number;
  end_ms: number;
  text: string;
  avg_logprob: number | null;
  no_speech_prob: number | null;
  words_json: TranscriptWordResponse[];
  raw_payload: Record<string, unknown>;
}

export interface SentenceUnitResponse {
  id: string;
  utterance_index: number;
  start_ms: number;
  end_ms: number;
  text: string;
  speaker_id: string | null;
  speaker_confidence: number | null;
  display_text: string;
  display_speaker_id: string | null;
  manual_text: string | null;
  manual_speaker_id: string | null;
  is_edited: boolean;
  source_segment_ids: number[];
  sentence_metadata: Record<string, unknown>;
  analysis_result: AnalysisResultResponse | null;
}

export interface CreateStreamSessionRequest {
  mime_type: string | null;
  original_filename: string | null;
  diarization_enabled: boolean;
  ingest_metadata: Record<string, unknown>;
}

export interface CreateLiveSessionRequest {
  mime_type: string | null;
  original_filename: string | null;
  sample_rate_hz: number | null;
  channel_count: number | null;
  session_metadata: Record<string, unknown>;
}

export interface StreamSessionResponse {
  id: string;
  status: string;
  mime_type: string | null;
  original_filename: string | null;
  storage_path: string;
  total_bytes: number;
  received_chunks: number;
  diarization_enabled: boolean;
  ingest_metadata: Record<string, unknown>;
  error_message: string | null;
  processing_job_id: string | null;
  processing_job_status: string | null;
  processing_stage: string | null;
  transcript_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface LiveAnalysisEventRequest {
  politeness_score: number;
  semantic_confidence_score: number;
  main_message_likelihood: number;
  analysis_model: string;
  analysis_payload: Record<string, unknown>;
}

export interface AppendLiveTranscriptEventRequest {
  event_type: "transcript.delta" | "transcript.final" | "analysis.delta" | "session.state";
  utterance_key: string | null;
  start_ms: number | null;
  end_ms: number | null;
  text: string | null;
  speaker_id: string | null;
  is_final: boolean;
  payload: Record<string, unknown>;
  analysis?: LiveAnalysisEventRequest | null;
}

export interface LiveTranscriptEventResponse {
  id: string;
  live_session_id: string;
  event_index: number;
  event_type: string;
  utterance_key: string | null;
  start_ms: number | null;
  end_ms: number | null;
  text: string | null;
  speaker_id: string | null;
  is_final: boolean;
  payload: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface LiveSessionResponse {
  id: string;
  status: string;
  mime_type: string | null;
  original_filename: string | null;
  sample_rate_hz: number | null;
  channel_count: number | null;
  storage_path: string;
  total_bytes: number;
  received_chunks: number;
  last_chunk_index: number;
  session_metadata: Record<string, unknown>;
  error_message: string | null;
  transcript_id: string | null;
  media_asset_id: string | null;
  processing_job_id: string | null;
  stopped_at: string | null;
  finalized_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface PlaybackDocument {
  transcriptId: string;
  conversationTitle: string | null;
  speakerLabels: Record<string, string>;
  reviewStatus: ReviewStatus;
  reviewedAt: string | null;
  segments: TranscriptSegmentResponse[];
  sentenceUnits: SentenceUnitResponse[];
}

export interface TranscriptReviewSentenceOverride {
  sentence_unit_id: string;
  manual_text: string | null;
  manual_speaker_id: string | null;
}

export interface TranscriptReviewUpdateRequest {
  conversation_title: string | null;
  speaker_labels: Record<string, string>;
  review_status: ReviewStatus;
  sentence_overrides: TranscriptReviewSentenceOverride[];
}
