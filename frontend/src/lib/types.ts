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
