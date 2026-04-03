export interface JobSummary {
  id: string;
  status: string;
  transcript_id: string | null;
  created_at: string;
  media_source_uri: string | null;
  media_mime_type: string | null;
  media_ingest_metadata: Record<string, unknown>;
  media_display_name: string | null;
}

export interface AnalysisResultResponse {
  politeness_score: number;
  semantic_confidence_score: number;
  main_message_likelihood: number;
  analysis_model: string;
  analysis_payload: Record<string, unknown>;
}

export interface SentenceUnitResponse {
  id: string;
  utterance_index: number;
  start_ms: number;
  end_ms: number;
  text: string;
  speaker_id: string | null;
  speaker_confidence: number | null;
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
  sentenceUnits: SentenceUnitResponse[];
}
