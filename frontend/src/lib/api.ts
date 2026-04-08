import type {
  CreateStreamSessionRequest,
  JobSummary,
  PlaybackDocument,
  TranscriptSegmentResponse,
  SentenceUnitResponse,
  StreamSessionResponse,
  TranscriptReviewUpdateRequest,
} from "./types";

const DEFAULT_BASE_URL = import.meta.env.VITE_ASR_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_CHUNK_SIZE = Number(import.meta.env.VITE_UPLOAD_CHUNK_SIZE_BYTES ?? 1024 * 1024 * 2);

export function buildJobMediaUrl(jobId: string): string {
  return `${DEFAULT_BASE_URL}/jobs/${jobId}/media`;
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${DEFAULT_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${path}`);
  }
  return (await response.json()) as T;
}

async function fetchLatestCompletedTranscriptId(): Promise<string> {
  const jobList = await fetchJson<{ jobs: JobSummary[] }>("/jobs?limit=50");
  const latest = jobList.jobs.find((job) => job.status === "completed" && job.transcript_id);
  if (!latest?.transcript_id) {
    throw new Error("No completed transcripts were found.");
  }
  return latest.transcript_id;
}

export async function fetchCompletedJobs(limit = 50): Promise<JobSummary[]> {
  const jobList = await fetchJson<{ jobs: JobSummary[] }>(`/jobs?limit=${limit}`);
  return jobList.jobs.filter((job) => job.status === "completed" && job.transcript_id);
}

export async function loadPlaybackDocument(transcriptId?: string): Promise<PlaybackDocument> {
  const resolvedTranscriptId = transcriptId || (await fetchLatestCompletedTranscriptId());
  const transcript = await fetchJson<{
    id: string;
    conversation_title: string | null;
    speaker_labels: Record<string, string>;
    review_status: PlaybackDocument["reviewStatus"];
    reviewed_at: string | null;
    segments: TranscriptSegmentResponse[];
    sentence_units: SentenceUnitResponse[];
  }>(
    `/transcripts/${resolvedTranscriptId}`,
  );
  return {
    transcriptId: resolvedTranscriptId,
    conversationTitle: transcript.conversation_title,
    speakerLabels: transcript.speaker_labels ?? {},
    reviewStatus: transcript.review_status,
    reviewedAt: transcript.reviewed_at,
    segments: [...(transcript.segments ?? [])].sort(
      (left, right) => left.segment_index - right.segment_index || left.start_ms - right.start_ms,
    ),
    sentenceUnits: [...transcript.sentence_units].sort(
      (left, right) => left.utterance_index - right.utterance_index || left.start_ms - right.start_ms,
    ),
  };
}

export async function createStreamSession(payload: CreateStreamSessionRequest): Promise<StreamSessionResponse> {
  const response = await fetch(`${DEFAULT_BASE_URL}/stream-sessions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Failed to create stream session (${response.status})`);
  }
  return (await response.json()) as StreamSessionResponse;
}

export async function uploadFileToStreamSession(
  sessionId: string,
  file: File,
  onProgress?: (uploadedBytes: number, totalBytes: number) => void,
): Promise<void> {
  let uploadedBytes = 0;

  for (let start = 0; start < file.size; start += DEFAULT_CHUNK_SIZE) {
    const end = Math.min(start + DEFAULT_CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);
    const response = await fetch(`${DEFAULT_BASE_URL}/stream-sessions/${sessionId}/chunks`, {
      method: "PUT",
      headers: {
        "content-type": "application/octet-stream",
      },
      body: chunk,
    });
    if (!response.ok) {
      throw new Error(`Failed to upload stream chunk (${response.status})`);
    }
    uploadedBytes = end;
    onProgress?.(uploadedBytes, file.size);
  }
}

export async function finalizeStreamSession(sessionId: string): Promise<StreamSessionResponse> {
  const response = await fetch(`${DEFAULT_BASE_URL}/stream-sessions/${sessionId}/finalize`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to finalize stream session (${response.status})`);
  }
  return (await response.json()) as StreamSessionResponse;
}

export async function fetchStreamSession(sessionId: string): Promise<StreamSessionResponse> {
  return fetchJson<StreamSessionResponse>(`/stream-sessions/${sessionId}`);
}

export async function saveTranscriptReview(
  transcriptId: string,
  payload: TranscriptReviewUpdateRequest,
): Promise<PlaybackDocument> {
  const response = await fetch(`${DEFAULT_BASE_URL}/transcripts/${transcriptId}/review`, {
    method: "PATCH",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Failed to save transcript review (${response.status})`);
  }
  const transcript = (await response.json()) as {
    id: string;
    conversation_title: string | null;
    speaker_labels: Record<string, string>;
    review_status: PlaybackDocument["reviewStatus"];
    reviewed_at: string | null;
    segments: TranscriptSegmentResponse[];
    sentence_units: SentenceUnitResponse[];
  };
  return {
    transcriptId: transcript.id,
    conversationTitle: transcript.conversation_title,
    speakerLabels: transcript.speaker_labels ?? {},
    reviewStatus: transcript.review_status,
    reviewedAt: transcript.reviewed_at,
    segments: [...(transcript.segments ?? [])].sort(
      (left, right) => left.segment_index - right.segment_index || left.start_ms - right.start_ms,
    ),
    sentenceUnits: [...transcript.sentence_units].sort(
      (left, right) => left.utterance_index - right.utterance_index || left.start_ms - right.start_ms,
    ),
  };
}

export async function deleteTranscript(transcriptId: string): Promise<void> {
  const response = await fetch(`${DEFAULT_BASE_URL}/transcripts/${transcriptId}`, {
    method: "DELETE",
  });
  if (response.ok) {
    return;
  }

  if (response.status === 405) {
    const fallbackResponse = await fetch(`${DEFAULT_BASE_URL}/transcripts/${transcriptId}/delete`, {
      method: "POST",
    });
    if (fallbackResponse.ok) {
      return;
    }
    throw new Error(`Failed to delete transcript (${fallbackResponse.status})`);
  }

  throw new Error(`Failed to delete transcript (${response.status})`);
}
