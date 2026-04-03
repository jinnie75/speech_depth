import type {
  CreateStreamSessionRequest,
  JobSummary,
  PlaybackDocument,
  SentenceUnitResponse,
  StreamSessionResponse,
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
  const transcript = await fetchJson<{ id: string; sentence_units: SentenceUnitResponse[] }>(
    `/transcripts/${resolvedTranscriptId}`,
  );
  return {
    transcriptId: resolvedTranscriptId,
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
