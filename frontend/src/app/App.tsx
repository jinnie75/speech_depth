import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";

import { ConversationLandscape } from "../components/ConversationLandscape";
import {
  buildJobMediaUrl,
  createStreamSession,
  fetchCompletedJobs,
  fetchStreamSession,
  finalizeStreamSession,
  loadPlaybackDocument,
  uploadFileToStreamSession,
} from "../lib/api";
import type { JobSummary, PlaybackDocument, StreamSessionResponse } from "../lib/types";

const TRANSCRIPT_ID = import.meta.env.VITE_TRANSCRIPT_ID;
const DEFAULT_MEDIA_SRC = import.meta.env.VITE_MEDIA_SRC ?? "";
const POLL_INTERVAL_MS = 2000;
const TRANSCRIPT_RETRY_LIMIT = 8;
const FALLBACK_SPEAKER_ID = "UNKNOWN_SPEAKER";
const FALLBACK_SPEAKER_LABEL = "Speaker";
const PRELOAD_TRANSCRIPT_ID = DEFAULT_MEDIA_SRC ? TRANSCRIPT_ID : undefined;
const DEFAULT_SPEAKER_COUNT = 2;

type SpeakerCount = 1 | 2;

interface SpeakerSummary {
  id: string;
  label: string;
  firstSeenMs: number;
  totalDurationMs: number;
  averagePoliteness: number;
}

interface UtteranceSummary {
  id: string;
  speakerId: string;
  text: string;
  startMs: number;
  endMs: number;
  durationMs: number;
  politenessScore: number;
}

interface ProcessedTranscriptOption {
  jobId: string;
  transcriptId: string;
  label: string;
  mediaSrc: string;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function formatSpeakerLabel(speakerId: string, index: number): string {
  if (speakerId === FALLBACK_SPEAKER_ID) {
    return FALLBACK_SPEAKER_LABEL;
  }
  if (speakerId.startsWith("SPEAKER_")) {
    const suffix = Number.parseInt(speakerId.replace("SPEAKER_", ""), 10);
    if (!Number.isNaN(suffix)) {
      return `Speaker ${suffix + 1}`;
    }
  }
  return index >= 0 ? `Speaker ${index + 1}` : speakerId;
}

function buildUtterances(document: PlaybackDocument): UtteranceSummary[] {
  return [...document.sentenceUnits]
    .map((sentence) => ({
      id: sentence.id,
      speakerId: sentence.speaker_id || FALLBACK_SPEAKER_ID,
      text: sentence.text,
      startMs: sentence.start_ms,
      endMs: sentence.end_ms,
      durationMs: Math.max(1, sentence.end_ms - sentence.start_ms),
      politenessScore: sentence.analysis_result?.politeness_score ?? 0.5,
    }))
    .sort((left, right) => left.startMs - right.startMs || left.endMs - right.endMs);
}

function buildSpeakerSummaries(utterances: UtteranceSummary[]): SpeakerSummary[] {
  const firstSeen = new Map<string, number>();
  const totalDurationMs = new Map<string, number>();
  const politenessTotals = new Map<string, number>();
  const politenessCounts = new Map<string, number>();

  for (const utterance of utterances) {
    const knownStart = firstSeen.get(utterance.speakerId);
    if (knownStart === undefined || utterance.startMs < knownStart) {
      firstSeen.set(utterance.speakerId, utterance.startMs);
    }
    totalDurationMs.set(
      utterance.speakerId,
      (totalDurationMs.get(utterance.speakerId) ?? 0) + utterance.durationMs,
    );
    politenessTotals.set(
      utterance.speakerId,
      (politenessTotals.get(utterance.speakerId) ?? 0) + utterance.politenessScore * 10,
    );
    politenessCounts.set(utterance.speakerId, (politenessCounts.get(utterance.speakerId) ?? 0) + 1);
  }

  return Array.from(firstSeen.entries())
    .sort((left, right) => left[1] - right[1] || left[0].localeCompare(right[0]))
    .map(([id, firstSeenMs], index) => ({
      id,
      label: formatSpeakerLabel(id, index),
      firstSeenMs,
      totalDurationMs: totalDurationMs.get(id) ?? 0,
      averagePoliteness: politenessCounts.get(id)
        ? (politenessTotals.get(id) ?? 0) / (politenessCounts.get(id) ?? 1)
        : 5,
    }));
}

function findActiveUtterance(utterances: UtteranceSummary[], currentTimeMs: number): UtteranceSummary | null {
  let latestStarted: UtteranceSummary | null = null;

  for (const utterance of utterances) {
    if (utterance.startMs <= currentTimeMs && currentTimeMs <= utterance.endMs) {
      return utterance;
    }
    if (utterance.startMs <= currentTimeMs) {
      latestStarted = utterance;
    } else {
      break;
    }
  }

  return latestStarted;
}

function fileToObjectUrl(file: File): string {
  return URL.createObjectURL(file);
}

function getProcessedTranscriptLabel(job: JobSummary): string {
  const preferredLabel = job.media_display_name?.trim();
  if (preferredLabel) {
    return preferredLabel;
  }

  const sourceName = job.media_source_uri?.split("/").pop()?.trim();
  if (sourceName) {
    return sourceName;
  }

  return job.transcript_id ?? "Processed transcript";
}

function getProcessedTranscriptMediaSrc(job: JobSummary): string {
  const sourceUri = job.media_source_uri?.trim();
  if (!sourceUri) {
    return "";
  }

  if (sourceUri.startsWith("http://") || sourceUri.startsWith("https://")) {
    return sourceUri;
  }

  return buildJobMediaUrl(job.id);
}

function describeStreamStatus(streamSession: StreamSessionResponse | null, uploadProgress: number | null): string | null {
  if (!streamSession) {
    return null;
  }

  if (streamSession.status === "open") {
    const suffix = uploadProgress === null ? "" : ` ${Math.round(uploadProgress * 100)}%`;
    return `Uploading${suffix}`;
  }
  if (streamSession.status === "queued") {
    return "Uploaded. Waiting for worker.";
  }
  if (streamSession.status === "processing") {
    return `Processing${streamSession.processing_stage ? `: ${streamSession.processing_stage}` : ""}`;
  }
  if (streamSession.status === "completed") {
    return "Transcript ready.";
  }
  if (streamSession.status === "failed") {
    return streamSession.error_message || "Processing failed.";
  }
  return streamSession.status;
}

export function App() {
  const [document, setDocument] = useState<PlaybackDocument | null>(null);
  const [isAwaitingReplacementTranscript, setIsAwaitingReplacementTranscript] = useState(false);
  const [transcriptLoadStatus, setTranscriptLoadStatus] = useState<"idle" | "loading" | "ready" | "empty">("idle");
  const [error, setError] = useState<string | null>(null);
  const [mediaSrc, setMediaSrc] = useState<string>(DEFAULT_MEDIA_SRC);
  const [mediaName, setMediaName] = useState<string>(DEFAULT_MEDIA_SRC ? "Configured media source" : "No file selected");
  const [playbackStarted, setPlaybackStarted] = useState(false);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const [streamSession, setStreamSession] = useState<StreamSessionResponse | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [speakerCount, setSpeakerCount] = useState<SpeakerCount>(DEFAULT_SPEAKER_COUNT);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [processedTranscriptOptions, setProcessedTranscriptOptions] = useState<ProcessedTranscriptOption[]>([]);
  const [selectedProcessedTranscriptId, setSelectedProcessedTranscriptId] = useState("");
  const mediaObjectUrlRef = useRef<string | null>(null);
  const pollingSessionIdRef = useRef<string | null>(null);

  const refreshProcessedTranscriptOptions = async (): Promise<ProcessedTranscriptOption[]> => {
    const jobs = await fetchCompletedJobs();
    const options = jobs
      .map((job) => ({
        jobId: job.id,
        transcriptId: job.transcript_id ?? "",
        label: getProcessedTranscriptLabel(job),
        mediaSrc: getProcessedTranscriptMediaSrc(job),
      }))
      .filter((option) => option.transcriptId);
    setProcessedTranscriptOptions(options);
    return options;
  };

  useEffect(() => {
    let active = true;

    refreshProcessedTranscriptOptions()
      .then(async (options) => {
        if (!PRELOAD_TRANSCRIPT_ID || !active) {
          return;
        }

        const payload = await loadPlaybackDocument(PRELOAD_TRANSCRIPT_ID);
        if (!active) {
          return;
        }
        setDocument(payload);
        setSelectedProcessedTranscriptId(
          options.some((option) => option.transcriptId === payload.transcriptId) ? payload.transcriptId : "",
        );
        setTranscriptLoadStatus(payload.sentenceUnits.length > 0 ? "ready" : "empty");
      })
      .catch((caughtError: unknown) => {
        if (!active) {
          return;
        }
        setError(caughtError instanceof Error ? caughtError.message : "Unknown error");
      });

    return () => {
      active = false;
      pollingSessionIdRef.current = null;
      if (mediaObjectUrlRef.current) {
        URL.revokeObjectURL(mediaObjectUrlRef.current);
      }
    };
  }, []);

  const utterances = document ? buildUtterances(document) : [];
  const speakers = buildSpeakerSummaries(utterances);
  const activeUtterance =
    !isAwaitingReplacementTranscript && (playbackStarted || currentTimeMs > 0)
      ? findActiveUtterance(utterances, currentTimeMs)
      : null;
  const landscapeSpeakers = !isAwaitingReplacementTranscript
    ? speakers.map((speaker, index) => ({
        id: speaker.id,
        label: speaker.label,
        side: index % 2 === 0 ? ("left" as const) : ("right" as const),
        opacity: activeUtterance?.speakerId === speaker.id ? 1 : 0.5,
        totalDurationMs: speaker.totalDurationMs,
        averagePoliteness: speaker.averagePoliteness,
      }))
    : [];
  const landscapeUtterances = !isAwaitingReplacementTranscript
    ? utterances.map((utterance, index) => ({
        id: utterance.id,
        speakerId: utterance.speakerId,
        politenessScore: utterance.politenessScore,
        progress: clamp((currentTimeMs - utterance.startMs) / utterance.durationMs, 0, 1),
        order: index,
      }))
    : [];

  useEffect(() => {
    if (!streamSession?.id) {
      return;
    }
    if (!["queued", "processing"].includes(streamSession.status)) {
      return;
    }

    let active = true;
    pollingSessionIdRef.current = streamSession.id;

    const poll = async () => {
      if (!active || pollingSessionIdRef.current !== streamSession.id) {
        return;
      }

      try {
        const nextSession = await fetchStreamSession(streamSession.id);
        if (!active || pollingSessionIdRef.current !== streamSession.id) {
          return;
        }

        if (nextSession.status === "completed" && nextSession.transcript_id) {
          setTranscriptLoadStatus("loading");
          let loadedDocument: PlaybackDocument | null = null;

          for (let attempt = 0; attempt < TRANSCRIPT_RETRY_LIMIT; attempt += 1) {
            const nextDocument = await loadPlaybackDocument(nextSession.transcript_id);
            if (!active || pollingSessionIdRef.current !== streamSession.id) {
              return;
            }
            if (nextDocument.sentenceUnits.length > 0) {
              loadedDocument = nextDocument;
              break;
            }
            await new Promise((resolve) => window.setTimeout(resolve, POLL_INTERVAL_MS));
          }

          if (!active || pollingSessionIdRef.current !== streamSession.id) {
            return;
          }

          if (loadedDocument) {
            setDocument(loadedDocument);
            setStreamSession(nextSession);
            setSelectedProcessedTranscriptId(nextSession.transcript_id ?? loadedDocument.transcriptId);
            setTranscriptLoadStatus("ready");
            setError(null);
            setUploadProgress(1);
            setIsAwaitingReplacementTranscript(false);
            setPlaybackStarted(false);
            setCurrentTimeMs(0);
            void refreshProcessedTranscriptOptions();
            return;
          }

          setTranscriptLoadStatus("empty");
          setStreamSession(nextSession);
          setError("Transcript finished processing, but no sentence units were returned by the API.");
          return;
        }

        setStreamSession(nextSession);

        if (nextSession.status === "failed") {
          return;
        }

        window.setTimeout(poll, POLL_INTERVAL_MS);
      } catch (caughtError: unknown) {
        if (!active) {
          return;
        }
        setError(caughtError instanceof Error ? caughtError.message : "Unknown polling error");
      }
    };

    const timeoutId = window.setTimeout(poll, POLL_INTERVAL_MS);
    return () => {
      active = false;
      window.clearTimeout(timeoutId);
    };
  }, [streamSession?.id, streamSession?.status]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setPendingFile(file);
    setSelectedProcessedTranscriptId("");
    setError(null);
    event.target.value = "";
  };

  const handleUploadSubmit = async () => {
    if (!pendingFile) {
      return;
    }

    const file = pendingFile;

    if (mediaObjectUrlRef.current) {
      URL.revokeObjectURL(mediaObjectUrlRef.current);
    }

    const objectUrl = fileToObjectUrl(file);
    mediaObjectUrlRef.current = objectUrl;
    setMediaSrc(objectUrl);
    setMediaName(file.name);
    setPlaybackStarted(false);
    setCurrentTimeMs(0);
    setError(null);
    setStreamSession(null);
    setSelectedProcessedTranscriptId("");
    setUploadProgress(0);
    setIsUploading(true);
    setIsAwaitingReplacementTranscript(true);
    setTranscriptLoadStatus("idle");
    pollingSessionIdRef.current = null;

    try {
      const createdSession = await createStreamSession({
        mime_type: file.type || null,
        original_filename: file.name,
        diarization_enabled: true,
        ingest_metadata: {
          frontend_uploaded: true,
          diarization_num_speakers: speakerCount,
        },
      });
      setStreamSession(createdSession);

      await uploadFileToStreamSession(createdSession.id, file, (uploadedBytes, totalBytes) => {
        setUploadProgress(totalBytes > 0 ? uploadedBytes / totalBytes : 0);
      });

      const finalizedSession = await finalizeStreamSession(createdSession.id);
      setStreamSession(finalizedSession);
      setUploadProgress(1);
      setPendingFile(null);
    } catch (caughtError: unknown) {
      setError(caughtError instanceof Error ? caughtError.message : "Unknown upload error");
      setIsAwaitingReplacementTranscript(false);
    } finally {
      setIsUploading(false);
    }
  };

  const handleProcessedTranscriptChange = async (event: ChangeEvent<HTMLSelectElement>) => {
    const transcriptId = event.target.value;
    setSelectedProcessedTranscriptId(transcriptId);

    if (!transcriptId) {
      return;
    }

    const selectedOption = processedTranscriptOptions.find((option) => option.transcriptId === transcriptId);
    setTranscriptLoadStatus("loading");
    setError(null);
    setIsAwaitingReplacementTranscript(false);
    setPendingFile(null);
    setPlaybackStarted(false);
    setStreamSession(null);
    setUploadProgress(null);

    if (mediaObjectUrlRef.current) {
      URL.revokeObjectURL(mediaObjectUrlRef.current);
      mediaObjectUrlRef.current = null;
    }

    try {
      const payload = await loadPlaybackDocument(transcriptId);
      const latestTimestamp = payload.sentenceUnits.reduce((largest, sentence) => Math.max(largest, sentence.end_ms), 0);
      setDocument(payload);
      setMediaSrc(selectedOption?.mediaSrc ?? "");
      setMediaName(selectedOption?.label ?? `Transcript ${transcriptId}`);
      setCurrentTimeMs(selectedOption?.mediaSrc ? 0 : latestTimestamp);
      setTranscriptLoadStatus(payload.sentenceUnits.length > 0 ? "ready" : "empty");
    } catch (caughtError: unknown) {
      setError(caughtError instanceof Error ? caughtError.message : "Unknown transcript selection error");
    }
  };

  if (error) {
    return (
      <main className="shell shell--status">
        <section className="status-card">
          <p className="status-card__eyebrow">Contour Playback</p>
          <h1>Transcript data failed to load</h1>
          <p>{error}</p>
          <p>Start the FastAPI server and make sure analyzed sentence units exist for the selected transcript.</p>
        </section>
      </main>
    );
  }

  const shouldRenderTranscriptPlayback = !isAwaitingReplacementTranscript;
  const activeTranscriptId = streamSession?.transcript_id || document?.transcriptId || "No transcript loaded";

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero__player">
          <div className="hero__controls">
            <label className="file-input">
              <span>Add File</span>
              <input type="file" accept="video/*" onChange={handleFileChange} />
            </label>
            <label className="speaker-mode">
              <span>Processed</span>
              <select value={selectedProcessedTranscriptId} onChange={handleProcessedTranscriptChange} disabled={isUploading}>
                <option value="">Select processed file</option>
                {processedTranscriptOptions.map((option) => (
                  <option key={option.transcriptId} value={option.transcriptId}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="speaker-mode">
              <span>Speakers</span>
              <select
                value={speakerCount}
                onChange={(event) => setSpeakerCount(Number.parseInt(event.target.value, 10) as SpeakerCount)}
                disabled={isUploading}
              >
                <option value="2">2</option>
                <option value="1">1</option>
              </select>
            </label>
            <button
              type="button"
              className="submit-upload"
              onClick={handleUploadSubmit}
              disabled={!pendingFile || isUploading}
            >
              Submit
            </button>
          </div>
          <div className="hero__media-meta">
            <span>{mediaName}</span>
            <span>{activeTranscriptId}</span>
            <span>{speakerCount} speaker{speakerCount === 1 ? "" : "s"}</span>
            {pendingFile ? <span>ready {pendingFile.name}</span> : null}
            {describeStreamStatus(streamSession, uploadProgress) ? (
              <span>{describeStreamStatus(streamSession, uploadProgress)}</span>
            ) : null}
          </div>
          <video
            className="hero__video"
            src={mediaSrc || undefined}
            controls
            preload="metadata"
            onPlay={() => setPlaybackStarted(true)}
            onTimeUpdate={(event) => setCurrentTimeMs(event.currentTarget.currentTime * 1000)}
            onSeeked={(event) => setCurrentTimeMs(event.currentTarget.currentTime * 1000)}
          />
        </div>

      </section>

      <section className="terrain-grid">
        {!shouldRenderTranscriptPlayback || landscapeSpeakers.length === 0 ? (
          <div className="empty-state">
            <p>
              {isUploading
                ? "Uploading and processing the selected file."
                : document
                ? "The transcript is loaded, but no speaker ranges can be drawn yet."
                : "Choose a file to create a transcript-driven playback."}
            </p>
            <p>
              {!shouldRenderTranscriptPlayback
                ? "Previous transcript playback is hidden until the new file finishes processing."
                : isUploading
                ? "The transcript will switch automatically when the backend finishes processing."
                : document
                ? "Speaker diarization did not yield enough conversation structure for the shared landscape."
                : "No startup transcript is preloaded in upload mode."}
            </p>
          </div>
        ) : (
          <ConversationLandscape
            speakers={landscapeSpeakers}
            utterances={landscapeUtterances}
            activeSpeakerId={activeUtterance?.speakerId ?? null}
            activeTranscriptText={activeUtterance?.text ?? null}
          />
        )}
      </section>
    </main>
  );
}
