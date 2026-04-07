import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";

import { ConversationLandscape } from "../components/ConversationLandscape";
import {
  buildJobMediaUrl,
  createStreamSession,
  deleteTranscript,
  fetchCompletedJobs,
  fetchStreamSession,
  finalizeStreamSession,
  loadPlaybackDocument,
  saveTranscriptReview,
  uploadFileToStreamSession,
} from "../lib/api";
import type { JobSummary, PlaybackDocument, ReviewStatus, StreamSessionResponse } from "../lib/types";

const TRANSCRIPT_ID = import.meta.env.VITE_TRANSCRIPT_ID;
const DEFAULT_MEDIA_SRC = import.meta.env.VITE_MEDIA_SRC ?? "";
const POLL_INTERVAL_MS = 2000;
const TRANSCRIPT_RETRY_LIMIT = 8;
const FALLBACK_SPEAKER_ID = "UNKNOWN_SPEAKER";
const FALLBACK_SPEAKER_LABEL = "Speaker";
const PRELOAD_TRANSCRIPT_ID = DEFAULT_MEDIA_SRC ? TRANSCRIPT_ID : undefined;
const DEFAULT_SPEAKER_COUNT = 2;
const SPEAKER_COUNT_OPTIONS: SpeakerCount[] = [1, 2, 3];

type SpeakerCount = 1 | 2 | 3;
type AppMode = "select" | "review" | "playback";

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
  hasContourSignal: boolean;
  substanceExcerpt: string | null;
}

interface ProcessedTranscriptOption {
  jobId: string;
  transcriptId: string;
  label: string;
  mediaSrc: string;
  reviewStatus: ReviewStatus;
}

interface ReviewSentenceDraft {
  sentenceId: string;
  text: string;
  speakerId: string | null;
}

interface ReviewDraft {
  conversationTitle: string;
  speakerLabels: Record<string, string>;
  sentences: Record<string, ReviewSentenceDraft>;
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 6.5v11l9-5.5z" fill="currentColor" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 6h4v12H7zM13 6h4v12h-4z" fill="currentColor" />
    </svg>
  );
}

function VolumeIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M5 10h4l5-4v12l-5-4H5zM17 9.2a4.2 4.2 0 0 1 0 5.6M18.9 6.8a7.4 7.4 0 0 1 0 10.4"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function MuteIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 10h4l5-4v12l-5-4H5z" fill="none" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.8" />
      <path d="M17 9l4 6M21 9l-4 6" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalizeLabel(value: string | null | undefined): string {
  return value?.trim() ?? "";
}

function formatSpeakerLabel(speakerId: string, index: number, speakerLabels: Record<string, string> = {}): string {
  const namedLabel = normalizeLabel(speakerLabels[speakerId]);
  if (namedLabel) {
    return namedLabel;
  }
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

function formatTimestamp(totalMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(totalMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function hasAnalysisSignal(value: unknown): boolean {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }

  return Object.values(value as Record<string, unknown>).some((entry) => {
    if (Array.isArray(entry)) {
      return entry.length > 0;
    }
    if (entry && typeof entry === "object") {
      return Object.keys(entry as Record<string, unknown>).length > 0;
    }
    return Boolean(entry);
  });
}

function shouldIncreaseContourLines(analysisPayload: Record<string, unknown> | null | undefined): boolean {
  if (!analysisPayload) {
    return false;
  }

  const hedgingPayload = analysisPayload.hedging;
  const substancePayload = analysisPayload.substance;
  return hasAnalysisSignal(hedgingPayload) || hasAnalysisSignal(substancePayload);
}

function stringifyAnalysisPayload(analysisPayload: Record<string, unknown> | null | undefined): string {
  if (!analysisPayload) {
    return "{}";
  }
  return JSON.stringify(analysisPayload, null, 2);
}

function extractSubstanceExcerpt(analysisPayload: Record<string, unknown> | null | undefined): string | null {
  const excerpt = analysisPayload?.substance;
  if (!excerpt || typeof excerpt !== "object" || Array.isArray(excerpt)) {
    return null;
  }

  const value = (excerpt as Record<string, unknown>).excerpt;
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function buildUtterances(document: PlaybackDocument): UtteranceSummary[] {
  return [...document.sentenceUnits]
    .map((sentence) => ({
      id: sentence.id,
      speakerId: sentence.display_speaker_id || FALLBACK_SPEAKER_ID,
      text: sentence.display_text,
      startMs: sentence.start_ms,
      endMs: sentence.end_ms,
      durationMs: Math.max(1, sentence.end_ms - sentence.start_ms),
      politenessScore: sentence.analysis_result?.politeness_score ?? 0.5,
      hasContourSignal: shouldIncreaseContourLines(sentence.analysis_result?.analysis_payload),
      substanceExcerpt: extractSubstanceExcerpt(sentence.analysis_result?.analysis_payload),
    }))
    .sort((left, right) => left.startMs - right.startMs || left.endMs - right.endMs);
}

function buildSpeakerSummaries(
  utterances: UtteranceSummary[],
  speakerLabels: Record<string, string>,
): SpeakerSummary[] {
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
      label: formatSpeakerLabel(id, index, speakerLabels),
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

function findPreviousUtterance(
  utterances: UtteranceSummary[],
  activeUtterance: UtteranceSummary | null,
): UtteranceSummary | null {
  if (!activeUtterance) {
    return null;
  }

  const activeIndex = utterances.findIndex((utterance) => utterance.id === activeUtterance.id);
  if (activeIndex <= 0) {
    return null;
  }

  return utterances[activeIndex - 1] ?? null;
}

function fileToObjectUrl(file: File): string {
  return URL.createObjectURL(file);
}

function getProcessedTranscriptLabel(job: JobSummary): string {
  const conversationTitle = job.conversation_title?.trim();
  if (conversationTitle) {
    return conversationTitle;
  }

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

function detectSpeakerIds(document: PlaybackDocument | null): string[] {
  if (!document) {
    return [];
  }
  const seen = new Set<string>();
  const detected: string[] = [];

  for (const sentence of document.sentenceUnits) {
    if (!sentence.speaker_id || seen.has(sentence.speaker_id)) {
      continue;
    }
    seen.add(sentence.speaker_id);
    detected.push(sentence.speaker_id);
  }

  return detected;
}

function buildReviewDraft(document: PlaybackDocument): ReviewDraft {
  const sentences: Record<string, ReviewSentenceDraft> = {};
  for (const sentence of document.sentenceUnits) {
    sentences[sentence.id] = {
      sentenceId: sentence.id,
      text: sentence.display_text,
      speakerId: sentence.display_speaker_id,
    };
  }

  return {
    conversationTitle: document.conversationTitle ?? "",
    speakerLabels: { ...document.speakerLabels },
    sentences,
  };
}

function normalizeSpeakerLabels(labels: Record<string, string>, allowedSpeakerIds: string[]): Record<string, string> {
  const allowed = new Set(allowedSpeakerIds);
  return Object.fromEntries(
    Object.entries(labels)
      .filter(([speakerId]) => allowed.has(speakerId))
      .map(([speakerId, label]) => [speakerId, normalizeLabel(label)])
      .filter(([, label]) => label.length > 0),
  );
}

function areSpeakerLabelsEqual(
  left: Record<string, string>,
  right: Record<string, string>,
  allowedSpeakerIds: string[],
): boolean {
  const normalizedLeft = normalizeSpeakerLabels(left, allowedSpeakerIds);
  const normalizedRight = normalizeSpeakerLabels(right, allowedSpeakerIds);
  return JSON.stringify(normalizedLeft) === JSON.stringify(normalizedRight);
}

function isReviewDraftDirty(document: PlaybackDocument | null, draft: ReviewDraft | null): boolean {
  if (!document || !draft) {
    return false;
  }

  const speakerIds = detectSpeakerIds(document);
  if (normalizeLabel(draft.conversationTitle) !== normalizeLabel(document.conversationTitle)) {
    return true;
  }
  if (!areSpeakerLabelsEqual(draft.speakerLabels, document.speakerLabels, speakerIds)) {
    return true;
  }

  return document.sentenceUnits.some((sentence) => {
    const sentenceDraft = draft.sentences[sentence.id];
    if (!sentenceDraft) {
      return false;
    }
    return sentenceDraft.text !== sentence.display_text || sentenceDraft.speakerId !== sentence.display_speaker_id;
  });
}

function buildReviewPayload(document: PlaybackDocument, draft: ReviewDraft, reviewStatus: ReviewStatus) {
  const speakerIds = detectSpeakerIds(document);
  const speakerLabels = normalizeSpeakerLabels(draft.speakerLabels, speakerIds);
  const sentenceOverrides = document.sentenceUnits
    .map((sentence) => {
      const sentenceDraft = draft.sentences[sentence.id];
      if (!sentenceDraft) {
        return null;
      }

      const manualText = sentenceDraft.text === sentence.text ? null : sentenceDraft.text;
      const manualSpeakerId = sentenceDraft.speakerId === sentence.speaker_id ? null : sentenceDraft.speakerId;
      if (manualText === null && manualSpeakerId === null) {
        return null;
      }

      return {
        sentence_unit_id: sentence.id,
        manual_text: manualText,
        manual_speaker_id: manualSpeakerId,
      };
    })
    .filter((override): override is NonNullable<typeof override> => override !== null);

  return {
    conversation_title: normalizeLabel(draft.conversationTitle) || null,
    speaker_labels: speakerLabels,
    review_status: reviewStatus,
    sentence_overrides: sentenceOverrides,
  };
}

export function App() {
  const [mode, setMode] = useState<AppMode>("select");
  const [document, setDocument] = useState<PlaybackDocument | null>(null);
  const [reviewDraft, setReviewDraft] = useState<ReviewDraft | null>(null);
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
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const [mediaDurationMs, setMediaDurationMs] = useState(0);
  const [isMediaPlaying, setIsMediaPlaying] = useState(false);
  const [isMediaMuted, setIsMediaMuted] = useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const mediaObjectUrlRef = useRef<string | null>(null);
  const pollingSessionIdRef = useRef<string | null>(null);
  const mediaElementRef = useRef<HTMLVideoElement | null>(null);

  const refreshProcessedTranscriptOptions = async (): Promise<ProcessedTranscriptOption[]> => {
    const jobs = await fetchCompletedJobs();
    const options = jobs
      .map((job) => ({
        jobId: job.id,
        transcriptId: job.transcript_id ?? "",
        label: getProcessedTranscriptLabel(job),
        mediaSrc: getProcessedTranscriptMediaSrc(job),
        reviewStatus: job.review_status,
      }))
      .filter((option) => option.transcriptId);
    setProcessedTranscriptOptions(options);
    return options;
  };

  const loadTranscriptIntoApp = async ({
    transcriptId,
    mediaSource,
    mediaLabel,
    selectedTranscriptId,
    preferredMode,
  }: {
    transcriptId: string;
    mediaSource: string;
    mediaLabel: string;
    selectedTranscriptId: string;
    preferredMode?: AppMode;
  }) => {
    setTranscriptLoadStatus("loading");
    const payload = await loadPlaybackDocument(transcriptId);
    setDocument(payload);
    setMediaSrc(mediaSource);
    setMediaName(mediaLabel);
    setSelectedProcessedTranscriptId(selectedTranscriptId);
    setPlaybackStarted(false);
    setCurrentTimeMs(0);
    setError(null);
    setPendingFile(null);
    setUploadProgress(null);
    setStreamSession(null);
    setTranscriptLoadStatus(payload.sentenceUnits.length > 0 ? "ready" : "empty");
    setMode(preferredMode ?? (payload.reviewStatus === "completed" ? "playback" : "review"));
  };

  useEffect(() => {
    let active = true;

    refreshProcessedTranscriptOptions()
      .then(async (options) => {
        if (!PRELOAD_TRANSCRIPT_ID || !active) {
          return;
        }

        const matchingOption = options.find((option) => option.transcriptId === PRELOAD_TRANSCRIPT_ID);
        const payload = await loadPlaybackDocument(PRELOAD_TRANSCRIPT_ID);
        if (!active) {
          return;
        }

        setDocument(payload);
        setMediaSrc(matchingOption?.mediaSrc ?? DEFAULT_MEDIA_SRC);
        setMediaName(matchingOption?.label ?? "Configured media source");
        setSelectedProcessedTranscriptId(matchingOption?.transcriptId ?? "");
        setTranscriptLoadStatus(payload.sentenceUnits.length > 0 ? "ready" : "empty");
        setMode(payload.reviewStatus === "completed" ? "playback" : "review");
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

  useEffect(() => {
    if (!document) {
      setReviewDraft(null);
      return;
    }

    setReviewDraft(buildReviewDraft(document));
    setSaveState("idle");
  }, [document]);

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
            setSelectedProcessedTranscriptId(nextSession.transcript_id);
            setTranscriptLoadStatus("ready");
            setError(null);
            setUploadProgress(1);
            setMode("review");
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

  const utterances = document ? buildUtterances(document) : [];
  const speakerIds = detectSpeakerIds(document);
  const speakerLabels = reviewDraft?.speakerLabels ?? document?.speakerLabels ?? {};
  const speakers = buildSpeakerSummaries(utterances, speakerLabels);
  const activeUtterance = playbackStarted || currentTimeMs > 0 ? findActiveUtterance(utterances, currentTimeMs) : null;
  const excerptUtterance = findPreviousUtterance(utterances, activeUtterance);
  const activeSentence =
    document?.sentenceUnits.find((sentence) => sentence.id === activeUtterance?.id) ?? null;
  const isReviewDirty = isReviewDraftDirty(document, reviewDraft);
  const isTranscriptProcessing =
    streamSession !== null && ["open", "queued", "processing"].includes(streamSession.status);
  const landscapeSpeakers = speakers.map((speaker, index) => ({
    id: speaker.id,
    label: speaker.label,
    side: index % 2 === 0 ? ("left" as const) : ("right" as const),
    opacity: activeUtterance?.speakerId === speaker.id ? 1 : 0.5,
    totalDurationMs: speaker.totalDurationMs,
    averagePoliteness: speaker.averagePoliteness,
  }));
  const landscapeUtterances = utterances.map((utterance, index) => ({
    id: utterance.id,
    speakerId: utterance.speakerId,
    politenessScore: utterance.politenessScore,
    hasContourSignal: utterance.hasContourSignal,
    substanceExcerpt: utterance.substanceExcerpt,
    progress: clamp((currentTimeMs - utterance.startMs) / utterance.durationMs, 0, 1),
    order: index,
  }));

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setPendingFile(file);
    setSelectedProcessedTranscriptId("");
    setError(null);
    setMode("select");
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
    setMediaDurationMs(0);
    setIsMediaPlaying(false);
    setError(null);
    setStreamSession(null);
    setUploadProgress(0);
    setIsUploading(true);
    setTranscriptLoadStatus("idle");
    pollingSessionIdRef.current = null;
    setMode("select");

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
    } finally {
      setIsUploading(false);
    }
  };

  const handleProcessedTranscriptSelect = async (option: ProcessedTranscriptOption) => {
    try {
      await loadTranscriptIntoApp({
        transcriptId: option.transcriptId,
        mediaSource: option.mediaSrc,
        mediaLabel: option.label,
        selectedTranscriptId: option.transcriptId,
      });
    } catch (caughtError: unknown) {
      setError(caughtError instanceof Error ? caughtError.message : "Unknown transcript selection error");
    }
  };

  const updateDraftTitle = (value: string) => {
    setReviewDraft((currentDraft) => (currentDraft ? { ...currentDraft, conversationTitle: value } : currentDraft));
    setSaveState("idle");
  };

  const updateDraftSpeakerLabel = (speakerId: string, value: string) => {
    setReviewDraft((currentDraft) =>
      currentDraft
        ? {
          ...currentDraft,
          speakerLabels: {
            ...currentDraft.speakerLabels,
            [speakerId]: value,
          },
        }
        : currentDraft,
    );
    setSaveState("idle");
  };

  const updateDraftSentence = (sentenceId: string, updates: Partial<ReviewSentenceDraft>) => {
    setReviewDraft((currentDraft) => {
      if (!currentDraft) {
        return currentDraft;
      }
      const existingSentence = currentDraft.sentences[sentenceId];
      if (!existingSentence) {
        return currentDraft;
      }
      return {
        ...currentDraft,
        sentences: {
          ...currentDraft.sentences,
          [sentenceId]: {
            ...existingSentence,
            ...updates,
          },
        },
      };
    });
    setSaveState("idle");
  };

  const resetDraftSentence = (sentenceId: string) => {
    if (!document) {
      return;
    }
    const sourceSentence = document.sentenceUnits.find((sentence) => sentence.id === sentenceId);
    if (!sourceSentence) {
      return;
    }
    updateDraftSentence(sentenceId, {
      text: sourceSentence.text,
      speakerId: sourceSentence.speaker_id,
    });
  };

  const handleSaveReview = async (nextStatus: ReviewStatus) => {
    if (!document || !reviewDraft) {
      return;
    }

    setSaveState("saving");
    setError(null);

    try {
      const nextDocument = await saveTranscriptReview(
        document.transcriptId,
        buildReviewPayload(document, reviewDraft, nextStatus),
      );
      setDocument(nextDocument);
      setSelectedProcessedTranscriptId(nextDocument.transcriptId);
      setMode(nextStatus === "completed" ? "playback" : "review");
      setSaveState("saved");
      void refreshProcessedTranscriptOptions();
    } catch (caughtError: unknown) {
      setSaveState("idle");
      setError(caughtError instanceof Error ? caughtError.message : "Unknown save error");
    }
  };

  const confirmDeleteTranscript = async () => {
    if (!document) {
      return;
    }

    setSaveState("saving");
    setError(null);

    try {
      await deleteTranscript(document.transcriptId);
      setIsDeleteConfirmOpen(false);
      setDocument(null);
      setReviewDraft(null);
      setMediaSrc("");
      setMediaName("No file selected");
      setPlaybackStarted(false);
      setCurrentTimeMs(0);
      setMediaDurationMs(0);
      setIsMediaPlaying(false);
      setPendingFile(null);
      setSelectedProcessedTranscriptId("");
      setTranscriptLoadStatus("idle");
      setStreamSession(null);
      setUploadProgress(null);
      setMode("select");
      setSaveState("idle");
      await refreshProcessedTranscriptOptions();
    } catch (caughtError: unknown) {
      setSaveState("idle");
      setError(caughtError instanceof Error ? caughtError.message : "Unknown delete error");
    }
  };

  const syncMediaState = (element: HTMLVideoElement) => {
    setCurrentTimeMs(element.currentTime * 1000);
    setMediaDurationMs(Number.isFinite(element.duration) ? element.duration * 1000 : 0);
    setIsMediaPlaying(!element.paused);
    setIsMediaMuted(element.muted);
  };

  const handleMediaMetadata = () => {
    if (mediaElementRef.current) {
      syncMediaState(mediaElementRef.current);
    }
  };

  const handleMediaTimeUpdate = () => {
    if (mediaElementRef.current) {
      syncMediaState(mediaElementRef.current);
    }
  };

  const handleMediaPlay = () => {
    setPlaybackStarted(true);
    setIsMediaPlaying(true);
  };

  const handleMediaPause = () => {
    setIsMediaPlaying(false);
  };

  const handleMediaVolumeChange = () => {
    if (mediaElementRef.current) {
      syncMediaState(mediaElementRef.current);
    }
  };

  const toggleMediaPlayback = async () => {
    if (!mediaElementRef.current) {
      return;
    }
    if (mediaElementRef.current.paused) {
      await mediaElementRef.current.play();
      return;
    }
    mediaElementRef.current.pause();
  };

  const handleMediaSeek = (nextTimeMs: number) => {
    if (!mediaElementRef.current) {
      return;
    }
    mediaElementRef.current.currentTime = nextTimeMs / 1000;
    setCurrentTimeMs(nextTimeMs);
  };

  const toggleMediaMute = () => {
    if (!mediaElementRef.current) {
      return;
    }
    mediaElementRef.current.muted = !mediaElementRef.current.muted;
    setIsMediaMuted(mediaElementRef.current.muted);
  };

  const renderCompactMediaControls = () => {
    if (!mediaSrc) {
      return (
        <div className="empty-state">
          <p>No media preview is available for this transcript.</p>
        </div>
      );
    }

    return (
      <>
        <video
          ref={mediaElementRef}
          className="review-media__element"
          src={mediaSrc}
          preload="metadata"
          onLoadedMetadata={handleMediaMetadata}
          onPlay={handleMediaPlay}
          onPause={handleMediaPause}
          onTimeUpdate={handleMediaTimeUpdate}
          onSeeked={handleMediaTimeUpdate}
          onVolumeChange={handleMediaVolumeChange}
        />
        <div className="review-media-controls">
          <button
            type="button"
            className="ghost-button review-media-controls__button review-media-controls__icon-button"
            onClick={() => void toggleMediaPlayback()}
            aria-label={isMediaPlaying ? "Pause media" : "Play media"}
            title={isMediaPlaying ? "Pause" : "Play"}
          >
            {isMediaPlaying ? <PauseIcon /> : <PlayIcon />}
          </button>
          <div className="review-media-controls__timeline">
            <span>{formatTimestamp(currentTimeMs)}</span>
            <input
              type="range"
              min={0}
              max={Math.max(mediaDurationMs, 0)}
              step={100}
              value={Math.min(currentTimeMs, mediaDurationMs || currentTimeMs)}
              onChange={(event) => handleMediaSeek(Number.parseInt(event.target.value, 10))}
            />
            <span>{formatTimestamp(mediaDurationMs)}</span>
          </div>
          <button
            type="button"
            className="ghost-button review-media-controls__button review-media-controls__icon-button"
            onClick={toggleMediaMute}
            aria-label={isMediaMuted ? "Unmute media" : "Mute media"}
            title={isMediaMuted ? "Unmute" : "Mute"}
          >
            {isMediaMuted ? <MuteIcon /> : <VolumeIcon />}
          </button>
        </div>
      </>
    );
  };

  if (error) {
    return (
      <main className="shell shell--status">
        <section className="status-card">
          <p className="status-card__eyebrow">Contour Playback</p>
          <h1>Something went wrong</h1>
          <p>{error}</p>
          <button type="button" className="ghost-button" onClick={() => setError(null)}>
            Dismiss
          </button>
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      {mode === "select" ? (
        <>
          <section className="onboarding-stack">
            <article className="surface-card onboarding-choice onboarding-choice--upload">
              <div className="onboarding-choice__header">
                <p className="eyebrow">Option 1</p>
                <h2>Upload a file</h2>
              </div>

              {isTranscriptProcessing ? (
                <div className="upload-processing" aria-live="polite">
                  <h3>Processing...</h3>
                  <div className="upload-processing__bar" aria-hidden="true">
                    <span className="upload-processing__bar-fill" />
                  </div>
                  <p className="upload-processing__copy">
                    {describeStreamStatus(streamSession, uploadProgress) ?? "Your transcript is processing."}
                  </p>
                </div>
              ) : !pendingFile ? (
                <label className="upload-dropzone">
                  <input type="file" accept=".mov,.mp4,video/mp4,video/quicktime" onChange={handleFileChange} />
                  <span className="upload-dropzone__step">Step 1</span>
                  <strong>add file</strong>
                  <span className="upload-dropzone__hint">Choose a `.mov` or `.mp4` to start a new transcript.</span>
                </label>
              ) : (
                <div className="upload-workflow">
                  <div className="upload-workflow__file">
                    <p className="upload-workflow__step">Step 1 complete</p>
                    <h3>{pendingFile.name}</h3>
                    <label className="ghost-button upload-workflow__replace">
                      choose another file
                      <input type="file" accept=".mov,.mp4,video/mp4,video/quicktime" onChange={handleFileChange} />
                    </label>
                  </div>

                  <div className="upload-workflow__controls">
                    <div className="speaker-step">
                      <span className="speaker-step__label">Step 2 Select speakers</span>
                      <div className="speaker-step__options" role="group" aria-label="Expected number of speakers">
                        {SPEAKER_COUNT_OPTIONS.map((option) => (
                          <button
                            key={option}
                            type="button"
                            className={`speaker-step__option${speakerCount === option ? " speaker-step__option--active" : ""}`}
                            aria-pressed={speakerCount === option}
                            onClick={() => setSpeakerCount(option)}
                            disabled={isUploading}
                          >
                            {option}
                          </button>
                        ))}
                      </div>
                    </div>

                    <button
                      type="button"
                      className="submit-upload"
                      onClick={handleUploadSubmit}
                      disabled={!pendingFile || isUploading}
                    >
                      Submit
                    </button>
                  </div>
                </div>
              )}

              {pendingFile || streamSession ? (
                <div className="status-strip onboarding-choice__status">
                  {pendingFile && !isTranscriptProcessing ? (
                    <span>{speakerCount} speaker{speakerCount === 1 ? "" : "s"} selected</span>
                  ) : null}
                  {pendingFile && !isTranscriptProcessing ? <span>Ready to upload</span> : null}
                  {describeStreamStatus(streamSession, uploadProgress) ? (
                    <span>{describeStreamStatus(streamSession, uploadProgress)}</span>
                  ) : null}
                </div>
              ) : null}
            </article>

            <div className="onboarding-divider" aria-hidden="true">
              <span>or</span>
            </div>

            <article className="surface-card onboarding-choice onboarding-choice--select">
              <div className="onboarding-choice__header">
                <p className="eyebrow">Option 2</p>
                <h2>Choose a saved conversation</h2>
              </div>
              {processedTranscriptOptions.length === 0 ? (
                <div className="empty-state">
                  <p>No processed transcripts are available yet.</p>
                  <p>Upload a file and let the worker finish transcription to populate this list.</p>
                </div>
              ) : (
                <div className="transcript-list">
                  {processedTranscriptOptions.map((option) => (
                    <button
                      type="button"
                      key={option.transcriptId}
                      className={`transcript-option${selectedProcessedTranscriptId === option.transcriptId ? " transcript-option--active" : ""
                        }`}
                      onClick={() => void handleProcessedTranscriptSelect(option)}
                    >
                      <span className="transcript-option__title">{option.label}</span>
                      <span className="transcript-option__meta">{option.reviewStatus.replace("_", " ")}</span>
                    </button>
                  ))}
                </div>
              )}
            </article>
          </section>
        </>
      ) : null}

      {mode === "review" && document && reviewDraft ? (
        <>
          <section className="review-top-stack">
            <section className="review-toolbar">
              <div className="review-toolbar__title">
                <p className="eyebrow">Transcript Review</p>
                <input
                  className="title-input"
                  value={reviewDraft.conversationTitle}
                  onChange={(event) => updateDraftTitle(event.target.value)}
                  placeholder="Name this conversation"
                />
              </div>
              <div className="review-toolbar__actions">
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => setIsDeleteConfirmOpen(true)}
                  disabled={saveState === "saving"}
                >
                  Delete Transcript
                </button>
                <button type="button" className="ghost-button" onClick={() => setMode("select")}>
                  Back to Files
                </button>
              </div>
            </section>

            <article className="surface-card review-media">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Media</p>
                  <h2>{mediaName}</h2>
                </div>
                <div className="status-strip">
                  <span>{document.transcriptId}</span>
                  <span>{speakerIds.length || 0} detected speaker{speakerIds.length === 1 ? "" : "s"}</span>
                  <span>{document.reviewStatus.replace("_", " ")}</span>
                </div>
              </div>
              {renderCompactMediaControls()}
            </article>
          </section>

          <section className="review-editor-shell">
            <article className="surface-card review-editor">
              <div className="section-heading review-editor__heading">
                <div>
                  <p className="eyebrow">Transcript Editor</p>
                  <h2>Assign names and correct the transcript</h2>
                </div>
                <p className="surface-note">Every row is editable. Update speaker names, reassign speakers, and revise transcript text in place.</p>
              </div>

              {speakerIds.length > 0 ? (
                <div className="speaker-name-grid">
                  {speakerIds.map((speakerId, index) => (
                    <label key={speakerId} className="speaker-name-card">
                      <span>{formatSpeakerLabel(speakerId, index)}</span>
                      <input
                        value={reviewDraft.speakerLabels[speakerId] ?? ""}
                        onChange={(event) => updateDraftSpeakerLabel(speakerId, event.target.value)}
                        placeholder={`Name ${formatSpeakerLabel(speakerId, index)}`}
                      />
                    </label>
                  ))}
                </div>
              ) : (
                <div className="empty-state">
                  <p>No speaker diarization labels were detected for this transcript.</p>
                  <p>You can still title the conversation and edit the sentence text below.</p>
                </div>
              )}

              <div className="transcript-grid transcript-grid--header" aria-hidden="true">
                <span>Timestamp</span>
                <span>Speaker</span>
                <span>Text</span>
                <span className="transcript-grid__actions-label">Actions</span>
              </div>

              <div className="transcript-editor-list">
                {document.sentenceUnits.map((sentence) => {
                  const sentenceDraft = reviewDraft.sentences[sentence.id];
                  const isActive = activeUtterance?.id === sentence.id;
                  const speakerIndex = speakerIds.indexOf(sentenceDraft?.speakerId ?? sentence.speaker_id ?? "");

                  return (
                    <div key={sentence.id} className={`transcript-row transcript-row--editing${isActive ? " transcript-row--active" : ""}`}>
                      <div className="transcript-grid">
                        <span className="transcript-row__time">{formatTimestamp(sentence.start_ms)}</span>
                        {speakerIds.length > 0 && sentenceDraft ? (
                          <label className="speaker-mode transcript-row__speaker-field">
                            <select
                              value={sentenceDraft.speakerId ?? ""}
                              onChange={(event) =>
                                updateDraftSentence(sentence.id, {
                                  speakerId: event.target.value || null,
                                })
                              }
                            >
                              <option value="">Unassigned</option>
                              {speakerIds.map((speakerId, index) => (
                                <option key={speakerId} value={speakerId}>
                                  {formatSpeakerLabel(speakerId, index, reviewDraft.speakerLabels)}
                                </option>
                              ))}
                            </select>
                          </label>
                        ) : (
                          <span className="transcript-row__speaker transcript-row__speaker--plain">
                            {sentenceDraft?.speakerId && speakerIndex >= 0
                              ? formatSpeakerLabel(sentenceDraft.speakerId, speakerIndex, reviewDraft.speakerLabels)
                              : "Unassigned"}
                          </span>
                        )}
                        {sentenceDraft ? (
                          <label className="transcript-row__field transcript-row__field--text">
                            <textarea
                              value={sentenceDraft.text}
                              onChange={(event) => updateDraftSentence(sentence.id, { text: event.target.value })}
                              rows={2}
                            />
                          </label>
                        ) : null}
                        <div className="transcript-row__actions">
                          <button type="button" className="ghost-button" onClick={() => resetDraftSentence(sentence.id)}>
                            Reset
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="review-editor__footer">
                <span className="save-indicator">
                  {saveState === "saving" ? "Saving..." : saveState === "saved" ? "Saved" : isReviewDirty ? "Unsaved changes" : "Up to date"}
                </span>
                <div className="review-editor__footer-actions">
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => void handleSaveReview("in_progress")}
                    disabled={saveState === "saving" || !isReviewDirty}
                  >
                    Save Draft
                  </button>
                  <button
                    type="button"
                    className="submit-upload"
                    onClick={() => void handleSaveReview("completed")}
                    disabled={saveState === "saving"}
                  >
                    Save and Continue
                  </button>
                </div>
              </div>
            </article>
          </section>
        </>
      ) : null}

      {isDeleteConfirmOpen ? (
        <div className="modal-backdrop" role="presentation">
          <div className="modal-card" role="alertdialog" aria-modal="true" aria-labelledby="delete-transcript-title">
            <p className="eyebrow">Confirm Delete</p>
            <h2 id="delete-transcript-title">Delete this transcript?</h2>
            <p className="surface-note">This removes the transcript from the library and cannot be undone.</p>
            <div className="modal-card__actions">
              <button type="button" className="ghost-button" onClick={() => setIsDeleteConfirmOpen(false)} disabled={saveState === "saving"}>
                Cancel
              </button>
              <button type="button" className="submit-upload" onClick={() => void confirmDeleteTranscript()} disabled={saveState === "saving"}>
                {saveState === "saving" ? "Deleting..." : "Delete Transcript"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {mode === "playback" && document ? (
        <>
          <section className="playback-header">
            <div>
              <p className="eyebrow">Visualization</p>
              <h1>{document.conversationTitle || mediaName}</h1>
              <p className="lede">Review is complete. You can still reopen the transcript editor at any time.</p>
            </div>
            <div className="playback-header__actions">
              <button type="button" className="ghost-button" onClick={() => setMode("select")}>
                Back to Files
              </button>
              <button type="button" className="submit-upload" onClick={() => setMode("review")}>
                Edit Transcript
              </button>
            </div>
          </section>

          <section className="playback-media surface-card">
            <div className="status-strip">
              <span>{mediaName}</span>
              <span>{document.transcriptId}</span>
              <span>{speakers.length} speaker{speakers.length === 1 ? "" : "s"}</span>
            </div>
            {renderCompactMediaControls()}
          </section>

          <section className="terrain-grid">
            {transcriptLoadStatus === "empty" || landscapeSpeakers.length === 0 ? (
              <div className="empty-state">
                <p>The transcript is loaded, but no speaker ranges can be drawn yet.</p>
                <p>Review speaker assignments in the transcript editor if you want a richer conversation map.</p>
              </div>
            ) : (
              <ConversationLandscape
                speakers={landscapeSpeakers}
                utterances={landscapeUtterances}
                activeUtteranceId={excerptUtterance?.id ?? null}
                activeSpeakerId={activeUtterance?.speakerId ?? null}
                activeTranscriptText={activeUtterance?.text ?? null}
              />
            )}
          </section>

          <section className="current-payload surface-card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Current Payload</p>
                <h2>Active sentence analysis</h2>
              </div>
              <p className="surface-note">This updates with playback and shows the current sentence&apos;s `analysis_payload`.</p>
            </div>
            {activeSentence ? (
              <div className="current-payload__body">
                <p className="current-payload__text">{activeSentence.display_text}</p>
                <pre className="current-payload__code">
                  {stringifyAnalysisPayload(activeSentence.analysis_result?.analysis_payload)}
                </pre>
              </div>
            ) : (
              <div className="empty-state">
                <p>Start playback to inspect the active sentence payload.</p>
              </div>
            )}
          </section>
        </>
      ) : null}
    </main>
  );
}
