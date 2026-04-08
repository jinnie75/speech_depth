import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";

import { ConversationLandscape } from "../components/ConversationLandscape";
import type {
  ConversationLandscapeMarginNote,
  ConversationLandscapeTranscript,
  ConversationLandscapeTranscriptToken,
} from "../components/ConversationLandscape";
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
import type {
  JobSummary,
  PlaybackDocument,
  ReviewStatus,
  SentenceUnitResponse,
  StreamSessionResponse,
  TranscriptSegmentResponse,
} from "../lib/types";

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
  analysisPayload: Record<string, unknown> | null;
  wordRanges: WordRange[];
  wordTimings: TimedWord[];
  contourSignals: TimedSignal[];
}

interface WordRange {
  start: number;
  end: number;
}

interface TimedWord {
  startMs: number;
  endMs: number;
}

interface TimedSignal {
  startMs: number;
  endMs: number;
}

interface UtterancePlaybackState {
  progress: number;
  visibleText: string | null;
  visibleWordCount: number;
  contourSignalCount: number;
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

function extractAnalysisMatches(value: unknown): string[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }

  const matches = (value as Record<string, unknown>).matches;
  if (!Array.isArray(matches)) {
    return [];
  }

  return Array.from(
    new Set(
      matches
        .filter((entry): entry is string => typeof entry === "string")
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0),
    ),
  );
}

function buildMatchVariants(match: string): string[] {
  const normalized = match.toLowerCase();
  const variants = new Set<string>([normalized, normalized.replace(/'/g, "’"), normalized.replace(/’/g, "'")]);

  const contractionPairs: Array<[string, string]> = [
    ["i am", "i'm"],
    ["i do not", "i don't"],
    ["do not", "don't"],
    ["cannot", "can't"],
  ];

  contractionPairs.forEach(([expanded, contracted]) => {
    if (normalized.includes(expanded)) {
      variants.add(normalized.replaceAll(expanded, contracted));
      variants.add(normalized.replaceAll(expanded, contracted.replace(/'/g, "’")));
    }
  });

  return Array.from(variants).filter((variant) => variant.length > 0);
}

function markMatchCoverage(coverage: boolean[], text: string, matches: string[]): void {
  const loweredText = text.toLowerCase();

  matches.forEach((match) => {
    buildMatchVariants(match).forEach((variant) => {
      let searchIndex = 0;

      while (searchIndex < loweredText.length) {
        const matchIndex = loweredText.indexOf(variant, searchIndex);
        if (matchIndex === -1) {
          break;
        }

        for (let index = matchIndex; index < matchIndex + variant.length; index += 1) {
          coverage[index] = true;
        }

        searchIndex = matchIndex + Math.max(variant.length, 1);
      }
    });
  });
}

function findMatchRanges(text: string, matches: string[]): WordRange[] {
  const loweredText = text.toLowerCase();
  const seen = new Set<string>();
  const ranges: WordRange[] = [];

  matches.forEach((match) => {
    buildMatchVariants(match).forEach((variant) => {
      let searchIndex = 0;

      while (searchIndex < loweredText.length) {
        const matchIndex = loweredText.indexOf(variant, searchIndex);
        if (matchIndex === -1) {
          break;
        }

        const rangeKey = `${matchIndex}:${matchIndex + variant.length}`;
        if (!seen.has(rangeKey)) {
          seen.add(rangeKey);
          ranges.push({
            start: matchIndex,
            end: matchIndex + variant.length,
          });
        }

        searchIndex = matchIndex + Math.max(variant.length, 1);
      }
    });
  });

  return ranges.sort((left, right) => left.start - right.start || left.end - right.end);
}

function rangesOverlap(left: WordRange, right: WordRange): boolean {
  return left.start < right.end && right.start < left.end;
}

function expandRangeToClause(text: string, range: WordRange): WordRange {
  const isClauseBoundary = (character: string): boolean => /[,:;.!?\n]/.test(character);
  let start = clamp(range.start, 0, text.length);
  let end = clamp(range.end, 0, text.length);

  while (start > 0 && !isClauseBoundary(text[start - 1])) {
    start -= 1;
  }
  while (start < text.length && /\s|["'([{]/.test(text[start])) {
    start += 1;
  }

  while (end < text.length && !isClauseBoundary(text[end])) {
    end += 1;
  }
  while (end > start && /\s|[)"'\]]/.test(text[end - 1])) {
    end -= 1;
  }

  return { start, end };
}

function buildClauseRanges(text: string, ranges: WordRange[]): WordRange[] {
  const seen = new Set<string>();

  return ranges
    .map((range) => expandRangeToClause(text, range))
    .filter((range) => range.end > range.start)
    .filter((range) => {
      const key = `${range.start}:${range.end}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

function buildActiveTranscript(
  utterance: UtteranceSummary | null,
  analysisPayload: Record<string, unknown> | null | undefined,
  visibleWordCount: number,
  currentTimeMs: number,
): ConversationLandscapeTranscript | null {
  if (!utterance || !utterance.text || visibleWordCount <= 0) {
    return null;
  }

  const transcriptText = utterance.text;
  const hedgingMatches = extractAnalysisMatches(analysisPayload?.hedging);
  const substanceMatches = extractAnalysisMatches(analysisPayload?.substance);
  const hedgeRanges = findMatchRanges(transcriptText, hedgingMatches);
  const substanceRanges = findMatchRanges(transcriptText, substanceMatches);
  const tokens: ConversationLandscapeTranscriptToken[] = [];
  const visibleRanges = utterance.wordRanges.slice(0, visibleWordCount);
  let cursor = 0;

  visibleRanges.forEach((wordRange, index) => {
    if (cursor < wordRange.start) {
      tokens.push({
        id: `${utterance.id}-gap-${cursor}`,
        kind: "text",
        text: transcriptText.slice(cursor, wordRange.start),
      });
    }

    const wordTiming = utterance.wordTimings[index];
    const wordDurationMs = Math.max((wordTiming?.endMs ?? utterance.endMs) - (wordTiming?.startMs ?? utterance.startMs), 1);

    tokens.push({
      id: `${utterance.id}-word-${index}-${wordRange.start}-${wordRange.end}`,
      kind: "word",
      text: transcriptText.slice(wordRange.start, wordRange.end),
      start: wordRange.start,
      end: wordRange.end,
      isHedge: hedgeRanges.some((range) => rangesOverlap(range, wordRange)),
      isSubstance: substanceRanges.some((range) => rangesOverlap(range, wordRange)),
      dropStartMs: (wordTiming?.startMs ?? utterance.startMs) + Math.min(wordDurationMs * 0.16, 90),
      dropDurationMs: 1600 + Math.min(wordDurationMs * 2.5, 420),
    });

    cursor = wordRange.end;
  });

  return {
    utteranceId: utterance.id,
    currentTimeMs,
    tokens: tokens.filter((token) => token.text.length > 0),
  };
}

function stringifyAnalysisPayload(analysisPayload: Record<string, unknown> | null | undefined): string {
  if (!analysisPayload) {
    return "{}";
  }
  return JSON.stringify(analysisPayload, null, 2);
}

function extractWordRanges(text: string): WordRange[] {
  return Array.from(text.matchAll(/\S+/g), (match) => ({
    start: match.index ?? 0,
    end: (match.index ?? 0) + match[0].length,
  }));
}

function buildFallbackWordTimings(wordCount: number, startMs: number, endMs: number): TimedWord[] {
  if (wordCount <= 0) {
    return [];
  }

  const durationMs = Math.max(endMs - startMs, wordCount);
  return Array.from({ length: wordCount }, (_, index) => {
    const nextIndex = index + 1;
    const wordStartMs = startMs + Math.floor((durationMs * index) / wordCount);
    const wordEndMs = startMs + Math.ceil((durationMs * nextIndex) / wordCount);
    return {
      startMs: wordStartMs,
      endMs: Math.max(wordStartMs + 1, wordEndMs),
    };
  });
}

function resolveSentenceSegmentWords(
  sentence: SentenceUnitResponse,
  segments: TranscriptSegmentResponse[],
): TimedWord[] {
  const relevantSegmentIndices = new Set(sentence.source_segment_ids);
  const candidateWords = segments
    .filter((segment) => relevantSegmentIndices.has(segment.segment_index))
    .flatMap((segment) => segment.words_json ?? [])
    .filter((word) => {
      const midpoint = word.start_ms + (word.end_ms - word.start_ms) / 2;
      return sentence.start_ms <= midpoint && midpoint < sentence.end_ms;
    })
    .sort((left, right) => left.start_ms - right.start_ms || left.end_ms - right.end_ms);

  return candidateWords.map((word) => ({
    startMs: Math.max(sentence.start_ms, word.start_ms),
    endMs: Math.min(sentence.end_ms, Math.max(word.end_ms, word.start_ms + 1)),
  }));
}

function buildSentenceWordTimings(
  sentence: SentenceUnitResponse,
  segments: TranscriptSegmentResponse[],
  wordRanges: WordRange[],
): TimedWord[] {
  if (wordRanges.length === 0) {
    return [];
  }

  const segmentWords = resolveSentenceSegmentWords(sentence, segments);
  if (segmentWords.length === wordRanges.length) {
    return segmentWords;
  }

  return buildFallbackWordTimings(wordRanges.length, sentence.start_ms, sentence.end_ms);
}

function fallbackSignalTiming(
  signalRange: WordRange,
  utteranceTextLength: number,
  utteranceStartMs: number,
  utteranceEndMs: number,
): TimedSignal {
  const durationMs = Math.max(utteranceEndMs - utteranceStartMs, 1);
  const startRatio = utteranceTextLength <= 0 ? 0 : signalRange.start / utteranceTextLength;
  const endRatio = utteranceTextLength <= 0 ? 1 : signalRange.end / utteranceTextLength;
  const startMs = utteranceStartMs + Math.floor(durationMs * startRatio);
  const endMs = utteranceStartMs + Math.ceil(durationMs * endRatio);

  return {
    startMs,
    endMs: Math.max(startMs + 1, endMs),
  };
}

function resolveRangeTiming(
  range: WordRange,
  utteranceTextLength: number,
  wordRanges: WordRange[],
  wordTimings: TimedWord[],
  utteranceStartMs: number,
  utteranceEndMs: number,
): TimedSignal {
  const overlappingWordIndexes = wordRanges
    .map((wordRange, index) => ({ wordRange, index }))
    .filter(({ wordRange }) => range.start < wordRange.end && wordRange.start < range.end)
    .map(({ index }) => index);

  if (overlappingWordIndexes.length === 0) {
    return fallbackSignalTiming(range, utteranceTextLength, utteranceStartMs, utteranceEndMs);
  }

  const firstWordTiming = wordTimings[overlappingWordIndexes[0]];
  const lastWordTiming = wordTimings[overlappingWordIndexes[overlappingWordIndexes.length - 1]];

  if (!firstWordTiming || !lastWordTiming) {
    return fallbackSignalTiming(range, utteranceTextLength, utteranceStartMs, utteranceEndMs);
  }

  return {
    startMs: firstWordTiming.startMs,
    endMs: Math.max(firstWordTiming.startMs + 1, lastWordTiming.endMs),
  };
}

function buildContourSignals(
  text: string,
  analysisPayload: Record<string, unknown> | null | undefined,
  wordRanges: WordRange[],
  wordTimings: TimedWord[],
  utteranceStartMs: number,
  utteranceEndMs: number,
): TimedSignal[] {
  const hedgingMatches = extractAnalysisMatches(analysisPayload?.hedging);
  const matchRanges = findMatchRanges(text, hedgingMatches);

  return matchRanges.map((matchRange) =>
    resolveRangeTiming(matchRange, text.length, wordRanges, wordTimings, utteranceStartMs, utteranceEndMs),
  );
}

function buildMarginNotes(utterances: UtteranceSummary[]): ConversationLandscapeMarginNote[] {
  return utterances.flatMap((utterance) => {
    const substanceMatches = extractAnalysisMatches(utterance.analysisPayload?.substance);
    const clauseRanges = buildClauseRanges(utterance.text, findMatchRanges(utterance.text, substanceMatches));

    return clauseRanges
      .map((range, index) => {
        const timing = resolveRangeTiming(
          range,
          utterance.text.length,
          utterance.wordRanges,
          utterance.wordTimings,
          utterance.startMs,
          utterance.endMs,
        );

        return {
          id: `${utterance.id}-substance-clause-${index}-${range.start}-${range.end}`,
          text: utterance.text.slice(range.start, range.end).trim(),
          utteranceId: utterance.id,
          appearAtMs: timing.endMs + 120,
          sourceStart: range.start,
          sourceEnd: range.end,
          settleDurationMs: 3000,
        };
      })
      .filter((note) => note.text.length > 0);
  });
}

function buildUtterances(document: PlaybackDocument): UtteranceSummary[] {
  return [...document.sentenceUnits]
    .map((sentence) => {
      const wordRanges = extractWordRanges(sentence.display_text);
      const wordTimings = buildSentenceWordTimings(sentence, document.segments, wordRanges);

      return {
        id: sentence.id,
        speakerId: sentence.display_speaker_id || FALLBACK_SPEAKER_ID,
        text: sentence.display_text,
        startMs: sentence.start_ms,
        endMs: sentence.end_ms,
        durationMs: Math.max(1, sentence.end_ms - sentence.start_ms),
        politenessScore: sentence.analysis_result?.politeness_score ?? 0.5,
        analysisPayload: sentence.analysis_result?.analysis_payload ?? null,
        wordRanges,
        wordTimings,
        contourSignals: buildContourSignals(
          sentence.display_text,
          sentence.analysis_result?.analysis_payload,
          wordRanges,
          wordTimings,
          sentence.start_ms,
          sentence.end_ms,
        ),
      };
    })
    .sort((left, right) => left.startMs - right.startMs || left.endMs - right.endMs);
}

function getUtterancePlaybackState(utterance: UtteranceSummary, currentTimeMs: number): UtterancePlaybackState {
  if (utterance.wordRanges.length === 0 || utterance.wordTimings.length === 0) {
    const progress = clamp((currentTimeMs - utterance.startMs) / utterance.durationMs, 0, 1);
    const contourSignalCount = utterance.contourSignals.reduce((total, signal) => {
      if (currentTimeMs <= signal.startMs) {
        return total;
      }
      if (currentTimeMs >= signal.endMs) {
        return total + 1;
      }
      const signalDurationMs = Math.max(signal.endMs - signal.startMs, 1);
      return total + clamp((currentTimeMs - signal.startMs) / signalDurationMs, 0, 1);
    }, 0);
    return {
      progress,
      visibleText: progress > 0 ? utterance.text : null,
      visibleWordCount: progress > 0 ? utterance.wordRanges.length : 0,
      contourSignalCount,
    };
  }

  if (currentTimeMs < utterance.wordTimings[0].startMs) {
    return { progress: 0, visibleText: null, visibleWordCount: 0, contourSignalCount: 0 };
  }

  const totalWords = utterance.wordTimings.length;
  let completedWords = 0;
  let progress = 1;

  for (let index = 0; index < totalWords; index += 1) {
    const wordTiming = utterance.wordTimings[index];

    if (currentTimeMs < wordTiming.startMs) {
      progress = completedWords / totalWords;
      break;
    }

    if (currentTimeMs < wordTiming.endMs) {
      const wordDurationMs = Math.max(wordTiming.endMs - wordTiming.startMs, 1);
      const partialProgress = clamp((currentTimeMs - wordTiming.startMs) / wordDurationMs, 0, 1);
      progress = (index + partialProgress) / totalWords;
      completedWords = index + 1;
      break;
    }

    completedWords = index + 1;
  }

  const visibleText =
    completedWords > 0 ? utterance.text.slice(0, utterance.wordRanges[completedWords - 1].end).trimEnd() : null;
  const contourSignalCount = utterance.contourSignals.reduce((total, signal) => {
    if (currentTimeMs <= signal.startMs) {
      return total;
    }
    if (currentTimeMs >= signal.endMs) {
      return total + 1;
    }
    const signalDurationMs = Math.max(signal.endMs - signal.startMs, 1);
    return total + clamp((currentTimeMs - signal.startMs) / signalDurationMs, 0, 1);
  }, 0);

  return {
    progress: clamp(progress, 0, 1),
    visibleText,
    visibleWordCount: completedWords,
    contourSignalCount,
  };
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

function describeProcessingProgress(streamSession: StreamSessionResponse | null, uploadProgress: number | null): string | null {
  if (!streamSession) {
    return null;
  }

  if (streamSession.status === "open") {
    const uploadWeight = uploadProgress === null ? 10 : Math.max(5, Math.min(20, Math.round(uploadProgress * 20)));
    return `${uploadWeight}% complete · Uploading file`;
  }

  if (streamSession.status === "queued") {
    return "25% complete · Waiting for worker";
  }

  if (streamSession.status === "processing") {
    if (streamSession.processing_stage === "transcription") {
      return "50% complete · Transcription";
    }
    if (streamSession.processing_stage === "diarization") {
      return "75% complete · Speaker diarization";
    }
    if (streamSession.processing_stage === "analysis") {
      return "90% complete · Analysis";
    }
    return "60% complete · Processing";
  }

  if (streamSession.status === "completed") {
    return "100% complete · Transcript ready";
  }

  if (streamSession.status === "failed") {
    return streamSession.error_message || "Processing failed.";
  }

  return null;
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
  const activeUtterancePlaybackState = activeUtterance ? getUtterancePlaybackState(activeUtterance, currentTimeMs) : null;
  const landscapeUtterances = utterances.map((utterance, index) => {
    const playbackState = getUtterancePlaybackState(utterance, currentTimeMs);

    return {
      id: utterance.id,
      speakerId: utterance.speakerId,
      politenessScore: utterance.politenessScore,
      contourSignalCount: playbackState.contourSignalCount,
      progress: playbackState.progress,
      order: index,
    };
  });
  const marginNotes = buildMarginNotes(utterances);
  const activeTranscript = buildActiveTranscript(
    activeUtterance,
    activeSentence?.analysis_result?.analysis_payload,
    activeUtterancePlaybackState?.visibleWordCount ?? 0,
    currentTimeMs,
  );

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

  const renderCompactMediaControls = ({ minimal = false }: { minimal?: boolean } = {}) => {
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
        <div className={`review-media-controls${minimal ? " review-media-controls--minimal" : ""}`}>
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
                  <div className="upload-processing__bar" aria-hidden="true" />
                  <p className="upload-processing__copy">
                    {describeProcessingProgress(streamSession, uploadProgress) ?? "Your transcript is processing."}
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

              {(pendingFile || streamSession) && !isTranscriptProcessing ? (
                <div className="status-strip onboarding-choice__status">
                  {pendingFile ? <span>{speakerCount} speaker{speakerCount === 1 ? "" : "s"} selected</span> : null}
                  {pendingFile ? <span>Ready to upload</span> : null}
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
                <button
                  type="button"
                  className="submit-upload"
                  onClick={() => void handleSaveReview("completed")}
                  disabled={saveState === "saving"}
                >
                  Save and Continue
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
            <div className="playback-header__summary">
              <p className="eyebrow">Visualization</p>
              <div className="playback-header__title-row">
                <h1>{document.conversationTitle || mediaName}</h1>
                <div className="playback-header__actions">
                  <button type="button" className="ghost-button" onClick={() => setMode("select")}>
                    Back to Files
                  </button>
                  <button type="button" className="submit-upload" onClick={() => setMode("review")}>
                    Edit Transcript
                  </button>
                </div>
              </div>
              <div className="playback-header__transport">{renderCompactMediaControls({ minimal: true })}</div>
            </div>
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
                activeSpeakerId={activeUtterance?.speakerId ?? null}
                activeTranscript={activeTranscript}
                marginNotes={marginNotes}
                currentTimeMs={currentTimeMs}
              />
            )}
          </section>
        </>
      ) : null}
    </main>
  );
}
