import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";

import { ConversationLandscape } from "../components/ConversationLandscape";
import type {
  ConversationLandscapeHandle,
  ConversationLandscapeMarginNote,
  ConversationLandscapeTranscript,
  ConversationLandscapeTranscriptToken,
} from "../components/ConversationLandscape";
import {
  appendLiveSessionEvent,
  buildJobMediaUrl,
  createLiveSession,
  createStreamSession,
  deleteTranscript,
  fetchCompletedJobs,
  finalizeLiveSession,
  fetchLiveSession,
  fetchStreamSession,
  finalizeStreamSession,
  loadPlaybackDocument,
  saveTranscriptReview,
  stopLiveSession,
  uploadChunkToLiveSession,
  uploadFileToStreamSession,
} from "../lib/api";
import type {
  AppendLiveTranscriptEventRequest,
  ArchivePreviewResponse,
  JobSummary,
  LiveAnalysisEventRequest,
  LiveSessionResponse,
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
const LIVE_CAPTURE_TIMESLICE_MS = 1200;
const ARCHIVE_PAGE_SIZE = 12;
const ARCHIVE_PREVIEW_BATCH_SIZE = 4;
const LIVE_HEDGE_PHRASES = [
  "i think",
  "maybe",
  "probably",
  "perhaps",
  "kind of",
  "sort of",
  "i guess",
  "i feel like",
  "might",
  "could",
];
const LIVE_SUBSTANCE_PHRASES = [
  "i need",
  "i want",
  "i feel",
  "i'm feeling",
  "i am feeling",
  "i'm worried",
  "i am worried",
  "i'm upset",
  "i am upset",
  "i'm frustrated",
  "i am frustrated",
  "i'm afraid",
  "i am afraid",
  "help",
];

type SpeakerCount = 1 | 2 | 3;
type AppMode = "select" | "create" | "about" | "live" | "review" | "playback";

type LiveCaptureState = "idle" | "starting" | "recording" | "stopping" | "finalizing";

interface SpeechRecognitionAlternativeLike {
  transcript: string;
}

interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: SpeechRecognitionAlternativeLike;
}

interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
}

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

interface SpeechRecognitionConstructorLike {
  new(): SpeechRecognitionLike;
}

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
  createdAt: string;
  reviewStatus: ReviewStatus;
  archivePreview: ArchivePreviewData | null;
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

interface LiveTranscriptEntry {
  utteranceKey: string;
  text: string;
  startMs: number;
  endMs: number;
  isFinal: boolean;
  analysis: LiveAnalysisEventRequest | null;
}

interface ArchivePreviewData {
  speakers: Array<{
    id: string;
    label: string;
    side: "left" | "right";
    opacity: number;
    totalDurationMs: number;
    averagePoliteness: number;
  }>;
  utterances: Array<{
    id: string;
    speakerId: string;
    politenessScore: number;
    contourSignalCount: number;
    progress: number;
    order: number;
  }>;
  activeSpeakerId: string | null;
  activeTranscript: ConversationLandscapeTranscript | null;
  marginNotes: ConversationLandscapeMarginNote[];
  currentTimeMs: number;
}

interface ArchivePreviewCardProps {
  isActive: boolean;
  option: ProcessedTranscriptOption;
  preview: ArchivePreviewData | null | undefined;
  onSelect: (option: ProcessedTranscriptOption) => void;
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

function ArchivePreviewCard({ isActive, option, preview, onSelect }: ArchivePreviewCardProps) {
  const cardRef = useRef<HTMLButtonElement | null>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (isVisible) {
      return;
    }

    const cardNode = cardRef.current;
    if (!cardNode) {
      return;
    }

    if (typeof IntersectionObserver === "undefined") {
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting || entry.intersectionRatio > 0)) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "240px" },
    );

    observer.observe(cardNode);
    return () => {
      observer.disconnect();
    };
  }, [isVisible]);

  return (
    <button
      ref={cardRef}
      type="button"
      className={`archive-card${isActive ? " archive-card--active" : ""}`}
      onClick={() => onSelect(option)}
    >
      <div className="archive-card__meta">
        <span className="archive-card__title">{option.label}</span>
        <span className="archive-card__date">{formatTranscriptCreatedAt(option.createdAt)}</span>
      </div>
      <div className="archive-card__preview">
        {preview ? (
          isVisible ? (
            <div className="archive-card__preview-frame" aria-hidden="true">
              <div className="archive-card__preview-landscape">
                <ConversationLandscape
                  speakers={preview.speakers}
                  utterances={preview.utterances}
                  activeSpeakerId={preview.activeSpeakerId}
                  activeTranscript={preview.activeTranscript}
                  marginNotes={preview.marginNotes}
                  currentTimeMs={preview.currentTimeMs}
                  snapshotMode="final"
                  staticPreview
                />
              </div>
            </div>
          ) : (
            <div className="archive-card__placeholder">
              <p>Preview ready.</p>
            </div>
          )
        ) : preview === null ? (
          <div className="archive-card__placeholder">
            <p>Final image unavailable.</p>
          </div>
        ) : (
          <div className="archive-card__placeholder">
            <p>Preparing final image...</p>
          </div>
        )}
      </div>
    </button>
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

function deserializeArchivePreview(preview: ArchivePreviewResponse | null | undefined): ArchivePreviewData | null {
  if (!preview) {
    return null;
  }

  return {
    speakers: preview.speakers.map((speaker) => ({
      id: speaker.id,
      label: speaker.label,
      side: speaker.side,
      opacity: speaker.opacity,
      totalDurationMs: speaker.total_duration_ms,
      averagePoliteness: speaker.average_politeness,
    })),
    utterances: preview.utterances.map((utterance) => ({
      id: utterance.id,
      speakerId: utterance.speaker_id,
      politenessScore: utterance.politeness_score,
      contourSignalCount: utterance.contour_signal_count,
      progress: utterance.progress,
      order: utterance.order,
    })),
    activeSpeakerId: preview.active_speaker_id,
    activeTranscript: preview.active_transcript
      ? {
        utteranceId: preview.active_transcript.utterance_id,
        currentTimeMs: preview.active_transcript.current_time_ms,
        tokens: preview.active_transcript.tokens.map((token) =>
          token.kind === "word"
            ? {
              id: token.id,
              kind: "word",
              text: token.text,
              start: token.start ?? 0,
              end: token.end ?? token.text.length,
              isHedge: !!token.is_hedge,
              isSubstance: !!token.is_substance,
              dropStartMs: token.drop_start_ms ?? 0,
              dropDurationMs: token.drop_duration_ms ?? 0,
            }
            : {
              id: token.id,
              kind: "text",
              text: token.text,
            },
        ),
      }
      : null,
    marginNotes: preview.margin_notes.map((note) => ({
      id: note.id,
      text: note.text,
      speakerId: note.speaker_id,
      utteranceId: note.utterance_id,
      appearAtMs: note.appear_at_ms,
      sourceStart: note.source_start,
      sourceEnd: note.source_end,
      settleDurationMs: note.settle_duration_ms,
    })),
    currentTimeMs: preview.current_time_ms,
  };
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
          speakerId: utterance.speakerId,
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

function getUtteranceDropEndMs(utterance: UtteranceSummary): number {
  const hedgingMatches = extractAnalysisMatches(utterance.analysisPayload?.hedging);
  const hedgeRanges = findMatchRanges(utterance.text, hedgingMatches);

  return utterance.wordRanges.reduce((latestDropEndMs, wordRange, index) => {
    if (!hedgeRanges.some((range) => rangesOverlap(range, wordRange))) {
      return latestDropEndMs;
    }

    const wordTiming = utterance.wordTimings[index];
    const wordStartMs = wordTiming?.startMs ?? utterance.startMs;
    const wordEndMs = wordTiming?.endMs ?? utterance.endMs;
    const wordDurationMs = Math.max(wordEndMs - wordStartMs, 1);
    const dropStartMs = wordStartMs + Math.min(wordDurationMs * 0.16, 90);
    const dropDurationMs = 1600 + Math.min(wordDurationMs * 2.5, 420);

    return Math.max(latestDropEndMs, dropStartMs + dropDurationMs);
  }, utterance.endMs);
}

function getConversationFinalSnapshotTimeMs(
  utterances: UtteranceSummary[],
  marginNotes: ConversationLandscapeMarginNote[],
  mediaDurationMs: number,
): number {
  const utteranceEndMs = utterances.map((utterance) => utterance.endMs);
  const utteranceDropEndMs = utterances.map((utterance) => getUtteranceDropEndMs(utterance));
  const marginNoteEndMs = marginNotes.map((note) => note.appearAtMs + note.settleDurationMs);

  return Math.max(mediaDurationMs, ...utteranceEndMs, ...utteranceDropEndMs, ...marginNoteEndMs, 0);
}

function buildSnapshotFileName(value: string): string {
  const normalized = value
    .trim()
    .replace(/[\\/:"*?<>|]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  return `${normalized || "conversation-landscape-final"}.png`;
}

function buildProcessedTranscriptOptions(jobs: JobSummary[]): ProcessedTranscriptOption[] {
  return jobs
    .map((job) => ({
      jobId: job.id,
      transcriptId: job.transcript_id ?? "",
      label: getProcessedTranscriptLabel(job),
      mediaSrc: getProcessedTranscriptMediaSrc(job),
      createdAt: job.created_at,
      reviewStatus: job.review_status,
      archivePreview: deserializeArchivePreview(job.archive_preview),
    }))
    .filter((option) => option.transcriptId)
    .sort((left, right) => {
      const leftTime = Date.parse(left.createdAt);
      const rightTime = Date.parse(right.createdAt);
      return (Number.isNaN(rightTime) ? 0 : rightTime) - (Number.isNaN(leftTime) ? 0 : leftTime);
    });
}

function mergeProcessedTranscriptOptions(
  currentOptions: ProcessedTranscriptOption[],
  incomingOptions: ProcessedTranscriptOption[],
): ProcessedTranscriptOption[] {
  const byTranscriptId = new Map<string, ProcessedTranscriptOption>();

  currentOptions.forEach((option) => {
    byTranscriptId.set(option.transcriptId, option);
  });

  incomingOptions.forEach((option) => {
    byTranscriptId.set(option.transcriptId, option);
  });

  return Array.from(byTranscriptId.values()).sort((left, right) => {
    const leftTime = Date.parse(left.createdAt);
    const rightTime = Date.parse(right.createdAt);
    return (Number.isNaN(rightTime) ? 0 : rightTime) - (Number.isNaN(leftTime) ? 0 : leftTime);
  });
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

function formatTranscriptCreatedAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function buildArchivePreviewData(document: PlaybackDocument): ArchivePreviewData | null {
  const utterances = buildUtterances(document);
  const speakers = buildSpeakerSummaries(utterances, document.speakerLabels ?? {});

  if (speakers.length === 0 || utterances.length === 0) {
    return null;
  }

  const marginNotes = buildMarginNotes(utterances);
  const finalSnapshotTimeMs = getConversationFinalSnapshotTimeMs(utterances, marginNotes, 0);
  const finalActiveUtterance = findActiveUtterance(utterances, finalSnapshotTimeMs);
  const finalActiveSentence =
    document.sentenceUnits.find((sentence) => sentence.id === finalActiveUtterance?.id) ?? null;
  const finalActiveUtterancePlaybackState = finalActiveUtterance
    ? getUtterancePlaybackState(finalActiveUtterance, finalSnapshotTimeMs)
    : null;

  return {
    speakers: speakers.map((speaker, index) => ({
      id: speaker.id,
      label: speaker.label,
      side: index % 2 === 0 ? ("left" as const) : ("right" as const),
      opacity: finalActiveUtterance?.speakerId === speaker.id ? 1 : 0.5,
      totalDurationMs: speaker.totalDurationMs,
      averagePoliteness: speaker.averagePoliteness,
    })),
    utterances: utterances.map((utterance, index) => {
      const playbackState = getUtterancePlaybackState(utterance, finalSnapshotTimeMs);

      return {
        id: utterance.id,
        speakerId: utterance.speakerId,
        politenessScore: utterance.politenessScore,
        contourSignalCount: playbackState.contourSignalCount,
        progress: playbackState.progress,
        order: index,
      };
    }),
    activeSpeakerId: finalActiveUtterance?.speakerId ?? null,
    activeTranscript: buildActiveTranscript(
      finalActiveUtterance,
      finalActiveSentence?.analysis_result?.analysis_payload,
      finalActiveUtterancePlaybackState?.visibleWordCount ?? 0,
      finalSnapshotTimeMs,
    ),
    marginNotes,
    currentTimeMs: finalSnapshotTimeMs,
  };
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

function getSpeechRecognitionConstructor(): SpeechRecognitionConstructorLike | null {
  if (typeof window === "undefined") {
    return null;
  }
  const recognitionWindow = window as Window & {
    SpeechRecognition?: SpeechRecognitionConstructorLike;
    webkitSpeechRecognition?: SpeechRecognitionConstructorLike;
  };
  return recognitionWindow.SpeechRecognition ?? recognitionWindow.webkitSpeechRecognition ?? null;
}

function isLiveCaptureSupported(): boolean {
  return typeof window !== "undefined" && typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;
}

function isLiveTranscriptionSupported(): boolean {
  return getSpeechRecognitionConstructor() !== null;
}

function getPreferredLiveMimeType(): string {
  if (typeof MediaRecorder === "undefined") {
    return "audio/webm";
  }
  if (typeof MediaRecorder.isTypeSupported !== "function") {
    return "audio/webm";
  }
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return candidates.find((value) => MediaRecorder.isTypeSupported(value)) ?? "audio/webm";
}

function buildLiveAnalysis(text: string): LiveAnalysisEventRequest {
  const normalized = text.toLowerCase();
  const hedgingMatches = LIVE_HEDGE_PHRASES.filter((phrase) => normalized.includes(phrase));
  const substanceMatches = LIVE_SUBSTANCE_PHRASES.filter((phrase) => normalized.includes(phrase));

  const politenessBase = hedgingMatches.length > 0 ? 0.68 : 0.52;
  const semanticConfidenceBase = substanceMatches.length > 0 ? 0.76 : 0.58;
  const mainMessageBase = text.trim().length > 32 ? 0.74 : 0.56;

  return {
    politeness_score: clamp(politenessBase + hedgingMatches.length * 0.05, 0, 1),
    semantic_confidence_score: clamp(semanticConfidenceBase + substanceMatches.length * 0.06, 0, 1),
    main_message_likelihood: clamp(mainMessageBase, 0, 1),
    analysis_model: "live-heuristic:mvp-client",
    analysis_payload: {
      hedging: { matches: hedgingMatches },
      substance: { matches: substanceMatches },
    },
  };
}

function upsertLiveTranscriptEntry(
  entries: LiveTranscriptEntry[],
  nextEntry: LiveTranscriptEntry,
): LiveTranscriptEntry[] {
  const existingIndex = entries.findIndex((entry) => entry.utteranceKey === nextEntry.utteranceKey);
  if (existingIndex === -1) {
    return [...entries, nextEntry].sort((left, right) => left.startMs - right.startMs || left.endMs - right.endMs);
  }

  const updatedEntries = [...entries];
  updatedEntries[existingIndex] = {
    ...updatedEntries[existingIndex],
    ...nextEntry,
  };
  return updatedEntries.sort((left, right) => left.startMs - right.startMs || left.endMs - right.endMs);
}

function formatLiveCaptureState(state: LiveCaptureState, session: LiveSessionResponse | null): string {
  if (state === "starting") {
    return "Starting microphone and live session";
  }
  if (state === "recording") {
    return session?.received_chunks ? `Recording live · ${session.received_chunks} chunks saved` : "Recording live";
  }
  if (state === "stopping") {
    return "Stopping capture and flushing transcript";
  }
  if (state === "finalizing") {
    return "Finalizing live session";
  }
  return "Ready";
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
  const [liveSession, setLiveSession] = useState<LiveSessionResponse | null>(null);
  const [liveCaptureState, setLiveCaptureState] = useState<LiveCaptureState>("idle");
  const [liveTranscriptEntries, setLiveTranscriptEntries] = useState<LiveTranscriptEntry[]>([]);
  const [liveNotice, setLiveNotice] = useState<string | null>(null);
  const [liveManualText, setLiveManualText] = useState("");
  const [isSavingLiveManualText, setIsSavingLiveManualText] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [speakerCount, setSpeakerCount] = useState<SpeakerCount>(DEFAULT_SPEAKER_COUNT);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [processedTranscriptOptions, setProcessedTranscriptOptions] = useState<ProcessedTranscriptOption[]>([]);
  const [archiveTotalCount, setArchiveTotalCount] = useState(0);
  const [isArchivePageLoading, setIsArchivePageLoading] = useState(false);
  const [archivePreviews, setArchivePreviews] = useState<Record<string, ArchivePreviewData | null>>({});
  const [selectedProcessedTranscriptId, setSelectedProcessedTranscriptId] = useState("");
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const [mediaDurationMs, setMediaDurationMs] = useState(0);
  const [isMediaPlaying, setIsMediaPlaying] = useState(false);
  const [isMediaMuted, setIsMediaMuted] = useState(false);
  const [isPlaybackComplete, setIsPlaybackComplete] = useState(false);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
  const mediaObjectUrlRef = useRef<string | null>(null);
  const pollingSessionIdRef = useRef<string | null>(null);
  const mediaElementRef = useRef<HTMLVideoElement | null>(null);
  const playbackFrameRef = useRef<number | null>(null);
  const liveMediaStreamRef = useRef<MediaStream | null>(null);
  const liveMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const liveRecognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const liveChunkIndexRef = useRef(0);
  const liveStartedAtRef = useRef<number | null>(null);
  const liveTranscriptDraftsRef = useRef<Record<string, { startMs: number; text: string; isFinal: boolean }>>({});
  const liveStopRequestedRef = useRef(false);
  const liveCaptureStateRef = useRef<LiveCaptureState>("idle");
  const liveLocalChunksRef = useRef<Blob[]>([]);
  const liveUploadQueueRef = useRef<Promise<void>>(Promise.resolve());
  const liveRecognitionRestartEnabledRef = useRef(false);
  const landscapeRef = useRef<ConversationLandscapeHandle | null>(null);
  const exportLandscapeRef = useRef<ConversationLandscapeHandle | null>(null);

  const refreshProcessedTranscriptOptions = async ({
    reset = false,
  }: {
    reset?: boolean;
  } = {}): Promise<ProcessedTranscriptOption[]> => {
    const offset = reset ? 0 : processedTranscriptOptions.length;
    const existingOptions = reset ? [] : processedTranscriptOptions;

    setIsArchivePageLoading(true);
    try {
      const jobPage = await fetchCompletedJobs(ARCHIVE_PAGE_SIZE, offset);
      const incomingOptions = buildProcessedTranscriptOptions(jobPage.jobs);
      const nextOptions = reset
        ? incomingOptions
        : mergeProcessedTranscriptOptions(existingOptions, incomingOptions);

      setArchiveTotalCount(jobPage.total);
      setProcessedTranscriptOptions(nextOptions);
      setArchivePreviews((currentPreviews) => {
        const validTranscriptIds = new Set(nextOptions.map((option) => option.transcriptId));
        const retainedPreviews = Object.fromEntries(
          Object.entries(currentPreviews).filter(([transcriptId]) => validTranscriptIds.has(transcriptId)),
        );
        const previewsFromJobs = Object.fromEntries(
          nextOptions
            .filter((option) => option.archivePreview !== null)
            .map((option) => [option.transcriptId, option.archivePreview] as const),
        );
        return {
          ...retainedPreviews,
          ...previewsFromJobs,
        };
      });
      return nextOptions;
    } finally {
      setIsArchivePageLoading(false);
    }
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
    setArchivePreviews((currentPreviews) => ({
      ...currentPreviews,
      [transcriptId]: buildArchivePreviewData(payload),
    }));
    setMediaSrc(mediaSource);
    setMediaName(mediaLabel);
    setSelectedProcessedTranscriptId(selectedTranscriptId);
    setPlaybackStarted(false);
    setCurrentTimeMs(0);
    setIsPlaybackComplete(false);
    setError(null);
    setPendingFile(null);
    setUploadProgress(null);
    setStreamSession(null);
    setTranscriptLoadStatus(payload.sentenceUnits.length > 0 ? "ready" : "empty");
    setMode(preferredMode ?? (payload.reviewStatus === "completed" ? "playback" : "review"));
  };

  useEffect(() => {
    let active = true;

    refreshProcessedTranscriptOptions({ reset: true })
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
        setArchivePreviews((currentPreviews) => ({
          ...currentPreviews,
          [PRELOAD_TRANSCRIPT_ID]: buildArchivePreviewData(payload),
        }));
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
    liveCaptureStateRef.current = liveCaptureState;
  }, [liveCaptureState]);

  useEffect(() => {
    return () => {
      liveStopRequestedRef.current = true;
      liveRecognitionRestartEnabledRef.current = false;
      liveRecognitionRef.current?.stop();
      liveRecognitionRef.current = null;
      liveMediaRecorderRef.current?.stop();
      liveMediaRecorderRef.current = null;
      liveMediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      liveMediaStreamRef.current = null;
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
    if (mode !== "select") {
      return;
    }

    const missingOptions = processedTranscriptOptions
      .filter((option) => archivePreviews[option.transcriptId] === undefined)
      .slice(0, ARCHIVE_PREVIEW_BATCH_SIZE);

    if (missingOptions.length === 0) {
      return;
    }

    let active = true;

    void Promise.all(
      missingOptions.map(async (option) => {
        try {
          const playbackDocument = await loadPlaybackDocument(option.transcriptId);
          return [option.transcriptId, buildArchivePreviewData(playbackDocument)] as const;
        } catch {
          return [option.transcriptId, null] as const;
        }
      }),
    ).then((entries) => {
      if (!active) {
        return;
      }

      setArchivePreviews((currentPreviews) => {
        const nextPreviews = { ...currentPreviews };
        for (const [transcriptId, preview] of entries) {
          if (nextPreviews[transcriptId] === undefined) {
            nextPreviews[transcriptId] = preview;
          }
        }
        return nextPreviews;
      });
    });

    return () => {
      active = false;
    };
  }, [archivePreviews, mode, processedTranscriptOptions]);

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
            void refreshProcessedTranscriptOptions({ reset: true });
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

  useEffect(() => {
    if (!isMediaPlaying) {
      if (playbackFrameRef.current !== null) {
        window.cancelAnimationFrame(playbackFrameRef.current);
        playbackFrameRef.current = null;
      }
      return;
    }

    const tick = () => {
      const mediaElement = mediaElementRef.current;
      if (!mediaElement) {
        playbackFrameRef.current = null;
        return;
      }

      setCurrentTimeMs(mediaElement.currentTime * 1000);

      if (mediaElement.paused || mediaElement.ended) {
        playbackFrameRef.current = null;
        return;
      }

      playbackFrameRef.current = window.requestAnimationFrame(tick);
    };

    playbackFrameRef.current = window.requestAnimationFrame(tick);

    return () => {
      if (playbackFrameRef.current !== null) {
        window.cancelAnimationFrame(playbackFrameRef.current);
        playbackFrameRef.current = null;
      }
    };
  }, [isMediaPlaying]);

  const shouldBuildConversationState = !!document && (mode === "review" || mode === "playback");
  const utterances = shouldBuildConversationState ? buildUtterances(document) : [];
  const speakerIds = shouldBuildConversationState ? detectSpeakerIds(document) : [];
  const speakerLabels = shouldBuildConversationState ? reviewDraft?.speakerLabels ?? document?.speakerLabels ?? {} : {};
  const speakers = shouldBuildConversationState ? buildSpeakerSummaries(utterances, speakerLabels) : [];
  const isReviewDirty = mode === "review" ? isReviewDraftDirty(document, reviewDraft) : false;
  const isTranscriptProcessing =
    streamSession !== null && ["open", "queued", "processing"].includes(streamSession.status);
  const hasMoreArchives = processedTranscriptOptions.length < archiveTotalCount;
  const marginNotes = shouldBuildConversationState ? buildMarginNotes(utterances) : [];
  const finalSnapshotTimeMs = getConversationFinalSnapshotTimeMs(utterances, marginNotes, mediaDurationMs);
  const landscapeTimeMs = isPlaybackComplete ? finalSnapshotTimeMs : currentTimeMs;
  const activeUtterance = playbackStarted || landscapeTimeMs > 0 ? findActiveUtterance(utterances, landscapeTimeMs) : null;
  const activeSentence =
    shouldBuildConversationState ? document.sentenceUnits.find((sentence) => sentence.id === activeUtterance?.id) ?? null : null;
  const finalActiveUtterance =
    playbackStarted || finalSnapshotTimeMs > 0 ? findActiveUtterance(utterances, finalSnapshotTimeMs) : null;
  const finalActiveSentence =
    shouldBuildConversationState
      ? document.sentenceUnits.find((sentence) => sentence.id === finalActiveUtterance?.id) ?? null
      : null;
  const landscapeSpeakers = speakers.map((speaker, index) => ({
    id: speaker.id,
    label: speaker.label,
    side: index % 2 === 0 ? ("left" as const) : ("right" as const),
    opacity: activeUtterance?.speakerId === speaker.id ? 1 : 0.5,
    totalDurationMs: speaker.totalDurationMs,
    averagePoliteness: speaker.averagePoliteness,
  }));
  const finalLandscapeSpeakers = speakers.map((speaker, index) => ({
    id: speaker.id,
    label: speaker.label,
    side: index % 2 === 0 ? ("left" as const) : ("right" as const),
    opacity: finalActiveUtterance?.speakerId === speaker.id ? 1 : 0.5,
    totalDurationMs: speaker.totalDurationMs,
    averagePoliteness: speaker.averagePoliteness,
  }));
  const activeUtterancePlaybackState = activeUtterance ? getUtterancePlaybackState(activeUtterance, landscapeTimeMs) : null;
  const finalActiveUtterancePlaybackState = finalActiveUtterance
    ? getUtterancePlaybackState(finalActiveUtterance, finalSnapshotTimeMs)
    : null;
  const landscapeUtterances = utterances.map((utterance, index) => {
    const playbackState = getUtterancePlaybackState(utterance, landscapeTimeMs);

    return {
      id: utterance.id,
      speakerId: utterance.speakerId,
      politenessScore: utterance.politenessScore,
      contourSignalCount: playbackState.contourSignalCount,
      progress: playbackState.progress,
      order: index,
    };
  });
  const finalLandscapeUtterances = utterances.map((utterance, index) => {
    const playbackState = getUtterancePlaybackState(utterance, finalSnapshotTimeMs);

    return {
      id: utterance.id,
      speakerId: utterance.speakerId,
      politenessScore: utterance.politenessScore,
      contourSignalCount: playbackState.contourSignalCount,
      progress: playbackState.progress,
      order: index,
    };
  });
  const activeTranscript = buildActiveTranscript(
    activeUtterance,
    activeSentence?.analysis_result?.analysis_payload,
    activeUtterancePlaybackState?.visibleWordCount ?? 0,
    landscapeTimeMs,
  );
  const finalActiveTranscript = buildActiveTranscript(
    finalActiveUtterance,
    finalActiveSentence?.analysis_result?.analysis_payload,
    finalActiveUtterancePlaybackState?.visibleWordCount ?? 0,
    finalSnapshotTimeMs,
  );
  const liveCaptureSupported = isLiveCaptureSupported();
  const liveTranscriptionSupported = isLiveTranscriptionSupported();
  const latestLiveEntry = liveTranscriptEntries[liveTranscriptEntries.length - 1] ?? null;
  const finalizedLiveEntryCount = liveTranscriptEntries.filter((entry) => entry.isFinal).length;

  const appendLiveEntry = (entry: LiveTranscriptEntry) => {
    setLiveTranscriptEntries((currentEntries) => upsertLiveTranscriptEntry(currentEntries, entry));
  };

  const persistLiveEvent = async (
    sessionId: string,
    payload: AppendLiveTranscriptEventRequest,
    nextEntry?: LiveTranscriptEntry,
  ) => {
    const event = await appendLiveSessionEvent(sessionId, payload);
    if (nextEntry) {
      appendLiveEntry(nextEntry);
    }
    return event;
  };

  const appendLiveTranscriptLine = async ({
    sessionId,
    utteranceKey,
    text,
    startMs,
    endMs,
  }: {
    sessionId: string;
    utteranceKey: string;
    text: string;
    startMs: number;
    endMs: number;
  }) => {
    const nextEntry: LiveTranscriptEntry = {
      utteranceKey,
      text,
      startMs,
      endMs,
      isFinal: true,
      analysis: buildLiveAnalysis(text),
    };

    await persistLiveEvent(
      sessionId,
      {
        event_type: "transcript.final",
        utterance_key: utteranceKey,
        start_ms: startMs,
        end_ms: endMs,
        text,
        speaker_id: "SPEAKER_00",
        is_final: true,
        payload: {},
      },
      nextEntry,
    );

    if (nextEntry.analysis) {
      await persistLiveEvent(sessionId, {
        event_type: "analysis.delta",
        utterance_key: utteranceKey,
        start_ms: startMs,
        end_ms: endMs,
        text: null,
        speaker_id: "SPEAKER_00",
        is_final: true,
        payload: {},
        analysis: nextEntry.analysis,
      });
    }
  };

  const handleSpeechRecognitionResult = async (
    sessionId: string,
    event: SpeechRecognitionEventLike,
  ): Promise<void> => {
    const elapsedMs = Math.max(0, (liveStartedAtRef.current ? Date.now() - liveStartedAtRef.current : 0));

    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      const transcriptText = result?.[0]?.transcript?.trim();
      if (!transcriptText) {
        continue;
      }

      const utteranceKey = `utt-${index}`;
      const existingDraft = liveTranscriptDraftsRef.current[utteranceKey];
      const startMs =
        existingDraft?.startMs ??
        Math.max(0, elapsedMs - Math.max(transcriptText.split(/\s+/).length * 420, 900));
      const nextEntry: LiveTranscriptEntry = {
        utteranceKey,
        text: transcriptText,
        startMs,
        endMs: Math.max(startMs + 1, elapsedMs),
        isFinal: result.isFinal,
        analysis: result.isFinal ? buildLiveAnalysis(transcriptText) : null,
      };

      if (
        existingDraft &&
        existingDraft.text === transcriptText &&
        existingDraft.isFinal === result.isFinal
      ) {
        continue;
      }

      liveTranscriptDraftsRef.current[utteranceKey] = {
        startMs,
        text: transcriptText,
        isFinal: result.isFinal,
      };

      const transcriptPayload: AppendLiveTranscriptEventRequest = {
        event_type: result.isFinal ? "transcript.final" : "transcript.delta",
        utterance_key: utteranceKey,
        start_ms: nextEntry.startMs,
        end_ms: nextEntry.endMs,
        text: transcriptText,
        speaker_id: "SPEAKER_00",
        is_final: result.isFinal,
        payload: {},
      };

      await persistLiveEvent(sessionId, transcriptPayload, nextEntry);

      if (result.isFinal) {
        setLiveNotice(null);
      }

      if (result.isFinal && nextEntry.analysis) {
        await persistLiveEvent(sessionId, {
          event_type: "analysis.delta",
          utterance_key: utteranceKey,
          start_ms: nextEntry.startMs,
          end_ms: nextEntry.endMs,
          text: null,
          speaker_id: "SPEAKER_00",
          is_final: true,
          payload: {},
          analysis: nextEntry.analysis,
        });
      }
    }
  };

  const handleStartLiveSession = async () => {
    if (!liveCaptureSupported) {
      setError("Live recording is not available in this browser.");
      return;
    }

    setError(null);
    setLiveNotice(
      liveTranscriptionSupported
        ? null
        : "Browser speech recognition is unavailable here. Recording will continue, and you can add manual transcript lines during capture.",
    );
    setMode("live");
    setLiveCaptureState("starting");
    setLiveTranscriptEntries([]);
    setLiveManualText("");
    setDocument(null);
    setReviewDraft(null);
    setSelectedProcessedTranscriptId("");
    setTranscriptLoadStatus("idle");
    setMediaSrc("");
    setMediaName("Live session");
    setPlaybackStarted(false);
    setCurrentTimeMs(0);
    setMediaDurationMs(0);
    setIsMediaPlaying(false);
    setIsPlaybackComplete(false);
    setStreamSession(null);
    setUploadProgress(null);
    liveChunkIndexRef.current = 0;
    liveTranscriptDraftsRef.current = {};
    liveStopRequestedRef.current = false;
    liveLocalChunksRef.current = [];
    liveUploadQueueRef.current = Promise.resolve();

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      liveMediaStreamRef.current = mediaStream;

      const mimeType = getPreferredLiveMimeType();
      const createdLiveSession = await createLiveSession({
        mime_type: mimeType,
        original_filename: `live-session-${Date.now()}.webm`,
        sample_rate_hz: null,
        channel_count: mediaStream.getAudioTracks().length || 1,
        session_metadata: {
          frontend_live: true,
          speech_recognition: true,
        },
      });
      setLiveSession(createdLiveSession);
      liveStartedAtRef.current = Date.now();

      const mediaRecorder = new MediaRecorder(mediaStream, { mimeType });
      liveMediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (recordedEvent: BlobEvent) => {
        if (!recordedEvent.data || recordedEvent.data.size === 0) {
          return;
        }

        liveLocalChunksRef.current.push(recordedEvent.data);
        const nextChunkIndex = liveChunkIndexRef.current;
        liveChunkIndexRef.current += 1;

        liveUploadQueueRef.current = liveUploadQueueRef.current
          .then(async () => {
            await uploadChunkToLiveSession(createdLiveSession.id, recordedEvent.data, nextChunkIndex);
            const refreshedSession = await fetchLiveSession(createdLiveSession.id);
            setLiveSession(refreshedSession);
          })
          .catch((caughtError: unknown) => {
            setError(caughtError instanceof Error ? caughtError.message : "Unknown live upload error");
          });
      };
      mediaRecorder.start(LIVE_CAPTURE_TIMESLICE_MS);

      const SpeechRecognitionCtor = getSpeechRecognitionConstructor();
      if (SpeechRecognitionCtor) {
        const recognition = new SpeechRecognitionCtor();
        liveRecognitionRef.current = recognition;
        liveRecognitionRestartEnabledRef.current = true;
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";
        recognition.onerror = (recognitionEvent) => {
          const errorCode = recognitionEvent.error?.trim() || "unknown";
          liveRecognitionRestartEnabledRef.current = false;
          setLiveNotice(
            errorCode === "network"
              ? "Browser speech recognition hit a network error. Recording is still running, but live transcript updates are paused. You can keep recording and add manual transcript lines below."
              : `Live transcription is paused (${errorCode}). Recording is still running, and you can add manual transcript lines below.`,
          );
          recognition.stop();
        };
        recognition.onend = () => {
          if (liveStopRequestedRef.current || !liveRecognitionRestartEnabledRef.current) {
            return;
          }
          if (liveCaptureStateRef.current === "recording") {
            try {
              recognition.start();
            } catch {
              // Browsers can throw while restarting after rapid end/start cycles.
            }
          }
        };
        recognition.onresult = (recognitionEvent) => {
          void handleSpeechRecognitionResult(createdLiveSession.id, recognitionEvent).catch((caughtError: unknown) => {
            setLiveNotice(
              caughtError instanceof Error ? caughtError.message : "Live transcription is paused, but recording continues.",
            );
          });
        };
        recognition.start();
      }

      setLiveCaptureState("recording");
    } catch (caughtError: unknown) {
      liveRecognitionRef.current?.stop();
      liveRecognitionRef.current = null;
      liveRecognitionRestartEnabledRef.current = false;
      liveMediaRecorderRef.current?.stop();
      liveMediaRecorderRef.current = null;
      liveMediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      liveMediaStreamRef.current = null;
      setLiveCaptureState("idle");
      setLiveSession(null);
      setMode("create");
      setError(caughtError instanceof Error ? caughtError.message : "Unknown live session error");
    }
  };

  const handleStopLiveSession = async () => {
    if (!liveSession) {
      setMode("create");
      return;
    }

    setLiveCaptureState("stopping");
    liveStopRequestedRef.current = true;
    liveRecognitionRestartEnabledRef.current = false;
    liveRecognitionRef.current?.stop();
    liveRecognitionRef.current = null;

    const recorder = liveMediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      await new Promise<void>((resolve) => {
        recorder.onstop = () => resolve();
        recorder.stop();
      });
    }
    liveMediaRecorderRef.current = null;
    liveMediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    liveMediaStreamRef.current = null;
    await liveUploadQueueRef.current;

    setLiveCaptureState("finalizing");

    try {
      const stoppedSession = await stopLiveSession(liveSession.id);
      setLiveSession(stoppedSession);
      const finalizedSession = await finalizeLiveSession(liveSession.id);
      setLiveSession(finalizedSession);

      if (mediaObjectUrlRef.current) {
        URL.revokeObjectURL(mediaObjectUrlRef.current);
      }
      const liveBlob = new Blob(liveLocalChunksRef.current, { type: finalizedSession.mime_type || "audio/webm" });
      const objectUrl = URL.createObjectURL(liveBlob);
      mediaObjectUrlRef.current = objectUrl;

      if (finalizedSession.transcript_id) {
        await loadTranscriptIntoApp({
          transcriptId: finalizedSession.transcript_id,
          mediaSource: objectUrl,
          mediaLabel: finalizedSession.original_filename || "Live session",
          selectedTranscriptId: "",
          preferredMode: "review",
        });
      } else {
        setMode("create");
      }

      setLiveCaptureState("idle");
      setLiveSession(null);
      setLiveTranscriptEntries([]);
      await refreshProcessedTranscriptOptions({ reset: true });
    } catch (caughtError: unknown) {
      setLiveCaptureState("idle");
      setError(caughtError instanceof Error ? caughtError.message : "Unknown live finalize error");
    }
  };

  const handleAddManualLiveTranscript = async () => {
    if (!liveSession) {
      return;
    }

    const normalizedText = liveManualText.trim();
    if (!normalizedText) {
      return;
    }

    setIsSavingLiveManualText(true);
    try {
      const lastEntryEndMs = liveTranscriptEntries[liveTranscriptEntries.length - 1]?.endMs ?? 0;
      const nowMs = Math.max(
        lastEntryEndMs + 1,
        liveStartedAtRef.current ? Date.now() - liveStartedAtRef.current : lastEntryEndMs + 1,
      );
      const startMs = Math.max(0, nowMs - Math.max(normalizedText.split(/\s+/).length * 450, 1200));
      const utteranceKey = `manual-${Date.now()}`;

      await appendLiveTranscriptLine({
        sessionId: liveSession.id,
        utteranceKey,
        text: normalizedText,
        startMs,
        endMs: nowMs,
      });

      setLiveManualText("");
      setLiveNotice(null);
    } catch (caughtError: unknown) {
      setLiveNotice(
        caughtError instanceof Error
          ? caughtError.message
          : "Could not add the manual transcript line, but recording is still running.",
      );
    } finally {
      setIsSavingLiveManualText(false);
    }
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setPendingFile(file);
    setSelectedProcessedTranscriptId("");
    setError(null);
    setMode("create");
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
    setIsPlaybackComplete(false);
    setError(null);
    setStreamSession(null);
    setUploadProgress(0);
    setIsUploading(true);
    setTranscriptLoadStatus("idle");
    pollingSessionIdRef.current = null;
    setMode("create");

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

  const handleLoadMoreArchives = async () => {
    if (isArchivePageLoading || processedTranscriptOptions.length >= archiveTotalCount) {
      return;
    }

    try {
      await refreshProcessedTranscriptOptions();
    } catch (caughtError: unknown) {
      setError(caughtError instanceof Error ? caughtError.message : "Unknown archive loading error");
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
      setArchivePreviews((currentPreviews) => ({
        ...currentPreviews,
        [nextDocument.transcriptId]: buildArchivePreviewData(nextDocument),
      }));
      setSelectedProcessedTranscriptId(nextDocument.transcriptId);
      setMode(nextStatus === "completed" ? "playback" : "review");
      setSaveState("saved");
      void refreshProcessedTranscriptOptions({ reset: true });
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
      setIsPlaybackComplete(false);
      setPendingFile(null);
      setSelectedProcessedTranscriptId("");
      setTranscriptLoadStatus("idle");
      setStreamSession(null);
      setUploadProgress(null);
      setMode("select");
      setSaveState("idle");
      await refreshProcessedTranscriptOptions({ reset: true });
    } catch (caughtError: unknown) {
      setSaveState("idle");
      setError(caughtError instanceof Error ? caughtError.message : "Unknown delete error");
    }
  };

  const syncMediaCurrentTime = (element: HTMLVideoElement) => {
    setCurrentTimeMs(element.currentTime * 1000);
  };

  const syncMediaState = (element: HTMLVideoElement) => {
    syncMediaCurrentTime(element);
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
    setIsPlaybackComplete(false);
  };

  const handleMediaPause = () => {
    if (mediaElementRef.current) {
      syncMediaCurrentTime(mediaElementRef.current);
      if (mediaElementRef.current.ended) {
        setIsPlaybackComplete(true);
      }
    }
    setIsMediaPlaying(false);
  };

  const handleMediaEnded = () => {
    if (mediaElementRef.current) {
      syncMediaState(mediaElementRef.current);
    }
    setIsMediaPlaying(false);
    setIsPlaybackComplete(true);
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
    syncMediaCurrentTime(mediaElementRef.current);
    setIsPlaybackComplete(mediaDurationMs > 0 && nextTimeMs >= mediaDurationMs - 100);
  };

  const toggleMediaMute = () => {
    if (!mediaElementRef.current) {
      return;
    }
    mediaElementRef.current.muted = !mediaElementRef.current.muted;
    setIsMediaMuted(mediaElementRef.current.muted);
  };

  const handleDownloadSnapshot = async () => {
    const didDownload = await exportLandscapeRef.current?.downloadSnapshotImage(
      buildSnapshotFileName(document?.conversationTitle || mediaName),
    );

    if (!didDownload) {
      setError("Failed to generate the final PNG snapshot.");
    }
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
          onEnded={handleMediaEnded}
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
        <section className="archive-shell">
          <header className="archive-header">
            <div className="archive-header__copy">
              <h1>OF TERRAINS WE SPEAK</h1>
            </div>
            <div className="archive-header__actions">
              <button type="button" className="submit-upload" onClick={() => setMode("create")}>
                Create New
              </button>
              <button type="button" className="about-link-button archive-header__about" onClick={() => setMode("about")}>
                ABOUT
              </button>
            </div>
          </header>

          {processedTranscriptOptions.length === 0 && isArchivePageLoading ? (
            <article className="surface-card archive-empty-state">
              <h2>Loading archived conversations...</h2>
              <p className="surface-note">Pulling in the first page of saved transcripts.</p>
            </article>
          ) : processedTranscriptOptions.length === 0 ? (
            <article className="surface-card archive-empty-state">
              <h2>No archived conversations yet</h2>
              <p className="surface-note">
                Start a live session or upload a file to begin building the gallery.
              </p>
              <button type="button" className="submit-upload" onClick={() => setMode("create")}>
                Create the first conversation
              </button>
            </article>
          ) : (
            <>
              <section className="archive-gallery" aria-label="Archived conversations">
                {processedTranscriptOptions.map((option) => {
                  const preview = archivePreviews[option.transcriptId];

                  return (
                    <ArchivePreviewCard
                      key={option.transcriptId}
                      isActive={selectedProcessedTranscriptId === option.transcriptId}
                      option={option}
                      preview={preview}
                      onSelect={(nextOption) => void handleProcessedTranscriptSelect(nextOption)}
                    />
                  );
                })}
              </section>
              {hasMoreArchives ? (
                <div className="archive-gallery__actions">
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => void handleLoadMoreArchives()}
                    disabled={isArchivePageLoading}
                  >
                    {isArchivePageLoading ? "Loading..." : "Load More"}
                  </button>
                </div>
              ) : null}
            </>
          )}
        </section>
      ) : null}

      {mode === "create" ? (
        <section className="create-shell">
          <header className="create-shell__hero">
            <div className="create-shell__hero-row">
              <div className="create-shell__copy">
                <h1>OF TERRAINS WE SPEAK</h1>
              </div>
              <div className="create-shell__actions">
                <button type="button" className="submit-upload" onClick={() => setMode("select")}>
                  Back to Archives
                </button>
                <button type="button" className="about-link-button create-shell__about" onClick={() => setMode("about")}>
                  ABOUT
                </button>
              </div>
            </div>
            <div className="create-shell__intro">
              <p className="surface-note">
                Choose how you want to begin the next conversation.
              </p>
            </div>
          </header>

          <section className="onboarding-stack">
            <article className="surface-card onboarding-choice onboarding-choice--live">
              <div className="onboarding-choice__header">
                <h2>Start a live session</h2>
              </div>
              <div className="live-launch">
                <p className="surface-note">
                  Record from your microphone, get rolling transcript updates in the browser, and drop straight into the review flow when you stop.
                </p>
                <div className="live-launch__actions">
                  <button
                    type="button"
                    className="submit-upload"
                    onClick={() => void handleStartLiveSession()}
                    disabled={!liveCaptureSupported || liveCaptureState !== "idle"}
                  >
                    Start live capture
                  </button>
                  <p className="live-launch__support">
                    {liveCaptureSupported && liveTranscriptionSupported
                      ? "Mic capture and browser speech recognition are available."
                      : liveCaptureSupported
                        ? "Mic capture is available. If browser speech recognition is blocked, you can still record and add manual transcript lines."
                        : "This browser does not expose microphone capture for the live MVP."}
                  </p>
                </div>
              </div>
            </article>

            <article className="surface-card onboarding-choice onboarding-choice--upload">
              <div className="onboarding-choice__header">
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
                  <span className="upload-dropzone__step">Step 1 upload audio file</span>
                  <strong>click to add file</strong>
                  <span className="upload-dropzone__hint">Choose a .mov or .mp4 file to visualize</span>
                </label>
              ) : (
                <div className="upload-workflow">
                  <div className="upload-workflow__file">
                    <p className="upload-workflow__step">Step 1 upload audio file</p>
                    <h3>{pendingFile.name}</h3>
                    <label className="ghost-button upload-workflow__replace">
                      choose another file
                      <input type="file" accept=".mov,.mp4,video/mp4,video/quicktime" onChange={handleFileChange} />
                    </label>
                  </div>

                  <div className="upload-workflow__controls">
                    <span className="speaker-step__label">Step 2 Select number of speakers</span>
                    <div className="upload-workflow__actions">
                      <div className="speaker-step">
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
                </div>
              )}
            </article>
          </section>
        </section>
      ) : null}

      {mode === "about" ? (
        <section className="about-shell">
          <header className="about-shell__header">
            <h1>OF TERRAINS WE SPEAK</h1>
            <div className="about-shell__actions">
              <button type="button" className="submit-upload" onClick={() => setMode("select")}>
                Back to Main
              </button>
              <button type="button" className="submit-upload" onClick={() => setMode("create")}>
                Create New
              </button>
            </div>
          </header>

          <section className="about-shell__content">
            <p>
              Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et
              dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip
              ex ea commodo consequat.
            </p>
            <p>
              Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
              Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est
              laborum.
            </p>
          </section>
        </section>
      ) : null}

      {mode === "live" ? (
        <section className="live-shell">
          <article className="surface-card live-panel live-panel--hero">
            <div className="live-panel__header">
              <div>
                <p className="eyebrow">Live Capture</p>
                <h2>{liveSession?.original_filename || "Microphone session"}</h2>
              </div>
              <div className="live-status-cluster">
                <span className={`live-status-pill live-status-pill--${liveCaptureState}`}>
                  {formatLiveCaptureState(liveCaptureState, liveSession)}
                </span>
                <span className="live-status-pill">
                  {formatTimestamp(latestLiveEntry?.endMs ?? 0)}
                </span>
              </div>
            </div>

            <div className="live-panel__actions">
              <button
                type="button"
                className="ghost-button"
                onClick={() => setMode("create")}
                disabled={liveCaptureState !== "idle"}
              >
                Back to Create
              </button>
              <button
                type="button"
                className="submit-upload"
                onClick={() => void handleStopLiveSession()}
                disabled={!liveSession || !["recording", "starting"].includes(liveCaptureState)}
              >
                Stop and Review
              </button>
            </div>

            {liveNotice ? (
              <div className="live-notice" role="status" aria-live="polite">
                <p>{liveNotice}</p>
              </div>
            ) : null}
          </article>

          <div className="live-grid">
            <article className="surface-card live-panel">
              <div className="section-heading">
                <div>
                  <h2>Rolling transcript</h2>
                  <p className="surface-note">Interim lines stay soft until the browser marks them final.</p>
                </div>
              </div>

              {liveTranscriptEntries.length === 0 ? (
                <div className="empty-state">
                  <p>Listening for speech…</p>
                  <p>Start talking and the transcript will begin to appear here.</p>
                </div>
              ) : (
                <div className="live-transcript-list">
                  {liveTranscriptEntries.map((entry) => (
                    <article
                      key={entry.utteranceKey}
                      className={`live-transcript-row${entry.isFinal ? " live-transcript-row--final" : " live-transcript-row--interim"}`}
                    >
                      <div className="live-transcript-row__meta">
                        <span>{formatTimestamp(entry.startMs)}</span>
                        <span>{entry.isFinal ? "final" : "listening"}</span>
                      </div>
                      <p>{entry.text}</p>
                    </article>
                  ))}
                </div>
              )}
            </article>

            <article className="surface-card live-panel">
              <div className="section-heading">
                <div>
                  <h2>Live analysis</h2>
                  <p className="surface-note">Client-side heuristics mirror the backend analysis shape for now.</p>
                </div>
              </div>

              <div className="live-metrics">
                <div className="live-metric-card">
                  <span>Final utterances</span>
                  <strong>{finalizedLiveEntryCount}</strong>
                </div>
                <div className="live-metric-card">
                  <span>Saved chunks</span>
                  <strong>{liveSession?.received_chunks ?? 0}</strong>
                </div>
              </div>

              {latestLiveEntry?.analysis ? (
                <div className="live-analysis-card">
                  <div className="live-analysis-card__scores">
                    <span>Politeness {Math.round(latestLiveEntry.analysis.politeness_score * 100)}%</span>
                    <span>Confidence {Math.round(latestLiveEntry.analysis.semantic_confidence_score * 100)}%</span>
                    <span>Main message {Math.round(latestLiveEntry.analysis.main_message_likelihood * 100)}%</span>
                  </div>
                  <pre>{stringifyAnalysisPayload(latestLiveEntry.analysis.analysis_payload)}</pre>
                </div>
              ) : (
                <div className="empty-state">
                  <p>No finalized analysis yet.</p>
                  <p>Once a line lands as final, its heuristic summary will appear here.</p>
                </div>
              )}

              <div className="live-manual-entry">
                <label className="transcript-row__field transcript-row__field--text">
                  <span>Add a manual transcript line</span>
                  <textarea
                    value={liveManualText}
                    onChange={(event) => setLiveManualText(event.target.value)}
                    rows={3}
                    placeholder="Type a transcript line here if browser speech recognition is unavailable."
                  />
                </label>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => void handleAddManualLiveTranscript()}
                  disabled={!liveSession || isSavingLiveManualText || !liveManualText.trim()}
                >
                  {isSavingLiveManualText ? "Saving line..." : "Add line"}
                </button>
              </div>
            </article>
          </div>
        </section>
      ) : null}

      {mode === "review" && document && reviewDraft ? (
        <>
          <section className="review-top-stack">
            <article className="surface-card review-media review-header-card">
              <section className="review-toolbar">
                <div className="review-toolbar__title">
                  <input
                    className="title-input"
                    value={reviewDraft.conversationTitle}
                    onChange={(event) => updateDraftTitle(event.target.value)}
                    placeholder="Name of the conversation"
                  />
                </div>
                <div className="review-toolbar__actions">
                  <button type="button" className="ghost-button" onClick={() => setMode("select")}>
                    Back to Archive
                  </button>
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => setIsDeleteConfirmOpen(true)}
                    disabled={saveState === "saving"}
                  >
                    Delete Transcript
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

              <div className="section-heading">
                <div>
                  <h2>{reviewDraft.conversationTitle.trim() || mediaName}</h2>
                </div>
              </div>
              {renderCompactMediaControls()}
            </article>
          </section>

          <section className="review-editor-shell">
            <article className="surface-card review-editor">
              <div className="section-heading review-editor__heading">
                <div>
                  <h2>Name the speakers and edit the transcript</h2>
                </div>
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
                <span>Transcript</span>
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
                  {saveState === "saving" ? "Saving..." : saveState === "saved" ? "Saved" : isReviewDirty ? "Click to save new changes" : ""}
                </span>
                <div className="review-editor__footer-actions">
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
              <div className="playback-header__title-row">
                <h2>{document.conversationTitle || mediaName}</h2>
                <div className="playback-header__actions">
                  <button type="button" className="ghost-button" onClick={() => setMode("select")}>
                    Back to Archive
                  </button>
                  <button type="button" className="ghost-button" onClick={() => setMode("review")}>
                    Edit Transcript
                  </button>
                  <button type="button" className="submit-upload" onClick={() => void handleDownloadSnapshot()}>
                    Download Final PNG
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
                ref={landscapeRef}
                speakers={landscapeSpeakers}
                utterances={landscapeUtterances}
                activeSpeakerId={activeUtterance?.speakerId ?? null}
                activeTranscript={activeTranscript}
                marginNotes={marginNotes}
                currentTimeMs={landscapeTimeMs}
                snapshotMode={isPlaybackComplete ? "final" : "live"}
              />
            )}
            {transcriptLoadStatus !== "empty" && landscapeSpeakers.length > 0 ? (
              <div className="conversation-landscape__export-proxy" aria-hidden="true">
                <ConversationLandscape
                  ref={exportLandscapeRef}
                  speakers={finalLandscapeSpeakers}
                  utterances={finalLandscapeUtterances}
                  activeSpeakerId={finalActiveUtterance?.speakerId ?? null}
                  activeTranscript={finalActiveTranscript}
                  marginNotes={marginNotes}
                  currentTimeMs={finalSnapshotTimeMs}
                  snapshotMode="final"
                />
              </div>
            ) : null}
          </section>
        </>
      ) : null}
    </main>
  );
}
