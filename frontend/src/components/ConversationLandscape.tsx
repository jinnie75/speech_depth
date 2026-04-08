import { useEffect, useLayoutEffect, useRef, useState } from "react";

interface ConversationLandscapeSpeaker {
  id: string;
  label: string;
  side: "left" | "right";
  opacity: number;
  totalDurationMs: number;
  averagePoliteness: number;
}

interface ConversationLandscapeUtterance {
  id: string;
  speakerId: string;
  politenessScore: number;
  contourSignalCount: number;
  progress: number;
  order: number;
}

export interface ConversationLandscapeTranscriptTextToken {
  id: string;
  kind: "text";
  text: string;
}

export interface ConversationLandscapeTranscriptWordToken {
  id: string;
  kind: "word";
  text: string;
  start: number;
  end: number;
  isHedge: boolean;
  isSubstance: boolean;
  dropStartMs: number;
  dropDurationMs: number;
}

export interface ConversationLandscapeMarginNote {
  id: string;
  text: string;
  utteranceId: string;
  appearAtMs: number;
  sourceStart: number;
  sourceEnd: number;
  settleDurationMs: number;
}

export type ConversationLandscapeTranscriptToken =
  | ConversationLandscapeTranscriptTextToken
  | ConversationLandscapeTranscriptWordToken;

export interface ConversationLandscapeTranscript {
  utteranceId: string;
  currentTimeMs: number;
  tokens: ConversationLandscapeTranscriptToken[];
}

interface ConversationLandscapeProps {
  speakers: ConversationLandscapeSpeaker[];
  utterances: ConversationLandscapeUtterance[];
  activeSpeakerId: string | null;
  activeTranscript: ConversationLandscapeTranscript | null;
  marginNotes: ConversationLandscapeMarginNote[];
  currentTimeMs: number;
}

interface Point {
  x: number;
  y: number;
}

interface SpeakerRegion {
  startX: number;
  endX: number;
  midX: number;
}

interface SpeakerCluster {
  tail: Point;
  utteranceCount: number;
}

interface SpeakerSection {
  speaker: ConversationLandscapeSpeaker;
  color: string;
  outerColor: string;
  levelCount: number;
  washHref: string | null;
  paths: { level: number; d: string }[];
}

interface WordPosition {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface LaunchPosition {
  left: number;
  top: number;
}

interface MarginNoteSlot {
  leftPercent: number;
  topPercent: number;
  widthCh: number;
}

const VIEWBOX_WIDTH = 1200;
const VIEWBOX_HEIGHT = 620;
const MAP_PADDING_X = 0;
const MAP_PADDING_Y = 0;
const MAP_WIDTH = VIEWBOX_WIDTH - MAP_PADDING_X * 2;
const MAP_HEIGHT = VIEWBOX_HEIGHT - MAP_PADDING_Y * 2;
const GRID_ROWS = 112;
const GRID_COLS = Math.max(156, Math.round((MAP_WIDTH / MAP_HEIGHT) * GRID_ROWS));
const SPEAKER_COLORS = ["#0a8f87", "#8a4c1b", "#d26f1b", "#73586d"];
const SPEAKER_OUTER_COLORS = ["#a7dbd6", "#d8b59b", "#f1c59f", "#c6b2c0"];
const MARGIN_NOTE_SLOTS: MarginNoteSlot[] = [
  { leftPercent: 8, topPercent: 61, widthCh: 18 },
  { leftPercent: 28, topPercent: 64, widthCh: 16 },
  { leftPercent: 49, topPercent: 60, widthCh: 20 },
  { leftPercent: 71, topPercent: 66, widthCh: 15 },
  { leftPercent: 15, topPercent: 73, widthCh: 17 },
  { leftPercent: 37, topPercent: 76, widthCh: 18 },
  { leftPercent: 59, topPercent: 72, widthCh: 16 },
  { leftPercent: 79, topPercent: 78, widthCh: 15 },
  { leftPercent: 7, topPercent: 84, widthCh: 16 },
  { leftPercent: 30, topPercent: 89, widthCh: 17 },
  { leftPercent: 55, topPercent: 86, widthCh: 18 },
  { leftPercent: 74, topPercent: 91, widthCh: 16 },
];

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function lerp(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
}

function hashString(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hashUnit(value: string): number {
  return hashString(value) / 4294967295;
}

function hashSigned(value: string): number {
  return hashUnit(value) * 2 - 1;
}

function hexToRgb(color: string): { r: number; g: number; b: number } {
  const normalized = color.replace("#", "");
  const expanded =
    normalized.length === 3
      ? normalized
        .split("")
        .map((character) => `${character}${character}`)
        .join("")
      : normalized;

  return {
    r: Number.parseInt(expanded.slice(0, 2), 16),
    g: Number.parseInt(expanded.slice(2, 4), 16),
    b: Number.parseInt(expanded.slice(4, 6), 16),
  };
}

function mixColors(startColor: string, endColor: string, amount: number): string {
  const start = hexToRgb(startColor);
  const end = hexToRgb(endColor);
  const mix = clamp(amount, 0, 1);

  return `rgb(${Math.round(lerp(start.r, end.r, mix))}, ${Math.round(lerp(start.g, end.g, mix))}, ${Math.round(
    lerp(start.b, end.b, mix),
  )})`;
}

function mixRgb(
  startColor: string,
  endColor: string,
  amount: number,
): { r: number; g: number; b: number } {
  const start = hexToRgb(startColor);
  const end = hexToRgb(endColor);
  const mix = clamp(amount, 0, 1);

  return {
    r: Math.round(lerp(start.r, end.r, mix)),
    g: Math.round(lerp(start.g, end.g, mix)),
    b: Math.round(lerp(start.b, end.b, mix)),
  };
}

function buildFieldWashDataUrl(
  field: number[][],
  maxFieldValue: number,
  innerColor: string,
  outerColor: string,
): string | null {
  if (maxFieldValue <= 0.04 || typeof document === "undefined") {
    return null;
  }

  const canvas = document.createElement("canvas");
  canvas.width = GRID_COLS;
  canvas.height = GRID_ROWS;
  const context = canvas.getContext("2d");

  if (!context) {
    return null;
  }

  const image = context.createImageData(GRID_COLS, GRID_ROWS);

  for (let yIndex = 0; yIndex < GRID_ROWS; yIndex += 1) {
    for (let xIndex = 0; xIndex < GRID_COLS; xIndex += 1) {
      const normalized = clamp(field[xIndex][yIndex] / maxFieldValue, 0, 1);
      const density = Math.pow(normalized, 0.86);
      const fade = clamp((normalized - 0.035) / 0.965, 0, 1);
      const alpha = Math.pow(fade, 1.35) * 0.72;
      const color = mixRgb(outerColor, innerColor, Math.pow(density, 0.72));
      const pixelIndex = (yIndex * GRID_COLS + xIndex) * 4;

      image.data[pixelIndex] = color.r;
      image.data[pixelIndex + 1] = color.g;
      image.data[pixelIndex + 2] = color.b;
      image.data[pixelIndex + 3] = Math.round(alpha * 255);
    }
  }

  context.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

function measureTranscriptWordPositions(
  canvas: HTMLDivElement | null,
  wordNodes: Map<string, HTMLSpanElement>,
): Record<string, WordPosition> {
  if (!canvas) {
    return {};
  }

  const canvasRect = canvas.getBoundingClientRect();
  const positions: Record<string, WordPosition> = {};

  wordNodes.forEach((node, id) => {
    const wordRect = node.getBoundingClientRect();
    positions[id] = {
      left: wordRect.left - canvasRect.left,
      top: wordRect.top - canvasRect.top,
      width: wordRect.width,
      height: wordRect.height,
    };
  });

  return positions;
}

function areWordPositionsEqual(
  left: Record<string, WordPosition>,
  right: Record<string, WordPosition>,
): boolean {
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);

  if (leftKeys.length !== rightKeys.length) {
    return false;
  }

  return leftKeys.every((key) => {
    const leftPosition = left[key];
    const rightPosition = right[key];

    return (
      rightPosition !== undefined &&
      leftPosition.left === rightPosition.left &&
      leftPosition.top === rightPosition.top &&
      leftPosition.width === rightPosition.width &&
      leftPosition.height === rightPosition.height
    );
  });
}

function areLaunchPositionsEqual(
  left: Record<string, LaunchPosition>,
  right: Record<string, LaunchPosition>,
): boolean {
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);

  if (leftKeys.length !== rightKeys.length) {
    return false;
  }

  return leftKeys.every((key) => {
    const leftPosition = left[key];
    const rightPosition = right[key];

    return rightPosition !== undefined && leftPosition.left === rightPosition.left && leftPosition.top === rightPosition.top;
  });
}

function buildVisibleMarginNotes(
  marginNotes: ConversationLandscapeMarginNote[],
  currentTimeMs: number,
): Array<{
  note: ConversationLandscapeMarginNote;
  leftPercent: number;
  topPercent: number;
  widthCh: number;
  revealProgress: number;
}> {
  const revealedNotes = [...marginNotes]
    .filter((note) => currentTimeMs >= note.appearAtMs)
    .sort((left, right) => left.appearAtMs - right.appearAtMs || left.id.localeCompare(right.id));
  const occupiedSlots = new Map<number, (typeof revealedNotes)[number]>();
  const slotQueue: number[] = [];

  revealedNotes.forEach((note) => {
    const preferredSlots = MARGIN_NOTE_SLOTS.map((_, slotIndex) => ({
      slotIndex,
      score: hashUnit(`${note.id}-slot-${slotIndex}`),
    })).sort((left, right) => left.score - right.score);

    const freeSlot = preferredSlots.find(({ slotIndex }) => !occupiedSlots.has(slotIndex));
    const chosenSlotIndex =
      freeSlot?.slotIndex ??
      (() => {
        const recycledSlotIndex = slotQueue.shift();
        return recycledSlotIndex ?? 0;
      })();

    occupiedSlots.set(chosenSlotIndex, note);
    slotQueue.push(chosenSlotIndex);
  });

  return slotQueue.map((slotIndex) => {
    const note = occupiedSlots.get(slotIndex);
    const slot = MARGIN_NOTE_SLOTS[slotIndex];

    return {
      note: note as ConversationLandscapeMarginNote,
      leftPercent: slot.leftPercent,
      topPercent: slot.topPercent,
      widthCh: slot.widthCh,
      revealProgress: clamp((currentTimeMs - (note?.appearAtMs ?? currentTimeMs)) / 420, 0, 1),
    };
  });
}

function getContourStroke(pathIndex: number, pathCount: number, outerColor: string, innerColor: string): string {
  if (pathCount <= 1) {
    return innerColor;
  }

  const innerness = pathIndex / (pathCount - 1);
  return mixColors(outerColor, innerColor, innerness);
}

function createGrid(): number[][] {
  return Array.from({ length: GRID_COLS }, () => Array<number>(GRID_ROWS).fill(0));
}

function cloneGrid(source: number[][]): number[][] {
  return source.map((column) => [...column]);
}

function gridX(index: number): number {
  return MAP_PADDING_X + (index / (GRID_COLS - 1)) * MAP_WIDTH;
}

function gridY(index: number): number {
  return MAP_PADDING_Y + (index / (GRID_ROWS - 1)) * MAP_HEIGHT;
}

function addRidgeStamp(
  field: number[][],
  centerX: number,
  centerY: number,
  lengthRadius: number,
  crossRadius: number,
  angle: number,
  amplitude: number,
): void {
  const influenceRadius = Math.max(lengthRadius, crossRadius) * 2.8;
  const minX = clamp(
    Math.floor(((centerX - influenceRadius - MAP_PADDING_X) / MAP_WIDTH) * (GRID_COLS - 1)),
    0,
    GRID_COLS - 1,
  );
  const maxX = clamp(
    Math.ceil(((centerX + influenceRadius - MAP_PADDING_X) / MAP_WIDTH) * (GRID_COLS - 1)),
    0,
    GRID_COLS - 1,
  );
  const minY = clamp(
    Math.floor(((centerY - influenceRadius - MAP_PADDING_Y) / MAP_HEIGHT) * (GRID_ROWS - 1)),
    0,
    GRID_ROWS - 1,
  );
  const maxY = clamp(
    Math.ceil(((centerY + influenceRadius - MAP_PADDING_Y) / MAP_HEIGHT) * (GRID_ROWS - 1)),
    0,
    GRID_ROWS - 1,
  );
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  for (let xIndex = minX; xIndex <= maxX; xIndex += 1) {
    const x = gridX(xIndex);
    for (let yIndex = minY; yIndex <= maxY; yIndex += 1) {
      const y = gridY(yIndex);
      const dx = x - centerX;
      const dy = y - centerY;
      const along = (dx * cos + dy * sin) / Math.max(lengthRadius, 1);
      const across = (-dx * sin + dy * cos) / Math.max(crossRadius, 1);
      const ridgeDistance = along * along * 0.38 + across * across * 2.8;
      const ridgeCore = Math.exp(-ridgeDistance);
      const terrace = 0.82 + Math.cos(along * Math.PI * 1.8) * 0.1;
      field[xIndex][yIndex] += amplitude * ridgeCore * terrace;
    }
  }
}

function blurField(field: number[][], passes: number): number[][] {
  let current = cloneGrid(field);

  for (let pass = 0; pass < passes; pass += 1) {
    const next = createGrid();
    for (let xIndex = 0; xIndex < GRID_COLS; xIndex += 1) {
      for (let yIndex = 0; yIndex < GRID_ROWS; yIndex += 1) {
        let total = 0;
        let weightTotal = 0;

        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const sampleX = xIndex + offsetX;
          if (sampleX < 0 || sampleX >= GRID_COLS) {
            continue;
          }
          for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
            const sampleY = yIndex + offsetY;
            if (sampleY < 0 || sampleY >= GRID_ROWS) {
              continue;
            }
            const weight = offsetX === 0 && offsetY === 0 ? 4 : offsetX === 0 || offsetY === 0 ? 2 : 1;
            total += current[sampleX][sampleY] * weight;
            weightTotal += weight;
          }
        }

        next[xIndex][yIndex] = weightTotal > 0 ? total / weightTotal : current[xIndex][yIndex];
      }
    }
    current = next;
  }

  return current;
}

function addTerritoryWash(
  field: number[][],
  seedKey: string,
  spokenStrength: number,
  talkShare: number,
  region: SpeakerRegion,
): void {
  const phaseA = hashUnit(`${seedKey}-wash-a`) * Math.PI * 2;
  const phaseB = hashUnit(`${seedKey}-wash-b`) * Math.PI * 2;
  const phaseC = hashUnit(`${seedKey}-wash-c`) * Math.PI * 2;
  const focusX = clamp((region.midX - MAP_PADDING_X) / MAP_WIDTH, 0, 1);
  const focusY = 0.5 + hashSigned(`${seedKey}-wash-y`) * 0.08;
  const regionStart = clamp((region.startX - MAP_PADDING_X) / MAP_WIDTH, 0, 1);
  const regionEnd = clamp((region.endX - MAP_PADDING_X) / MAP_WIDTH, 0, 1);
  const regionSpan = Math.max(regionEnd - regionStart, 0.12);

  for (let xIndex = 0; xIndex < GRID_COLS; xIndex += 1) {
    const nx = xIndex / (GRID_COLS - 1);
    for (let yIndex = 0; yIndex < GRID_ROWS; yIndex += 1) {
      const ny = yIndex / (GRID_ROWS - 1);
      const dx = nx - focusX;
      const dy = ny - focusY;
      const broadHill = Math.exp(-(dx * dx / Math.max(regionSpan * regionSpan * 0.9, 0.03) + dy * dy / 0.24));
      const regionMask =
        nx < regionStart
          ? Math.exp(-((regionStart - nx) * (regionStart - nx)) / 0.005)
          : nx > regionEnd
            ? Math.exp(-((nx - regionEnd) * (nx - regionEnd)) / 0.005)
            : 1;
      const wave =
        Math.sin((nx * 1.35 + ny * 0.72) * Math.PI * 2 + phaseA) * 0.5 +
        Math.cos((nx * 0.8 - ny * 1.12) * Math.PI * 2 + phaseB) * 0.35 +
        Math.sin((nx * 2.4 + ny * 1.8) * Math.PI + phaseC) * 0.2;
      const value =
        (broadHill * 0.18 + (wave + 1.05) * 0.035) *
        spokenStrength *
        (0.85 + talkShare * 0.35) *
        regionMask;

      field[xIndex][yIndex] += value;
    }
  }
}

function getSpeakerRegion(
  speakers: ConversationLandscapeSpeaker[],
  speaker: ConversationLandscapeSpeaker,
): SpeakerRegion {
  const mapStartX = MAP_PADDING_X;
  const mapEndX = VIEWBOX_WIDTH - MAP_PADDING_X;
  const orderedSpeakers = speakers;
  const regionGap = orderedSpeakers.length > 1 ? 18 : 0;
  const availableWidth = Math.max(
    mapEndX - mapStartX - regionGap * Math.max(orderedSpeakers.length - 1, 0),
    MAP_WIDTH * 0.72,
  );
  const totalDuration = orderedSpeakers.reduce(
    (sum, entry) => sum + Math.max(entry.totalDurationMs, 1),
    0,
  );

  let cursorX = mapStartX;
  let regionStartX = mapStartX;
  let regionEndX = mapEndX;

  orderedSpeakers.forEach((entry, index) => {
    const widthShare = Math.max(entry.totalDurationMs, 1) / Math.max(totalDuration, 1);
    const rawWidth =
      index === orderedSpeakers.length - 1
        ? mapEndX - cursorX
        : availableWidth * widthShare;
    const nextCursorX = Math.min(mapEndX, cursorX + rawWidth);

    if (entry.id === speaker.id) {
      regionStartX = cursorX;
      regionEndX = nextCursorX;
    }

    cursorX = Math.min(mapEndX, nextCursorX + regionGap);
  });

  const slotInset = Math.min(Math.max((regionEndX - regionStartX) * 0.06, 10), 24);
  const startX = Math.min(regionStartX + slotInset, regionEndX - 20);
  const endX = Math.max(regionEndX - slotInset, regionStartX + 20);

  return {
    startX,
    endX,
    midX: (startX + endX) * 0.5,
  };
}

function scoreRegionPoint(
  field: number[][],
  occupiedField: number[][],
  point: Point,
  desiredY: number,
  region: SpeakerRegion,
): number {
  const localDensity = sampleField(field, point);
  const occupiedDensity = sampleField(occupiedField, point);
  const edgeDistance = Math.min(point.x - region.startX, region.endX - point.x);
  const edgePenalty = edgeDistance < 34 ? (34 - edgeDistance) * 0.03 : 0;
  const verticalPenalty = (Math.abs(point.y - desiredY) / MAP_HEIGHT) * 0.42;

  return localDensity * 1.2 + occupiedDensity * 2.5 + edgePenalty + verticalPenalty;
}

function findAvailableSeedPoint(
  field: number[][],
  occupiedField: number[][],
  utteranceId: string,
  desiredY: number,
  region: SpeakerRegion,
  clusters: SpeakerCluster[],
): { point: Point; score: number } {
  const verticalOffsets = [-0.12, -0.06, 0, 0.06, 0.12];
  const xSteps = Math.max(4, Math.round((region.endX - region.startX) / 78));
  let bestPoint = {
    x: region.midX,
    y: clamp(desiredY, MAP_PADDING_Y + 34, VIEWBOX_HEIGHT - MAP_PADDING_Y - 34),
  };
  let bestScore = Number.POSITIVE_INFINITY;

  for (let xIndex = 0; xIndex < xSteps; xIndex += 1) {
    const xAmount = xSteps === 1 ? 0.5 : xIndex / (xSteps - 1);
    for (const verticalOffset of verticalOffsets) {
      const candidate = {
        x: clamp(
          lerp(region.startX + 24, region.endX - 24, xAmount) +
            hashSigned(`${utteranceId}-seed-x-${xIndex}-${verticalOffset}`) * Math.min((region.endX - region.startX) * 0.025, 10),
          region.startX + 24,
          region.endX - 24,
        ),
        y: clamp(
          desiredY +
            verticalOffset * MAP_HEIGHT +
            hashSigned(`${utteranceId}-seed-y-${xIndex}-${verticalOffset}`) * 8,
          MAP_PADDING_Y + 34,
          VIEWBOX_HEIGHT - MAP_PADDING_Y - 34,
        ),
      };
      const clusterPenalty = clusters.reduce((penalty, cluster) => {
        const distance = Math.hypot(candidate.x - cluster.tail.x, candidate.y - cluster.tail.y);
        if (distance >= 120) {
          return penalty;
        }
        return penalty + (120 - distance) * 0.004;
      }, 0);
      const score = scoreRegionPoint(field, occupiedField, candidate, desiredY, region) + clusterPenalty;

      if (score < bestScore) {
        bestScore = score;
        bestPoint = candidate;
      }
    }
  }

  return { point: bestPoint, score: bestScore };
}

function chooseClusterStart(
  field: number[][],
  occupiedField: number[][],
  utteranceId: string,
  desiredY: number,
  region: SpeakerRegion,
  clusters: SpeakerCluster[],
): { point: Point; clusterIndex: number | null } {
  const newSeed = findAvailableSeedPoint(field, occupiedField, utteranceId, desiredY, region, clusters);
  let bestPoint = newSeed.point;
  let bestClusterIndex: number | null = null;
  let bestScore = newSeed.score + 0.42;

  clusters.forEach((cluster, clusterIndex) => {
    const continuationScore =
      scoreRegionPoint(field, occupiedField, cluster.tail, desiredY, region) +
      (Math.abs(cluster.tail.y - desiredY) / MAP_HEIGHT) * 0.36 +
      cluster.utteranceCount * 0.03;

    if (continuationScore < bestScore) {
      bestScore = continuationScore;
      bestPoint = cluster.tail;
      bestClusterIndex = clusterIndex;
    }
  });

  return { point: bestPoint, clusterIndex: bestClusterIndex };
}

function sampleField(field: number[][], point: Point): number {
  const normalizedX = clamp((point.x - MAP_PADDING_X) / MAP_WIDTH, 0, 1);
  const normalizedY = clamp((point.y - MAP_PADDING_Y) / MAP_HEIGHT, 0, 1);
  const xIndex = clamp(Math.round(normalizedX * (GRID_COLS - 1)), 0, GRID_COLS - 1);
  const yIndex = clamp(Math.round(normalizedY * (GRID_ROWS - 1)), 0, GRID_ROWS - 1);
  return field[xIndex][yIndex];
}

function pickDirectionAwayFromDensity(
  field: number[][],
  occupiedField: number[][],
  utteranceId: string,
  startPoint: Point,
  regionStartX: number,
  regionEndX: number,
  stepDistance: number,
  verticalDistance: number,
): number {
  const baseAngle = hashUnit(`${utteranceId}-direction`) * Math.PI * 2;
  const offsets = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5];
  let bestAngle = baseAngle;
  let bestScore = Number.POSITIVE_INFINITY;

  for (const offset of offsets) {
    const candidateAngle = baseAngle + offset * 0.55;
    const candidatePoint = {
      x: clamp(startPoint.x + Math.cos(candidateAngle) * stepDistance, regionStartX + 24, regionEndX - 24),
      y: clamp(startPoint.y + Math.sin(candidateAngle) * verticalDistance, MAP_PADDING_Y + 30, VIEWBOX_HEIGHT - MAP_PADDING_Y - 30),
    };
    const midwayPoint = {
      x: lerp(startPoint.x, candidatePoint.x, 0.55),
      y: lerp(startPoint.y, candidatePoint.y, 0.55),
    };
    const densityScore =
      sampleField(field, candidatePoint) * 1.1 +
      sampleField(field, midwayPoint) * 0.9 +
      sampleField(occupiedField, candidatePoint) * 2.2 +
      sampleField(occupiedField, midwayPoint) * 1.6;
    const edgeDistance = Math.min(candidatePoint.x - regionStartX, regionEndX - candidatePoint.x);
    const edgePenalty = edgeDistance < 42 ? (42 - edgeDistance) * 0.02 : 0;
    const tieBreaker = hashUnit(`${utteranceId}-candidate-${offset}`) * 0.001;
    const score = densityScore + edgePenalty + tieBreaker;

    if (score < bestScore) {
      bestScore = score;
      bestAngle = candidateAngle;
    }
  }

  return bestAngle;
}

function suppressOverlap(field: number[][], occupiedField: number[][]): number[][] {
  const next = createGrid();
  for (let xIndex = 0; xIndex < GRID_COLS; xIndex += 1) {
    for (let yIndex = 0; yIndex < GRID_ROWS; yIndex += 1) {
      next[xIndex][yIndex] = field[xIndex][yIndex] * Math.exp(-occupiedField[xIndex][yIndex] * 1.4);
    }
  }
  return next;
}

function accumulateField(target: number[][], source: number[][], scale: number): void {
  for (let xIndex = 0; xIndex < GRID_COLS; xIndex += 1) {
    for (let yIndex = 0; yIndex < GRID_ROWS; yIndex += 1) {
      target[xIndex][yIndex] += source[xIndex][yIndex] * scale;
    }
  }
}

function quadraticPoint(start: Point, control: Point, end: Point, amount: number): Point {
  const inverse = 1 - amount;
  return {
    x: inverse * inverse * start.x + 2 * inverse * amount * control.x + amount * amount * end.x,
    y: inverse * inverse * start.y + 2 * inverse * amount * control.y + amount * amount * end.y,
  };
}

function quadraticTangent(start: Point, control: Point, end: Point, amount: number): Point {
  const inverse = 1 - amount;
  return {
    x: 2 * inverse * (control.x - start.x) + 2 * amount * (end.x - control.x),
    y: 2 * inverse * (control.y - start.y) + 2 * amount * (end.y - control.y),
  };
}

function addBezierStroke(
  field: number[][],
  strokeId: string,
  start: Point,
  control: Point,
  end: Point,
  radiusX: number,
  radiusY: number,
  amplitude: number,
  progress: number,
): Point {
  const revealedProgress = clamp(progress, 0, 1);
  const segmentCount = 26;
  const revealedSegments = Math.max(1, Math.ceil(segmentCount * revealedProgress));
  let lastPoint = start;

  for (let index = 0; index <= revealedSegments; index += 1) {
    const t = (index / segmentCount) * revealedProgress;
    const point = quadraticPoint(start, control, end, t);
    const tangent = quadraticTangent(start, control, end, t);
    const angle = Math.atan2(
      Math.abs(tangent.y) < 0.0001 ? end.y - start.y : tangent.y,
      Math.abs(tangent.x) < 0.0001 ? end.x - start.x : tangent.x,
    );
    const wobbleX = hashSigned(`${strokeId}-x-${index}`) * radiusX * 0.08;
    const wobbleY = hashSigned(`${strokeId}-y-${index}`) * radiusY * 0.08;
    const envelope = 0.7 + Math.sin((t / Math.max(revealedProgress, 0.0001)) * Math.PI) * 0.3;

    addRidgeStamp(
      field,
      point.x + wobbleX,
      point.y + wobbleY,
      radiusX * (0.82 + envelope * 0.18),
      radiusY * (0.28 + envelope * 0.08),
      angle,
      amplitude * envelope,
    );

    if (index % 4 === 0) {
      addRidgeStamp(
        field,
        point.x - wobbleX * 0.4,
        point.y - wobbleY * 0.4,
        radiusX * 1.55,
        radiusY * 0.34,
        angle + Math.sin(t * Math.PI * 3) * 0.2,
        amplitude * 0.22,
      );
    }

    lastPoint = point;
  }

  const finalAngle = Math.atan2(end.y - control.y, end.x - control.x);
  addRidgeStamp(
    field,
    lastPoint.x + hashSigned(`${strokeId}-cap-x`) * radiusX * 0.12,
    lastPoint.y + hashSigned(`${strokeId}-cap-y`) * radiusY * 0.12,
    radiusX * 0.96,
    radiusY * 0.26,
    finalAngle,
    amplitude * 0.52,
  );

  return lastPoint;
}

function interpolatePoint(
  x1: number,
  y1: number,
  v1: number,
  x2: number,
  y2: number,
  v2: number,
  level: number,
): Point {
  if (Math.abs(v2 - v1) < 0.00001) {
    return { x: (x1 + x2) * 0.5, y: (y1 + y2) * 0.5 };
  }

  const amount = clamp((level - v1) / (v2 - v1), 0, 1);
  return {
    x: lerp(x1, x2, amount),
    y: lerp(y1, y2, amount),
  };
}

function warpContourPoint(point: Point): Point {
  const nx = clamp((point.x - MAP_PADDING_X) / MAP_WIDTH, 0, 1);
  const ny = clamp((point.y - MAP_PADDING_Y) / MAP_HEIGHT, 0, 1);
  const warpX =
    Math.sin((nx * 7.4 + ny * 2.1) * Math.PI) * 4.2 +
    Math.cos((nx * 2.8 - ny * 6.3) * Math.PI) * 2.6;
  const warpY =
    Math.cos((nx * 6.2 + ny * 2.7) * Math.PI) * 4 +
    Math.sin((nx * 3.1 - ny * 7.1) * Math.PI) * 2.2;

  return {
    x: clamp(point.x + warpX, MAP_PADDING_X, VIEWBOX_WIDTH - MAP_PADDING_X),
    y: clamp(point.y + warpY, MAP_PADDING_Y, VIEWBOX_HEIGHT - MAP_PADDING_Y),
  };
}

function buildContourPath(field: number[][], level: number): string {
  const segments: string[] = [];

  for (let yIndex = 0; yIndex < GRID_ROWS - 1; yIndex += 1) {
    for (let xIndex = 0; xIndex < GRID_COLS - 1; xIndex += 1) {
      const aVal = field[xIndex][yIndex];
      const bVal = field[xIndex + 1][yIndex];
      const cVal = field[xIndex + 1][yIndex + 1];
      const dVal = field[xIndex][yIndex + 1];

      const x1 = gridX(xIndex);
      const y1 = gridY(yIndex);
      const x2 = gridX(xIndex + 1);
      const y2 = gridY(yIndex + 1);

      const ab = interpolatePoint(x1, y1, aVal, x2, y1, bVal, level);
      const bc = interpolatePoint(x2, y1, bVal, x2, y2, cVal, level);
      const cd = interpolatePoint(x2, y2, cVal, x1, y2, dVal, level);
      const da = interpolatePoint(x1, y2, dVal, x1, y1, aVal, level);

      let state = 0;
      if (aVal >= level) state |= 1;
      if (bVal >= level) state |= 2;
      if (cVal >= level) state |= 4;
      if (dVal >= level) state |= 8;

      const pushSegment = (start: Point, end: Point) => {
        const warpedStart = warpContourPoint(start);
        const warpedEnd = warpContourPoint(end);
        segments.push(
          `M ${warpedStart.x.toFixed(1)} ${warpedStart.y.toFixed(1)} L ${warpedEnd.x.toFixed(1)} ${warpedEnd.y.toFixed(1)}`,
        );
      };

      switch (state) {
        case 0:
        case 15:
          break;
        case 1:
        case 14:
          pushSegment(da, ab);
          break;
        case 2:
        case 13:
          pushSegment(ab, bc);
          break;
        case 3:
        case 12:
          pushSegment(da, bc);
          break;
        case 4:
        case 11:
          pushSegment(bc, cd);
          break;
        case 5:
          pushSegment(da, ab);
          pushSegment(bc, cd);
          break;
        case 6:
        case 9:
          pushSegment(ab, cd);
          break;
        case 7:
        case 8:
          pushSegment(da, cd);
          break;
        case 10:
          pushSegment(ab, bc);
          pushSegment(cd, da);
          break;
        default:
          break;
      }
    }
  }

  return segments.join(" ");
}

export function ConversationLandscape({
  speakers,
  utterances,
  activeSpeakerId,
  activeTranscript,
  marginNotes,
  currentTimeMs,
}: ConversationLandscapeProps) {
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const wordRefs = useRef(new Map<string, HTMLSpanElement>());
  const [wordPositions, setWordPositions] = useState<Record<string, WordPosition>>({});
  const [marginNoteLaunchPositions, setMarginNoteLaunchPositions] = useState<Record<string, LaunchPosition>>({});
  const maxSpeakerDurationMs = speakers.reduce((largest, speaker) => Math.max(largest, speaker.totalDurationMs), 1);
  const totalUtteranceCount = Math.max(utterances.length, 1);
  const occupancyField = createGrid();
  const sectionsById = new Map<string, SpeakerSection>();
  const buildOrder = [...speakers].sort((left, right) => right.totalDurationMs - left.totalDurationMs);

  buildOrder.forEach((speaker) => {
    const speakerIndex = speakers.findIndex((entry) => entry.id === speaker.id);
    const color = SPEAKER_COLORS[speakerIndex % SPEAKER_COLORS.length];
    const outerColor = SPEAKER_OUTER_COLORS[speakerIndex % SPEAKER_OUTER_COLORS.length];
    const speakerUtterances = utterances.filter((utterance) => utterance.speakerId === speaker.id);
    const talkShare = clamp(speaker.totalDurationMs / maxSpeakerDurationMs, 0, 1);
    const field = createGrid();
    const region = getSpeakerRegion(speakers, speaker);
    const progressedUtterances = speakerUtterances.filter((utterance) => utterance.progress > 0);
    const spokenStrength = clamp(
      progressedUtterances.reduce((total, utterance) => total + utterance.progress, 0) / Math.max(speakerUtterances.length, 1),
      0,
      1,
    );

    if (spokenStrength > 0) {
      addTerritoryWash(field, speaker.id, spokenStrength, talkShare, region);
    }

    let contourSignalCount = 0;
    const clusters: SpeakerCluster[] = [];
    let previousCenterY: number | null = null;

    speakerUtterances.forEach((utterance, utteranceIndex) => {
      if (utterance.progress <= 0) {
        return;
      }

      contourSignalCount += utterance.contourSignalCount;

      const globalSpread = utterance.order / totalUtteranceCount;
      const localSpread =
        speakerUtterances.length <= 1 ? 0.5 : utteranceIndex / Math.max(speakerUtterances.length - 1, 1);
      const spread = lerp(globalSpread, localSpread, 0.72);
      const wavePhase = hashUnit(`${speaker.id}-path`) * Math.PI * 2;
      const targetCenterY = clamp(
        MAP_PADDING_Y +
          MAP_HEIGHT *
            (0.5 +
              Math.sin(spread * Math.PI * 1.15 + wavePhase) * 0.12 +
              Math.cos(spread * Math.PI * 2.1 + wavePhase * 0.55) * 0.06 +
              hashSigned(`${utterance.id}-y`) * 0.025),
        MAP_PADDING_Y + 40,
        VIEWBOX_HEIGHT - MAP_PADDING_Y - 40,
      );
      const centerY =
        previousCenterY === null
          ? targetCenterY
          : clamp(lerp(previousCenterY, targetCenterY, 0.3), MAP_PADDING_Y + 40, VIEWBOX_HEIGHT - MAP_PADDING_Y - 40);
      previousCenterY = centerY;

      const regionWidth = region.endX - region.startX;
      const stepDistance = regionWidth * lerp(0.22, 0.42, talkShare);
      const verticalDistance = MAP_HEIGHT * lerp(0.06, 0.11, talkShare);
      const startSelection = chooseClusterStart(field, occupancyField, utterance.id, centerY, region, clusters);
      const startPoint = startSelection.point;
      const directionAngle = pickDirectionAwayFromDensity(
        field,
        occupancyField,
        utterance.id,
        startPoint,
        region.startX,
        region.endX,
        stepDistance,
        verticalDistance,
      );

      const endPoint = {
        x: clamp(
          startPoint.x + Math.cos(directionAngle) * stepDistance,
          region.startX + 24,
          region.endX - 24,
        ),
        y: clamp(
          startPoint.y + Math.sin(directionAngle) * verticalDistance + hashSigned(`${utterance.id}-end-y`) * 10,
          MAP_PADDING_Y + 30,
          VIEWBOX_HEIGHT - MAP_PADDING_Y - 30,
        ),
      };

      const bendAmount = hashSigned(`${utterance.id}-bend`) * regionWidth * 0.14;
      const perpendicularOffset = hashSigned(`${utterance.id}-perp`) * Math.min(stepDistance, verticalDistance) * 0.48;
      const controlPoint = {
        x: clamp(
          lerp(startPoint.x, endPoint.x, 0.5) + bendAmount - Math.sin(directionAngle) * perpendicularOffset,
          region.startX + 24,
          region.endX - 24,
        ),
        y: clamp(
          lerp(startPoint.y, endPoint.y, 0.5) + hashSigned(`${utterance.id}-control-y`) * 24 + Math.cos(directionAngle) * perpendicularOffset,
          MAP_PADDING_Y + 28,
          VIEWBOX_HEIGHT - MAP_PADDING_Y - 28,
        ),
      };

      const radiusX = lerp(76, 148, talkShare);
      const radiusY = lerp(60, 118, talkShare);
      const amplitude = lerp(0.9, 0.62, talkShare) + utterance.politenessScore * 0.26;

      const nextTail = addBezierStroke(
        field,
        utterance.id,
        startPoint,
        controlPoint,
        endPoint,
        radiusX,
        radiusY,
        amplitude,
        utterance.progress,
      );

      if (startSelection.clusterIndex === null) {
        clusters.push({ tail: nextTail, utteranceCount: 1 });
      } else {
        clusters[startSelection.clusterIndex] = {
          tail: nextTail,
          utteranceCount: clusters[startSelection.clusterIndex].utteranceCount + 1,
        };
      }
    });

    const smoothedField = blurField(field, 1);
    const separatedField = suppressOverlap(smoothedField, occupancyField);
    accumulateField(occupancyField, separatedField, 0.55);
    const maxFieldValue = separatedField.reduce((largest, column) => Math.max(largest, ...column), 0);
    const levelCount = Math.ceil(clamp(2 + contourSignalCount, 2, 13));
    const levels =
      progressedUtterances.length === 0 || maxFieldValue <= 0.04
        ? []
        : Array.from({ length: levelCount }, (_, levelIndex) =>
          lerp(maxFieldValue * 0.08, maxFieldValue * 0.84, (levelIndex + 1) / (levelCount + 1)),
        );

    sectionsById.set(speaker.id, {
      speaker,
      color,
      outerColor,
      levelCount: levels.length,
      washHref: buildFieldWashDataUrl(separatedField, maxFieldValue, color, outerColor),
      paths: levels
        .map((level) => ({
          level,
          d: buildContourPath(separatedField, level),
        }))
        .filter((path) => path.d.length > 0),
    });
  });

  const speakerSections = speakers
    .map((speaker) => sectionsById.get(speaker.id))
    .filter((section): section is NonNullable<typeof section> => section !== undefined);
  const activeSection = activeSpeakerId ? sectionsById.get(activeSpeakerId) : undefined;
  const transcriptColor = activeSection?.color ?? "var(--ink)";
  const transcriptCurrentTimeMs = activeTranscript?.currentTimeMs ?? currentTimeMs;
  const canvasWidth = canvasRef.current?.clientWidth ?? 0;
  const canvasHeight = canvasRef.current?.clientHeight ?? 0;
  const transcriptWordTokens = activeTranscript?.tokens.filter(
    (token): token is ConversationLandscapeTranscriptWordToken => token.kind === "word",
  ) ?? [];
  const visibleMarginNotes = buildVisibleMarginNotes(marginNotes, transcriptCurrentTimeMs);

  const fallingWords = transcriptWordTokens
    .map((token) => {
      if (!token.isHedge) {
        return null;
      }

      const position = wordPositions[token.id];
      if (!position || transcriptCurrentTimeMs < token.dropStartMs) {
        return null;
      }

      const progress = clamp((transcriptCurrentTimeMs - token.dropStartMs) / token.dropDurationMs, 0, 1);
      if (progress >= 1) {
        return null;
      }

      return {
        token,
        position,
        progress,
        driftX: hashSigned(`${token.id}-drift`) * 34,
        rotation: hashSigned(`${token.id}-rotation`) * 18,
      };
    })
    .filter((item): item is NonNullable<typeof item> => item !== null);

  useLayoutEffect(() => {
    const nextPositions = measureTranscriptWordPositions(canvasRef.current, wordRefs.current);
    setWordPositions((currentPositions) =>
      areWordPositionsEqual(currentPositions, nextPositions) ? currentPositions : nextPositions,
    );
  }, [activeTranscript]);

  useLayoutEffect(() => {
    if (!activeTranscript) {
      return;
    }

    const activeNoteLaunches = marginNotes
      .filter((note) => note.utteranceId === activeTranscript.utteranceId)
      .reduce<Record<string, LaunchPosition>>((positions, note) => {
        const matchingTokens = transcriptWordTokens.filter(
          (token) => note.sourceStart < token.end && token.start < note.sourceEnd,
        );

        if (matchingTokens.length === 0) {
          return positions;
        }

        const tokenPositions = matchingTokens
          .map((token) => wordPositions[token.id])
          .filter((position): position is WordPosition => position !== undefined);

        if (tokenPositions.length === 0) {
          return positions;
        }

        const firstPosition = tokenPositions[0];
        const top = tokenPositions.reduce((minimum, position) => Math.min(minimum, position.top), firstPosition.top);
        const left = tokenPositions.reduce((minimum, position) => Math.min(minimum, position.left), firstPosition.left);

        positions[note.id] = { left, top };
        return positions;
      }, {});

    setMarginNoteLaunchPositions((currentLaunchPositions) => {
      const mergedLaunchPositions = { ...currentLaunchPositions, ...activeNoteLaunches };
      return areLaunchPositionsEqual(currentLaunchPositions, mergedLaunchPositions)
        ? currentLaunchPositions
        : mergedLaunchPositions;
    });
  }, [activeTranscript, marginNotes, transcriptWordTokens, wordPositions]);

  useEffect(() => {
    if (!canvasRef.current || typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver(() => {
      const nextPositions = measureTranscriptWordPositions(canvasRef.current, wordRefs.current);
      setWordPositions((currentPositions) =>
        areWordPositionsEqual(currentPositions, nextPositions) ? currentPositions : nextPositions,
      );
    });

    observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <section className="conversation-landscape" aria-label="Conversation landscape">
      <div ref={canvasRef} className="conversation-landscape__canvas">
        {activeTranscript ? (
          <div
            className={`conversation-landscape__transcript${activeSection?.speaker.side === "right" ? " conversation-landscape__transcript--right" : ""
              }`}
            style={{ color: transcriptColor }}
            aria-live="polite"
          >
            {activeSection ? (
              <p className="conversation-landscape__transcript-label">{activeSection.speaker.label}</p>
            ) : null}
            <p className="conversation-landscape__transcript-copy">
              {activeTranscript.tokens.map((token) => {
                if (token.kind === "text") {
                  return <span key={token.id}>{token.text}</span>;
                }

                const isDropping = token.isHedge && activeTranscript.currentTimeMs >= token.dropStartMs;

                return (
                  <span
                    key={token.id}
                    ref={(node) => {
                      if (node) {
                        wordRefs.current.set(token.id, node);
                      } else {
                        wordRefs.current.delete(token.id);
                      }
                    }}
                    className={[
                      "conversation-landscape__transcript-word",
                      token.isHedge ? "conversation-landscape__transcript-word--hedging" : "",
                      isDropping ? "conversation-landscape__transcript-word--detached" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    {token.text}
                  </span>
                );
              })}
            </p>
          </div>
        ) : null}
        <div className="conversation-landscape__margin-notes" aria-hidden="true">
          {visibleMarginNotes.map(({ note, leftPercent, topPercent, widthCh, revealProgress }) => (
            (() => {
              const launchPosition = marginNoteLaunchPositions[note.id];
              const slotLeft = (leftPercent / 100) * canvasWidth;
              const slotTop = (topPercent / 100) * canvasHeight;
              const settleProgress = clamp(
                (transcriptCurrentTimeMs - note.appearAtMs) / Math.max(note.settleDurationMs, 1),
                0,
                1,
              );
              const swayPhase = hashUnit(`${note.id}-phase`) * Math.PI * 2;
              const swayAmplitude = lerp(20, 42, hashUnit(`${note.id}-sway`));
              const settleX = Math.pow(settleProgress, 0.82);
              const settleY = Math.pow(settleProgress, 1.18);
              const flutterX = Math.sin(settleProgress * Math.PI * 2.55 + swayPhase) * swayAmplitude * Math.pow(1 - settleProgress, 0.58);
              const flutterY = Math.cos(settleProgress * Math.PI * 1.55 + swayPhase) * 14 * Math.pow(1 - settleProgress, 0.92);
              const rotateDeg = Math.sin(settleProgress * Math.PI * 1.8 + swayPhase) * 5.5 * Math.pow(1 - settleProgress, 0.86);
              const translateX = launchPosition
                ? lerp(launchPosition.left - slotLeft, 0, settleX) + flutterX
                : flutterX * 0.45;
              const translateY = launchPosition
                ? lerp(launchPosition.top - slotTop, 0, settleY) + flutterY
                : lerp(26, 0, revealProgress) + flutterY * 0.45;

              return (
                <p
                  key={note.id}
                  className="conversation-landscape__margin-note"
                  style={{
                    left: `${leftPercent}%`,
                    top: `${topPercent}%`,
                    width: `${widthCh}ch`,
                    opacity: `${lerp(0.18, 0.72, revealProgress)}`,
                    transform: `translate3d(${translateX}px, ${translateY}px, 0) rotate(${rotateDeg}deg)`,
                  }}
                >
                  {note.text}
                </p>
              );
            })()
          ))}
        </div>
        <div className="conversation-landscape__falling-words" style={{ color: transcriptColor }} aria-hidden="true">
          {fallingWords.map(({ token, position, progress, driftX, rotation }) => {
            const swayPhase = hashUnit(`${token.id}-phase`) * Math.PI * 2;
            const sway = Math.sin(progress * Math.PI * 3.15 + swayPhase) * Math.abs(driftX) * 1.18 * Math.pow(1 - progress, 0.4);
            const glideX = driftX * Math.pow(progress, 0.96);
            const fallY = lerp(0, 248, Math.pow(progress, 1.18));
            const bobY = Math.cos(progress * Math.PI * 2.1 + swayPhase) * 12 * Math.pow(1 - progress, 0.72);
            const rotateDeg =
              Math.sin(progress * Math.PI * 2.2 + swayPhase) * 8 * Math.pow(1 - progress, 0.52) +
              rotation * Math.pow(progress, 0.84) * 0.5;

            return (
              <span
                key={`${activeTranscript?.utteranceId ?? "utterance"}-${token.id}`}
                className={[
                  "conversation-landscape__falling-word",
                  token.isHedge ? "conversation-landscape__falling-word--hedging" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                style={{
                  left: `${position.left}px`,
                  top: `${position.top}px`,
                  width: `${Math.max(position.width, 1)}px`,
                  height: `${Math.max(position.height, 1)}px`,
                  transform: `translate3d(${glideX + sway}px, ${fallY + bobY}px, 0) rotate(${rotateDeg}deg)`,
                  opacity: `${clamp(1 - Math.pow(progress, 1.35), 0, 1)}`,
                }}
              >
                {token.text}
              </span>
            );
          })}
        </div>
        <svg
          className="conversation-landscape__svg"
          viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
          role="img"
          aria-label="Collective terrain map drawn from the conversation"
        >
          {speakerSections.map(({ speaker, washHref }) =>
            washHref ? (
              <image
                key={`${speaker.id}-wash`}
                href={washHref}
                x={MAP_PADDING_X}
                y={MAP_PADDING_Y}
                width={MAP_WIDTH}
                height={MAP_HEIGHT}
                preserveAspectRatio="none"
                opacity={speaker.opacity * 0.92}
              />
            ) : null,
          )}
          {speakerSections.map(({ speaker, color, outerColor, paths }) => (
            <g key={speaker.id} opacity={speaker.opacity}>
              {paths.map((path, pathIndex) => (
                <path
                  key={`${speaker.id}-${pathIndex}`}
                  d={path.d}
                  fill="none"
                  stroke={getContourStroke(pathIndex, paths.length, outerColor, color)}
                  strokeOpacity={lerp(0.76, 0.98, paths.length <= 1 ? 1 : pathIndex / (paths.length - 1))}
                  strokeWidth={1.45}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ))}
            </g>
          ))}
        </svg>
      </div>
    </section>
  );
}
