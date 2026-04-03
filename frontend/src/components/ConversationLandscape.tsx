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
  progress: number;
  order: number;
}

interface ConversationLandscapeProps {
  speakers: ConversationLandscapeSpeaker[];
  utterances: ConversationLandscapeUtterance[];
  activeSpeakerId: string | null;
  activeTranscriptText: string | null;
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

const VIEWBOX_WIDTH = 1200;
const VIEWBOX_HEIGHT = 620;
const GRID_COLS = 156;
const GRID_ROWS = 112;
const MAP_PADDING_X = 34;
const MAP_PADDING_Y = 28;
const MAP_WIDTH = VIEWBOX_WIDTH - MAP_PADDING_X * 2;
const MAP_HEIGHT = VIEWBOX_HEIGHT - MAP_PADDING_Y * 2;
const SPEAKER_COLORS = ["#0a8f87", "#8a4c1b", "#4b6c84", "#73586d"];
const SPEAKER_OUTER_COLORS = ["#a7dbd6", "#d8b59b", "#9fb8ca", "#c6b2c0"];

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

function getContourStroke(pathIndex: number, pathCount: number, outerColor: string, innerColor: string): string {
  if (pathCount <= 1) {
    return innerColor;
  }

  const innerness = pathIndex / (pathCount - 1);
  return mixColors(outerColor, innerColor, innerness);
}

function formatDuration(totalDurationMs: number): string {
  const seconds = totalDurationMs / 1000;
  if (seconds >= 10) {
    return `${seconds.toFixed(1)}s`;
  }
  return `${seconds.toFixed(2)}s`;
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

function addGaussianStamp(
  field: number[][],
  centerX: number,
  centerY: number,
  radiusX: number,
  radiusY: number,
  amplitude: number,
): void {
  const minX = clamp(Math.floor(((centerX - radiusX * 2.6 - MAP_PADDING_X) / MAP_WIDTH) * (GRID_COLS - 1)), 0, GRID_COLS - 1);
  const maxX = clamp(Math.ceil(((centerX + radiusX * 2.6 - MAP_PADDING_X) / MAP_WIDTH) * (GRID_COLS - 1)), 0, GRID_COLS - 1);
  const minY = clamp(Math.floor(((centerY - radiusY * 2.6 - MAP_PADDING_Y) / MAP_HEIGHT) * (GRID_ROWS - 1)), 0, GRID_ROWS - 1);
  const maxY = clamp(Math.ceil(((centerY + radiusY * 2.6 - MAP_PADDING_Y) / MAP_HEIGHT) * (GRID_ROWS - 1)), 0, GRID_ROWS - 1);

  for (let xIndex = minX; xIndex <= maxX; xIndex += 1) {
    const x = gridX(xIndex);
    for (let yIndex = minY; yIndex <= maxY; yIndex += 1) {
      const y = gridY(yIndex);
      const dx = (x - centerX) / Math.max(radiusX, 1);
      const dy = (y - centerY) / Math.max(radiusY, 1);
      const distance = dx * dx + dy * dy;
      field[xIndex][yIndex] += amplitude * Math.exp(-distance * 1.9);
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
  const verticalOffsets = [-0.22, -0.12, 0, 0.12, 0.22];
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
            hashSigned(`${utteranceId}-seed-x-${xIndex}-${verticalOffset}`) * Math.min((region.endX - region.startX) * 0.05, 18),
          region.startX + 24,
          region.endX - 24,
        ),
        y: clamp(
          desiredY +
            verticalOffset * MAP_HEIGHT +
            hashSigned(`${utteranceId}-seed-y-${xIndex}-${verticalOffset}`) * 16,
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
  let bestScore = newSeed.score + 0.16;

  clusters.forEach((cluster, clusterIndex) => {
    const continuationScore =
      scoreRegionPoint(field, occupiedField, cluster.tail, desiredY, region) +
      (Math.abs(cluster.tail.y - desiredY) / MAP_HEIGHT) * 0.52 +
      cluster.utteranceCount * 0.045;

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
    const wobbleX = hashSigned(`${strokeId}-x-${index}`) * radiusX * 0.08;
    const wobbleY = hashSigned(`${strokeId}-y-${index}`) * radiusY * 0.08;
    const envelope = 0.7 + Math.sin((t / Math.max(revealedProgress, 0.0001)) * Math.PI) * 0.3;

    addGaussianStamp(
      field,
      point.x + wobbleX,
      point.y + wobbleY,
      radiusX * (0.56 + envelope * 0.14),
      radiusY * (0.54 + envelope * 0.12),
      amplitude * envelope,
    );

    if (index % 4 === 0) {
      addGaussianStamp(
        field,
        point.x - wobbleX * 0.4,
        point.y - wobbleY * 0.4,
        radiusX * 1.4,
        radiusY * 1.34,
        amplitude * 0.3,
      );
    }

    lastPoint = point;
  }

  addGaussianStamp(
    field,
    lastPoint.x + hashSigned(`${strokeId}-cap-x`) * radiusX * 0.12,
    lastPoint.y + hashSigned(`${strokeId}-cap-y`) * radiusY * 0.12,
    radiusX * 0.68,
    radiusY * 0.62,
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
        segments.push(`M ${start.x.toFixed(1)} ${start.y.toFixed(1)} L ${end.x.toFixed(1)} ${end.y.toFixed(1)}`);
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
  activeTranscriptText,
}: ConversationLandscapeProps) {
  const maxSpeakerDurationMs = speakers.reduce((largest, speaker) => Math.max(largest, speaker.totalDurationMs), 1);
  const totalUtteranceCount = Math.max(utterances.length, 1);
  const occupancyField = createGrid();
  const sectionsById = new Map<
    string,
    {
      speaker: ConversationLandscapeSpeaker;
      color: string;
      outerColor: string;
      levelCount: number;
      paths: { level: number; d: string }[];
    }
  >();
  const buildOrder = [...speakers].sort((left, right) => right.totalDurationMs - left.totalDurationMs);

  buildOrder.forEach((speaker) => {
    const speakerIndex = speakers.findIndex((entry) => entry.id === speaker.id);
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

    let politeContourCount = 0;
    const clusters: SpeakerCluster[] = [];

    speakerUtterances.forEach((utterance, utteranceIndex) => {
      if (utterance.progress <= 0) {
        return;
      }

      if (utterance.politenessScore > 0.5) {
        politeContourCount += 1;
      }

      const globalSpread = utterance.order / totalUtteranceCount;
      const wavePhase = hashUnit(`${speaker.id}-path`) * Math.PI * 2;
      const centerY = clamp(
        MAP_PADDING_Y +
          MAP_HEIGHT *
            (0.5 +
              Math.sin(globalSpread * Math.PI * 1.8 + wavePhase) * 0.22 +
              Math.cos(globalSpread * Math.PI * 3.4 + wavePhase * 0.7) * 0.12 +
              hashSigned(`${utterance.id}-y`) * 0.06),
        MAP_PADDING_Y + 40,
        VIEWBOX_HEIGHT - MAP_PADDING_Y - 40,
      );

      const regionWidth = region.endX - region.startX;
      const stepDistance = regionWidth * lerp(0.22, 0.42, talkShare);
      const verticalDistance = MAP_HEIGHT * lerp(0.09, 0.18, talkShare);
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
          startPoint.y + Math.sin(directionAngle) * verticalDistance + hashSigned(`${utterance.id}-end-y`) * 18,
          MAP_PADDING_Y + 30,
          VIEWBOX_HEIGHT - MAP_PADDING_Y - 30,
        ),
      };

      const bendAmount = hashSigned(`${utterance.id}-bend`) * regionWidth * 0.22;
      const perpendicularOffset = hashSigned(`${utterance.id}-perp`) * Math.min(stepDistance, verticalDistance) * 0.9;
      const controlPoint = {
        x: clamp(
          lerp(startPoint.x, endPoint.x, 0.5) + bendAmount - Math.sin(directionAngle) * perpendicularOffset,
          region.startX + 24,
          region.endX - 24,
        ),
        y: clamp(
          lerp(startPoint.y, endPoint.y, 0.5) + hashSigned(`${utterance.id}-control-y`) * 52 + Math.cos(directionAngle) * perpendicularOffset,
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

    const smoothedField = blurField(field, 3);
    const separatedField = suppressOverlap(smoothedField, occupancyField);
    accumulateField(occupancyField, separatedField, 0.55);
    const maxFieldValue = separatedField.reduce((largest, column) => Math.max(largest, ...column), 0);
    const levelCount = Math.max(1, politeContourCount + 1);
    const levels =
      progressedUtterances.length === 0 || maxFieldValue <= 0.04
        ? []
        : Array.from({ length: levelCount }, (_, levelIndex) =>
            lerp(maxFieldValue * 0.12, maxFieldValue * 0.78, (levelIndex + 1) / (levelCount + 1)),
          );

    sectionsById.set(speaker.id, {
      speaker,
      color: SPEAKER_COLORS[speakerIndex % SPEAKER_COLORS.length],
      outerColor: SPEAKER_OUTER_COLORS[speakerIndex % SPEAKER_OUTER_COLORS.length],
      levelCount: levels.length,
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

  return (
    <section className="conversation-landscape" aria-label="Conversation landscape">
      <div className="conversation-landscape__meta">
        {speakerSections.map(({ speaker, color, levelCount }) => {
          const isActive = activeSpeakerId === speaker.id;
          return (
            <div
              key={speaker.id}
              className={`conversation-landscape__chip${isActive ? " conversation-landscape__chip--active" : ""}`}
              style={{
                borderColor: `${color}3d`,
                backgroundColor: `${color}10`,
                color,
                opacity: speaker.opacity,
              }}
            >
              <span className="conversation-landscape__chip-label">{speaker.label}</span>
              <span className="conversation-landscape__chip-copy">
                {formatDuration(speaker.totalDurationMs)} spoken · polite {speaker.averagePoliteness.toFixed(1)} · contour lines{" "}
                {levelCount}
              </span>
            </div>
          );
        })}
      </div>

      <svg
        className="conversation-landscape__svg"
        viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
        role="img"
        aria-label="Collective terrain map drawn from the conversation"
      >
        <rect x="0" y="0" width={VIEWBOX_WIDTH} height={VIEWBOX_HEIGHT} fill="#fbf7ef" />
        <rect
          x={MAP_PADDING_X}
          y={MAP_PADDING_Y}
          width={MAP_WIDTH}
          height={MAP_HEIGHT}
          fill="none"
          stroke="rgba(23, 24, 28, 0.08)"
        />
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
      {activeTranscriptText ? (
        <div
          className={`conversation-landscape__transcript${
            activeSection?.speaker.side === "right" ? " conversation-landscape__transcript--right" : ""
          }`}
          style={{ color: activeSection?.color ?? "var(--ink)" }}
          aria-live="polite"
        >
          {activeSection ? (
            <p className="conversation-landscape__transcript-label">{activeSection.speaker.label}</p>
          ) : null}
          <p className="conversation-landscape__transcript-copy">{activeTranscriptText}</p>
        </div>
      ) : null}
    </section>
  );
}
