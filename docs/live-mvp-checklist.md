# Live MVP Checklist

This checklist scopes the first build slice for a near-real-time recording experience that can later converge into the existing review and playback flow.

## Phase 1: Backend foundations

- [x] Add persistent `live_sessions` model for live capture state.
- [x] Add persistent `live_transcript_events` model for append-only transcript and analysis events.
- [x] Create live-session service for session creation, chunk intake, stop, finalize, and event logging.
- [x] Expose REST endpoints for `create`, `status`, `chunks`, `events`, `stop`, and `finalize`.
- [x] Create the provisional transcript at live-session start so later UI work has a stable transcript id.
- [x] Keep a real recording file on disk from the first audio chunk onward.
- [x] Add backend tests for the new live-session lifecycle.

## Phase 2: Transcript projection

- [x] Project `transcript.final` events into canonical `sentence_units`.
- [x] Project `analysis.delta` events into canonical `analysis_results`.
- [x] Rebuild transcript `full_text` from finalized live utterances.
- [ ] Add revision history table so user edits and model revisions are both preserved.
- [ ] Add sentence locking to protect manual edits from later reconciliation.

## Phase 3: Frontend live mode

- [x] Add `live` app mode in the React client.
- [x] Add browser microphone capture with `MediaRecorder`.
- [x] Add live session client helpers in `frontend/src/lib/api.ts`.
- [x] Render rolling transcript with provisional vs final styling.
- [ ] Show live heuristic overlays in the visualization.
- [x] Route completed live sessions into the existing review flow.

## Phase 4: Reconciliation and replay parity

- [ ] Add full-session reconciliation pass from the saved recording.
- [ ] Align finalized transcript timestamps to replay media.
- [ ] Merge reconciliation results without overwriting locked user edits.
- [ ] Create a job/reconciliation pathway that does not duplicate media assets or transcripts.
- [ ] Add replay annotations for edits and heuristic updates over time.

## Known Constraints

- Live diarization is intentionally deferred for the MVP.
- The current implementation starts with REST event ingestion; WebSocket fan-out comes next.
- Finalization is a lightweight archive step for now, not yet a full offline reconciliation pipeline.
