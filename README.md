# ASR Viz Backend

Backend for media ingestion, timestamped transcription, optional speaker diarization, sentence-level analysis, and structured storage.

## Components

- FastAPI API for job creation and retrieval
- SQLAlchemy models for media, jobs, transcripts, segments, sentence units, and analysis
- Worker loop backed by the database queue
- Provider interfaces for ASR, diarization, and analysis
- Scene clustering persisted from sentence-level transcript analysis

## Quick Start

1. Run `./scripts/setup_local.sh`.
2. Add any remaining keys to `.env.local` if the setup prompt did not collect them.
3. Start the API and worker:

```bash
./scripts/setup_local.sh
./scripts/run_api.sh
./scripts/run_worker.sh
```

The setup script creates or reuses `.venv`, installs the backend with `faster-whisper` and diarization dependencies, seeds `.env.local` and `frontend/.env` from their examples, and prompts for a Hugging Face token on first run. Real transcription is the default: `ENABLE_MOCK_TRANSCRIPTION=false`.

## Local CLI Workflow

Use the CLI for local testing without writing inline Python:

```bash
python3 -m asr_viz.cli submit /absolute/path/to/file.mp4 --mime-type video/mp4
python3 -m asr_viz.cli worker --once
python3 -m asr_viz.cli status
python3 -m asr_viz.cli transcript --sentences 10
```

Run `python3 -m asr_viz.cli worker` to keep processing queued jobs continuously.

The worker wrapper accepts the same flags, for example:

```bash
./scripts/run_worker.sh --once
```

## Frontend Expanse

A new frontend scaffold lives in [`frontend/`](/Users/jinheeshim/Documents/ITP/thesis/v5_asr_viz/frontend/README.md). It reads completed transcript analysis from the API and maps each sentence into a conversation landscape:

- speaker regions are sized by total speaking duration
- contour lines grow when a sentence has non-empty `hedging` or `substance` data in `analysis_payload`
- the active utterance is shown beneath the visualizer during playback
- raw `analysis_payload` JSON is printed below the visualizer for each sentence

The frontend expects the existing `/jobs` and `/transcripts/{id}` endpoints and does not require schema changes to the backend.

## Environment Variables

- `DATABASE_URL`
- `APP_ENV`
- `AUTO_CREATE_SCHEMA`
- `AUTH_PROVIDER`
- `REQUIRE_AUTH`
- `AUTH_DEV_USER_ID`
- `ANONYMOUS_SESSION_COOKIE_NAME`
- `ANONYMOUS_SESSION_COOKIE_MAX_AGE_SECONDS`
- `ANONYMOUS_SESSION_COOKIE_SECURE`
- `ANONYMOUS_SESSION_COOKIE_DOMAIN`
- `CLERK_ISSUER_URL`
- `CLERK_AUDIENCE`
- `CLERK_JWKS_URL`
- `HUGGINGFACE_TOKEN`
- `ASR_MODEL`
- `DIARIZATION_MODEL`
- `DIARIZATION_NUM_SPEAKERS`
- `JOB_POLL_INTERVAL_SECONDS`
- `ENABLE_MOCK_TRANSCRIPTION`
- `MAX_UPLOAD_SIZE_BYTES`
- `PER_USER_STORAGE_QUOTA_BYTES`
- `LIVE_MODE_ENABLED`
- `STORAGE_BACKEND`
- `MEDIA_STORAGE_DIR`
- `SIGNED_URL_TTL_SECONDS`
- `R2_ACCOUNT_ID`
- `R2_BUCKET_NAME`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

## Public Deployment Foundation

The backend now includes the first public-web foundation slice:

- owner scoping on jobs, transcripts, stream sessions, and live sessions
- anonymous browser-session ownership via `HttpOnly` cookie, with optional Clerk bearer auth support
- per-upload and per-user storage limits
- Alembic scaffolding for managed schema migrations

For local development and anonymous MVP deployment, `REQUIRE_AUTH=false` uses a durable anonymous session cookie so uploads, transcript edits, and playback stay attached to the same browser session across pages and refreshes.

For production with anonymous-first access:

1. set `REQUIRE_AUTH=false`
2. set `AUTH_PROVIDER=clerk` only if you want to accept optional Clerk bearer tokens later
3. configure `DATABASE_URL` for Postgres
4. set cookie security vars for your deployed environment
5. run migrations before starting the API:

```bash
PYTHONPATH=src alembic upgrade head
```

Deployment setup is tracked in the checklist at [docs/public-deployment.md](/Users/jinheeshim/Documents/ITP/thesis/v5_asr_viz/docs/public-deployment.md).

## Speaker Diarization

Speaker diarization is optional and off by default. To enable it:

```bash
./scripts/setup_local.sh
python3 -m asr_viz.cli submit /absolute/path/to/file.mp4 --mime-type video/mp4 --diarization
./scripts/run_worker.sh --once
```

Set `DIARIZATION_NUM_SPEAKERS=2` in `.env.local` if you want to pin the speaker count.

If you are using the default `pyannote/speaker-diarization-3.1` model, the Hugging Face account behind `HUGGINGFACE_TOKEN` must also have accepted access to that gated repository. A valid token without model access will now leave the transcript completed, but speaker labels will be skipped and the job will record the diarization warning in `stage_details`.

Sentence units will then include `speaker_id` and `speaker_confidence` when the diarization model can assign them.

## Scene Clusters

The backend now persists scene clusters derived from consecutive `sentence_units`. Each scene stores:

- start/end timestamps
- dominant speaker and speaker mix
- aggregated politeness / semantic confidence / main-message scores
- topic label and confidence
- membership rows linking scenes back to sentence units

Useful endpoints:

```bash
GET /transcripts/{transcript_id}/scenes
POST /transcripts/{transcript_id}/scenes/rebuild
```

## Streaming Ingestion

The backend now includes a first streaming-ingestion slice that accepts uploaded media chunks, persists them to disk, and then hands the finalized file into the normal processing queue.

Useful endpoints:

```bash
POST /stream-sessions
PUT /stream-sessions/{session_id}/chunks
POST /stream-sessions/{session_id}/finalize
GET /stream-sessions/{session_id}
```

This is an upload/finalize workflow, not live incremental ASR yet. The finalized stream session creates a normal queued processing job.
