# ASR Viz Backend

Backend for media ingestion, timestamped transcription, optional speaker diarization, sentence-level analysis, and structured storage.

## Components

- FastAPI API for job creation and retrieval
- SQLAlchemy models for media, jobs, transcripts, segments, sentence units, and analysis
- Worker loop backed by the database queue
- Provider interfaces for ASR, diarization, and analysis
- Scene clustering persisted from sentence-level transcript analysis

## Quick Start

1. Set `DATABASE_URL` to a Postgres URL for normal use. If omitted, the app falls back to local SQLite for development.
2. Optionally set `OPENAI_API_KEY` and `ANALYSIS_MODEL`.
3. Install dependencies and run:

```bash
uvicorn asr_viz.api.main:app --reload
python3 -m asr_viz.worker
```

## Local CLI Workflow

Use the CLI for local testing without writing inline Python:

```bash
python3 -m asr_viz.cli submit /absolute/path/to/file.mp4 --mime-type video/mp4
python3 -m asr_viz.cli worker --once
python3 -m asr_viz.cli status
python3 -m asr_viz.cli transcript --sentences 10
```

Run `python3 -m asr_viz.cli worker` to keep processing queued jobs continuously.

## Frontend Expanse

A new frontend scaffold lives in [`frontend/`](/Users/jinheeshim/Documents/ITP/thesis/v5_asr_viz/frontend/README.md). It reads completed transcript analysis from the API and maps each sentence into a 3D "expanse" scene:

- time becomes horizontal position
- semantic confidence lifts the node vertically
- topics and speaker turns spread the scene in depth
- main-message likelihood affects scale
- topic shifts intensify glow and trail curvature

The frontend expects the existing `/jobs` and `/transcripts/{id}` endpoints and does not require schema changes to the backend.

## Environment Variables

- `DATABASE_URL`
- `OPENAI_API_KEY`
- `HUGGINGFACE_TOKEN`
- `ASR_MODEL`
- `ANALYSIS_MODEL`
- `DIARIZATION_MODEL`
- `DIARIZATION_NUM_SPEAKERS`
- `JOB_POLL_INTERVAL_SECONDS`
- `ENABLE_MOCK_TRANSCRIPTION`

## Speaker Diarization

Speaker diarization is optional and off by default. To enable it:

```bash
pip install -e '.[diarization]'
export HUGGINGFACE_TOKEN=your_token_here
export DIARIZATION_NUM_SPEAKERS=2
python3 -m asr_viz.cli submit /absolute/path/to/file.mp4 --mime-type video/mp4 --diarization
python3 -m asr_viz.cli worker --once
```

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
