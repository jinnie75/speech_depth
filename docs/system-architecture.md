# System Architecture

This diagram reflects the current implementation in the repository across the frontend, FastAPI backend, background worker, storage layer, and model/provider integrations.

## Runtime Architecture

```mermaid
flowchart LR
    user([User])

    subgraph client["Client Layer"]
        browser["Browser UI<br/>React 19 + Vite<br/>frontend/src/app/App.tsx"]
        cli["Local CLI<br/>submit/status/transcript/worker<br/>src/asr_viz/cli.py"]
    end

    subgraph api["API Layer"]
        fastapi["FastAPI app<br/>job, transcript, media, review,<br/>stream-session endpoints<br/>src/asr_viz/api/main.py"]
        streamsvc["Streaming ingestion service<br/>create / append chunks / finalize<br/>src/asr_viz/services/streaming.py"]
        jobsvc["Job creation service<br/>src/asr_viz/services/jobs.py"]
    end

    subgraph worker["Background Processing"]
        workerloop["Worker loop<br/>polls DB queue<br/>src/asr_viz/worker.py"]
        pipeline["ProcessingPipeline<br/>claim -> transcribe -> diarize -> analyze -> persist<br/>src/asr_viz/services/pipeline.py"]
        mediasvc["Media resolver<br/>local file or URL download<br/>src/asr_viz/services/media.py"]
        segmenter["Sentence segmentation<br/>ASR segments -> sentence candidates<br/>src/asr_viz/pipeline/segmentation.py"]
    end

    subgraph providers["Provider Layer"]
        whisper["Transcription provider<br/>Faster Whisper or mock text reader<br/>src/asr_viz/providers/transcription.py"]
        pyannote["Optional diarization provider<br/>Pyannote or no-op<br/>src/asr_viz/providers/diarization.py"]
        analysis["Heuristic analysis provider<br/>hedging + substance extraction<br/>src/asr_viz/providers/analysis_v2.py"]
        lexicons["Local NLP resources<br/>.lexicons/nrc_emolex<br/>.nltk_data"]
    end

    subgraph storage["Persistence And Files"]
        db["SQLite / SQLAlchemy DB<br/>asr_viz.db"]
        media["Media storage<br/>.media/"]
        streamfiles["Chunk upload storage<br/>.media/stream_ingestion/"]
        remote["Remote media URL"]
    end

    user --> browser
    user --> cli

    browser -->|"GET /jobs, /transcripts/{id}, /jobs/{id}/media<br/>PATCH /transcripts/{id}/review"| fastapi
    browser -->|"POST /stream-sessions<br/>PUT /chunks<br/>POST /finalize"| fastapi
    cli -->|"submit job / inspect state / run worker"| fastapi
    cli -->|"direct DB-backed commands"| db
    cli -->|"optional local worker command"| workerloop

    fastapi --> streamsvc
    fastapi --> jobsvc
    fastapi <--> db
    fastapi --> media

    streamsvc <--> db
    streamsvc --> streamfiles
    streamsvc -->|"finalize creates MediaAsset + ProcessingJob"| jobsvc

    jobsvc <--> db

    workerloop -->|"claim queued job"| db
    workerloop --> pipeline
    pipeline <--> db
    pipeline --> mediasvc
    mediasvc --> media
    mediasvc --> remote

    pipeline --> whisper
    whisper -->|"ASR segments + word timestamps"| segmenter
    segmenter -->|"sentence candidates"| pipeline

    pipeline -->|"if diarization_enabled"| pyannote
    pyannote -->|"speaker-tagged sentences"| pipeline

    pipeline --> analysis
    analysis --> lexicons
    analysis -->|"analysis payload + scores"| pipeline

    pipeline -->|"persist transcript, segments,<br/>sentence units, analysis results"| db

    db -->|"completed jobs, transcript sentences,<br/>analysis payload, review metadata"| fastapi
    media -->|"playback asset"| fastapi
    fastapi -->|"JSON + media responses"| browser
```

## Core Persistence Model

```mermaid
erDiagram
    MEDIA_ASSETS ||--o{ PROCESSING_JOBS : "source for"
    MEDIA_ASSETS ||--o{ TRANSCRIPTS : "media for"
    PROCESSING_JOBS o|--|| TRANSCRIPTS : "produces"
    STREAM_INGESTION_SESSIONS o|--o| PROCESSING_JOBS : "finalizes into"
    TRANSCRIPTS ||--o{ TRANSCRIPT_SEGMENTS : "contains"
    TRANSCRIPTS ||--o{ SENTENCE_UNITS : "contains"
    SENTENCE_UNITS ||--o| ANALYSIS_RESULTS : "has"

    MEDIA_ASSETS {
        string id PK
        string source_type
        string source_uri
        string mime_type
        string checksum
        json ingest_metadata
    }

    PROCESSING_JOBS {
        string id PK
        string media_asset_id FK
        string transcript_id FK
        string status
        string current_stage
        boolean diarization_enabled
        string asr_model_version
        string diarization_model_version
        string analysis_model_version
        json stage_details
    }

    STREAM_INGESTION_SESSIONS {
        string id PK
        string status
        string storage_path
        int total_bytes
        int received_chunks
        string processing_job_id FK
    }

    TRANSCRIPTS {
        string id PK
        string media_asset_id FK
        string job_id FK
        string language_code
        text full_text
        json transcript_metadata
    }

    TRANSCRIPT_SEGMENTS {
        string id PK
        string transcript_id FK
        int segment_index
        int start_ms
        int end_ms
        text text
        json words_json
    }

    SENTENCE_UNITS {
        string id PK
        string transcript_id FK
        int utterance_index
        int start_ms
        int end_ms
        text text
        string speaker_id
        float speaker_confidence
        json sentence_metadata
    }

    ANALYSIS_RESULTS {
        string id PK
        string sentence_unit_id FK
        float politeness_score
        float semantic_confidence_score
        float main_message_likelihood
        json analysis_payload
    }
```

## Notes

- The frontend is a separate Vite app that consumes the backend API; FastAPI is not serving the SPA bundle in this repository.
- Streaming ingestion is upload-and-finalize, not live incremental ASR.
- Diarization is conditional: the worker uses Pyannote only when diarization is enabled and a Hugging Face token is configured.
- URL-based media sources are downloaded into `.media/` on demand before transcription.
