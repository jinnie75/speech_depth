# Of Terrains We Speak

Of Terrains We Speak is an interactive conversation visualizer that reveals patterns of hedging, emotion, and vulnerability, and how different they can look across cultures, contexts, and relationships.

Each speaker is represented by different colored terrains that grows as the conversation progresses, the area proportional to how much they’ve spoken. Every time there is an expression of uncertainty or a hedge, a level of elevation is added to the topographic map. Expressions of emotions, vulnerability and reflections on life are drawn directly into part of the map. 

This Github Repo is for setting up a local application that runs on your computer.


# Backend

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

The setup script creates or reuses `.venv`, installs the backend with `faster-whisper` and pyannote `speaker-diarization` dependencies, and seeds `.env.local` and `frontend/.env` from their examples. It prompts for a Hugging Face token on first run for the diarization(= identification of speakers) stack. To use this model, the Hugging Face token needs to be granted access, for which you can follow steps on their [Hugging Face page](https://huggingface.co/pyannote/speaker-diarization-3.1). A valid token without model access will  leave the transcript completed, but speaker labels will be skipped and the job will record the diarization warning in `stage_details`.

## Speaker Diarization

Speaker diarization is controlled globally through `.env.local`. It runs for all jobs when `ENABLE_DIARIZATION=true` and a valid `HUGGINGFACE_TOKEN` is available. To disable it, set `ENABLE_DIARIZATION=false`.

Sentence units will then include `speaker_id` and `speaker_confidence` when the diarization model can assign them.

## Analysis

I took a rule-based approach to analysis, mainly looking for 'hedging' - word or phrase used in a sentence to express ambiguity, caution, or indecisiveness or what I defined as 'substance' - expressions of emotion, of what's expected of the speaker and how they think they're viewed by other people. The rules are defined in `./src/asr_viz/providers/analysis_v2.py`

# Frontend

Each speaker is represented by colored contour lines. 
- The number of contour lines grow when a sentence has non-empty `hedging` or `substance` data in `analysis_payload`.
- Speaker regions are sized by their respective total speaking duration.
- The active utterance is shown beneath the visualizer during playback.

Custom typeface is used: English uses a bundled handwriting-style font, and Korean is `야근하는 김주임체` from [NAVER CLOVA handwriting typefaces](https://clova.ai/handwriting/list.html).
