# ASR Expanse Frontend

This frontend reads transcript analysis from the existing FastAPI backend and projects it into a stacked 3D scene based on your p5 sketch logic.

## Data Pipeline

1. `GET /jobs?limit=50` finds the latest completed transcript unless `VITE_TRANSCRIPT_ID` is set.
2. `GET /transcripts/{id}` loads sentence units, transcript segments, and per-sentence analysis.
3. `buildSceneDocument()` derives a combined score for each sentence:
   `(1 - semantic_confidence_score + politeness_score + main_message_likelihood) / 3`
4. Sentences stay on the current plane while the score is `>= 5` on a 0-10 scale. If it drops below `5`, a new plane is pushed deeper into the stack and shrunk.
5. `ExpanseScene` renders those planes with centered sentence text and faint speaker traces.

## Visual Mapping

- plane break: combined score under the threshold
- plane depth: stack order
- plane scale: progressive shrink factor per stack
- text emphasis: newest sentence on the active plane is fully opaque
- persistent emphasis: sentences with `main_message_likelihood >= 0.9` remain bold and fully visible

## Run

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Start the backend separately with:

```bash
uvicorn asr_viz.api.main:app --reload
```
