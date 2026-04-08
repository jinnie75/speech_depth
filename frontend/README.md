# ASR Viz Frontend

React + Vite frontend for uploading audio, reviewing transcripts, and playing back processed conversations from the FastAPI backend.

## Development

```bash
npm install
npm run dev
```

The app expects the API at `http://127.0.0.1:8000` by default. Override it with `VITE_ASR_API_BASE_URL` when needed.

## Build

```bash
npm run build
```

The build now type-checks without emitting legacy `vite.config.js` or `vite.config.d.ts` artifacts before producing the Vite bundle.
