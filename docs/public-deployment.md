# Public Deployment Checklist

Use this checklist to deploy the MVP with:

- `Vercel` for the frontend
- `Render` web service for FastAPI
- `Render` background worker for transcription and diarization
- `Neon Postgres` for the database
- `Clerk` for authentication
- `Cloudflare R2` private bucket for uploads
- `Hugging Face` token for diarization

This version assumes:

- anonymous-cookie-first access for MVP
- no required account or login before upload/review/playback
- optional Clerk support later without redesigning ownership

## Secret Handling Rules

- [ ] Do not commit real secrets to GitHub.
- [ ] Do not hardcode secret values in source files.
- [ ] Set production secrets only in the Vercel, Render, Clerk, Neon, Cloudflare, and Hugging Face dashboards.
- [ ] Keep only example placeholder values in `.env.local.example` and `frontend/.env.example`.
- [ ] Use `.env.local` and `frontend/.env` only for local development.

## MVP Defaults

- [ ] Max single upload is `100 MB`.
- [ ] Per-user hard quota is `500 MB`.
- [ ] Diarization is enabled by default.
- [ ] Live mode stays hidden behind `LIVE_MODE_ENABLED=false`.
- [ ] Background job states are `queued`, `processing`, `completed`, and `failed`.

## 1. Neon Postgres

- [ ] Create a Neon project and database.
- [ ] Copy the pooled or direct Postgres connection string.
- [ ] Save it for Render as `DATABASE_URL`.
- [ ] If Neon gives you a `postgresql://...` or `postgres://...` URL, that is fine. The app now normalizes it to SQLAlchemy's `postgresql+psycopg://...` form internally.
- [ ] Do not commit the Postgres URL to GitHub.

## 2. Clerk

- [ ] Create a Clerk application.
- [ ] Save the frontend publishable key for Vercel as `VITE_CLERK_PUBLISHABLE_KEY` for later optional login.
- [ ] Save the Clerk issuer URL for Render as `CLERK_ISSUER_URL` for later optional bearer-token verification.
- [ ] If you use audience validation, set `CLERK_AUDIENCE`.
- [ ] If you want to pin the JWKS endpoint, set `CLERK_JWKS_URL`.
- [ ] Later, when custom domains are live, add the production frontend URL to Clerk allowed origins and redirect URLs.

## 3. Cloudflare R2

- [ ] Create a private-only R2 bucket.
- [ ] Record the bucket name for `R2_BUCKET_NAME`.
- [ ] Record the account ID for `R2_ACCOUNT_ID`.
- [ ] Create an access key pair.
- [ ] Save the access key ID as `R2_ACCESS_KEY_ID`.
- [ ] Save the secret access key as `R2_SECRET_ACCESS_KEY`.
- [ ] Do not expose the bucket publicly.
- [ ] Plan to use presigned upload and download URLs only.

## 4. Hugging Face

- [ ] Create a Hugging Face access token with access to the diarization model you intend to use.
- [ ] If you use `pyannote/speaker-diarization-3.1`, sign in to Hugging Face with the same account and explicitly accept access to that gated repository before deploying the token.
- [ ] Save it in Render as `HUGGINGFACE_TOKEN`.
- [ ] Do not commit the token to GitHub.

## 5. Render Web Service

- [ ] Create a Render web service for the FastAPI app.
- [ ] Configure dashboard environment variables only.
- [ ] Make sure the build installs the app dependencies from `pyproject.toml`, including the Postgres driver.
- [ ] Set `APP_ENV=production`.
- [ ] Set `DATABASE_URL`.
- [ ] Set `AUTO_CREATE_SCHEMA=false`.
- [ ] Set `AUTH_PROVIDER=clerk` if you want optional Clerk token support later, otherwise `disabled`.
- [ ] Set `REQUIRE_AUTH=false` for anonymous-cookie-first MVP access.
- [ ] Set `ANONYMOUS_SESSION_COOKIE_NAME=asr_viz_anon_session`.
- [ ] Set `ANONYMOUS_SESSION_COOKIE_MAX_AGE_SECONDS=15552000`.
- [ ] Set `ANONYMOUS_SESSION_COOKIE_SECURE=true`.
- [ ] Set `ANONYMOUS_SESSION_COOKIE_DOMAIN` only after you move to a stable custom domain strategy.
- [ ] Set `CLERK_ISSUER_URL` if `AUTH_PROVIDER=clerk`.
- [ ] Set `CLERK_AUDIENCE` if used.
- [ ] Set `CLERK_JWKS_URL` if used.
- [ ] Set `CORS_ORIGINS` to the current Vercel preview URL, and later `https://app.mydomain.com`.
- [ ] Set `ENABLE_MOCK_TRANSCRIPTION=false`.
- [ ] Set `ASR_MODEL=small`.
- [ ] Set `DIARIZATION_MIN_SPEAKERS=1`.
- [ ] Set `DIARIZATION_MAX_SPEAKERS=3`.
- [ ] Set `MAX_UPLOAD_SIZE_BYTES=104857600`.
- [ ] Set `PER_USER_STORAGE_QUOTA_BYTES=524288000`.
- [ ] Set `LIVE_MODE_ENABLED=false`.
- [ ] Set `STORAGE_BACKEND=r2` when the R2 upload flow is enabled in code.
- [ ] Set `SIGNED_URL_TTL_SECONDS=900`.
- [ ] Set `R2_ACCOUNT_ID`.
- [ ] Set `R2_BUCKET_NAME`.
- [ ] Set `R2_ACCESS_KEY_ID`.
- [ ] Set `R2_SECRET_ACCESS_KEY`.
- [ ] Set `HUGGINGFACE_TOKEN`.
- [ ] Run database migrations before serving production traffic:

```bash
PYTHONPATH=src alembic upgrade head
```

## 6. Render Worker

- [ ] Create a separate Render background worker service.
- [ ] Use the same codebase as the API.
- [ ] Configure the worker with dashboard environment variables only.
- [ ] Make sure the build installs the app dependencies from `pyproject.toml`, including the Postgres driver.
- [ ] Copy the same runtime env vars used by the API:
  `APP_ENV`, `DATABASE_URL`, `AUTH_PROVIDER`, `REQUIRE_AUTH`, `ENABLE_MOCK_TRANSCRIPTION`,
  `ASR_MODEL`, `DIARIZATION_MIN_SPEAKERS`, `DIARIZATION_MAX_SPEAKERS`,
  `MAX_UPLOAD_SIZE_BYTES`, `PER_USER_STORAGE_QUOTA_BYTES`, `LIVE_MODE_ENABLED`,
  `STORAGE_BACKEND`, `SIGNED_URL_TTL_SECONDS`, `R2_ACCOUNT_ID`, `R2_BUCKET_NAME`,
  `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `HUGGINGFACE_TOKEN`.
- [ ] Keep the worker CPU-only for MVP.
- [ ] Keep the API and worker independent so a future GPU worker can replace the current worker path.

## 7. Vercel Frontend

- [ ] Create a Vercel project for the `frontend` app.
- [ ] Configure dashboard environment variables only.
- [ ] Set `VITE_ASR_API_BASE_URL` to the Render API URL.
- [ ] Set `VITE_REQUIRE_AUTH=false` for anonymous-cookie-first MVP access.
- [ ] Set `VITE_CLERK_PUBLISHABLE_KEY` only when optional Clerk login is added to the frontend.
- [ ] Set `VITE_UPLOAD_CHUNK_SIZE_BYTES=2097152` unless you intentionally change it.
- [ ] Use the Vercel preview URL for early testing.
- [ ] Later, switch to `app.mydomain.com`.

## 8. Domain Cutover Later

- [ ] Point `app.mydomain.com` to Vercel.
- [ ] Point `api.mydomain.com` to Render.
- [ ] Update `CORS_ORIGINS` to include the final frontend URL.
- [ ] Update `ANONYMOUS_SESSION_COOKIE_DOMAIN` if you want a stricter shared cookie strategy under your final domain.
- [ ] Update Clerk allowed origins and redirect URLs to the final frontend URL.

## 9. Current Status Notes

- [ ] Owner scoping is already implemented for jobs, transcripts, stream sessions, and live sessions.
- [ ] Anonymous session ownership is already implemented with `HttpOnly` cookies.
- [ ] Local development still works with `REQUIRE_AUTH=false`.
- [ ] Upload size and per-user quota enforcement are already implemented.
- [ ] Alembic scaffolding is already in the repo.
- [ ] Full R2-backed upload and presigned URL flow is still a remaining implementation step.
- [ ] Clerk-aware frontend session wiring is still a remaining implementation step.
