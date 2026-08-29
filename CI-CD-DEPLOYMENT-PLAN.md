# CI/CD & Deployment Plan — NLP Music Recommendation API

> Target: **Render (Web Service)**. Why: the ML model must stay loaded in memory and warm;
> Render runs a single long-lived container (no cold-start model re-download), deploys straight
> from a connected GitHub repo, has a free tier, and is the least setup for a sem-4 project.
> Serverless (Vercel/Lambda) is a poor fit because every cold start would re-download the 90 MB model.

## Current state & blockers

- **Stack:** FastAPI + uvicorn, `sentence-transformers` (`all-MiniLM-L6-v2`), pandas, numpy,
  scikit-learn, VADER sentiment.
- **Runtime footprint (loaded at import):**
  - `songs_with_mood.parquet` — 61 MB
  - `embeddings.npy` — 44 MB
  - `SentenceTransformer('all-MiniLM-L6-v2')` — ~90 MB, downloaded on first run
- **Blocker 1 — git is 2 GB.** `embeddings.npy` (44M) and `songs_with_mood.parquet` (61M) are
  tracked in git. Bloats `.git` to ~2 GB; slows pushes; some platforms reject oversized repos.
- **Blocker 2 — no deploy plumbing.** No `requirements.txt`, `Dockerfile`, tests, or CI config.
- **Not needed at runtime:** `song_lyrics_preprocessed.parquet` (4.9 GB) is already gitignored —
  keep it out of the image.

## Phase 0 — Fix the git repo (do first, blocks everything)

```bash
git rm --cached embeddings.npy songs_with_mood.parquet
```

Add to `.gitignore`:

```gitignore
*.npy
*.parquet
```

Rewrite history to drop the blobs so `.git` shrinks to KBs:
- `git filter-repo --invert-paths --path embeddings.npy --path songs_with_mood.parquet`, **or**
- squash to a single fresh commit (fine this early in the project), **or**
- start a clean new repo.

## Phase 1 — Project plumbing

### 1a. `requirements.txt` (pinned)

```
fastapi==<latest>
uvicorn[standard]==<latest>
gunicorn==<latest>
sentence-transformers==<latest>
pandas==<latest>
numpy==<latest>
scikit-learn==<latest>
vaderSentiment==<latest>
pyarrow==<latest>
pytest==<latest>
httpx==<latest>
```

### 1b. `Dockerfile` (multi-stage)

- **Build stage:** install deps; **pre-download `all-MiniLM-L6-v2`** so the image already contains
  the model (no runtime download, no cold-start surprise). Copy `songs_with_mood.parquet` +
  `embeddings.npy` into the image.
- **Runtime stage:** `gunicorn -k uvicorn.workers.UvicornWorker api:app -b 0.0.0.0:$PORT`.
- Expected image size ~2 GB (sentence-transformers pulls in torch) — normal; Render handles it.

### 1c. `api.py` — add a health endpoint

Add `GET /health` returning `200` once the model is ready. Render uses this as the liveness check.

### 1d. `.gitignore`

```gitignore
*.npy
*.parquet
.idea/
.DS_Store
```

## Phase 2 — CI (GitHub Actions)

File: `.github/workflows/ci.yml`. On push/PR:

1. Install deps from `requirements.txt`.
2. Smoke test: boot the app (or `TestClient`) and assert `POST /recommend` returns 200 with a sane
   payload.
3. Build the Docker image and push to **GHCR** (GitHub Container Registry), tagged with the commit SHA.

## Phase 3 — CD (deploy to Render)

Option A (recommended): connect the repo to Render; Render builds from the `Dockerfile` and
auto-deploys on every push to `main`. CI just guards quality.

Option B: CI pushes the image to GHCR; Render pulls it via a webhook.

Result: `https://<app>.onrender.com/docs` → `POST /recommend`.

## Phase 4 — Verify

- Test the live endpoint with a `curl` example query.
- Confirm the model loads once at boot (Render logs) and responses stay < 100 ms.

## Out of scope (not recommended at this stage)

- Object storage for the data (overkill at 105 MB).
- GPU instances, load balancing, auth.

---

*Author: Sneh Prince Herenj | CS37034 (Sem 4)*
