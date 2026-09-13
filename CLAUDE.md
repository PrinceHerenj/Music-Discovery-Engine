# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NLP music recommendation API. Songs are embedded with `all-MiniLM-L6-v2` (384-D) and matched by cosine similarity to an encoded query; VADER sentiment scores a query's mood (happy/sad/neutral) to optionally filter results.

## Commands

- **Run the API** (dev server, `http://localhost:8000`, interactive docs at `/docs`):
  ```bash
  python api.py
  ```
  Use the project venv (`.venv/`, Python 3.14).
- **Install deps**: `pip install -r requirements.txt`.
- **Load test** (server must be running):
  ```bash
  locust -f locustfile.py --host http://localhost:8000            # web UI at http://localhost:8089
  locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 --run-time 30s
  ```
- **Generate data**: run `notebook.ipynb` to (re)produce `songs_with_mood.parquet` and `embeddings.npy` (both gitignored).
- No pytest suite is present; CI (`.github/workflows/ci.yml`) builds/pushes a Docker image and runs the Locust load test.

## Architecture

Two runtime modules plus a notebook pipeline:

- **`recommendation_engine.py`** — the whole engine. On **import**, it eagerly loads three large objects into memory and never unloads them:
  - `songs_with_mood.parquet` (~61 MB DataFrame of songs)
  - `embeddings.npy` (~44 MB precomputed song embeddings, memory-mapped)
  - `SentenceTransformer('all-MiniLM-L6-v2')` (~90 MB, downloaded on first run if not cached)

  Exposes `recommend(query, top_k=7, filter_mood=True)` (encodes query, cosine-scores, optional mood filter, returns sorted rows) and `infer_mood(query)`. Tunables `HAPPY_THRESHOLD`/`SAD_THRESHOLD`/`MODEL_NAME` are module-level constants.

- **`rag_engine.py`** — LangChain RAG layer (Augmentation + Generation) on top of the existing retrieval. `SongRetriever(BaseRetriever)` wraps `recommend()` and returns `Document`s (song facts in `page_content`, structured fields in `metadata`). `ask(query, top_k, filter_mood)` runs the full R→A→G flow: retrieve, format docs into a system prompt, generate an answer via `ChatOllama(model="llama3.2:3b")`, returning `(answer, [song metadata])`. Uses only stable `langchain-core` primitives (`BaseRetriever`, `ChatPromptTemplate`, `ChatOllama`) — the legacy `langchain.chains` (`create_retrieval_chain`/`create_stuff_documents_chain`) was removed in LangChain 1.x, so the chain is hand-wired in `ask()`.

- **`api.py`** — FastAPI app. `POST /recommend` returns raw JSON rows via `recommend()`; `POST /ask` returns `{"answer": ..., "songs": [...]}` via `ask()`.

- **`notebook.ipynb`** — offline pipeline that builds the `.parquet` (mood tagging) and `.npy` (embeddings). Re-run it if the data files are missing.

- **`locustfile.py`** — Locust `HttpUser` with two tasks: `POST /recommend` (weight 3) and `POST /ask` (weight 1).

## Key constraints

- **Data files are not committed** (`*.npy`, `*.parquet` gitignored) and must exist locally for the API to import — `api.py`/`recommendation_engine.py` crash on startup if `embeddings.npy` or `songs_with_mood.parquet` is absent. CI downloads them from the `v0.1.0-data` GitHub release before building.
- **Model loads at import time**, so every cold start re-downloads `all-MiniLM-L6-v2`; the single long-lived Render web-service container (one gunicorn worker) exists for this reason over serverless.
- **`/ask` depends on a running local Ollama server** with `llama3.2:3b` pulled (`ollama pull llama3.2:3b`). It is ~3–4 s per request on CPU; the README "<100ms" claim applies to `/recommend` only (~90 ms). The Locust load test's `/ask` task will fail without Ollama provisioned.
- The `songs_with_mood.parquet` and `embeddings.npy` files tracked in git bloated `.git` to ~2 GB; a plan to `git rm --cached` them and rewrite history was drafted but not fully executed.
