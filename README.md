# NLP Music Recommendation Engine

Music recommendation system using semantic embeddings and sentiment analysis, with an optional
RAG layer that answers natural-language queries about music.

## Quick Start

### 1. Install dependencies

Use the project venv (Python 3.14) or create one:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` pulls in the CPU-only PyTorch wheel, so the image stays lean.

### 2. Generate / obtain the data files

The engine needs two data files (both gitignored — not committed):

- `songs_with_mood.parquet` — song catalog with mood tags
- `embeddings.npy` — precomputed 384-D song embeddings

Run `notebook.ipynb` to build them from scratch, or download the prebuilt release:

```bash
gh release download v0.1.0-data \
  --pattern "songs_with_mood.parquet" \
  --pattern "embeddings.npy"
```

### 3. Run the API

```bash
python api.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

> **`/ask` additionally requires a local Ollama server** with `llama3.2:3b` pulled
> (`ollama pull llama3.2:3b`). `/recommend` and `/health` work without it.

## API Endpoints

### `GET /health`

Liveness check. Returns `200` with `{"status": "healthy", "model_loaded": true}` once the model
is ready.

### `POST /recommend`

Returns ranked songs as raw JSON.

**Request:**
```json
{
  "query": "sad jazz music",
  "top_k": 7,
  "filter_mood": true
}
```

**Response:**
```json
[
  {
    "title": "Blue in Green",
    "artist": "Bill Evans",
    "year": 1959,
    "mood": "sad",
    "scores": 0.85
  }
]
```

### `POST /ask`

RAG endpoint — retrieves matching songs, then generates a natural-language answer via
`llama3.2:3b`. Requires Ollama.

**Request:**
```json
{
  "query": "recommend some upbeat pop music",
  "top_k": 7,
  "filter_mood": true
}
```

**Response:**
```json
{
  "answer": "Here are a few upbeat pop tracks…",
  "songs": [
    { "title": "…", "artist": "…", "year": 2010, "mood": "happy", "score": 0.81 }
  ]
}
```

## How It Works

1. **Embedding Generation**: Uses `all-MiniLM-L6-v2` to generate 384-D semantic embeddings for
   all songs.
2. **Mood Classification**: VADER sentiment classifies songs as happy (score > 0.3), sad
   (< -0.3), or neutral.
3. **Query Processing**: Encodes the user query into an embedding.
4. **Similarity Search**: Computes cosine similarity between query and song embeddings.
5. **Filtering & Ranking**: Optionally filters by mood, returns top-k results sorted by similarity.
6. **RAG (optional, `/ask`)**: A LangChain `BaseRetriever` wraps the recommender, formats the
   retrieved songs into a prompt, and generates an answer via `ChatOllama`.

## Performance

- **`/recommend` response time**: ~90 ms average
- **`/ask` response time**: ~3–4 s (LLM generation on CPU)
- **Dataset**: 30,000 songs
- **Embedding dimension**: 384
- **Model**: `all-MiniLM-L6-v2`

## Example Queries

- "upbeat pop music"
- "sad jazz ballads"
- "energetic rock songs"
- "romantic slow songs"
- "motivational hip hop"

## Configuration

Adjustable parameters in `recommendation_engine.py`:

```python
HAPPY_THRESHOLD = 0.3       # Sentiment score threshold for happy mood
SAD_THRESHOLD = -0.3        # Sentiment score threshold for sad mood
MODEL_NAME = 'all-MiniLM-L6-v2'  # Sentence transformer model
```

The RAG model is set in `rag_engine.py` (`ChatOllama(model="llama3.2:3b", temperature=0)`).

## Docker

A multi-stage `Dockerfile` pre-caches the sentence-transformer weights and runs a single
gunicorn worker (to respect Render's memory limits):

```bash
docker build -t music-discovery-engine .
docker run -p 10000:10000 music-discovery-engine
```

## Load Testing

Locust tasks are tagged so the endpoints can be targeted independently. Start the API first,
then:

```bash
# Both endpoints (web UI at http://localhost:8089)
locust -f locustfile.py --host http://localhost:8000

# Only /recommend (weight 3)
locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 --run-time 30s --tags recommend

# Only /ask (weight 1) — needs Ollama running
locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 --run-time 30s --tags ask
```

## Troubleshooting

- **"embeddings.npy not found"**: Run `notebook.ipynb` to generate embeddings, or download the
  `v0.1.0-data` release.
- **"songs_with_mood.parquet not found"**: Run the mood-tagging section in `notebook.ipynb`.
- **`/ask` returns an error**: Ensure Ollama is running and `llama3.2:3b` is pulled.
- **Slow API responses**: The model loads once at import; confirm embeddings are memory-mapped and
  check system resources.

## References

- [Sentence-BERT](https://www.sbert.net/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)

## Author

Sneh Prince Herenj | CS37034 (Sem 4)
