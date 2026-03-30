# NLP Music Recommendation Engine

Music recommendation system using semantic embeddings and sentiment analysis.

## Quick Start

### Installation
```bash
pip install fastapi uvicorn sentence-transformers pandas numpy scikit-learn vaderSentiment pyarrow
```

### Run API
```bash
python api.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoint

**POST** `/recommend`

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

## How It Works

1. **Embedding Generation**: Uses all-MiniLM-L6-v2 model to generate 384-D semantic embeddings for all songs
2. **Mood Classification**: VADER sentiment analyzer classifies songs as happy (score > 0.3), sad (< -0.3), or neutral
3. **Query Processing**: Encodes user query into an embedding
4. **Similarity Search**: Computes cosine similarity between query and song embeddings
5. **Filtering & Ranking**: Optionally filters by mood, returns top-k results sorted by similarity

## Performance

- **Response Time**: < 100ms average
- **Dataset**: ~30,000 songs
- **Embedding Dimension**: 384
- **Model**: all-MiniLM-L6-v2

## Example Queries

- "upbeat pop music"
- "sad jazz ballads"
- "energetic rock songs"
- "romantic slow songs"
- "motivational hip hop"

## Configuration

Adjustable parameters in `recommendation_engine.py`:
```python
HAPPY_THRESHOLD = 0.3      # Sentiment score threshold for happy mood
SAD_THRESHOLD = -0.3       # Sentiment score threshold for sad mood
MODEL_NAME = 'all-MiniLM-L6-v2'  # Sentence transformer model
```

## Troubleshooting

- **"embeddings.npy not found"**: Run `notebook.ipynb` to generate embeddings
- **"songs_with_mood.parquet not found"**: Run the mood tagging section in `notebook.ipynb`
- **Slow API responses**: Ensure embeddings are loaded in memory and check system resources

## References

- [Sentence-BERT](https://www.sbert.net/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [FastAPI](https://fastapi.tiangolo.com/)

## Author

Sneh Prince Herenj | CS37034 (Sem 4)
