import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from sentence_transformers import SentenceTransformer

df = pd.read_parquet('./songs_with_mood.parquet')
embeddings = np.load('./embeddings.npy')
model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.metrics.pairwise import cosine_similarity

def infer_mood(query):
    score = analyzer.polarity_scores(query)['compound']
    if score > 0.3: return 'happy'
    elif score < -0.3: return 'sad'
    else: return 'neutral'

def recommend(query, top_k=7, filter_mood=True):
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]

    results = df.copy()
    results['scores'] = scores

    if filter_mood:
        mood = infer_mood(query)
        if mood:
            results = results[results['mood'] == mood]

    return results.sort_values('scores', ascending=False).head(top_k)[['title', 'artist', 'year', 'mood', 'scores']]
