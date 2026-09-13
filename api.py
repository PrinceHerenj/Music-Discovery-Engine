from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

torch.set_grad_enabled(False)

from recommendation_engine import recommend
from rag_engine import ask

app = FastAPI(title="Song Recommendation API", version="1.0.0")

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 7
    filter_mood: bool = True

class AskRequest(BaseModel):
    query: str
    top_k: int = 7
    filter_mood: bool = True

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "model" in globals()
    }

@app.post("/recommend")
def get_recommendation(request: RecommendRequest):
    results = recommend(
        query=request.query,
        top_k=request.top_k,
        filter_mood=request.filter_mood
    )
    return results.to_dict(orient="records")

@app.post("/ask")
def get_answer(request: AskRequest):
    answer, songs = ask(
        query=request.query,
        top_k=request.top_k,
        filter_mood=request.filter_mood
    )
    return {"answer": answer, "songs": songs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
