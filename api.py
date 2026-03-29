from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from recommendation_engine import recommend

app = FastAPI(title="Song Recommendation API", version="1.0.0")

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 7
    filter_mood: bool = True

@app.post("/recommend")
def get_recommendation(request: RecommendRequest):
    results = recommend(
        query=request.query,
        top_k=request.top_k,
        filter_mood=request.filter_mood
    )
    return results.to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)