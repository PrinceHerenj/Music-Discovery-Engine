from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"

def test_recommend_endpoint():
    payload = {
        "query": "chill acoustic guitar",
        "top_k": 2
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, (list, dict))