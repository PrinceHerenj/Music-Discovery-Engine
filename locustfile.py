from locust import HttpUser, task, between, tag


class MusicRecommendUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    @tag("recommend")
    def recommend(self):
        self.client.post(
            "/recommend",
            json={"query": "sad jazz music", "top_k": 7, "filter_mood": True},
        )

    @task(1)
    @tag("ask")
    def ask(self):
        self.client.post(
            "/ask",
            json={"query": "upbeat pop music", "top_k": 7, "filter_mood": True},
        )
