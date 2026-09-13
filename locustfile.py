from locust import HttpUser, task, between


class MusicRecommendUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def recommend(self):
        self.client.post(
            "/recommend",
            json={"query": "sad jazz music", "top_k": 7, "filter_mood": True},
        )

    @task(1)
    def ask(self):
        self.client.post(
            "/ask",
            json={"query": "upbeat pop music", "top_k": 7, "filter_mood": True},
        )
