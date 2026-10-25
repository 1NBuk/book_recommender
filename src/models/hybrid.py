
import pandas as pd
import numpy as np
from src.models.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    def __init__(self, config, models):
        super().__init__(config, name="Hybrid")
        self.models = models
        self.weights = config.get("models.hybrid.weights", {})
        self.seen_by_user = {}

    def fit(self, interactions, items=None, users=None):
        self.seen_by_user = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        self.is_fitted = True
        return self

    def _normalize(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return scores
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def recommend(self, user_id, n=10, exclude_seen=True):
        combined = {}

        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 0.0)
            if weight <= 0:
                continue

            try:
                recs = model.recommend(user_id, n=200, exclude_seen=exclude_seen)
            except Exception:
                continue

            if recs is None or recs.empty:
                continue

            norm_scores = self._normalize(recs["score"].values)
            for item_id, score in zip(recs["item_id"].values, norm_scores):
                combined[item_id] = combined.get(item_id, 0.0) + weight * score

        if not combined:
            return pd.DataFrame(columns=["user_id", "item_id", "score"])

        result = pd.DataFrame(
            list(combined.items()),
            columns=["item_id", "score"],
        )
        result = result.sort_values("score", ascending=False).head(n)
        result["user_id"] = user_id
        return result[["user_id", "item_id", "score"]]
