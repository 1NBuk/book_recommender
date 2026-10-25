
import numpy as np
import pandas as pd
from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def __init__(self, config):
        super().__init__(config, name="Popularity")

    def fit(self, interactions, items=None, users=None):
        min_ratings = self.config.get("models.popularity.min_ratings", 5)

        grouped = interactions[interactions["rating"] > 0].groupby("item_id").agg(
            mean_rating=("rating", "mean"),
            rating_count=("rating", "count"),
        ).reset_index()

        grouped = grouped[grouped["rating_count"] >= min_ratings]
        grouped["score"] = grouped["mean_rating"] * np.log1p(grouped["rating_count"])
        grouped = grouped.sort_values("score", ascending=False).reset_index(drop=True)

        self.popular_items = grouped
        self.seen_by_user = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        self.is_fitted = True
        return self

    def recommend(self, user_id, n=10, exclude_seen=True):
        candidates = self.popular_items
        if exclude_seen and user_id in self.seen_by_user:
            seen = self.seen_by_user[user_id]
            candidates = candidates[~candidates["item_id"].isin(seen)]

        result = candidates.head(n)[["item_id", "score"]].copy()
        result["user_id"] = user_id
        return result[["user_id", "item_id", "score"]]
