
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    def __init__(self, config):
        super().__init__(config, name="ContentBased")
        self.item_tfidf = None
        self.items = None
        self.item_id_to_row = {}
        self.user_profile_cache = {}

    def fit(self, interactions, items=None, items_tfidf=None):
        if items is None or items_tfidf is None:
            raise ValueError("ContentBased requires items and items_tfidf")

        self.items = items.reset_index(drop=True)
        self.item_tfidf = items_tfidf
        self.item_id_to_row = {item_id: i for i, item_id in enumerate(self.items["item_id"])}

        self.user_seen = interactions.groupby("user_id")["item_id"].apply(list).to_dict()
        self.user_ratings = interactions.groupby("user_id").apply(
            lambda df: dict(zip(df["item_id"], df["rating"]))
        ).to_dict()
        self.is_fitted = True
        return self

    def _build_user_profile(self, user_id):
        if user_id in self.user_profile_cache:
            return self.user_profile_cache[user_id]

        if user_id not in self.user_ratings:
            return None

        ratings = self.user_ratings[user_id]
        rows = []
        weights = []
        for item_id, rating in ratings.items():
            if item_id in self.item_id_to_row and rating > 0:
                rows.append(self.item_id_to_row[item_id])
                weights.append(rating)

        if not rows:
            return None

        sub_matrix = self.item_tfidf[rows]
        weights = np.array(weights).reshape(-1, 1)
        profile = sub_matrix.multiply(weights).sum(axis=0)
        profile = np.asarray(profile).reshape(1, -1)

        self.user_profile_cache[user_id] = profile
        return profile

    def recommend(self, user_id, n=10, exclude_seen=True):
        profile = self._build_user_profile(user_id)
        if profile is None:
            return pd.DataFrame(columns=["user_id", "item_id", "score"])

        scores = cosine_similarity(profile, self.item_tfidf).flatten()

        if exclude_seen and user_id in self.user_seen:
            seen = set(self.user_seen[user_id])
            mask = self.items["item_id"].isin(seen).values
            scores[mask] = -np.inf

        top_idx = np.argsort(scores)[::-1][:n]
        result = pd.DataFrame({
            "user_id": user_id,
            "item_id": self.items.iloc[top_idx]["item_id"].values,
            "score": scores[top_idx],
        })
        return result
