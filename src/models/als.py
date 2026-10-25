
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.models.base import BaseRecommender


class ALSRecommender(BaseRecommender):
    def __init__(self, config):
        super().__init__(config, name="ALS")
        self.model = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_items_matrix = None

    def fit(self, interactions, items=None, users=None):
        from implicit.als import AlternatingLeastSquares

        factors = self.config.get("models.als.factors", 64)
        regularization = self.config.get("models.als.regularization", 0.1)
        iterations = self.config.get("models.als.iterations", 30)
        random_state = self.config.get("runtime.seed", 42)

        user_ids = interactions["user_id"].unique()
        item_ids = interactions["item_id"].unique()

        self.user_to_idx = {u: i for i, u in enumerate(user_ids)}
        self.item_to_idx = {it: i for i, it in enumerate(item_ids)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        rows = interactions["user_id"].map(self.user_to_idx).values
        cols = interactions["item_id"].map(self.item_to_idx).values
        values = interactions["rating"].astype(float).values

        self.user_items_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_to_idx), len(self.item_to_idx)),
        )

        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self.model.fit(self.user_items_matrix)
        self.is_fitted = True
        return self

    def recommend(self, user_id, n=10, exclude_seen=True):
        if user_id not in self.user_to_idx:
            return pd.DataFrame(columns=["user_id", "item_id", "score"])

        user_idx = self.user_to_idx[user_id]
        ids, scores = self.model.recommend(
            user_idx,
            self.user_items_matrix[user_idx],
            N=n,
            filter_already_liked_items=exclude_seen,
        )

        result = pd.DataFrame({
            "user_id": user_id,
            "item_id": [self.idx_to_item[i] for i in ids],
            "score": scores,
        })
        return result
