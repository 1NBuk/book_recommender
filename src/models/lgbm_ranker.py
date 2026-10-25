
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.base import BaseRecommender


class LGBMRanker(BaseRecommender):
    def __init__(self, config):
        super().__init__(config, name="LGBM")
        self.model = None
        self.feature_columns = [
            "user_mean_rating", "user_interaction_count",
            "item_mean_rating", "item_interaction_count",
            "year", "average_rating", "ratings_count",
        ]
        self.user_features = None
        self.item_features = None
        self.metrics = {}

    def fit(self, interactions, items=None, users=None):
        if users is None or items is None:
            raise ValueError("LGBM requires users and items features")

        self.user_features = users
        self.item_features = items

        user_cols = ["user_id", "user_mean_rating", "user_interaction_count"]
        item_cols = ["item_id", "item_mean_rating", "item_interaction_count",
                     "year", "average_rating", "ratings_count"]

        available_user_cols = [c for c in user_cols if c in users.columns]
        available_item_cols = [c for c in item_cols if c in items.columns]

        df = interactions.merge(users[available_user_cols], on="user_id", how="left")
        df = df.merge(items[available_item_cols], on="item_id", how="left")

        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0
        df[self.feature_columns] = df[self.feature_columns].fillna(0)

        X = df[self.feature_columns]
        y = df["rating"].astype(float)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=self.config.get("runtime.seed", 42)
        )

        self.model = LGBMRegressor(
            n_estimators=self.config.get("models.lgbm.n_estimators", 500),
            learning_rate=self.config.get("models.lgbm.learning_rate", 0.03),
            num_leaves=self.config.get("models.lgbm.num_leaves", 31),
            random_state=self.config.get("runtime.seed", 42),
            verbose=-1,
        )
        self.model.fit(X_train, y_train)

        pred = self.model.predict(X_valid)
        self.metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_valid, pred))),
            "mae": float(mean_absolute_error(y_valid, pred)),
        }

        self.df = df
        self.is_fitted = True
        return self

    def score(self, user_id, item_ids):
        user_row = self.user_features[self.user_features["user_id"] == user_id]
        if user_row.empty:
            return np.zeros(len(item_ids))

        user_data = user_row.iloc[0]
        item_rows = self.item_features[self.item_features["item_id"].isin(item_ids)]

        if item_rows.empty:
            return np.zeros(len(item_ids))

        df = item_rows[["item_id"]].copy()
        df["user_mean_rating"] = user_data.get("user_mean_rating", 0)
        df["user_interaction_count"] = user_data.get("user_interaction_count", 0)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df[self.feature_columns] = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

        scores = self.model.predict(df[self.feature_columns])
        return scores

    def recommend(self, user_id, n=10, exclude_seen=True):
        user_row = self.user_features[self.user_features["user_id"] == user_id]
        if user_row.empty:
            return pd.DataFrame(columns=["user_id", "item_id", "score"])

        user_data = user_row.iloc[0]

        df = self.item_features.copy()
        df["user_mean_rating"] = user_data.get("user_mean_rating", 0)
        df["user_interaction_count"] = user_data.get("user_interaction_count", 0)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df[self.feature_columns] = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

        scores = self.model.predict(df[self.feature_columns])
        df["score"] = scores

        if exclude_seen:
            seen = set(self.df[self.df["user_id"] == user_id]["item_id"].unique())
            df = df[~df["item_id"].isin(seen)]

        result = df.nlargest(n, "score")[["item_id", "score"]].copy()
        result["user_id"] = user_id
        return result[["user_id", "item_id", "score"]]
