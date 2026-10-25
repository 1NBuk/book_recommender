
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from src.models.base import BaseRecommender


class LGBMGroupRanker(BaseRecommender):
    def __init__(self, config):
        super().__init__(config, name="LGBMGroupRanker")
        self.model = None
        self.feature_columns = [
            "user_mean_rating", "user_interaction_count",
            "item_mean_rating", "item_interaction_count",
            "year", "average_rating", "ratings_count",
        ]
        self.user_features = None
        self.item_features = None
        self.item_stats = None
        self.metrics = {}

    def _build_training_set(self, interactions, users, items):
        user_cols = ["user_id", "user_mean_rating", "user_interaction_count"]
        item_cols = ["item_id", "item_mean_rating", "item_interaction_count",
                     "year", "average_rating", "ratings_count"]

        avail_user_cols = [c for c in user_cols if c in users.columns]
        avail_item_cols = [c for c in item_cols if c in items.columns]

        df = interactions.merge(users[avail_user_cols], on="user_id", how="left")
        df = df.merge(items[avail_item_cols], on="item_id", how="left")

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["relevance"] = (df["rating"] >= 7).astype(int)

        df = df.sort_values("user_id").reset_index(drop=True)

        return df

    def fit(self, interactions, items=None, users=None):
        if users is None or items is None:
            raise ValueError("LGBMGroupRanker requires user and item features")

        self.user_features = users
        self.item_features = items

        df = self._build_training_set(interactions, users, items)

        grouped = df.groupby("user_id")
        group_sizes = grouped.size().values
        users_with_multiple = grouped.filter(lambda g: len(g) >= 2)["user_id"].unique()

        df = df[df["user_id"].isin(users_with_multiple)].reset_index(drop=True)
        df = df.sort_values("user_id").reset_index(drop=True)

        group_sizes = df.groupby("user_id").size().values

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(splitter.split(df, groups=df["user_id"]))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        train_groups = train_df.groupby("user_id").size().values
        valid_groups = valid_df.groupby("user_id").size().values

        train_data = lgb.Dataset(
            train_df[self.feature_columns],
            label=train_df["relevance"],
            group=train_groups,
        )
        valid_data = lgb.Dataset(
            valid_df[self.feature_columns],
            label=valid_df["relevance"],
            group=valid_groups,
            reference=train_data,
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "boosting_type": "gbdt",
            "num_leaves": self.config.get("models.lgbm.num_leaves", 31),
            "learning_rate": self.config.get("models.lgbm.learning_rate", 0.03),
            "n_estimators": self.config.get("models.lgbm.n_estimators", 500),
            "min_data_in_leaf": 5,
            "verbose": -1,
            "random_state": self.config.get("runtime.seed", 42),
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=params["n_estimators"],
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        self.df = df
        self.is_fitted = True
        return self

    def recommend(self, user_id, n=10, exclude_seen=True):
        user_row = self.user_features[self.user_features["user_id"] == user_id]
        if user_row.empty:
            return pd.DataFrame(columns=["user_id", "item_id", "score"])

        user_data = user_row.iloc[0]

        df = self.item_features.copy()
        df["user_mean_rating"] = float(user_data.get("user_mean_rating", 0))
        df["user_interaction_count"] = float(user_data.get("user_interaction_count", 0))

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

    def score(self, user_id, item_ids):
        user_row = self.user_features[self.user_features["user_id"] == user_id]
        if user_row.empty:
            return np.zeros(len(item_ids))

        user_data = user_row.iloc[0]
        item_rows = self.item_features[self.item_features["item_id"].isin(item_ids)]

        if item_rows.empty:
            return np.zeros(len(item_ids))

        df = item_rows[["item_id"]].copy()
        df["user_mean_rating"] = float(user_data.get("user_mean_rating", 0))
        df["user_interaction_count"] = float(user_data.get("user_interaction_count", 0))

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df[self.feature_columns] = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

        return self.model.predict(df[self.feature_columns])
