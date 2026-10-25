
import pandas as pd
import numpy as np


class UserFeatureBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, train_interactions):
        grouped = train_interactions.groupby("user_id")

        features = grouped.agg(
            user_interaction_count=("item_id", "count"),
            user_mean_rating=("rating", "mean"),
            user_std_rating=("rating", "std"),
            user_min_rating=("rating", "min"),
            user_max_rating=("rating", "max"),
            user_unique_items=("item_id", "nunique"),
            user_explicit_count=("is_explicit", "sum"),
        ).reset_index()

        features["user_std_rating"] = features["user_std_rating"].fillna(0)
        features["user_explicit_ratio"] = (
            features["user_explicit_count"] / features["user_interaction_count"]
        )

        features["user_activity_level"] = pd.cut(
            features["user_interaction_count"],
            bins=[0, 5, 20, 100, np.inf],
            labels=["cold", "low", "medium", "high"],
        ).astype(str)

        return features
