
import pandas as pd
import numpy as np


class ItemFeatureBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, train_interactions, items):
        grouped = train_interactions.groupby("item_id")

        stats = grouped.agg(
            item_interaction_count=("user_id", "count"),
            item_mean_rating=("rating", "mean"),
            item_std_rating=("rating", "std"),
            item_unique_users=("user_id", "nunique"),
            item_explicit_count=("is_explicit", "sum"),
        ).reset_index()

        stats["item_std_rating"] = stats["item_std_rating"].fillna(0)

        stats["item_popularity_level"] = pd.cut(
            stats["item_interaction_count"],
            bins=[0, 5, 20, 100, np.inf],
            labels=["niche", "low", "medium", "popular"],
        ).astype(str)

        meta_columns = [
            "item_id", "isbn", "title", "author", "year", "publisher",
            "tags", "average_rating", "ratings_count", "language", "content",
        ]
        available = [c for c in meta_columns if c in items.columns]

        merged = stats.merge(items[available], on="item_id", how="left")

        for col in ["year", "average_rating", "ratings_count"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        return merged

    def build_content_corpus(self, items):
        corpus = (
            items["title"].fillna("") + " " +
            items["author"].fillna("") + " " +
            items["tags"].fillna("") + " " +
            items["publisher"].fillna("")
        )
        return corpus.str.strip().str.lower()
