
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, config):
        self.config = config
        self.test_size = config.get("data.test_size", 0.2)
        self.val_size = config.get("data.val_size", 0.1)
        self.random_state = config.get("data.random_state", 42)
        self.strategy = config.get("data.split_strategy", "random")

    def split(self, interactions):
        if self.strategy == "random":
            return self._random_split(interactions)
        if self.strategy == "time_aware":
            return self._time_aware_split(interactions)
        raise ValueError(f"Unknown split strategy: {self.strategy}")

    def _random_split(self, interactions):
        train_val, test = train_test_split(
            interactions,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        val_ratio = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=self.random_state,
        )

        return {
            "train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }

    def _time_aware_split(self, interactions):
        if "timestamp" not in interactions.columns:
            raise ValueError("time_aware split requires timestamp column")

        interactions = interactions.sort_values("timestamp").reset_index(drop=True)
        n = len(interactions)
        test_start = int(n * (1 - self.test_size))
        val_start = int(test_start * (1 - self.val_size))

        return {
            "train": interactions.iloc[:val_start].reset_index(drop=True),
            "val": interactions.iloc[val_start:test_start].reset_index(drop=True),
            "test": interactions.iloc[test_start:].reset_index(drop=True),
        }
