
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    def __init__(self, config, name=None):
        self.config = config
        self.name = name or self.__class__.__name__
        self.is_fitted = False

    @abstractmethod
    def fit(self, interactions, items=None, users=None):
        pass

    @abstractmethod
    def recommend(self, user_id, n=10, exclude_seen=True):
        pass

    def recommend_batch(self, user_ids, n=10, exclude_seen=True):
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recommend(user_id, n=n, exclude_seen=exclude_seen)
        return results

    def score(self, user_id, item_ids):
        raise NotImplementedError(f"{self.name} does not implement score()")

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)

    def __repr__(self):
        return f"{self.name}(fitted={self.is_fitted})"
