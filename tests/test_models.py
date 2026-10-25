
import os
import sys
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.models.popularity import PopularityRecommender
from src.models.base import BaseRecommender


class DummyConfig:
    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        if key in self.data:  # flat keys like "models.popularity.min_ratings"
            return self.data[key]
        keys = key.split(".")
        value = self.data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


@pytest.fixture
def sample_interactions():
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
        "item_id": ["i1", "i2", "i3", "i1", "i2", "i1"],
        "rating": [5, 8, 3, 7, 6, 9],
        "interaction": [1, 1, 1, 1, 1, 1],
        "is_explicit": [1, 1, 1, 1, 1, 1],
    })


def test_popularity_fit(sample_interactions):
    cfg = DummyConfig({"models.popularity.min_ratings": 1})
    model = PopularityRecommender(cfg)
    model.fit(sample_interactions)

    assert model.is_fitted
    assert len(model.popular_items) > 0
    assert "score" in model.popular_items.columns


def test_popularity_recommend(sample_interactions):
    cfg = DummyConfig({"models.popularity.min_ratings": 1})
    model = PopularityRecommender(cfg)
    model.fit(sample_interactions)

    recs = model.recommend("u1", n=2)
    assert len(recs) <= 2
    assert "user_id" in recs.columns
    assert "item_id" in recs.columns
    assert "score" in recs.columns
    assert (recs["user_id"] == "u1").all()


def test_popularity_exclude_seen(sample_interactions):
    cfg = DummyConfig({"models.popularity.min_ratings": 1})
    model = PopularityRecommender(cfg)
    model.fit(sample_interactions)

    recs = model.recommend("u1", n=10, exclude_seen=True)
    seen = {"i1", "i2", "i3"}
    assert not any(item in seen for item in recs["item_id"])


def test_recommender_repr(sample_interactions):
    cfg = DummyConfig({"models.popularity.min_ratings": 1})
    model = PopularityRecommender(cfg)
    repr_str = repr(model)
    assert "Popularity" in repr_str
    assert "fitted=False" in repr_str

    model.fit(sample_interactions)
    repr_str = repr(model)
    assert "fitted=True" in repr_str


def test_base_recommender_is_abstract():
    with pytest.raises(TypeError):
        BaseRecommender(DummyConfig({}))
