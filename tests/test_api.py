import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.main import create_app
from src.api.store import ArtifactStore


@pytest.fixture
def artifacts_dir(tmp_path):
    items = pd.DataFrame(
        {
            "item_id": ["b1", "b2", "b3", "b4"],
            "isbn": ["1", "2", "3", "4"],
            "title": ["One", "Two", "Three", "Four"],
            "author": ["A", "B", "C", "D"],
            "year": [2001, 1999, 3000, None],  # 3000 is junk and must become null
            "publisher": ["P"] * 4,
            "image_url": [None, "http://img/2.jpg", None, None],
            "average_rating": [4.5, None, 3.0, 4.0],
        }
    )
    recs = pd.DataFrame(
        {
            "user_id": ["u1"] * 4 + ["u2"],
            "item_id": ["b2", "b1", "ghost", "b3", "b4"],  # "ghost" has no metadata
            "rank": [0, 1, 2, 3, 0],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )
    popular = pd.DataFrame({"item_id": ["b4", "b3", "b1"], "score": [9.0, 8.0, 7.0]})
    items.to_parquet(tmp_path / "items.parquet", index=False)
    recs.to_parquet(tmp_path / "recommendations.parquet", index=False)
    popular.to_parquet(tmp_path / "popular.parquet", index=False)
    (tmp_path / "model_info.json").write_text(
        json.dumps({"model": "hybrid", "created_at": "2026-01-01T00:00:00+00:00", "metrics": {"ndcg@10": 0.05}})
    )
    return tmp_path


def make_client(artifacts_dir, **kw):
    settings = Settings(artifacts_dir=artifacts_dir, **kw)
    return TestClient(create_app(settings))


def test_health_and_ready(artifacts_dir):
    with make_client(artifacts_dir) as c:
        assert c.get("/health/live").json() == {"status": "ok"}
        assert c.get("/health/ready").json() == {"status": "ready"}


def test_personalized_recommendations(artifacts_dir):
    with make_client(artifacts_dir) as c:
        r = c.get("/v1/users/u1/recommendations", params={"n": 10})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "personalized"
        assert body["model"] == "hybrid"
        # "ghost" is dropped, order by rank preserved, ranks are 1-based and contiguous
        assert [i["item_id"] for i in body["items"]] == ["b2", "b1", "b3"]
        assert [i["rank"] for i in body["items"]] == [1, 2, 3]
        assert body["count"] == 3
        assert "x-request-id" in r.headers


def test_n_limits_results(artifacts_dir):
    with make_client(artifacts_dir) as c:
        body = c.get("/v1/users/u1/recommendations", params={"n": 2}).json()
        assert body["count"] == 2


def test_cold_start_fallback(artifacts_dir):
    with make_client(artifacts_dir) as c:
        body = c.get("/v1/users/new_user/recommendations", params={"n": 2}).json()
        assert body["source"] == "popular_fallback"
        assert [i["item_id"] for i in body["items"]] == ["b4", "b3"]


@pytest.mark.parametrize("n", [0, -1, 101, "abc"])
def test_invalid_n(artifacts_dir, n):
    with make_client(artifacts_dir) as c:
        assert c.get("/v1/users/u1/recommendations", params={"n": n}).status_code == 422


def test_invalid_user_id(artifacts_dir):
    with make_client(artifacts_dir) as c:
        assert c.get("/v1/users/bad%20id%0A/recommendations").status_code == 422


def test_book_lookup_and_cleaning(artifacts_dir):
    with make_client(artifacts_dir) as c:
        assert c.get("/v1/books/b3").json()["year"] is None
        assert c.get("/v1/books/b2").json()["image_url"] == "http://img/2.jpg"
        assert c.get("/v1/books/nope").status_code == 404


def test_popular_and_model(artifacts_dir):
    with make_client(artifacts_dir) as c:
        assert c.get("/v1/popular", params={"n": 2}).json()["count"] == 2
        info = c.get("/v1/model").json()
        assert info["n_users"] == 2 and info["n_items"] == 4 and info["metrics"]["ndcg@10"] == 0.05


def test_api_key_enforced(artifacts_dir):
    with make_client(artifacts_dir, api_key="secret") as c:
        assert c.get("/v1/popular").status_code == 401
        assert c.get("/v1/popular", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/popular", headers={"X-API-Key": "secret"}).status_code == 200
        # infra endpoints stay open for probes and Prometheus
        assert c.get("/health/live").status_code == 200
        assert c.get("/metrics").status_code == 200


def test_metrics_exposed(artifacts_dir):
    with make_client(artifacts_dir) as c:
        c.get("/v1/users/u1/recommendations")
        c.get("/v1/users/zzz/recommendations")
        text = c.get("/metrics").text
        assert "bookrec_http_requests_total" in text
        assert 'path="/v1/users/{user_id}/recommendations"' in text  # route template, not raw path
        assert 'bookrec_recommendations_total{source="personalized"}' in text
        assert 'bookrec_recommendations_total{source="popular_fallback"}' in text
        assert "bookrec_artifact_users 2.0" in text


def test_missing_artifacts_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        ArtifactStore.load(tmp_path)
