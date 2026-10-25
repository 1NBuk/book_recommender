from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("bookrec.store")

REQUIRED_FILES = ("recommendations.parquet", "items.parquet", "popular.parquet", "model_info.json")


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _book_record(row: dict) -> dict:
    book = {k: _clean(v) for k, v in row.items()}
    year = book.get("year")
    if year is not None:
        year = int(year)
        book["year"] = year if 1000 <= year <= 2100 else None
    if book.get("average_rating") is not None:
        book["average_rating"] = float(book["average_rating"])
    return book


class ArtifactStore:
    """In-memory view over exported serving artifacts (read-only after load)."""

    def __init__(self, recs: dict, books: dict, popular: list, info: dict):
        self._recs = recs
        self._books = books
        self._popular = popular
        self.info = info

    @classmethod
    def load(cls, directory: Path | str) -> "ArtifactStore":
        d = Path(directory)
        missing = [f for f in REQUIRED_FILES if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing artifacts in {d.resolve()}: {', '.join(missing)}")

        info = json.loads((d / "model_info.json").read_text(encoding="utf-8"))

        items = pd.read_parquet(d / "items.parquet")
        books = {rec["item_id"]: _book_record(rec) for rec in items.to_dict("records")}

        recs_df = pd.read_parquet(d / "recommendations.parquet")
        total = len(recs_df)
        recs_df = recs_df[recs_df["item_id"].isin(books.keys())]
        dropped = total - len(recs_df)
        if total:
            log.info(
                "recommendations: %d rows, %d dropped without metadata (%.1f%%)",
                total,
                dropped,
                100.0 * dropped / total,
            )
        recs_df = recs_df.sort_values(["user_id", "rank"])
        recs: dict[str, tuple[list[str], list[float]]] = {}
        for user_id, grp in recs_df.groupby("user_id", sort=False):
            recs[str(user_id)] = (grp["item_id"].tolist(), [round(float(s), 6) for s in grp["score"]])

        pop_df = pd.read_parquet(d / "popular.parquet").sort_values("score", ascending=False)
        popular = [(iid, round(float(s), 6)) for iid, s in zip(pop_df["item_id"], pop_df["score"]) if iid in books]

        log.info("loaded: %d users, %d books, %d popular", len(recs), len(books), len(popular))
        return cls(recs, books, popular, info)

    # ---- properties -------------------------------------------------------
    @property
    def n_users(self) -> int:
        return len(self._recs)

    @property
    def n_items(self) -> int:
        return len(self._books)

    @property
    def n_popular(self) -> int:
        return len(self._popular)

    # ---- queries ----------------------------------------------------------
    def _decorate(self, pairs: list[tuple[str, float]]) -> list[dict]:
        return [
            {**self._books[item_id], "rank": rank, "score": score}
            for rank, (item_id, score) in enumerate(pairs, start=1)
        ]

    def recommend(self, user_id: str, n: int) -> tuple[str, list[dict]]:
        entry = self._recs.get(user_id)
        if entry is not None:
            ids, scores = entry
            return "personalized", self._decorate(list(zip(ids[:n], scores[:n])))
        return "popular_fallback", self._decorate(self._popular[:n])

    def popular(self, n: int) -> list[dict]:
        return self._decorate(self._popular[:n])

    def book(self, item_id: str) -> Optional[dict]:
        return self._books.get(item_id)
