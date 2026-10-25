"""Build the small set of serving artifacts from the training outputs.

    python scripts/export_artifacts.py --root . --out artifacts --model hybrid

Reads:  feature_store/items_active.parquet, feature_store/item_features.parquet,
        experiments/recommendations_cache/<model>_top100.parquet,
        experiments/metrics/final_results.json
Writes: <out>/recommendations.parquet, items.parquet, popular.parquet, model_info.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

METRIC_KEYS = ["ndcg@10", "hit_rate@10", "precision@10", "recall@10", "map@10", "coverage", "personalization"]
META_COLS = ["item_id", "isbn", "title", "author", "year", "publisher", "image_url", "average_rating"]


def build_items(root: Path) -> pd.DataFrame:
    active = pd.read_parquet(root / "feature_store/items_active.parquet")
    feats = pd.read_parquet(root / "feature_store/item_features.parquet")

    active = active.rename(columns={"Image-URL-M": "image_url"})
    if "image_url" not in feats.columns:
        feats["image_url"] = None

    cols = [c for c in META_COLS if c in active.columns]
    frames = [active[cols], feats[[c for c in META_COLS if c in feats.columns]]]
    items = pd.concat(frames, ignore_index=True).drop_duplicates("item_id", keep="first")
    items = items[items["title"].notna() & (items["title"].astype(str).str.strip() != "")]
    for col in META_COLS:
        if col not in items.columns:
            items[col] = None
    return items[META_COLS].reset_index(drop=True)


def build_popular(root: Path, items: pd.DataFrame, top: int = 500, min_count: int = 5) -> pd.DataFrame:
    # same formula as PopularityRecommender: mean_rating * log1p(count)
    feats = pd.read_parquet(root / "feature_store/item_features.parquet")
    feats = feats[feats["item_interaction_count"] >= min_count].copy()
    feats["score"] = feats["item_mean_rating"] * np.log1p(feats["item_interaction_count"])
    feats = feats[feats["item_id"].isin(items["item_id"])]
    return feats.nlargest(top, "score")[["item_id", "score"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--model", default="hybrid", help="name of <model>_top100.parquet in recommendations_cache")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = build_items(root)
    popular = build_popular(root, items)
    recs = pd.read_parquet(root / f"experiments/recommendations_cache/{args.model}_top100.parquet")
    recs = recs[["user_id", "item_id", "rank", "score"]].copy()
    recs["rank"] = recs["rank"].astype("int32")
    recs["score"] = recs["score"].astype("float32")
    # rank in the cache is 0-based: the API exposes 1-based ranks itself, so keep order only
    recs = recs.sort_values(["user_id", "rank"])

    metrics: dict[str, float] = {}
    results = root / "experiments/metrics/final_results.json"
    if results.exists():
        all_metrics = json.loads(results.read_text(encoding="utf-8"))
        metrics = {k: float(all_metrics[args.model][k]) for k in METRIC_KEYS if k in all_metrics.get(args.model, {})}

    now = datetime.now(timezone.utc)
    info = {
        "model": args.model,
        "created_at": now.isoformat(timespec="seconds"),
        "created_at_unix": int(now.timestamp()),
        "metrics": metrics,
    }

    recs.to_parquet(out / "recommendations.parquet", index=False, compression="zstd")
    items.to_parquet(out / "items.parquet", index=False, compression="zstd")
    popular.to_parquet(out / "popular.parquet", index=False, compression="zstd")
    (out / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"users={recs['user_id'].nunique()} recs={len(recs)} items={len(items)} popular={len(popular)}")


if __name__ == "__main__":
    main()
