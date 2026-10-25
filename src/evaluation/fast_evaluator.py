
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from src.evaluation.ranking_metrics import evaluate_user
from src.evaluation.business_metrics import diversity_intra_list, novelty, personalization


class FastEvaluator:
    def __init__(self, config):
        self.config = config
        self.k_values = tuple(config.get("evaluation.k_values", [5, 10, 20]))

    def build_ground_truth(self, test_interactions):
        gt = defaultdict(set)
        for row in test_interactions.itertuples():
            gt[row.user_id].add(row.item_id)
        return dict(gt)

    def evaluate_ranking_from_cache(self, cache_df, ground_truth, k_values=None):
        if k_values is None:
            k_values = self.k_values

        recs_by_user = cache_df.groupby("user_id")["item_id"].apply(list).to_dict()

        all_metrics = defaultdict(list)
        n_evaluated = 0
        max_k = max(k_values)

        for user_id, recs in recs_by_user.items():
            relevant = ground_truth.get(user_id, set())
            if not relevant:
                continue
            recommended = recs[:max_k]
            user_metrics = evaluate_user(recommended, relevant, k_values)
            for key, value in user_metrics.items():
                all_metrics[key].append(value)
            n_evaluated += 1

        result = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in all_metrics.items()
        }
        result["n_users_evaluated"] = n_evaluated
        return result

    def evaluate_business_from_cache(self, cache_df, item_features, item_popularity,
                                     total_items, k=10, compute_personalization=True):
        start = time.time()

        recs_by_user = cache_df.groupby("user_id").apply(
            lambda df: df.nsmallest(k, "rank")["item_id"].tolist()
        ).to_dict()

        all_items_flat = []
        recommended_items = set()
        for items in recs_by_user.values():
            all_items_flat.extend(items)
            recommended_items.update(items)

        coverage = len(recommended_items) / total_items if total_items > 0 else 0.0

        diversity_scores = []
        for items in list(recs_by_user.values())[:500]:
            diversity_scores.append(diversity_intra_list(items, item_features, "tags"))
        diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0

        novelty_score = novelty(all_items_flat, item_popularity)

        personalization_score = 0.0
        if compute_personalization:
            sample = dict(list(recs_by_user.items())[:500])
            personalization_score = personalization(sample)

        return {
            "coverage": float(coverage),
            "diversity": float(diversity),
            "novelty": float(novelty_score),
            "personalization": float(personalization_score),
            "n_users_evaluated": len(recs_by_user),
            "eval_time": round(time.time() - start, 2),
        }
