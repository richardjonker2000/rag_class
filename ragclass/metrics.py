"""Retrieval (IR) metrics.

    ranked:   list of retrieved document ids, best first
    relevant: set of relevant ids (binary labels) or dict id -> grade (graded labels)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _rel(relevant, d) -> float:
    return float(relevant.get(d, 0)) if isinstance(relevant, dict) else float(d in relevant)


def precision_at_k(ranked, relevant, k: int) -> float:
    """Of the k results shown, what fraction is relevant?"""
    return sum(_rel(relevant, d) > 0 for d in ranked[:k]) / k


def recall_at_k(ranked, relevant, k: int) -> float:
    """Of all relevant documents, what fraction appears in the top k?"""
    return sum(_rel(relevant, d) > 0 for d in ranked[:k]) / max(1, len(relevant))


def hit_at_k(ranked, relevant, k: int) -> float:
    """Is there at least one relevant document in the top k? (a.k.a. success@k)"""
    return float(any(_rel(relevant, d) > 0 for d in ranked[:k]))


def reciprocal_rank(ranked, relevant, k: int | None = None) -> float:
    """1 / position of the first relevant result (0 if none). Averaged over queries = MRR."""
    for i, d in enumerate(ranked[:k], start=1):
        if _rel(relevant, d) > 0:
            return 1.0 / i
    return 0.0


def average_precision(ranked, relevant, k: int | None = None) -> float:
    """Mean of precision@i at every rank i where a relevant document appears. Averaged = MAP."""
    hits, total = 0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if _rel(relevant, d) > 0:
            hits += 1
            total += hits / i
    return total / max(1, len(relevant))


def ndcg_at_k(ranked, relevant, k: int) -> float:
    """Discounted cumulative gain, normalised by the best possible ordering.
    Rewards relevant results more the higher they are ranked (gain / log2(rank + 1))."""
    dcg = sum((2 ** _rel(relevant, d) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    grades = sorted((relevant.values() if isinstance(relevant, dict) else [1] * len(relevant)), reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades[:k], start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(run: dict, qrels: dict, ks=(1, 5, 10)) -> pd.Series:
    """Average every metric over all queries. run: qid -> ranked ids, qrels: qid -> relevant ids."""
    return per_query(run, qrels, ks).mean(numeric_only=True)


def per_query(run: dict, qrels: dict, ks=(1, 5, 10)) -> pd.DataFrame:
    rows = []
    for qid, relevant in qrels.items():
        ranked = list(run.get(qid, []))
        r = {"qid": qid, "MRR@10": reciprocal_rank(ranked, relevant, 10),
             "MAP@100": average_precision(ranked, relevant, 100)}
        for k in ks:
            r[f"P@{k}"] = precision_at_k(ranked, relevant, k)
            r[f"R@{k}"] = recall_at_k(ranked, relevant, k)
            r[f"nDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
        r["Hit@10"] = hit_at_k(ranked, relevant, 10)
        r["R@100"] = recall_at_k(ranked, relevant, 100)
        rows.append(r)
    cols = ["qid", "P@1", "P@5", "P@10", "R@10", "R@100", "Hit@10", "MRR@10", "MAP@100", "nDCG@10"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    """How often is system B *not* better than A when we resample the queries? (~ a p-value)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), (n, len(a)))
    return float(((b[idx] - a[idx]).mean(1) <= 0).mean())
