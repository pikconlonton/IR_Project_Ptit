from __future__ import annotations

import math
from typing import Dict, List, Sequence


def precision_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = 0
    for doc_id in ranked[:k]:
        if relevant.get(doc_id, 0) > 0:
            hits += 1
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    total_rel = sum(1 for v in relevant.values() if v > 0)
    if total_rel == 0:
        return 0.0
    hits = 0
    for doc_id in ranked[:k]:
        if relevant.get(doc_id, 0) > 0:
            hits += 1
    return hits / total_rel


def mrr(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / idx
    return 0.0


def average_precision(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    """AP@k for binary or graded qrels (treat rel>0 as relevant).

    AP = mean precision at ranks where a relevant doc is retrieved.
    """
    total_rel = sum(1 for v in relevant.values() if v > 0)
    if total_rel == 0:
        return 0.0

    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            hits += 1
            sum_prec += hits / i

    # Standard AP divides by total relevant in the collection (not by hits).
    return sum_prec / total_rel


def dcg_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    """DCG@k with graded relevance.

    Uses: (2^rel - 1) / log2(i+1)
    where i is 1-indexed rank.
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        rel = float(relevant.get(doc_id, 0))
        if rel <= 0:
            continue
        dcg += (2.0**rel - 1.0) / math.log2(i + 1.0)
    return dcg


def ndcg_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    """nDCG@k with graded relevance."""
    ideal_ranked = [doc_id for doc_id, rel in sorted(relevant.items(), key=lambda x: x[1], reverse=True) if rel > 0]
    idcg = dcg_at_k(ideal_ranked, relevant, k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k: int | Sequence[int] = 10,
) -> Dict[str, float]:
    """Evaluate a run against qrels.

    Parameters
    - run[qid] = ranked list of doc_ids
    - qrels[qid][doc_id] = relevance (int)
    - k: int or list of cutoffs
    """

    ks = [int(k)] if isinstance(k, int) else [int(x) for x in k]
    ks = sorted(set([x for x in ks if x > 0]))
    if not ks:
        raise ValueError("k must contain at least one positive cutoff")

    # Accumulators per cutoff
    ps = {cut: [] for cut in ks}
    rs = {cut: [] for cut in ks}
    mrrs = {cut: [] for cut in ks}
    maps = {cut: [] for cut in ks}
    ndcgs = {cut: [] for cut in ks}

    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])
        for cut in ks:
            ps[cut].append(precision_at_k(ranked, relevant, cut))
            rs[cut].append(recall_at_k(ranked, relevant, cut))
            mrrs[cut].append(mrr(ranked, relevant, cut))
            maps[cut].append(average_precision(ranked, relevant, cut))
            ndcgs[cut].append(ndcg_at_k(ranked, relevant, cut))

    def avg(x: List[float]) -> float:
        return sum(x) / max(1, len(x))

    out: Dict[str, float] = {"num_queries": float(len(qrels))}
    for cut in ks:
        out[f"P@{cut}"] = avg(ps[cut])
        out[f"R@{cut}"] = avg(rs[cut])
        out[f"MRR@{cut}"] = avg(mrrs[cut])
        out[f"MAP@{cut}"] = avg(maps[cut])
        out[f"nDCG@{cut}"] = avg(ndcgs[cut])
    return out
