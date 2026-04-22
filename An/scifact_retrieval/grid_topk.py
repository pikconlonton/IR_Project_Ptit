from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .bm25 import BM25Index
from .io_utils import iter_queries_in_qrels, load_corpus, load_queries, load_split_qrels
from .metrics import evaluate_run
from .mmr import MMRReranker


@dataclass(frozen=True)
class GridResult:
    bm25_topk: int
    mmr_topk: int
    metrics: Dict[str, float]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid search for bm25-topk and mmr-topk")
    p.add_argument("--data-dir", type=str, default="scifact", help="Path to scifact folder")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument(
        "--max-queries",
        type=int,
        default=300,
        help="Limit the number of queries evaluated (<=0 means no limit). Useful for faster grid runs.",
    )
    p.add_argument(
        "--bm25-topk-grid",
        type=str,
        default="20,50,100,200,500",
        help="Comma-separated list of BM25 candidate sizes",
    )
    p.add_argument(
        "--mmr-topk-grid",
        type=str,
        default="5,10,20",
        help="Comma-separated list of final topk sizes after MMR",
    )
    p.add_argument("--lambda-param", type=float, default=0.7)
    p.add_argument(
        "--eval-ks",
        type=str,
        default="5,10,20",
        help="Comma-separated cutoffs to evaluate (must be >0)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional path to write JSONL results (one row per grid point)",
    )
    return p


def _parse_int_grid(s: str) -> List[int]:
    out: List[int] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    out = sorted(set([x for x in out if x > 0]))
    if not out:
        raise ValueError("grid must contain at least one positive integer")
    return out


def _evaluate_one(
    *,
    queries_text: List[Tuple[str, str]],
    qrels: Dict[str, Dict[str, int]],
    bm25: BM25Index,
    doc_id_to_index: Dict[str, int],
    doc_tfidf: np.ndarray,
    vectorizer: TfidfVectorizer,
    bm25_topk: int,
    mmr_topk: int,
    lambda_param: float,
    eval_ks: Sequence[int],
) -> Dict[str, float]:
    reranker = MMRReranker(lambda_param=float(lambda_param))
    run: Dict[str, List[str]] = {}

    # IMPORTANT: `evaluate_run` iterates over qrels keys, not run keys.
    # If we only evaluate a subset of queries, we must also subset qrels,
    # otherwise `num_queries` (and averages) will still reflect the full split.
    eval_qids = {qid for qid, _ in queries_text}
    qrels_eval = {qid: rels for qid, rels in qrels.items() if qid in eval_qids}

    for qid, qtext in tqdm(queries_text, desc=f"Grid bm25={bm25_topk} mmr={mmr_topk}", leave=False):
        bm25_hits = bm25.search(qtext, topk=int(bm25_topk))
        cand_ids = [doc_id for doc_id, _ in bm25_hits]
        if not cand_ids:
            run[qid] = []
            continue

        cand_idx = [doc_id_to_index[cid] for cid in cand_ids]
        cand_vecs = doc_tfidf[cand_idx].astype(np.float32)
        q_vec = vectorizer.transform([qtext]).toarray()[0].astype(np.float32)  # pyright: ignore

        selected_ids = reranker.rerank(
            query_vec=q_vec,
            cand_vecs=cand_vecs,
            cand_ids=cand_ids,
            topk=int(mmr_topk),
        )
        run[qid] = selected_ids

    return evaluate_run(run, qrels_eval, k=list(eval_ks))


def main() -> None:
    args = build_argparser().parse_args()
    np.random.seed(args.seed)

    bm25_grid = _parse_int_grid(args.bm25_topk_grid)
    mmr_grid = _parse_int_grid(args.mmr_topk_grid)
    eval_ks = _parse_int_grid(args.eval_ks)

    data_dir = Path(args.data_dir)
    corpus = load_corpus(data_dir)
    queries = load_queries(data_dir)
    qrels = load_split_qrels(data_dir, args.split)
    queries = iter_queries_in_qrels(queries, qrels)

    if int(args.max_queries) > 0:
        queries = queries[: int(args.max_queries)]

    queries_text = [(q.query_id, q.text) for q in queries]

    doc_ids = [d.doc_id for d in corpus]
    doc_texts = [d.full_text for d in corpus]
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    bm25 = BM25Index.build(doc_ids=doc_ids, docs_text=doc_texts)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=200_000,
        ngram_range=(1, 2),
    )
    doc_tfidf_sparse = vectorizer.fit_transform(doc_texts)
    doc_tfidf = doc_tfidf_sparse.toarray()  # pyright: ignore

    # Ensure sensible grid: bm25-topk must be >= mmr-topk
    grid_points: List[Tuple[int, int]] = []
    for b in bm25_grid:
        for m in mmr_grid:
            if b >= m:
                grid_points.append((b, m))

    results: List[GridResult] = []
    for b, m in grid_points:
        metrics = _evaluate_one(
            queries_text=queries_text,
            qrels=qrels,
            bm25=bm25,
            doc_id_to_index=doc_id_to_index,
            doc_tfidf=doc_tfidf,
            vectorizer=vectorizer,
            bm25_topk=b,
            mmr_topk=m,
            lambda_param=float(args.lambda_param),
            eval_ks=eval_ks,
        )
        results.append(GridResult(bm25_topk=b, mmr_topk=m, metrics=metrics))

    # Sort by the "closest" cutoff to mmr_topk for MRR then nDCG.
    def _score_key(r: GridResult) -> Tuple[float, float]:
        cut = min(eval_ks, key=lambda x: abs(x - r.mmr_topk))
        return (float(r.metrics.get(f"MRR@{cut}", 0.0)), float(r.metrics.get(f"nDCG@{cut}", 0.0)))

    results_sorted = sorted(results, key=_score_key, reverse=True)

    # Print a compact table.
    header = ["bm25_topk", "mmr_topk"]
    for k in eval_ks:
        header += [f"MRR@{k}", f"nDCG@{k}", f"R@{k}"]
    print("\t".join(header))
    for r in results_sorted:
        row: List[str] = [str(r.bm25_topk), str(r.mmr_topk)]
        for k in eval_ks:
            row.append(f"{r.metrics.get(f'MRR@{k}', 0.0):.6f}")
            row.append(f"{r.metrics.get(f'nDCG@{k}', 0.0):.6f}")
            row.append(f"{r.metrics.get(f'R@{k}', 0.0):.6f}")
        print("\t".join(row))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results_sorted:
                f.write(
                    json.dumps(
                        {
                            "bm25_topk": r.bm25_topk,
                            "mmr_topk": r.mmr_topk,
                            **r.metrics,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
