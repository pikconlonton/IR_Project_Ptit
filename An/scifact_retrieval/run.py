from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .bm25 import BM25Index
from .io_utils import iter_queries_in_qrels, load_corpus, load_queries, load_split_qrels
from .metrics import evaluate_run
from .mmr import MMRReranker


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SciFact BM25 retrieval + MMR rerank")
    p.add_argument("--data-dir", type=str, default="scifact", help="Path to scifact folder")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--bm25-topk", type=int, default=100)
    p.add_argument("--mmr-topk", type=int, default=10)
    p.add_argument("--lambda-param", type=float, default=0.7)
    p.add_argument("--out", type=str, default="", help="Write run as JSONL (qid + doc_ids)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)

    corpus = load_corpus(data_dir)
    queries = load_queries(data_dir)
    qrels = load_split_qrels(data_dir, args.split)
    queries = iter_queries_in_qrels(queries, qrels)

    doc_ids = [d.doc_id for d in corpus]
    doc_texts = [d.full_text for d in corpus]
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    bm25 = BM25Index.build(doc_ids=doc_ids, docs_text=doc_texts)

    # TF-IDF for MMR sim computations.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=200_000,
        ngram_range=(1, 2),
    )
    doc_tfidf = vectorizer.fit_transform(doc_texts)

    reranker = MMRReranker(lambda_param=float(args.lambda_param))

    run: Dict[str, List[str]] = {}

    for q in tqdm(queries, desc=f"Retrieving ({args.split})"):
        bm25_hits = bm25.search(q.text, topk=int(args.bm25_topk))
        cand_ids = [doc_id for doc_id, _ in bm25_hits]

    # Candidate vectors
        cand_idx = [doc_id_to_index[cid] for cid in cand_ids]
        cand_vecs = doc_tfidf[cand_idx].toarray().astype(np.float32) # pyright: ignore
        q_vec = vectorizer.transform([q.text]).toarray()[0].astype(np.float32) # pyright: ignore

        selected_ids = reranker.rerank(
            query_vec=q_vec,
            cand_vecs=cand_vecs,
            cand_ids=cand_ids,
            topk=int(args.mmr_topk),
        )
        run[q.query_id] = selected_ids

    # Report a couple of common cutoffs.
    metrics = evaluate_run(run, qrels, k=[5, int(args.mmr_topk)])
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for qid, ranked in run.items():
                f.write(json.dumps({"query_id": qid, "doc_ids": ranked}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
