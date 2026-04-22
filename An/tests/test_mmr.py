import numpy as np

from scifact_retrieval.mmr import MMRReranker


def test_mmr_prefers_diversity_when_lambda_low():
    # Query is close to doc0 and doc1, doc2 is orthogonal-ish.
    q = np.array([1.0, 0.0], dtype=np.float32)
    docs = np.array([
        [1.0, 0.0],   # very relevant
        [0.9, 0.1],   # similar to doc0
        [0.0, 1.0],   # diverse
    ], dtype=np.float32)
    ids = ["d0", "d1", "d2"]

    reranker = MMRReranker(lambda_param=0.2)
    ranked = reranker.rerank(q, docs, ids, topk=2)

    assert ranked[0] == "d0"
    assert ranked[1] == "d2"
