from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    """Cosine similarity for row vectors in x."""
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    xn = x / denom
    return xn @ xn.T


@dataclass
class MMRReranker:
    """MMR re-ranker.

    Contract:
    - Inputs:
      - query_vec: shape (d,)
      - cand_vecs: shape (n, d)
      - cand_ids: list[str] length n
    - Output:
      - selected_ids: list[str] length <= topk
    """

    lambda_param: float = 0.7

    def rerank(
        self,
        query_vec: np.ndarray,
        cand_vecs: np.ndarray,
        cand_ids: Sequence[str],
        topk: int,
    ) -> List[str]:
        if topk <= 0:
            return []
        if len(cand_ids) == 0:
            return []

        n = cand_vecs.shape[0]
        topk = min(topk, n)

        # relevance: sim(query, doc)
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        dn = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
        rel = dn @ qn  # (n,)

        # diversity: sim(doc, already_selected)
        sim_dd = cosine_sim_matrix(cand_vecs)

        selected: List[int] = []
        remaining = set(range(n))

        # Start with the most relevant.
        first = int(np.argmax(rel))
        selected.append(first)
        remaining.remove(first)

        while len(selected) < topk and remaining:
            best_i = None
            best_score = -1e9
            for i in remaining:
                max_sim_to_selected = float(np.max(sim_dd[i, selected])) if selected else 0.0
                mmr_score = float(self.lambda_param * rel[i] - (1.0 - self.lambda_param) * max_sim_to_selected)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_i = i
            assert best_i is not None
            selected.append(best_i)
            remaining.remove(best_i)

        return [str(cand_ids[i]) for i in selected]
