from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _try_get_nltk_stopwords(language: str = "english") -> Optional[Set[str]]:
    """Best-effort NLTK stopword loader.

    - If NLTK (or the corpus) isn't installed, returns None.
    - If stopwords are missing, tries to download them quietly.
    """

    try:
        import nltk  # type: ignore
        from nltk.corpus import stopwords  # type: ignore
    except Exception:
        return None

    try:
        words = set(stopwords.words(language))
        return words
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words(language))
            return words
        except Exception:
            return None


def simple_tokenize(text: str, stopwords: Optional[Set[str]] = None) -> List[str]:
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


@dataclass
class BM25Index:
    doc_ids: List[str]
    bm25: BM25Okapi
    tokenized_docs: List[List[str]]
    stopwords: Optional[Set[str]] = None

    @classmethod
    def build(
        cls,
        doc_ids: Sequence[str],
        docs_text: Sequence[str],
        *,
        remove_stopwords: bool = True,
        stopwords_language: str = "english",
    ) -> "BM25Index":
        sw = _try_get_nltk_stopwords(stopwords_language) if remove_stopwords else None
        tokenized_docs = [simple_tokenize(t, stopwords=sw) for t in docs_text]
        bm25 = BM25Okapi(tokenized_docs)
        return cls(doc_ids=list(doc_ids), bm25=bm25, tokenized_docs=tokenized_docs, stopwords=sw)

    def search(self, query: str, topk: int) -> List[Tuple[str, float]]:
        q_tokens = simple_tokenize(query, stopwords=self.stopwords)
        scores = self.bm25.get_scores(q_tokens)
        # rank_bm25 returns a numpy array
        ranked_idx = scores.argsort()[::-1][:topk]
        results: List[Tuple[str, float]] = []
        for i in ranked_idx:
            results.append((self.doc_ids[int(i)], float(scores[int(i)])))
        return results
