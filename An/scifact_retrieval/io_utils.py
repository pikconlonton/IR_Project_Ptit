from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        # Title is often useful for retrieval.
        return f"{self.title}. {self.text}".strip()


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_corpus(data_dir: Path) -> List[CorpusDoc]:
    corpus_path = data_dir / "corpus.jsonl"
    docs: List[CorpusDoc] = []
    for obj in read_jsonl(corpus_path):
        docs.append(
            CorpusDoc(
                doc_id=str(obj.get("_id")),
                title=str(obj.get("title", "")),
                text=str(obj.get("text", "")),
            )
        )
    return docs


def load_queries(data_dir: Path) -> List[Query]:
    queries_path = data_dir / "queries.jsonl"
    queries: List[Query] = []
    for obj in read_jsonl(queries_path):
        queries.append(Query(query_id=str(obj.get("_id")), text=str(obj.get("text", ""))))
    return queries


def load_qrels_tsv(path: Path) -> Dict[str, Dict[str, int]]:
    """Return qrels as: qrels[qid][docid] = relevance (int)."""
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"]).strip()
            docid = str(row["corpus-id"]).strip()
            score = int(float(row.get("score", 0)))
            qrels.setdefault(qid, {})[docid] = score
    return qrels


def load_split_qrels(data_dir: Path, split: str) -> Dict[str, Dict[str, int]]:
    split = split.lower().strip()
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train|test, got: {split}")
    return load_qrels_tsv(data_dir / "qrels" / f"{split}.tsv")


def iter_queries_in_qrels(queries: List[Query], qrels: Dict[str, Dict[str, int]]) -> List[Query]:
    qset = set(qrels.keys())
    return [q for q in queries if q.query_id in qset]
