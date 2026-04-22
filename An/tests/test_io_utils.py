from pathlib import Path

from scifact_retrieval.io_utils import load_corpus, load_queries, load_split_qrels


def test_loaders_smoke():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "scifact"

    corpus = load_corpus(data_dir)
    queries = load_queries(data_dir)
    qrels = load_split_qrels(data_dir, "test")

    assert len(corpus) > 0
    assert len(queries) > 0
    assert len(qrels) > 0
