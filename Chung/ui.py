from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Lightweight re-implementation of the retrieval logic from ir_project.py
st.set_page_config(page_title="IR PRF UI", layout="wide")


@st.cache_data
def load_jsonl_dict(path: str) -> Dict[str, dict]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data[str(obj["_id"])] = obj
    return data


@st.cache_data
def load_qrels(path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep="\t")
    qrels = {}
    for _, row in df.iterrows():
        qid = str(row["query-id"]).strip()
        docid = str(row["corpus-id"]).strip()
        qrels.setdefault(qid, []).append(docid)
    return qrels


def tokenize(text: str, stemmer: PorterStemmer, stop_words: set) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [stemmer.stem(w) for w in text.split() if w not in stop_words]


def build_inverted_index(
    corpus: Dict[str, dict], stemmer: PorterStemmer, stop_words: set
):
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_lengths = {}
    doc_tokens = {}

    for doc_id, doc in corpus.items():
        text = ((doc.get("title", "") + " ") * 2) + doc.get("text", "")
        tokens = tokenize(text, stemmer, stop_words)
        doc_tokens[doc_id] = tokens
        doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            inverted_index[token][doc_id] += 1

    return inverted_index, doc_lengths, doc_tokens


class PRFRetriever:
    def __init__(
        self,
        inverted_index,
        doc_lengths,
        doc_tokens,
        avg_doc_length,
        num_docs,
        k1=1.2,
        b=0.5,
    ):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.doc_tokens = doc_tokens
        self.avg_doc_length = avg_doc_length
        self.num_docs = num_docs
        self.k1 = k1
        self.b = b

        self.idf = {}
        for term, doc_freqs in inverted_index.items():
            df = len(doc_freqs)
            self.idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)

    def bm25_score(self, query_tokens: List[str]) -> Dict[str, float]:
        scores = defaultdict(float)
        for term in query_tokens:
            if term in self.inverted_index:
                idf = self.idf.get(term, 0.0)
                for doc_id, tf in self.inverted_index[term].items():
                    doc_len = self.doc_lengths[doc_id]
                    norm = 1 - self.b + self.b * (doc_len / self.avg_doc_length)
                    score = idf * (self.k1 + 1) * tf / (self.k1 * norm + tf)
                    scores[doc_id] += score
        return scores

    def extract_expansion_terms(
        self, top_docs: List[str], num_terms: int = 5
    ) -> List[str]:
        term_scores = defaultdict(float)
        for doc_id in top_docs:
            for t in self.doc_tokens[doc_id]:
                term_scores[t] += self.idf.get(t, 0)
        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[
            :num_terms
        ]
        return [t for t, _ in top_terms]

    def retrieve(
        self,
        query_text: str,
        stemmer: PorterStemmer,
        stop_words: set,
        top_k: int = 100,
        prf_top_docs: int = 10,
        expansion_terms: int = 5,
    ) -> List[Tuple[str, float]]:
        query_tokens = tokenize(query_text, stemmer, stop_words)
        initial_scores = self.bm25_score(query_tokens)
        ranked = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)

        top_docs = [doc_id for doc_id, _ in ranked[:prf_top_docs]]
        exp_terms = self.extract_expansion_terms(top_docs, num_terms=expansion_terms)

        # boost original query tokens
        expanded_query = query_tokens * 3 + exp_terms
        final_scores = self.bm25_score(expanded_query)
        final_ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return final_ranked[:top_k]


# Metrics
def recall_at_k(results, relevant_docs, k):
    retrieved = [d for d, _ in results[:k]]
    hit = sum([1 for d in retrieved if d in relevant_docs])
    return hit / len(relevant_docs) if relevant_docs else 0


def precision_at_k(results, relevant_docs, k):
    retrieved = [d for d, _ in results[:k]]
    hit = sum([1 for d in retrieved if d in relevant_docs])
    return hit / k if k > 0 else 0


def average_precision_at_k(results, relevant_docs, k):
    retrieved = [d for d, _ in results[:k]]
    score = 0
    hit = 0
    for i, d in enumerate(retrieved):
        if d in relevant_docs:
            hit += 1
            score += hit / (i + 1)
    if len(relevant_docs) == 0:
        return 0
    return score / min(len(relevant_docs), k)


def dcg_at_k(results, relevant_docs, k):
    retrieved = [d for d, _ in results[:k]]
    return sum(
        [1 / math.log2(i + 2) for i, d in enumerate(retrieved) if d in relevant_docs]
    )


def ndcg_at_k(results, relevant_docs, k):
    dcg = dcg_at_k(results, relevant_docs, k)
    ideal = sum([1 / math.log2(i + 2) for i in range(min(len(relevant_docs), k))])
    return dcg / ideal if ideal > 0 else 0


def main():
    st.title("IR System — BM25 + PRF (SciFact)")

    st.sidebar.header("Data")
    data_dir = st.sidebar.text_input(
        "Local data directory (contains corpus.jsonl)", value="./scifact/scifact"
    )
    upload_corpus = st.sidebar.file_uploader(
        "Upload corpus.jsonl (optional)", type=["jsonl"]
    )
    upload_queries = st.sidebar.file_uploader(
        "Upload queries.jsonl (optional)", type=["jsonl"]
    )
    upload_qrels = st.sidebar.file_uploader(
        "Upload qrels.tsv (optional)", type=["tsv", "txt"]
    )

    st.sidebar.header("Retriever settings")
    top_k = st.sidebar.number_input(
        "Top K to return", min_value=1, max_value=500, value=100
    )
    prf_top_docs = st.sidebar.number_input(
        "PRF top docs", min_value=1, max_value=50, value=10
    )
    expansion_terms = st.sidebar.number_input(
        "Expansion terms", min_value=1, max_value=50, value=5
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Load data"):
        try:
            if upload_corpus is not None:
                corpus = json.loads(
                    "["
                    + ",".join(
                        [
                            l.decode()
                            for l in upload_corpus.getvalue().splitlines()
                            if l.strip()
                        ]
                    )
                    + "]"
                )
                corpus = {str(obj["_id"]): obj for obj in corpus}
            else:
                corpus = load_jsonl_dict(Path(data_dir) / "corpus.jsonl")

            if upload_queries is not None:
                queries = json.loads(
                    "["
                    + ",".join(
                        [
                            l.decode()
                            for l in upload_queries.getvalue().splitlines()
                            if l.strip()
                        ]
                    )
                    + "]"
                )
                queries = {str(obj["_id"]): obj for obj in queries}
            else:
                queries = load_jsonl_dict(Path(data_dir) / "queries.jsonl")

            if upload_qrels is not None:
                qrels = pd.read_csv(upload_qrels, sep="\t")
                qrels = qrels.groupby("query-id")["corpus-id"].apply(list).to_dict()
            else:
                qrels = load_qrels(Path(data_dir) / "qrels" / "test.tsv")

            st.session_state["corpus"] = corpus
            st.session_state["queries"] = queries
            st.session_state["qrels"] = qrels
            st.success("Data loaded")
        except Exception as e:
            st.error(f"Failed to load data: {e}")

    if "corpus" not in st.session_state:
        st.info("Load data first from the sidebar (or upload files).")
        return

    corpus = st.session_state["corpus"]
    queries = st.session_state.get("queries", {})
    qrels = st.session_state.get("qrels", {})

    # prepare index (cache)
    stemmer = PorterStemmer()
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set()

    with st.spinner("Building inverted index..."):
        inverted_index, doc_lengths, doc_tokens = build_inverted_index(
            corpus, stemmer, stop_words
        )
        avg_doc_length = np.mean(list(doc_lengths.values())) if doc_lengths else 0.0
        retriever = PRFRetriever(
            inverted_index,
            doc_lengths,
            doc_tokens,
            avg_doc_length,
            num_docs=len(corpus),
        )

    st.header("Single Query Search")
    qtext = st.text_area("Query text")
    if st.button("Search"):
        if not qtext:
            st.warning("Enter a query")
        else:
            results = retriever.retrieve(
                qtext,
                stemmer,
                stop_words,
                top_k=top_k,
                prf_top_docs=int(prf_top_docs),
                expansion_terms=int(expansion_terms),
            )
            st.write(f"Top {len(results)} results")
            for i, (doc_id, score) in enumerate(results, start=1):
                doc = corpus.get(doc_id, {})
                title = doc.get("title", "")
                text = doc.get("text", "")
                st.markdown(f"**{i}. {title}** — `{doc_id}`")
                st.caption(f"score: {score:.4f}")
                st.write((text or "")[:800])
                st.markdown("---")

    st.header("Evaluate on all queries")
    if st.button("Run evaluation (may be slow)"):
        if not qrels:
            st.warning("No qrels loaded; cannot evaluate")
        else:
            K_LIST = [1, 5, 10, 100]
            metrics = {
                "ndcg": {k: [] for k in K_LIST},
                "recall": {k: [] for k in K_LIST},
                "map": {k: [] for k in K_LIST},
                "precision": {k: [] for k in K_LIST},
            }
            progress = st.progress(0)
            q_items = list(qrels.items())
            for idx, (qid, rel_docs) in enumerate(q_items):
                query_obj = queries.get(qid)
                if not query_obj:
                    continue
                qtext = query_obj.get("text", "")
                results = retriever.retrieve(
                    qtext,
                    stemmer,
                    stop_words,
                    top_k=max(K_LIST),
                    prf_top_docs=int(prf_top_docs),
                    expansion_terms=int(expansion_terms),
                )
                for k in K_LIST:
                    metrics["recall"][k].append(recall_at_k(results, rel_docs, k))
                    metrics["precision"][k].append(precision_at_k(results, rel_docs, k))
                    metrics["map"][k].append(
                        average_precision_at_k(results, rel_docs, k)
                    )
                    metrics["ndcg"][k].append(ndcg_at_k(results, rel_docs, k))
                progress.progress((idx + 1) / len(q_items))

            st.subheader("Aggregate results (mean)")
            for metric in ["ndcg", "recall", "map", "precision"]:
                st.markdown(f"**{metric.upper()}**")
                cols = st.columns(len(K_LIST))
                for i, k in enumerate(K_LIST):
                    meanv = np.mean(metrics[metric][k]) if metrics[metric][k] else 0.0
                    cols[i].metric(f"@{k}", f"{meanv:.5f}")


if __name__ == "__main__":
    main()
