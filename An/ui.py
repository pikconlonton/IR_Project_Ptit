from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from scifact_retrieval.bm25 import BM25Index
from scifact_retrieval.io_utils import CorpusDoc
from scifact_retrieval.mmr import MMRReranker


def parse_jsonl_bytes(b: bytes) -> List[CorpusDoc]:
    lines = b.decode("utf-8").splitlines()
    docs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        docs.append(
            CorpusDoc(
                doc_id=str(obj.get("_id")),
                title=str(obj.get("title", "")),
                text=str(obj.get("text", "")),
            )
        )
    return docs


def docs_to_texts(docs: List[CorpusDoc]) -> List[str]:
    return [d.full_text for d in docs]


def main() -> None:
    st.set_page_config(page_title="SciFact Retrieval UI", layout="wide")
    st.title("SciFact — Simple Retrieval UI")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.header("Data")
        data_dir = st.text_input(
            "Local data directory (contains corpus.jsonl)", value="./scifact"
        )
        uploaded = st.file_uploader("Or upload corpus.jsonl", type=["jsonl"])
        st.markdown("---")

        st.header("Parameters")
        bm25_topk = st.number_input("BM25 top-k", min_value=1, max_value=1000, value=50)
        use_mmr = st.checkbox("Use MMR re-ranker", value=True)
        mmr_topk = st.number_input("MMR top-k", min_value=1, max_value=1000, value=10)
        lambda_param = st.slider(
            "MMR lambda (balancing relevance/diversity)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
        )
        st.markdown("---")
        st.write("Quick tips:")
        st.write(
            "- Provide a local `corpus.jsonl` folder in the data directory, or upload a `corpus.jsonl` file."
        )
        st.write(
            "- Each line should be a JSON object with `_id`, `title`, and `text` fields."
        )

    with col_right:
        st.header("Query")
        query = st.text_area("Enter your query text here", height=120)
        if st.button("Search"):
            # Load corpus
            if uploaded is not None:
                docs = parse_jsonl_bytes(uploaded.read())
            else:
                p = Path(data_dir)
                try:
                    from scifact_retrieval.io_utils import load_corpus

                    docs = load_corpus(Path(data_dir))
                except Exception as e:
                    st.error(f"Failed to load corpus from '{data_dir}': {e}")
                    docs = []

            if not docs:
                st.warning(
                    "No documents loaded. Upload a corpus.jsonl or point to a correct local path."
                )
                return

            doc_ids = [d.doc_id for d in docs]
            doc_texts = docs_to_texts(docs)

            # Build BM25
            with st.spinner("Building BM25 index..."):
                bm25 = BM25Index.build(doc_ids=doc_ids, docs_text=doc_texts)

            with st.spinner("Running BM25 search..."):
                bm25_hits = bm25.search(query, topk=int(bm25_topk))

            cand_ids = [doc_id for doc_id, _ in bm25_hits]
            cand_scores = {doc_id: score for doc_id, score in bm25_hits}

            # Prepare reranking
            if use_mmr:
                vectorizer = TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=200_000,
                    ngram_range=(1, 2),
                )
                doc_tfidf = vectorizer.fit_transform(doc_texts)
                doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

                cand_idx = [doc_id_to_index[cid] for cid in cand_ids]
                cand_vecs = doc_tfidf[cand_idx].toarray().astype(np.float32)
                q_vec = vectorizer.transform([query]).toarray()[0].astype(np.float32)

                reranker = MMRReranker(lambda_param=float(lambda_param))
                selected_ids = reranker.rerank(
                    query_vec=q_vec,
                    cand_vecs=cand_vecs,
                    cand_ids=cand_ids,
                    topk=int(mmr_topk),
                )
            else:
                selected_ids = cand_ids[: int(mmr_topk)]

            st.subheader("Results")
            for i, doc_id in enumerate(selected_ids, start=1):
                idx = next((j for j, d in enumerate(docs) if d.doc_id == doc_id), None)
                title = docs[idx].title if idx is not None else ""
                text = docs[idx].text if idx is not None else ""
                score = cand_scores.get(doc_id, None)
                st.markdown(f"**{i}. {title}**  ")
                if score is not None:
                    st.caption(f"BM25 score: {score:.4f}")
                # show a snippet
                snippet = text[:800] + ("..." if len(text) > 800 else "")
                st.write(snippet)
                st.markdown("---")


if __name__ == "__main__":
    main()
