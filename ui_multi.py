from __future__ import annotations

import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# We'll reuse the An implementation's modules for BM25 and MMR
try:
    from An.scifact_retrieval.bm25 import BM25Index  # type: ignore
    from An.scifact_retrieval.io_utils import CorpusDoc, load_corpus  # type: ignore
    from An.scifact_retrieval.mmr import MMRReranker  # type: ignore
except Exception:
    # Fallback imports from package layout (when running from repo root)
    try:
        from scifact_retrieval.bm25 import BM25Index  # type: ignore
        from scifact_retrieval.io_utils import CorpusDoc, load_corpus  # type: ignore
        from scifact_retrieval.mmr import MMRReranker  # type: ignore
    except Exception:
        BM25Index = None  # type: ignore
        CorpusDoc = None  # type: ignore
        load_corpus = None  # type: ignore
        MMRReranker = None  # type: ignore


def parse_jsonl_bytes(b: bytes) -> List[Dict]:
    lines = b.decode("utf-8").splitlines()
    docs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        docs.append(obj)
    return docs


# --- Chung PRF logic (copied/minimized) ---
def _tokenize(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    # lightweight: no stemming here to keep dependency small
    return [t for t in text.split() if t]


def build_inverted_index(corpus: Dict[str, dict]):
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_lengths = {}
    doc_tokens = {}
    for doc_id, doc in corpus.items():
        text = ((doc.get("title", "") + " ") * 2) + doc.get("text", "")
        tokens = _tokenize(text)
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

    def bm25_score(self, query_tokens: Sequence[str]) -> Dict[str, float]:
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
        top_k: int = 100,
        prf_top_docs: int = 10,
        expansion_terms: int = 5,
    ) -> List[Tuple[str, float]]:
        query_tokens = _tokenize(query_text)
        initial_scores = self.bm25_score(query_tokens)
        ranked = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc_id for doc_id, _ in ranked[:prf_top_docs]]
        exp_terms = self.extract_expansion_terms(top_docs, num_terms=expansion_terms)
        expanded_query = query_tokens * 3 + exp_terms
        final_scores = self.bm25_score(expanded_query)
        final_ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return final_ranked[:top_k]


# --- Hà ScienceIRSystem loader ---
def load_science_ir_system(path: Path):
    code_py = path / "code.py"
    if not code_py.exists():
        return None
    spec = importlib.util.spec_from_file_location("ha_code", str(code_py))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, "ScienceIRSystem", None)


def main():
    st.set_page_config(page_title="IR Multi-Method UI", layout="wide")
    st.title("IR Multi-Method")

    method = st.sidebar.selectbox(
        "Choose method", ["An (BM25+MMR)", "Chung (BM25+PRF)", "Hà (BM25+)"]
    )

    # Common data input
    uploaded = st.sidebar.file_uploader(
        "Upload corpus.jsonl (optional)", type=["jsonl"]
    )
    data_dir = st.sidebar.text_input("Local data directory", value="scifact")

    if method == "An (BM25+MMR)":
        st.header("An — BM25 + MMR")
        bm25_topk = st.number_input("BM25 top-k", min_value=1, max_value=1000, value=50)
        use_mmr = st.checkbox("Use MMR re-ranker", value=True)
        mmr_topk = st.number_input("MMR top-k", min_value=1, max_value=1000, value=10)
        lambda_param = st.slider("MMR lambda", min_value=0.0, max_value=1.0, value=0.7)

        query = st.text_area("Query text")
        if st.button("Search (An)"):
            # load corpus
            if uploaded is not None:
                docs_obj = parse_jsonl_bytes(uploaded.read())
                docs = [
                    CorpusDoc(
                        doc_id=str(o.get("_id")),
                        title=o.get("title", ""),
                        text=o.get("text", ""),
                    )
                    for o in docs_obj
                ]
            else:
                try:
                    docs = load_corpus(Path(data_dir))
                except Exception as e:
                    st.error(f"Failed to load corpus: {e}")
                    docs = []

            if not docs:
                st.warning("No documents loaded")
                return

            doc_ids = [d.doc_id for d in docs]
            doc_texts = [d.full_text for d in docs]
            bm25 = BM25Index.build(doc_ids=doc_ids, docs_text=doc_texts)
            bm25_hits = bm25.search(query, topk=int(bm25_topk))
            cand_ids = [doc_id for doc_id, _ in bm25_hits]
            cand_scores = {doc_id: score for doc_id, score in bm25_hits}

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
                selected = reranker.rerank(
                    query_vec=q_vec,
                    cand_vecs=cand_vecs,
                    cand_ids=cand_ids,
                    topk=int(mmr_topk),
                )
            else:
                selected = cand_ids[: int(mmr_topk)]

            st.subheader("Results")
            for i, doc_id in enumerate(selected, start=1):
                idx = next((j for j, d in enumerate(docs) if d.doc_id == doc_id), None)
                title = docs[idx].title if idx is not None else ""
                text = docs[idx].text if idx is not None else ""
                score = cand_scores.get(doc_id)
                st.markdown(f"**{i}. {title}**")
                if score is not None:
                    st.caption(f"BM25 score: {score:.4f}")
                st.write((text or "")[:800])
                st.markdown("---")

    elif method == "Chung (BM25+PRF)":
        st.header("Chung — BM25 + PRF")
        prf_top_docs = st.number_input(
            "PRF top docs", min_value=1, max_value=50, value=10
        )
        expansion_terms = st.number_input(
            "Expansion terms", min_value=1, max_value=50, value=5
        )
        top_k = st.number_input(
            "Top K to return", min_value=1, max_value=500, value=100
        )

        query = st.text_area("Query text")
        if st.button("Search (Chung)"):
            # load corpus
            if uploaded is not None:
                docs_raw = parse_jsonl_bytes(uploaded.read())
                corpus = {str(o.get("_id")): o for o in docs_raw}
            else:
                try:
                    corpus_list = parse_jsonl_bytes(
                        Path(data_dir).joinpath("corpus.jsonl").read_bytes()
                    )
                    corpus = {str(o.get("_id")): o for o in corpus_list}
                except Exception as e:
                    st.error(f"Failed to load corpus: {e}")
                    corpus = {}

            if not corpus:
                st.warning("No documents loaded")
                return

            inverted_index, doc_lengths, doc_tokens = build_inverted_index(corpus)
            avg_doc_length = np.mean(list(doc_lengths.values())) if doc_lengths else 0.0
            retriever = PRFRetriever(
                inverted_index,
                doc_lengths,
                doc_tokens,
                avg_doc_length,
                num_docs=len(corpus),
            )
            results = retriever.retrieve(
                query,
                top_k=int(top_k),
                prf_top_docs=int(prf_top_docs),
                expansion_terms=int(expansion_terms),
            )

            st.subheader("Results")
            for i, (doc_id, score) in enumerate(results, start=1):
                doc = corpus.get(doc_id, {})
                title = doc.get("title", "")
                text = doc.get("text", "")
                st.markdown(f"**{i}. {title}** — `{doc_id}`")
                st.caption(f"score: {score:.4f}")
                st.write((text or "")[:800])
                st.markdown("---")

    else:  # Hà — BM25+ (ScienceIRSystem)
        st.header("Hà — BM25+ (ScienceIRSystem)")
        
        data_path_input = st.text_input("Local Data path (fallback)", value="data/scifact")
        current_dir = Path(__file__).parent
        ha_folder = current_dir / "Hà"
        ha_cls = load_science_ir_system(ha_folder)
        
        if ha_cls is None:
            st.error(f"Không tìm thấy code.py tại {ha_folder}")
            return

        if "ha_system" not in st.session_state:
            st.session_state.ha_system = ha_cls(data_path_input)
        
        ha = st.session_state.ha_system

        # Nút nạp dữ liệu thủ công
        if st.button("🔄 Load & Index Data"):
            try:
                with st.spinner("Đang nạp dữ liệu..."):
                    if uploaded is not None:
                        content = uploaded.getvalue()
                        docs_raw = parse_jsonl_bytes(content)
                        ha.corpus = {}
                        for doc in docs_raw:
                            did = str(doc.get("_id") or doc.get("id"))
                            ha.corpus[did] = (doc.get("title", "") + " ") * 2 + doc.get("text", "")
                        st.success(f"✅ Đã nạp {len(ha.corpus)} tài liệu từ file Upload!")
                    else:
                        ha.load_data()
                        st.info("📁 Đã nạp dữ liệu từ thư mục local.")
                    
                    ha.build_index()
                    st.session_state.ha_system = ha # Lưu trạng thái đã index
                    st.success("🚀 Index đã sẵn sàng!")
            except Exception as e:
                st.error(f"Lỗi: {e}")

        st.divider()
        q = st.text_area("Nhập truy vấn:")
        k = st.number_input("Top-K results", min_value=1, value=10)
        
        if st.button("🔍 Search (Hà)"):
            # --- LOGIC TỰ ĐỘNG FIX LỖI "DỮ LIỆU TRỐNG" ---
            if ha.bm25 is None:
                if uploaded is not None:
                    with st.spinner("Phát hiện file upload, đang tự động Index..."):
                        content = uploaded.getvalue()
                        docs_raw = parse_jsonl_bytes(content)
                        ha.corpus = {str(d.get("_id")): (d.get("title", "") + " ") * 2 + d.get("text", "") for d in docs_raw}
                        ha.build_index()
                        st.session_state.ha_system = ha
                elif os.path.exists(data_path_input):
                    with st.spinner("Tự động nạp dữ liệu từ local path..."):
                        ha.load_data()
                        ha.build_index()
                        st.session_state.ha_system = ha
                else:
                    st.warning("⚠️ Dữ liệu trống! Vui lòng upload file hoặc kiểm tra đường dẫn local rồi nhấn 'Load & Index Data'.")
                    return

            # Thực hiện search sau khi đã đảm bảo có Index
            try:
                results = ha.retrieve(q, top_k=int(k))
                if not results:
                    st.info("Không có kết quả.")
                else:
                    for i, (doc_id, score) in enumerate(results.items(), start=1):
                        with st.container():
                            st.markdown(f"**{i}. Tài liệu {doc_id}** (Score: {score:.4f})")
                            text = ha.corpus.get(doc_id, "")
                            st.write(text[:800] + "...")
                            st.markdown("---")
            except Exception as e:
                st.error(f"Lỗi truy vấn: {e}")

if __name__ == "__main__":
    main()
