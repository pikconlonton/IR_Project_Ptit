from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import streamlit as st

# Import the existing IR system implementation
from code import ScienceIRSystem


def get_system(data_path: str) -> ScienceIRSystem:
    if "ir_system" not in st.session_state:
        st.session_state.ir_system = ScienceIRSystem(data_path)
    return st.session_state.ir_system


def main() -> None:
    st.set_page_config(page_title="Hà — BM25+ UI", layout="wide")
    st.title("BM25+ Retrieval — Hà's implementation")

    st.sidebar.header("Data & Actions")
    data_dir = st.sidebar.text_input("Local data directory", value="data/scifact")
    st.sidebar.markdown("\n")

    if st.sidebar.button("Load data"):
        ir = get_system(data_dir)
        try:
            ir.data_path = data_dir
            ir.load_data()
            st.sidebar.success(f"Loaded {len(ir.corpus)} docs")
        except Exception as e:
            st.sidebar.error(f"Load failed: {e}")

    if st.sidebar.button("Build index"):
        if "ir_system" not in st.session_state:
            st.sidebar.warning("Load data first")
        else:
            ir = st.session_state.ir_system
            try:
                ir.build_index()
                st.sidebar.success("Index built")
            except Exception as e:
                st.sidebar.error(f"Index build failed: {e}")

    if st.sidebar.button("Run evaluation"):
        if "ir_system" not in st.session_state:
            st.sidebar.warning("Load data and build index first")
        else:
            ir = st.session_state.ir_system
            try:
                # run_evaluation prints results to stdout; capture by showing st.info
                st.info("Running evaluation — this may take a while")
                ir.run_evaluation()
                st.success("Evaluation finished (see console output)")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    st.markdown("---")
    st.header("Single Query Search")
    q = st.text_area("Enter query text", height=120)
    k = st.number_input("Top-k", min_value=1, max_value=500, value=20)
    if st.button("Search"):
        if "ir_system" not in st.session_state:
            st.warning("Load data and build index first")
        else:
            ir = st.session_state.ir_system
            if ir.bm25 is None:
                st.warning("Index not built yet — building now")
                ir.build_index()
            try:
                results = ir.retrieve(q, top_k=int(k))
                if not results:
                    st.info("No results")
                else:
                    for i, (doc_id, score) in enumerate(results.items(), start=1):
                        doc_text = ir.corpus.get(doc_id, "")
                        snippet = doc_text[:600] + (
                            "..." if len(doc_text) > 600 else ""
                        )
                        st.markdown(f"**{i}. {doc_id}** — score: {score:.4f}")
                        st.write(snippet)
                        st.markdown("---")
            except Exception as e:
                st.error(f"Search failed: {e}")


if __name__ == "__main__":
    main()
