import json
import os
import re
import csv
import math
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Plus
from typing import Dict, List, Sequence

# Tải dữ liệu ngôn ngữ cần thiết
nltk.download('stopwords', quiet=True)

# --- CÁC HÀM ĐÁNH GIÁ THUẦN (MANUAL METRICS) ---
def precision_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    if k <= 0: return 0.0
    hits = sum(1 for doc_id in ranked[:k] if relevant.get(doc_id, 0) > 0)
    return hits / k

def recall_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    total_rel = sum(1 for v in relevant.values() if v > 0)
    if total_rel == 0: return 0.0
    hits = sum(1 for doc_id in ranked[:k] if relevant.get(doc_id, 0) > 0)
    return hits / total_rel

def average_precision(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    total_rel = sum(1 for v in relevant.values() if v > 0)
    if total_rel == 0: return 0.0
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            hits += 1
            sum_prec += hits / i
    return sum_prec / total_rel

def dcg_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        rel = float(relevant.get(doc_id, 0))
        if rel <= 0: continue
        dcg += (2.0**rel - 1.0) / math.log2(i + 1.0)
    return dcg

def ndcg_at_k(ranked: Sequence[str], relevant: Dict[str, int], k: int) -> float:
    ideal_ranked = [d for d, r in sorted(relevant.items(), key=lambda x: x[1], reverse=True) if r > 0]
    idcg = dcg_at_k(ideal_ranked, relevant, k)
    if idcg <= 0: return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg

# --- CLASS HỆ THỐNG TRUY VẤN ---
class ScienceIRSystem:
    def __init__(self, data_path="data/scifact"):
        self.data_path = data_path
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.corpus = {}
        self.queries = {}
        self.qrels_test = {}
        self.doc_ids = []
        self.bm25 = None

    def tokenize_advanced(self, text):
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [self.stemmer.stem(w) for w in text.split() if w not in self.stop_words]
        return tokens

    def load_data(self):
        print(f"📂 Đang tải dữ liệu từ: {os.path.abspath(self.data_path)}")
        # Load Corpus
        with open(os.path.join(self.data_path, "corpus.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                did = str(doc.get("_id") or doc.get("id"))
                self.corpus[did] = (doc.get("title", "") + " ") * 2 + doc.get("text", "")
        # Load Queries
        with open(os.path.join(self.data_path, "queries.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                self.queries[str(q.get("_id") or q.get("id"))] = q.get("text", "")
        # Load Qrels
        qrels_path = os.path.join(self.data_path, "qrels", "test.tsv")
        if os.path.exists(qrels_path):
            with open(qrels_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader, None)
                for row in reader:
                    q_id, d_id, score = str(row[0]), str(row[1]), int(row[2])
                    if q_id not in self.qrels_test: self.qrels_test[q_id] = {}
                    self.qrels_test[q_id][d_id] = score
        print(f"✅ Đã load {len(self.corpus)} tài liệu và {len(self.qrels_test)} test queries.")

    def build_index(self):
        self.doc_ids = list(self.corpus.keys())
        print("⏳ Đang xây dựng Index BM25+...")
        tokenized_corpus = [self.tokenize_advanced(self.corpus[did]) for did in self.doc_ids]
        self.bm25 = BM25Plus(tokenized_corpus, k1=1.2, b=0.5)
        print("✅ Indexing hoàn tất!")

    def retrieve(self, query_text, top_k=100):
        query_tokens = self.tokenize_advanced(query_text)
        if not query_tokens:
            return {} # Trả về dict rỗng để không lỗi .items()
            
        scores = self.bm25.get_scores(query_tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        
        # Trả về Dictionary để UI dùng được .items()
        return {self.doc_ids[i]: float(scores[i]) for i in top_idx}

    def run_evaluation(self, k_values=[1, 5, 10, 100]):
        print(f"⏳ Đang đánh giá trên {len(self.qrels_test)} queries...")
        
        # Lưu trữ kết quả theo từng mốc K
        metrics_results = {k: {"ndcg": [], "recall": [], "map": [], "p": []} for k in k_values}

        for q_id, relevant_docs in self.qrels_test.items():
            if q_id in self.queries:
                # Lấy danh sách doc_ids từ kết quả tìm kiếm
                ranked_list = self.retrieve(self.queries[q_id], top_k=max(k_values))
                
                for k in k_values:
                    metrics_results[k]["ndcg"].append(ndcg_at_k(ranked_list, relevant_docs, k))
                    metrics_results[k]["recall"].append(recall_at_k(ranked_list, relevant_docs, k))
                    metrics_results[k]["map"].append(average_precision(ranked_list, relevant_docs, k))
                    metrics_results[k]["p"].append(precision_at_k(ranked_list, relevant_docs, k))

        # Tính trung bình cộng
        final_data = {
            "Metric": ["NDCG", "Recall", "MAP", "Precision"],
        }
        for k in k_values:
            final_data[f"@{k}"] = [
                sum(metrics_results[k]["ndcg"]) / len(self.qrels_test),
                sum(metrics_results[k]["recall"]) / len(self.qrels_test),
                sum(metrics_results[k]["map"]) / len(self.qrels_test),
                sum(metrics_results[k]["p"]) / len(self.qrels_test)
            ]

        df_metrics = pd.DataFrame(final_data)
        print("\n" + "="*55)
        print(df_metrics.to_string(index=False))
        print("="*55)

if __name__ == "__main__":
    PATH_TO_DATA = "data/scifact" 
    ir_system = ScienceIRSystem(PATH_TO_DATA)
    if os.path.exists(PATH_TO_DATA):
        ir_system.load_data()
        ir_system.build_index()
        ir_system.run_evaluation()