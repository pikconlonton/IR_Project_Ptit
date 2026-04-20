import json
import os
import re
import csv
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Plus
from beir.retrieval.evaluation import EvaluateRetrieval

# Tải dữ liệu ngôn ngữ cần thiết (chỉ cần chạy lần đầu)
nltk.download('stopwords', quiet=True)

class ScienceIRSystem:
    def __init__(self, data_path="data/scifact"):
        """
        Khởi tạo hệ thống với đường dẫn thư mục dữ liệu local.
        Mặc định tìm trong thư mục 'data/scifact' cùng cấp với file code.
        """
        self.data_path = data_path
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.corpus = {}
        self.queries = {}
        self.qrels_test = {}
        self.doc_ids = []
        self.bm25 = None

    def tokenize_advanced(self, text):
        """Tiền xử lý: Chuyển lowercase, xóa ký tự đặc biệt, lọc stopwords và Stemming."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [self.stemmer.stem(w) for w in text.split() if w not in self.stop_words]
        return tokens

    def load_data(self):
        """Load dữ liệu từ các file jsonl và tsv cục bộ."""
        print(f"📂 Đang tải dữ liệu từ: {os.path.abspath(self.data_path)}")
        
        # 1. Load Corpus (Nhân đôi Title để tăng trọng số như trong notebook)
        corpus_path = os.path.join(self.data_path, "corpus.jsonl")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                did = str(doc.get("_id") or doc.get("id"))
                title = doc.get("title", "")
                text = doc.get("text", "")
                # Kỹ thuật Boosting: Nhân đôi tiêu đề
                self.corpus[did] = (title + " ") * 2 + text

        # 2. Load Queries
        queries_path = os.path.join(self.data_path, "queries.jsonl")
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                self.queries[str(q.get("_id") or q.get("id"))] = q.get("text", "")

        # 3. Load Qrels (Nhãn kiểm thử)
        qrels_path = os.path.join(self.data_path, "qrels", "test.tsv")
        if os.path.exists(qrels_path):
            with open(qrels_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader, None)  # Bỏ qua header
                for row in reader:
                    if len(row) < 3: continue
                    q_id, d_id, score = str(row[0]), str(row[1]), int(row[2])
                    if q_id not in self.qrels_test: 
                        self.qrels_test[q_id] = {}
                    self.qrels_test[q_id][d_id] = score
        
        print(f"✅ Đã load {len(self.corpus)} tài liệu và {len(self.qrels_test)} test queries.")

    def build_index(self):
        """Xây dựng bộ chỉ mục BM25+ với tham số k1=1.2, b=0.5."""
        self.doc_ids = list(self.corpus.keys())
        print("⏳ Đang tiền xử lý (Stemming) và xây dựng Index BM25+...")
        
        tokenized_corpus = [self.tokenize_advanced(self.corpus[did]) for did in self.doc_ids]
        self.bm25 = BM25Plus(tokenized_corpus, k1=1.2, b=0.5)
        print("✅ Indexing hoàn tất!")

    def retrieve(self, query_text, top_k=100):
        """Thực hiện truy vấn và trả về kết quả kèm score."""
        query_tokens = self.tokenize_advanced(query_text)
        scores = self.bm25.get_scores(query_tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        return {self.doc_ids[i]: float(scores[i]) for i in top_idx}

    def run_evaluation(self):
        """Chạy đánh giá và in bảng metrics NDCG, Recall, MAP, Precision."""
        print("⏳ Đang thực hiện tìm kiếm trên tập Test và tính toán chỉ số...")
        all_results = {}
        for q_id in self.qrels_test:
            if q_id in self.queries:
                all_results[q_id] = self.retrieve(self.queries[q_id], top_k=100)

        # Sử dụng BEIR để đánh giá
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            self.qrels_test, all_results, k_values=[1, 5, 10, 100]
        )

        # Định dạng bảng kết quả giống hệt notebook
        metrics_data = {
            "Metric": ["NDCG", "Recall", "MAP", "Precision"],
            "@1":   [ndcg["NDCG@1"],   recall["Recall@1"], _map["MAP@1"],   precision["P@1"]],
            "@5":   [ndcg["NDCG@5"],   recall["Recall@5"], _map["MAP@5"],   precision["P@5"]],
            "@10":  [ndcg["NDCG@10"],  recall["Recall@10"],_map["MAP@10"],  precision["P@10"]],
            "@100": [ndcg["NDCG@100"], recall["Recall@100"],_map["MAP@100"],precision["P@100"]],
        }
        df_metrics = pd.DataFrame(metrics_data)
        
        print("\n" + "="*55)
        print(df_metrics.to_string(index=False))
        print("="*55)

if __name__ == "__main__":
    # HƯỚNG DẪN: 
    # 1. Đặt các file corpus.jsonl, queries.jsonl vào thư mục 'data/scifact'
    # 2. Đặt file test.tsv vào thư mục 'data/scifact/qrels'
    
    # Bạn có thể thay đổi đường dẫn tại đây:
    PATH_TO_DATA = "data/scifact" 
    
    ir_system = ScienceIRSystem(PATH_TO_DATA)
    
    if not os.path.exists(PATH_TO_DATA):
        print(f"❌ Lỗi: Không tìm thấy thư mục dữ liệu tại {PATH_TO_DATA}")
    else:
        ir_system.load_data()
        ir_system.build_index()
        ir_system.run_evaluation()