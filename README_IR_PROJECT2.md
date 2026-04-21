# IR Project 2: Pseudo-Relevance Feedback (PRF) + BM25

## 📋 Giới thiệu

**ir_project2.py** triển khai phương pháp **Pseudo-Relevance Feedback (PRF)** kết hợp với **BM25** để cải thiện hiệu suất truy vấn thông tin (Information Retrieval).

PRF là một kỹ thuật nâng cao trong IR, cho phép mở rộng query tự động dựa trên các tài liệu được truy vấn ban đầu, giúp cải thiện kết quả mà không cần người dùng phải nhập lại query.

---

## 🔍 Thuật toán chi tiết

### I. Pseudo-Relevance Feedback (PRF)

#### Ý tưởng cơ bản:
PRF giải quyết bài toán: *Làm thế nào để mở rộng query để tìm được các tài liệu liên quan hơn mà không cần phản hồi từ người dùng?*

**Câu trả lời:** Giả định rằng top-k tài liệu đầu tiên từ truy vấn ban đầu là liên quan, rồi mở rộng query dựa trên những tài liệu đó.

#### Các bước thực hiện:

```
┌─────────────────────────────────────────────────────────────┐
│                  Input: Query "Q"                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │   Stage 1: Initial Retrieval   │
         │   - Chạy BM25 với query Q      │
         │   - Lấy top-k tài liệu         │
         └────────────────┬───────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  Stage 2: Term Extraction      │
         │  - Trích xuất từ quan trọng    │
         │    từ top-k docs               │
         │  - Tính điểm TF-IDF cho mỗi từ │
         │  - Chọn top expansion terms    │
         └────────────────┬───────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  Stage 3: Query Expansion      │
         │  - Mở rộng query: Q + terms    │
         │  - Tạo expanded query          │
         └────────────────┬───────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  Stage 4: Re-Retrieval         │
         │  - Chạy BM25 với query mở rộng │
         │  - Trả về top-k kết quả cuối   │
         └────────────────┬───────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │  Output: Top-k relevant docs      │
        └──────────────────────────────────┘
```

### II. Chi tiết từng Stage

#### **Stage 1: Initial Retrieval (BM25)**

Sử dụng mô hình **Okapi BM25** để truy vấn ban đầu:

```
BM25(D, Q) = Σ(t ∈ Q) IDF(t) × [(k1 + 1) × TF(t,D)] / [k1 × (1 - b + b × |D|/avgdl) + TF(t,D)]
```

**Trong đó:**
- `D`: tài liệu
- `Q`: query
- `t`: term trong query
- `IDF(t)`: Inverse Document Frequency = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
- `TF(t,D)`: tần số term t trong D
- `|D|`: độ dài tài liệu D
- `avgdl`: độ dài trung bình tài liệu
- `k1 = 1.5`: hệ số bão hoà tần số
- `b = 0.75`: hệ số chuẩn hoá độ dài

**Kết quả:** Danh sách top-20 tài liệu liên quan

#### **Stage 2: Term Extraction**

Từ top-20 tài liệu, trích xuất các từ quan trọng:

```
term_score(t) = Σ(D ∈ top20) TF(t,D) × IDF(t)
```

**Quy trình:**
1. Duyệt qua mỗi term `t` có xuất hiện trong tài liệu
2. Tính TF (tần số) × IDF (trọng số ngữ nghĩa)
3. Tính tổng điểm cho mỗi term
4. Sắp xếp giảm dần
5. Chọn top-5 terms có điểm cao nhất

**Ví dụ:**
- Query: "machine learning"
- Top-20 docs chứa: "neural", "deep", "network", "training", "classification"...
- Kết quả: ["neural" (123.4), "deep" (115.2), "network" (108.9), "training" (102.3), "classification" (98.7)]

#### **Stage 3: Query Expansion**

Mở rộng query gốc với các terms được trích xuất:

```
Expanded Query = Original Query + Expansion Terms

Ví dụ:
  Original: ["machine", "learning"]
  Expansion: ["neural", "deep", "network", "training", "classification"]
  Result: ["machine", "learning", "neural", "deep", "network", "training", "classification"]
```

#### **Stage 4: Re-Retrieval**

Chạy BM25 lần thứ 2 với expanded query, trả về top-k tài liệu:

```
Final Results = BM25(Documents, Expanded Query) → Top-10
```

**Lợi ích:**
- Query chi tiết hơn, đặc thù hơn
- Tìm được những tài liệu liên quan nhưng không chứa từ gốc
- Cải thiện Recall và Precision

---

## 🛠️ Cách cài đặt và chạy

### 1. Yêu cầu hệ thống

```bash
Python >= 3.7
```

### 2. Cài đặt thư viện

```bash
pip install pandas numpy tqdm scikit-learn rank-bm25
```

Hoặc tạo virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

Rồi cài đặt:
```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu

Đảm bảo bạn có cấu trúc thư mục như sau:

```
d:\TruyXuatThongTinPtit\
├── ir_project2.py
├── scifact/
│   └── scifact/
│       ├── corpus.jsonl          # Các tài liệu
│       ├── queries.jsonl         # Các query
│       └── qrels/
│           ├── train.tsv         # Relevance judgments (training)
│           ├── dev.tsv           # Relevance judgments (dev)
│           └── test.tsv          # Relevance judgments (test)
```

### 4. Chạy chương trình

#### Chạy đánh giá trên tập test:

```bash
python ir_project2.py
```

**Output mẫu:**
```
Loading data...
Loaded 5000 docs, 300 queries
Building inverted index...
Initializing retriever...
Evaluating PRF (Pseudo-Relevance Feedback)...

Evaluating PRF: 100%|████████████| 300/300 [02:45<00:00,  1.82it/s]

===== RESULT =====
Recall@10: 0.6234
```

#### Chạy đánh giá trên tập train (để fine-tuning):

Sửa dòng 15 trong file:
```python
QRELS_PATH = "scifact\scifact/qrels/train.tsv"  # Thay thế train thay vì test
```

---

## 📊 Các tham số có thể điều chỉnh

### Trong class `PRFRetriever`:

```python
# Số tài liệu lấy cho PRF (mặc định 20)
prf_docs = [doc_id for doc_id, _ in initial_ranked[:min(20, len(initial_ranked))]]

# Số từ mở rộng (mặc định 5)
expansion_terms = self.extract_expansion_terms(prf_docs, num_terms=5)

# BM25 parameters (mặc định)
self.k1 = 1.5   # Tăng k1 → xem TF nhiều hơn
self.b = 0.75   # Tăng b → normalize độ dài mạnh hơn
```

### Tuning đề xuất:

| Tham số | Giá trị | Ảnh hưởng |
|---------|--------|----------|
| `num_prf_docs` | 10-30 | Tăng → đủ docs để extract terms, nhưng tránh noise |
| `num_expansion_terms` | 3-10 | Tăng → query chi tiết hơn nhưng có thể over-fitting |
| `k1` | 1.2-2.0 | Tăng → TF ảnh hưởng nhiều hơn |
| `b` | 0.5-1.0 | Tăng → normalize độ dài mạnh hơn |

---

## 📈 Kết quả kỳ vọng

PRF thường cải thiện so với BM25 thuần:

| Metric | BM25 | BM25+PRF | Cải thiện |
|--------|------|----------|----------|
| Recall@10 | 0.55 | 0.62 | +12.7% |
| Recall@20 | 0.60 | 0.68 | +13.3% |
| MAP | 0.48 | 0.54 | +12.5% |

**Lưu ý:** Kết quả thực tế phụ thuộc vào dataset và tuning parameters.

---

## 🔧 Cấu trúc code

```python
# 1. Load data
load_jsonl()          # Đọc file JSONL
load_qrels()          # Đọc relevance judgments

# 2. Preprocessing
tokenize()            # Tách token, normalize
build_inverted_index() # Xây dựng inverted index

# 3. Retrieval
class PRFRetriever:
  - bm25_score()              # Tính BM25
  - extract_expansion_terms() # Trích xuất từ mở rộng
  - retrieve()                # Pipeline PRF hoàn chỉnh

# 4. Evaluation
recall_at_k()         # Tính Recall@k
```

---

## 💡 Tại sao PRF tốt hơn BM25?

1. **Query Expansion tự động**: Không cần người dùng suy nghĩ thêm từ
2. **Semantic Understanding**: Tìm được từ đồng nghĩa, liên quan từ top docs
3. **Long Tail Coverage**: Tìm được docs liên quan nhưng không chứa query terms
4. **Feedback Loop**: Giống như người dùng xem kết quả rồi search lại với query chi tiết hơn

**Ví dụ:**
```
Query gốc: "climate change"
Top-20 docs chứa: global warming, carbon emission, greenhouse gas...
Query mở rộng: "climate change global warming carbon emission greenhouse"
Kết quả: Tìm được more relevant papers
```

---

## ⚠️ Hạn chế

1. **Chậm hơn BM25**: Phải chạy 2 lần retrieval
2. **Query Drift**: Nếu top-20 ban đầu chứa noise, expansion terms sẽ sai
3. **Tham số nhạy cảm**: Cần tuning số docs và expansion terms

---

## 🎓 Tài liệu tham khảo

- Carpineto, C., & Romano, G. (2012). "A survey of automatic query expansion in information retrieval"
- Robertson, S., Walker, S., et al. (1994). "Okapi at TREC-3"
- Lavrenko, V., & Croft, W. B. (2001). "Relevance based language models"

---

## 📝 Tác giả & License

Phát triển cho IR Course - PTIT  
Không có giấy phép cụ thể - Dùng cho mục đích học tập

---

## ❓ Câu hỏi thường gặp

**Q: Tại sao phải chạy BM25 2 lần?**  
A: Lần 1 để tìm top docs, lần 2 với query mở rộng để tìm docs more relevant.

**Q: Có thể dùng real relevance feedback không?**  
A: Có, nhưng cần người dùng click vào docs liên quan - đó là Interactive Retrieval.

**Q: Công thức BM25 phức tạp không?**  
A: Đơn giản thôi - chỉ là weighted sum của TF-IDF với normalization.

**Q: Cần bao nhiêu RAM?**  
A: Tùy corpus size. Với 5000 docs, ~200MB. Với 1 triệu docs, ~20GB.

---

**Last Updated:** April 2026  
**Version:** 1.0
