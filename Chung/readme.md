# 📄 Information Retrieval System (BM25 + PRF) on SciFact

## 📌 Giới thiệu
Dự án này xây dựng một hệ thống **Information Retrieval (IR)** cổ điển để truy xuất tài liệu khoa học từ tập dữ liệu **:contentReference[oaicite:0]{index=0}**. Hệ thống sử dụng **BM25** làm hàm xếp hạng chính, kết hợp với **Pseudo-Relevance Feedback (PRF)** để mở rộng truy vấn và cải thiện chất lượng tìm kiếm, sau đó đánh giá bằng các metric chuẩn trong IR.

---

## ⚙️ Tổng quan pipeline

Query
↓
Tokenize + Preprocess
↓
BM25 Retrieval (lần 1)
↓
Top-K Documents
↓
Extract Expansion Terms (PRF)
↓
Expand Query
↓
BM25 Retrieval (lần 2)
↓
Final Ranking
↓
Evaluation (NDCG, Recall, MAP, Precision)


---

## 🔤 Tiền xử lý (Preprocessing)
Hệ thống áp dụng các bước xử lý văn bản cơ bản:
- Chuyển toàn bộ văn bản về lowercase  
- Loại bỏ ký tự đặc biệt  
- Loại bỏ stopwords (NLTK)  
- Stemming bằng Porter Stemmer  

Ví dụ:

"Studies are running quickly"
→ ["studi", "run", "quickli"]


---

## 📚 Inverted Index
Dữ liệu được tổ chức dưới dạng inverted index:

term → {doc_id: term_frequency}


Ngoài ra còn lưu:
- `doc_lengths`: độ dài mỗi document  
- `doc_tokens`: danh sách token của từng document  

Một kỹ thuật quan trọng được áp dụng là **boost title**:

text = (title + " ") * 2 + content

→ giúp tăng trọng số tiêu đề trong quá trình tìm kiếm.

---

## ⚙️ BM25 Retrieval
Hệ thống sử dụng thuật toán **:contentReference[oaicite:1]{index=1}** để tính điểm.

Công thức:

score = IDF * ((k1 + 1) * tf) / (k1 * norm + tf)


Trong đó:
- `tf`: số lần xuất hiện của từ trong document  
- `idf`: độ hiếm của từ  
- `norm`: chuẩn hóa theo độ dài document  
- `k1 = 1.2`, `b = 0.5`  

Ý nghĩa:
- Từ hiếm → quan trọng hơn  
- Xuất hiện nhiều → điểm cao hơn  
- Document dài → bị giảm điểm  

---

## 🔁 Pseudo-Relevance Feedback (PRF)
PRF giúp cải thiện truy vấn bằng cách sử dụng thông tin từ kết quả ban đầu.

Quy trình:
1. Retrieve lần 1 bằng BM25  
2. Lấy top 10 documents  
3. Trích xuất 5 từ quan trọng nhất (theo IDF)  
4. Mở rộng query:

expanded_query = query * 3 + expansion_terms

5. Retrieve lần 2  

Lợi ích:
- Tăng Recall đáng kể  
- Bổ sung ngữ cảnh cho query  
- Cải thiện chất lượng ranking  

---

## 📊 Evaluation Metrics
Hệ thống được đánh giá bằng các metric chuẩn:

- **NDCG@K**: đánh giá chất lượng ranking, ưu tiên document đúng ở top  
- **Recall@K**: tỷ lệ document đúng được tìm thấy  
- **MAP@K**: đánh giá thứ tự xuất hiện của document đúng  
- **Precision@K**: độ chính xác trong top K  

---

## 📈 Kết quả

========== FINAL RESULTS ==========

NDCG
@1: 0.49333 @5: 0.63477 @10: 0.65613 @100: 0.68409

RECALL
@1: 0.48056 @5: 0.75650 @10: 0.81633 @100: 0.93267

MAP
@1: 0.49333 @5: 0.58806 @10: 0.59860 @100: 0.60588

PRECISION
@1: 0.49333 @5: 0.16267 @10: 0.09033 @100: 0.01057


Nhận xét:
- PRF giúp tăng Recall rất rõ  
- NDCG và MAP cải thiện so với BM25 thuần  
- Precision giảm khi K tăng (đặc trưng bình thường của IR)  

---

## 🚀 Cách chạy
Cài đặt thư viện:

pip install pandas numpy tqdm nltk


Tải stopwords (chạy 1 lần):
```python
import nltk
nltk.download('stopwords')

Chuẩn bị dữ liệu:

scifact/
 ├── corpus.jsonl
 ├── queries.jsonl
 └── qrels/
      └── test.tsv
```

Chạy chương trình:

```
python ir_project.py
```

## Giao diện tương tác (UI)

Có thể thử nghiệm BM25+PRF tương tác và chạy đánh giá batch từ giao diện.

Chạy UI:

```
python -m pip install -r Chung/requirements.txt
streamlit run Chung/ui.py
```

UI cho phép upload hoặc trỏ tới folder data (`scifact/scifact`) chứa `corpus.jsonl`, `queries.jsonl`, và `qrels/test.tsv`.
