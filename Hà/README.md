# Scientific Information Retrieval System (BM25+)

Dự án triển khai hệ thống truy xuất thông tin (Information Retrieval) chuyên biệt cho các văn bản khoa học, sử dụng thuật toán **BM25+** trên tập dữ liệu **SciFact**. Hệ thống tập trung tối ưu hóa độ chính xác thông qua tiền xử lý ngôn ngữ tự nhiên và kỹ thuật hiệu chỉnh trọng số văn bản.

## 📌 Tính năng chính

* **Thuật toán BM25+**: Sử dụng biến thể cải tiến của BM25 với tham số $k1=1.2$ và $b=0.5$ để xử lý tốt sự khác biệt về độ dài tài liệu khoa học.
* **Tiền xử lý nâng cao**: Tích hợp **Porter Stemming** để đưa từ về dạng gốc và loại bỏ **Stopwords**, giúp tăng khả năng khớp từ khóa chính xác.
* **Kỹ thuật Boosting**: Thực hiện nhân đôi nội dung phần **Tiêu đề (Title)** để tăng trọng số cho các từ khóa xuất hiện tại đây.
* **Đánh giá chuẩn BEIR**: Tích hợp thư viện `beir` để tính toán các chỉ số chuyên sâu như NDCG, MAP, Recall và Precision.

## 📂 Cấu trúc thư mục dữ liệu

Để hệ thống hoạt động local, dữ liệu cần được tổ chức như sau:

```text
.
├── code.py                 # File thực thi chính
└── data/
    └── scifact/
        ├── corpus.jsonl    # Chứa 5,183 tài liệu khoa học
        ├── queries.jsonl   # Chứa 300 câu truy vấn test
        └── qrels/
            └── test.tsv    # File nhãn đúng (ground truth)
```

## 🛠 Yêu cầu cài đặt

Trước khi chạy, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết bằng lệnh sau:

```bash
pip install pandas nltk rank-bm25 beir
```

## 🚀 Hướng dẫn sử dụng

1. **Chuẩn bị dữ liệu**:  
   Đảm bảo dữ liệu đã được đặt đúng vị trí theo cấu trúc thư mục như mô tả ở phần [Cấu trúc thư mục](#-cấu-trúc-thư-mục).

2. **Chạy chương trình**:  
   Mở terminal và thực hiện lệnh sau:

   ```bash
   python code.py

3. **Quá trình thực hiện**:
    Hệ thống sẽ tự động thực hiện quy trình: Tải dữ liệu -> Tiền xử lý & Indexing -> Truy vấn -> Đánh giá chỉ số.

## Giao diện tương tác (UI)

Có một ứng dụng Streamlit nhẹ tại `Hà/ui.py` dùng lại `ScienceIRSystem` trong `code.py` để:
- Load data local
- Build index
- Search single query
- Chạy evaluation (in ra console)

Chạy UI:

```
python -m pip install -r Hà/requirements.txt
streamlit run Hà/ui.py
```

Sau khi chạy, mở URL do Streamlit in ra (thường http://localhost:8501).

## 📊 Kết quả thực nghiệm

Hệ thống đã được đánh giá chi tiết dựa trên tập dữ liệu SciFact với **300 truy vấn mẫu**. Kết quả thu được từ quá trình đánh giá như sau:

| Metric | @1 | @5 | @10 | @100 |
| :--- | :---: | :---: | :---: | :---: |
| **NDCG** | 0.54667 | 0.66229 | 0.68311 | 0.70947 |
| **Recall** | 0.53083 | 0.75511 | 0.81394 | 0.92756 |
| **MAP** | 0.53083 | 0.62577 | 0.63532 | 0.64190 |
| **Precision** | 0.54667 | 0.16333 | 0.09000 | 0.01047 |

### 🔍 Phân tích kết quả

Dựa trên bảng số liệu thực nghiệm, hệ thống cho thấy hiệu năng truy xuất ấn tượng:

* **Recall@100 (~92.7%)**: Hệ thống có khả năng bao phủ cực tốt. Điều này chứng tỏ phần lớn các tài liệu có liên quan trong cơ sở dữ liệu đều được tìm thấy và đưa vào danh sách kết quả trả về.
* **NDCG@10 (0.683)**: Hiệu quả xếp hạng đạt mức khá cao. Các tài liệu liên quan nhất có xu hướng xuất hiện ở những vị trí đầu tiên, giúp tối ưu hóa trải nghiệm tìm kiếm của người dùng.
* **Độ chính xác (Precision)**: Các chỉ số cho thấy sự cân bằng ổn định giữa việc tìm kiếm rộng và việc đảm bảo tính liên quan của các kết quả hàng đầu.

### 📝 Quy trình xử lý (Workflow)

1. **Data Loading**: Đọc dữ liệu từ file JSONL và thực hiện Boosting Title.
2. **Preprocessing**: Tokenization → Lowercase → Stopwords Removal → Porter Stemming.
3. **Indexing**: Xây dựng bộ chỉ mục dựa trên thuật toán BM25+.
4. **Retrieval**: Tiền xử lý truy vấn và tính điểm số tương quan với toàn bộ corpus.
5. **Evaluation**: So sánh kết quả với file test.tsv bằng thư viện BEIR.
