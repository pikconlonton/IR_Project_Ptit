# IR_Project_Ptit

Project chứa một số triển khai baseline cho bài toán Information Retrieval trên dataset SciFact.

Mục tiêu nhanh: có một UI duy nhất để thử 3 phương pháp (An: BM25+MMR, Chung: BM25+PRF, Hà: BM25+) — dùng `ui_multi.py`.

Chạy UI (shell)

1) Tạo virtualenv và kích hoạt (khuyến nghị):

```
python -m venv .venv
source .venv/bin/activate
```

2) Cài dependencies (cách an toàn: cài từng folder để tránh xung đột):

```
python -m pip install -r An/requirements.txt
python -m pip install -r Chung/requirements.txt
python -m pip install -r Hà/requirements.txt
```

Hoặc cài gộp (ít chính xác hơn nhưng nhanh):

```
python -m pip install streamlit scikit-learn rank-bm25 numpy pandas nltk beir
```

3) Chạy multi-method UI:

```
streamlit run ui_multi.py
```

4) (Tuỳ chọn) Nếu muốn chạy UI cho từng thư mục riêng:

```fish
streamlit run An/ui.py      # An: BM25 + MMR
streamlit run Chung/ui.py   # Chung: BM25 + PRF
streamlit run Hà/ui.py      # Hà: BM25+
```

Sau khi chạy lệnh `streamlit run ...`, mở trình duyệt theo URL do Streamlit in ra (thường http://localhost:8501).

Ghi chú ngắn
- Với corpus lớn, một số bước (TF-IDF, build index) có thể tốn thời gian và bộ nhớ. Tôi có thể thêm caching nếu muốn.
- Nếu có lỗi import khi chạy `ui_multi.py`, đảm bảo bạn đang chạy từ thư mục gốc của repo và các dependencies đã được cài.

Nếu muốn, tôi có thể thêm phần hướng dẫn cấu hình nhanh (ví dụ: cách chuẩn hoá đường dẫn dữ liệu `scifact/`) hoặc localize UI hoàn toàn sang tiếng Việt.

