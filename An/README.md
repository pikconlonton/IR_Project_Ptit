# SciFact retrieval: BM25 + MMR re-rank

Baseline truy xuất tài liệu cho SciFact theo 2 tầng:

1) **BM25**: lấy danh sách ứng viên theo mức độ liên quan (lexical matching)
2) **MMR (Maximal Marginal Relevance)**: re-rank lại danh sách ứng viên để **vừa liên quan vừa đa dạng** (giảm các tài liệu na ná nhau)

Phù hợp để làm baseline IR nhanh trước khi thay bằng dense retrieval / cross-encoder.

## Data layout

This repo expects exactly the folder you already have:

- `scifact/corpus.jsonl`
- `scifact/queries.jsonl`
- `scifact/qrels/{train.tsv,test.tsv}`

## Setup

Create a virtual env and install deps.

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

Nếu bạn muốn chạy unit test:

```fish
pip install pytest
pytest -q
```

## Giải thích phương pháp

### 1) BM25 (tầng candidate retrieval)

BM25 là scoring dựa trên tf-idf + độ dài văn bản. Với query *q* và document *d*, BM25 trả về điểm liên quan. Ở đây dùng thư viện `rank-bm25`.

Trong code:

- Mỗi document được ghép `title + text` rồi token hoá đơn giản.
- Lấy top-N theo `--bm25-topk` làm tập ứng viên.

### 2) MMR (tầng re-rank để tăng diversity)

MMR chọn lần lượt các document sao cho:

\[
	ext{score}(d) = \lambda \cdot \text{sim}(q, d) - (1-\lambda) \cdot \max_{d' \in S} \text{sim}(d, d')
\]

- **sim(q, d)**: mức liên quan giữa query và doc
- **sim(d, d')**: độ giống nhau giữa doc đang xét và các doc đã chọn (để phạt trùng lặp)
- **S**: tập doc đã chọn
- **\(\lambda\)** (`--lambda-param`): cân bằng liên quan vs đa dạng

Trong baseline này:

- `sim` được tính bằng **cosine similarity trên TF-IDF** (scikit-learn).
- Sau MMR, lấy top-K cuối cùng theo `--mmr-topk`.

## Chạy retrieval

Example: run on the test set and print metrics + save a run file.

```fish
python3 -m scifact_retrieval.run \
  --data-dir scifact \
  --split test \
  --bm25-topk 20 \
  --mmr-topk 5 \
  --lambda-param 0.7 \
  --out runs/run_test.jsonl
```

## Grid search `bm25-topk` và `mmr-topk`

Để so sánh ảnh hưởng của 2 tham số `--bm25-topk` (candidate size) và `--mmr-topk` (final top-k sau MMR) lên metrics, bạn có thể chạy script grid sau:

```fish
python -m scifact_retrieval.grid_topk \
  --data-dir scifact \
  --split train \
  --bm25-topk-grid 20,50,100,200,500 \
  --mmr-topk-grid 5,10,20 \
  --lambda-param 0.7 \
  --eval-ks 5,10,20 \
  --max-queries 300 \
  --out runs/grid_topk_results.jsonl
```

Script sẽ in ra bảng TSV (copy vào Excel/Google Sheets được) gồm các cột `MRR@k`, `nDCG@k`, `R@k` cho từng cấu hình.

### Gợi ý chọn grid

- Luôn đảm bảo: `bm25-topk >= mmr-topk`.
- Nếu chấm @10: thường thử `mmr-topk-grid=10` và `bm25-topk-grid=50,100,200,500`.
- Nếu chấm @20: thử `mmr-topk-grid=20` và `bm25-topk-grid=100,200,500,1000`.

### Giới hạn số query khi chạy grid

Tham số `--max-queries` giúp chạy nhanh (mặc định là 300 query):

- `--max-queries 300`: chỉ evaluate 300 query đầu tiên.
- `--max-queries 0`: chạy toàn bộ query trong split.

### Kết quả grid (300 queries, split=train)

Kết quả dưới đây được sinh bởi file: `runs/grid_topk_results_300.jsonl` (num_queries = 300).

Grid đã thử:

- `bm25-topk-grid`: 20, 50, 100
- `mmr-topk-grid`: 5, 10, 20
- `lambda-param`: 0.7

**Cấu hình tốt nhất trong grid này** (cao nhất theo `MRR@10`, đồng thời `nDCG@10` cũng cao nhất trong tập thử):

- `bm25-topk = 20`, `mmr-topk = 20`

Metrics tương ứng:

- `MRR@10 = 0.519829`
- `nDCG@10 = 0.561313`
- `R@20 = 0.861167`

> Lưu ý: đây chỉ là kết quả trên 300 query đầu tiên của split train. Để kết luận chắc hơn, bạn nên chạy thêm với `--max-queries 0` (full) hoặc lấy mẫu ngẫu nhiên nhiều lần.

### Tham số quan trọng

- `--data-dir`: đường dẫn tới folder `scifact/`
- `--split`: `train` hoặc `test` (để load qrels tương ứng và in metric)
- `--bm25-topk`: số ứng viên BM25 lấy ban đầu (tăng thì có thể tốt hơn nhưng chậm hơn)
- `--mmr-topk`: số kết quả cuối cùng sau MMR
- `--lambda-param`:
  - gần `1.0`: ưu tiên **liên quan** (gần giống BM25)
  - gần `0.0`: ưu tiên **đa dạng** (ít doc “giống nhau” hơn)
- `--out`: lưu output dạng JSONL

### Output

Chương trình in ra JSON metrics cơ bản (tính theo qrels của split):

- `P@K`: Precision@K
- `R@K`: Recall@K
- `MRR@K`: Mean Reciprocal Rank@K

File output JSONL (`--out`) có dạng:

- mỗi dòng: `{ "query_id": ..., "doc_ids": [...] }`

## Notes

- MMR dùng TF-IDF cosine similarity cho cả `sim(q,d)` và `sim(d,d')`.
- Baseline này **không** dùng stemming/lemmatization; nếu muốn cải thiện lexical matching có thể thử tokenizer tốt hơn.
