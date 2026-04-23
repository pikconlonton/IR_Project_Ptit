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

Kết quả dưới đây được sinh bởi file: `runs/grid_topk_results_2.jsonl` (num_queries = 300).

Grid đã thử:

- `bm25-topk-grid`: 20, 50, 100
- `mmr-topk-grid`: 5, 10, 20
- `lambda-param`: 0.7

Dưới đây là tóm tắt các cấu hình và một số metric chính (lấy trực tiếp từ file JSONL):

| bm25_topk | mmr_topk | P@1   | R@1   | MRR@1  | nDCG@1 | P@5   | R@5   | MRR@5  | nDCG@5 | P@10  | R@10  | MRR@10               | nDCG@10              | R@20  |
|---------:|--------:|:------|:------|:-------|:------|:------|:------|:-------|:------|:------|:------|:---------------------|:---------------------|:------|
| 1        | 1       | 0.5200 | 0.4987 | 0.5200 | 0.5200 | 0.1040 | 0.4987 | 0.5200 | 0.5032 | 0.0520 | 0.4987 | 0.5200               | 0.5032004268555648   | 0.4987 |
| 5        | 1       | 0.4333 | 0.4193 | 0.4333 | 0.4333 | 0.0867 | 0.4193 | 0.4333 | 0.4222 | 0.0433 | 0.4193 | 0.43333333333333335  | 0.42217135917450105  | 0.4193 |
| 5        | 5       | 0.4333 | 0.4193 | 0.4333 | 0.4333 | 0.1487 | 0.6846 | 0.5361 | 0.5678 | 0.0743 | 0.6846 | 0.5360555555555556   | 0.5678107707425408   | 0.6846 |
| 10       | 1       | 0.4267 | 0.4126 | 0.4267 | 0.4267 | 0.0853 | 0.4126 | 0.4267 | 0.4155 | 0.0427 | 0.4126 | 0.4266666666666667   | 0.41550469250783434  | 0.4126 |
| 10       | 5       | 0.4267 | 0.4126 | 0.4267 | 0.4267 | 0.1400 | 0.6529 | 0.5218 | 0.5486 | 0.0700 | 0.6529 | 0.5218333333333333   | 0.5486138270334566   | 0.6529 |
| 10       | 10      | 0.4267 | 0.4126 | 0.4267 | 0.4267 | 0.1400 | 0.6529 | 0.5218 | 0.5486 | 0.0880 | 0.7972 | 0.538398148148148    | 0.5959672168573347   | 0.7972 |
| 20       | 1       | 0.4200 | 0.4059 | 0.4200 | 0.4200 | 0.0840 | 0.4059 | 0.4200 | 0.4088 | 0.0420 | 0.4059 | 0.4200               | 0.4088380258411677   | 0.4059 |
| 20       | 5       | 0.4200 | 0.4059 | 0.4200 | 0.4200 | 0.1340 | 0.6262 | 0.5078 | 0.5317 | 0.0670 | 0.6262 | 0.5077777777777778   | 0.5317414293894098   | 0.6262 |
| 20       | 10      | 0.4200 | 0.4059 | 0.4200 | 0.4200 | 0.1340 | 0.6262 | 0.5078 | 0.5317 | 0.0773 | 0.7147 | 0.5198293650793651   | 0.5613134263918764   | 0.7147 |
| 20       | 20      | 0.4200 | 0.4059 | 0.4200 | 0.4200 | 0.1340 | 0.6262 | 0.5078 | 0.5317 | 0.0773 | 0.7147 | 0.5198293650793651   | 0.5613134263918764   | 0.8612 |
| 50       | 1       | 0.4133 | 0.3993 | 0.4133 | 0.4133 | 0.0827 | 0.3993 | 0.4133 | 0.4022 | 0.0413 | 0.3993 | 0.41333333333333333  | 0.40217135917450103  | 0.3993 |
| 50       | 5       | 0.4133 | 0.3993 | 0.4133 | 0.4133 | 0.1320 | 0.6162 | 0.4994 | 0.5229 | 0.0660 | 0.6162 | 0.49938888888888877  | 0.5229105833450013   | 0.6162 |
| 50       | 10      | 0.4133 | 0.3993 | 0.4133 | 0.4133 | 0.1320 | 0.6162 | 0.4994 | 0.5229 | 0.0743 | 0.6873 | 0.508743386243386    | 0.5462225163399965   | 0.6873 |
| 50       | 20      | 0.4133 | 0.3993 | 0.4133 | 0.4133 | 0.1320 | 0.6162 | 0.4994 | 0.5229 | 0.0743 | 0.6873 | 0.508743386243386    | 0.5462225163399965   | 0.7505 |
| 100      | 1       | 0.4133 | 0.3993 | 0.4133 | 0.4133 | 0.0827 | 0.3993 | 0.4133 | 0.4022 | 0.0413 | 0.3993 | 0.41333333333333333  | 0.40217135917450103  | 0.3993 |
| 100      | 5       | 0.4133 | 0.3993 | 0.4133 | 0.4133 | 0.1320 | 0.6162 | 0.4992 | 0.5228 | 0.0660 | 0.6162 | 0.4992222222222222   | 0.5227645041755384   | 0.6162 |
| 100      | 10      | 0.4133 | 0.3993 | 0.4133 | 0.4133 | 0.1320 | 0.6162 | 0.4992 | 0.5228 | 0.0733 | 0.6790 | 0.5074563492063489   | 0.5434306566397544   | 0.6790 |
| 100      | 20      | 0.4133 | 0.3993 | 0.4133 | 0.4133 | 0.1320 | 0.6162 | 0.4992 | 0.5228 | 0.0733 | 0.6790 | 0.5074563492063489   | 0.5434306566397544   | 0.7447 |

Theo tiêu chí `MRR@10` (cao nhất) thì cấu hình tốt nhất trong file này là:

- `bm25-topk = 10`, `mmr-topk = 10` — MRR@10 = 0.538398148148148, nDCG@10 = 0.5959672168573347, R@20 = 0.7971666666666667

Lưu ý: đây là kết quả trên 300 query đầu tiên của split `train`. Để rút ra kết luận chắc chắn hơn, hãy chạy lại với `--max-queries 0` (toàn bộ query) hoặc thực hiện nhiều mẫu/seed khác nhau.

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

## Giao diện tương tác (UI)

Có thể thử truy vấn tương tác và tinh chỉnh tham số (BM25 top-k, bật/tắt MMR, lambda, ...).

Chạy UI (fish):

```fish
python -m pip install -r An/requirements.txt
streamlit run An/ui.py
```

Trong UI có thể upload `corpus.jsonl` hoặc nhập đường dẫn tới folder `scifact/` local. UI cũng cho phép cấu hình `bm25_topk`, bật/tắt MMR và xem snippet kết quả.

Chương trình in ra JSON metrics cơ bản (tính theo qrels của split):

- `P@K`: Precision@K
- `R@K`: Recall@K
- `MRR@K`: Mean Reciprocal Rank@K

File output JSONL (`--out`) có dạng:

- mỗi dòng: `{ "query_id": ..., "doc_ids": [...] }`

## Notes

- MMR dùng TF-IDF cosine similarity cho cả `sim(q,d)` và `sim(d,d')`.
- Baseline này **không** dùng stemming/lemmatization; nếu muốn cải thiện lexical matching có thể thử tokenizer tốt hơn.
