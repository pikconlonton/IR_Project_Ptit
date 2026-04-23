import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np
import math
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

# đường dẫn dữ liệu
CORPUS_PATH = "scifact/scifact/corpus.jsonl"
QUERY_PATH = "scifact/scifact/queries.jsonl"
QRELS_PATH = "scifact/scifact/qrels/test.tsv"

# các mức K để evaluate
K_LIST = [1, 5, 10, 100]

# stopwords + stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# tokenize: lowercase + remove ký tự + remove stopword + stemming
def tokenize(text):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [stemmer.stem(w) for w in text.split() if w not in stop_words]


# load file jsonl
def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data[str(obj["_id"]).strip()] = obj
    return data


# load qrels (ground truth)
def load_qrels(path):
    df = pd.read_csv(path, sep="\t")
    qrels = {}
    for _, row in df.iterrows():
        qid = str(row["query-id"]).strip()
        docid = str(row["corpus-id"]).strip()
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(docid)
    return qrels


print("Loading data...")
corpus = load_jsonl(CORPUS_PATH)
queries = load_jsonl(QUERY_PATH)
qrels = load_qrels(QRELS_PATH)
print(f"Loaded {len(corpus)} docs, {len(queries)} queries")


# build inverted index: term -> {doc_id: tf}
def build_inverted_index(corpus):
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_lengths = {}
    doc_tokens = {}

    for doc_id, doc in corpus.items():
        # boost title (nhân đôi title)
        text = (doc.get("title", "") + " ") * 2 + doc.get("text", "")
        tokens = tokenize(text)

        doc_tokens[doc_id] = tokens
        doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            inverted_index[token][doc_id] += 1

    return inverted_index, doc_lengths, doc_tokens


print("Building inverted index...")
inverted_index, doc_lengths, doc_tokens = build_inverted_index(corpus)
avg_doc_length = np.mean(list(doc_lengths.values()))


# PRF Retriever (BM25 + query expansion)
class PRFRetriever:
    def __init__(
        self, inverted_index, doc_lengths, doc_tokens, avg_doc_length, num_docs
    ):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.doc_tokens = doc_tokens
        self.avg_doc_length = avg_doc_length
        self.num_docs = num_docs
        self.k1 = 1.2
        self.b = 0.5

        # tính IDF
        self.idf = {}
        for term, doc_freqs in inverted_index.items():
            df = len(doc_freqs)
            self.idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)

    # tính điểm BM25
    def bm25_score(self, query_tokens):
        scores = defaultdict(float)

        for term in query_tokens:
            if term in self.inverted_index:
                idf = self.idf[term]
                for doc_id, tf in self.inverted_index[term].items():
                    doc_len = self.doc_lengths[doc_id]
                    norm = 1 - self.b + self.b * (doc_len / self.avg_doc_length)
                    score = idf * (self.k1 + 1) * tf / (self.k1 * norm + tf)
                    scores[doc_id] += score

        return scores

    # lấy các term quan trọng từ top docs (PRF)
    def extract_expansion_terms(self, top_docs, num_terms=5):
        term_scores = defaultdict(float)

        for doc_id in top_docs:
            tokens = self.doc_tokens[doc_id]
            for t in tokens:
                term_scores[t] += self.idf.get(t, 0)

        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[
            :num_terms
        ]
        return [t for t, _ in top_terms]

    # retrieve: BM25 + PRF
    def retrieve(self, query_text, top_k=100):
        query_tokens = tokenize(query_text)

        # lần 1: BM25
        initial_scores = self.bm25_score(query_tokens)
        ranked = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)

        # lấy top docs để expand
        top_docs = [doc_id for doc_id, _ in ranked[:10]]
        expansion_terms = self.extract_expansion_terms(top_docs)

        # expand query (query gốc weight cao hơn)
        expanded_query = query_tokens * 3 + expansion_terms

        # lần 2: BM25
        final_scores = self.bm25_score(expanded_query)
        final_ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        return final_ranked[:top_k]


# ===== METRICS =====


def recall_at_k(results, relevant_docs, k):
    retrieved = [d for d, _ in results[:k]]
    hit = sum([1 for d in retrieved if d in relevant_docs])
    return hit / len(relevant_docs) if relevant_docs else 0


def precision_at_k(results, relevant_docs, k):
    retrieved = [d for d, _ in results[:k]]
    hit = sum([1 for d in retrieved if d in relevant_docs])
    return hit / k if k > 0 else 0


def average_precision_at_k(results, relevant_docs, k):
    retrieved = [d for d, _ in results[:k]]
    score = 0
    hit = 0

    for i, d in enumerate(retrieved):
        if d in relevant_docs:
            hit += 1
            score += hit / (i + 1)

    if len(relevant_docs) == 0:
        return 0

    return score / min(len(relevant_docs), k)


def dcg_at_k(results, relevant_docs, k):
    retrieved = [d for d, _ in results[:k]]
    return sum(
        [1 / math.log2(i + 2) for i, d in enumerate(retrieved) if d in relevant_docs]
    )


def ndcg_at_k(results, relevant_docs, k):
    dcg = dcg_at_k(results, relevant_docs, k)
    ideal = sum([1 / math.log2(i + 2) for i in range(min(len(relevant_docs), k))])
    return dcg / ideal if ideal > 0 else 0


print("Initializing retriever...")
retriever = PRFRetriever(
    inverted_index, doc_lengths, doc_tokens, avg_doc_length, len(corpus)
)

# lưu metric
metrics = {
    "ndcg": {k: [] for k in K_LIST},
    "recall": {k: [] for k in K_LIST},
    "map": {k: [] for k in K_LIST},
    "precision": {k: [] for k in K_LIST},
}

print("Evaluating...\n")

# loop qua từng query
for qid, rel_docs in tqdm(qrels.items()):
    if qid not in queries:
        continue

    query_text = queries[qid]["text"]
    results = retriever.retrieve(query_text, top_k=max(K_LIST))

    for k in K_LIST:
        metrics["recall"][k].append(recall_at_k(results, rel_docs, k))
        metrics["precision"][k].append(precision_at_k(results, rel_docs, k))
        metrics["map"][k].append(average_precision_at_k(results, rel_docs, k))
        metrics["ndcg"][k].append(ndcg_at_k(results, rel_docs, k))

# in kết quả
print("\n========== FINAL RESULTS ==========")

for metric in ["ndcg", "recall", "map", "precision"]:
    print(f"\n{metric.upper()}")
    for k in K_LIST:
        print(f"@{k}: {np.mean(metrics[metric][k]):.5f}", end="  ")
    print()
