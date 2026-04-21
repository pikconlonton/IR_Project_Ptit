import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import math
import re

CORPUS_PATH = "scifact\scifact\corpus.jsonl"
QUERY_PATH = "scifact\scifact\queries.jsonl"
QRELS_PATH = "scifact\scifact/qrels/test.tsv"

TOP_K = 10


def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data[str(obj["_id"]).strip()] = obj
    return data

def load_qrels(path):
    df = pd.read_csv(path, sep='\t')
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

def tokenize(text):
    """Simple tokenization and lowercasing"""
    text = text.lower()
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def build_inverted_index(corpus):
    """Build inverted index: term -> {doc_id: frequency}"""
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_lengths = {}
    doc_ids = []
    
    for doc_id, doc in corpus.items():
        text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        tokens = tokenize(text)
        doc_ids.append(doc_id)
        doc_lengths[doc_id] = len(tokens)
        
        for token in tokens:
            inverted_index[token][doc_id] += 1
    
    return inverted_index, doc_lengths, doc_ids

print("Building inverted index...")
inverted_index, doc_lengths, doc_ids = build_inverted_index(corpus)
avg_doc_length = np.mean(list(doc_lengths.values()))


class BM25Retriever:
    """Simplified BM25 implementation"""
    
    def __init__(self, inverted_index, doc_lengths, doc_ids, avg_doc_length, num_docs):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.doc_ids = doc_ids
        self.avg_doc_length = avg_doc_length
        self.num_docs = num_docs
        self.k1 = 1.5  # Term frequency saturation
        self.b = 0.75   # Length normalization
        
        # Calculate IDF
        self.idf = {}
        for term, doc_freqs in inverted_index.items():
            df = len(doc_freqs)
            self.idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
    
    def retrieve(self, query_text, top_k=10):
        query_tokens = tokenize(query_text)
        scores = defaultdict(float)
        
        for term in query_tokens:
            if term in self.inverted_index:
                idf = self.idf[term]
                
                for doc_id, tf in self.inverted_index[term].items():
                    doc_len = self.doc_lengths[doc_id]
                    
                    # BM25 formula
                    norm_factor = 1 - self.b + self.b * (doc_len / self.avg_doc_length)
                    bm25_score = idf * (self.k1 + 1) * tf / (self.k1 * norm_factor + tf)
                    
                    scores[doc_id] += bm25_score
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

class TFIDFRetriever:
    """Traditional TF-IDF with simpler implementation"""
    
    def __init__(self, inverted_index, doc_lengths, doc_ids, num_docs):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.doc_ids = doc_ids
        self.num_docs = num_docs
        
        # Calculate IDF for all terms
        self.idf = {}
        for term, doc_freqs in inverted_index.items():
            df = len(doc_freqs)  # Document frequency
            self.idf[term] = math.log(num_docs / (df + 1))
    
    def retrieve(self, query_text, top_k=10):
        query_tokens = tokenize(query_text)
        scores = defaultdict(float)
        
        for term in query_tokens:
            if term in self.inverted_index:
                idf = self.idf[term]
                for doc_id, tf in self.inverted_index[term].items():
                    # TF = raw term frequency
                    # IDF = log(N/df)
                    scores[doc_id] += tf * idf
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class LanguageModelRetriever:
    """Language model with Dirichlet smoothing"""
    
    def __init__(self, inverted_index, doc_lengths, doc_ids, num_docs):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.doc_ids = doc_ids
        self.num_docs = num_docs
        self.mu = 2000  # Dirichlet smoothing parameter
        
        # Calculate collection term frequencies
        self.coll_freqs = defaultdict(int)
        self.coll_length = 0
        for term, doc_freqs in inverted_index.items():
            for doc_id, freq in doc_freqs.items():
                self.coll_freqs[term] += freq
                self.coll_length += freq
    
    def retrieve(self, query_text, top_k=10):
        query_tokens = tokenize(query_text)
        scores = defaultdict(float)
        
        for doc_id in self.doc_ids:
            score = 0.0
            for term in query_tokens:
                if term in self.inverted_index and doc_id in self.inverted_index[term]:
                    tf = self.inverted_index[term][doc_id]
                else:
                    tf = 0
                
                # Dirichlet smoothing
                doc_len = self.doc_lengths[doc_id]
                coll_freq = self.coll_freqs[term]
                
                # Avoid log(0) with epsilon
                numerator = tf + self.mu * coll_freq / self.coll_length
                denominator = doc_len + self.mu
                prob = numerator / denominator
                
                if prob > 0:
                    score += math.log(prob)
            
            scores[doc_id] = score
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

class PRFRetriever:
    """
    Pseudo-Relevance Feedback: 
    1. Initial retrieval with BM25
    2. Extract top terms from top-k documents
    3. Re-retrieve with expanded query
    """
    
    def __init__(self, inverted_index, doc_lengths, doc_ids, avg_doc_length, num_docs):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.doc_ids = doc_ids
        self.avg_doc_length = avg_doc_length
        self.num_docs = num_docs
        self.k1 = 1.5
        self.b = 0.75
        
        # Calculate IDF
        self.idf = {}
        for term, doc_freqs in inverted_index.items():
            df = len(doc_freqs)
            self.idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
    
    def bm25_score(self, query_tokens, top_k_docs=None):
        """
        Score documents using BM25
        If top_k_docs provided, only score those documents
        """
        scores = defaultdict(float)
        
        for term in query_tokens:
            if term in self.inverted_index:
                idf = self.idf[term]
                
                docs_to_score = top_k_docs if top_k_docs else self.inverted_index[term].items()
                
                for doc_id, tf in self.inverted_index[term].items():
                    if top_k_docs and doc_id not in top_k_docs:
                        continue
                    
                    doc_len = self.doc_lengths[doc_id]
                    norm_factor = 1 - self.b + self.b * (doc_len / self.avg_doc_length)
                    bm25_score = idf * (self.k1 + 1) * tf / (self.k1 * norm_factor + tf)
                    scores[doc_id] += bm25_score
        
        return scores
    
    def extract_expansion_terms(self, top_docs, num_terms=10):
        """Extract important terms from top retrieved documents"""
        term_scores = defaultdict(float)
        
        for doc_id in top_docs:
            # Get all terms in this document
            doc_text = self.doc_ids  # Just for iteration
            
            for term, doc_freqs in self.inverted_index.items():
                if doc_id in doc_freqs:
                    tf = doc_freqs[doc_id]
                    idf = self.idf[term]
                    # Score = TF * IDF
                    term_scores[term] += tf * idf
        
        # Get top terms
        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:num_terms]
        return [term for term, _ in top_terms]
    
    def retrieve(self, query_text, top_k=10):
        query_tokens = tokenize(query_text)
        
        # Stage 1: Initial retrieval with BM25
        initial_scores = self.bm25_score(query_tokens)
        initial_ranked = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top documents for PRF (e.g., top 20)
        prf_docs = [doc_id for doc_id, _ in initial_ranked[:min(20, len(initial_ranked))]]
        
        # Stage 2: Extract expansion terms from top documents
        expansion_terms = self.extract_expansion_terms(prf_docs, num_terms=5)
        expanded_query = query_tokens + expansion_terms
        
        # Stage 3: Re-retrieve with expanded query
        final_scores = self.bm25_score(expanded_query)
        final_ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        return final_ranked[:top_k]


def recall_at_k(results, relevant_docs, k=10):
    retrieved = [doc_id for doc_id, _ in results[:k]]
    hit = sum([1 for d in retrieved if d in relevant_docs])
    return hit / len(relevant_docs) if relevant_docs else 0

print("Initializing retriever...")
prf_retriever = PRFRetriever(inverted_index, doc_lengths, doc_ids, avg_doc_length, len(corpus))

print("Evaluating PRF (Pseudo-Relevance Feedback)...\n")

recall_scores = []

for qid, rel_docs in tqdm(qrels.items(), desc="Evaluating PRF"):
    if qid not in queries:
        continue
    
    query_text = queries[qid]["text"]
    results = prf_retriever.retrieve(query_text, top_k=TOP_K)
    score = recall_at_k(results, rel_docs, k=TOP_K)
    recall_scores.append(score)

avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

