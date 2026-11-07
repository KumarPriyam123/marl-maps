"""
Retriever over a SMALL corpus (the union of all passages in the loaded dataset).

Two implementations behind one interface:
  - BM25Retriever  : pure-python Okapi BM25. Zero dependencies. Default.
  - DenseRetriever : E5-base-v2 embeddings + FAISS. Needs sentence-transformers
                     + faiss-cpu. Closer to the paper; still fine on CPU for a
                     few-thousand-passage micro-corpus.

A "passage" is a dict: {"title": str, "text": str}. retrieve() returns a list of
such dicts (top-k), each with an added "_score" field.
"""

from __future__ import annotations
import os
import re
import math
import json
import pickle
import collections
from typing import List, Dict


def _tok(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, passages: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [_tok(p["title"] + " " + p["text"]) for p in passages]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        df = collections.Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}
        self.tf = [collections.Counter(d) for d in self.docs]

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        q = _tok(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
            scores.append(s)
        order = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in order:
            p = dict(self.passages[i])
            p["_score"] = round(scores[i], 4)
            out.append(p)
        return out


class DenseRetriever:
    """E5-base-v2 + FAISS. Lazy-imports heavy deps so the smoke test never needs them."""
    def __init__(self, passages: List[Dict], model_name: str = "intfloat/e5-base-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import faiss
        self.passages = passages
        self.model = SentenceTransformer(model_name)
        texts = [f"passage: {p['title']} {p['text']}" for p in passages]
        emb = self.model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self._np = np

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True
                              ).astype("float32")
        scores, idx = self.index.search(q, k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            p = dict(self.passages[i])
            p["_score"] = round(float(s), 4)
            out.append(p)
        return out


def build_corpus(dataset: List[Dict]) -> List[Dict]:
    """Union of all passages across the dataset, de-duplicated by (title, text)."""
    seen, corpus = set(), []
    for ex in dataset:
        for p in ex.get("passages", []):
            key = (p["title"], p["text"])
            if key in seen:
                continue
            seen.add(key)
            corpus.append({"title": p["title"], "text": p["text"]})
    return corpus


def get_retriever(dataset: List[Dict], kind: str = "bm25"):
    corpus = build_corpus(dataset)
    if kind == "dense":
        import config
        return DenseRetriever(corpus, model_name=getattr(config, "DENSE_MODEL",
                                                         "intfloat/e5-base-v2"))
    return BM25Retriever(corpus)
