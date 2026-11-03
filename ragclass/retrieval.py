"""Retrievers: BM25 (sparse), dense bi-encoder, hybrid (reciprocal rank fusion) and cross-encoder
reranking. Deliberately small and readable: the same ideas you see in the Lab 1 notebook."""
from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def _topk(scores: np.ndarray, k: int):
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


class BM25:
    """Okapi BM25.  score(q, d) = sum over query terms t of
         idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avg_len))
    k1 controls how fast repeated terms saturate; b controls document-length normalisation."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float32)           # docs x vocab counts
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        norm = k1 * (1 - b + b * dl / dl.mean())
        w = tf.copy()
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        w.data = self.idf[tf.indices] * tf.data * (k1 + 1) / (tf.data + norm[rows])
        self.weights = w.tocsc()                                              # docs x vocab BM25 weights
        self.vocab = self.vec.vocabulary_

    def scores(self, query: str) -> np.ndarray:
        cols = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not cols:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(1)).ravel()

    def search(self, query: str, k: int = 10):
        return _topk(self.scores(query), k)

    def explain(self, query: str, doc: int) -> dict:
        """Contribution of each query term to the score of one document."""
        out = {}
        for t in dict.fromkeys(tokenize(query)):
            j = self.vocab.get(t)
            out[t] = 0.0 if j is None else float(self.weights[doc, j])
        return out


def _cached_embeddings(model_name: str, texts: list[str], compute):
    """Embeddings are saved to the course cache folder the first time, so later labs (or a re-run after a
    crash) load them in seconds instead of re-encoding the whole corpus."""
    import hashlib

    from .data import cache_dir

    h = hashlib.sha1(f"{model_name}|{len(texts)}|{texts[0][:300]}|{texts[-1][:300]}".encode()).hexdigest()[:12]
    path = cache_dir() / "embeddings" / f"{model_name.replace('/', '__')}_{h}.npy"
    if path.exists():
        return np.load(path)
    emb = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


class Dense:
    """Bi-encoder: embed documents once, embed the query, rank by cosine similarity."""

    def __init__(self, texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 128,
                 query_prefix: str | None = None, model=None, show_progress: bool = True):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model or SentenceTransformer(model_name, device=device)
        if device == "cuda":
            self.model.half()
        if query_prefix is None:   # bge models were trained with an instruction prefix on queries
            query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name else ""
        self.query_prefix = query_prefix
        self.emb = _cached_embeddings(model_name, texts, lambda: self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=show_progress).astype(np.float32))

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode([self.query_prefix + q for q in queries], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def scores(self, query: str) -> np.ndarray:
        return self.emb @ self.encode_queries([query])[0]

    def search(self, query: str, k: int = 10):
        return _topk(self.scores(query), k)

    def search_many(self, queries: list[str], k: int = 10):
        q = self.encode_queries(queries)
        return [_topk(self.emb @ v, k) for v in q]


def rrf(rankings: list, k: int = 60, top: int | None = None):
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank(d)).
    Uses only ranks, so the systems' score scales do not need to be comparable."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            fused[int(d)] = fused.get(int(d), 0.0) + 1.0 / (k + rank)
    order = sorted(fused, key=fused.get, reverse=True)[:top]
    return np.array(order), np.array([fused[d] for d in order])


class Reranker:
    """Cross-encoder: reads query and document *together* and outputs a relevance score.
    Much more accurate than a bi-encoder, much slower, so only used on the top candidates."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2", max_length: int = 512):
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length,
                                  device="cuda" if torch.cuda.is_available() else "cpu")

    def rerank(self, query: str, cand_idx, texts: list[str], top: int | None = None):
        cand_idx = np.asarray(cand_idx)
        if len(cand_idx) == 0:
            return cand_idx, np.array([])
        s = np.asarray(self.model.predict([(query, texts[i]) for i in cand_idx], show_progress_bar=False))
        order = np.argsort(-s)[:top]
        return cand_idx[order], s[order]


class HybridRetriever:
    """Everything together: BM25 + dense -> RRF -> (optional) cross-encoder rerank.
    `texts` are the searchable units (documents or chunks); `meta` is returned alongside."""

    def __init__(self, texts: list[str], meta: list[dict] | None = None, dense_model="BAAI/bge-small-en-v1.5",
                 reranker: str | None = "cross-encoder/ms-marco-MiniLM-L6-v2", show_progress=True):
        self.texts = texts
        self.meta = meta or [{} for _ in texts]
        self.bm25 = BM25(texts)
        self.dense = Dense(texts, dense_model, show_progress=show_progress)
        self.reranker = Reranker(reranker) if reranker else None

    def search(self, query: str, k: int = 5, candidates: int = 50, rerank: bool = True) -> list[dict]:
        b, _ = self.bm25.search(query, candidates)
        d, _ = self.dense.search(query, candidates)
        idx, sc = rrf([b, d], top=candidates)
        if rerank and self.reranker is not None:
            idx, sc = self.reranker.rerank(query, idx, self.texts, top=k)
        return [{"rank": r + 1, "score": float(s), "text": self.texts[i], **self.meta[i]}
                for r, (i, s) in enumerate(zip(idx[:k], sc[:k]))]
