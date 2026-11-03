"""Chunkers. Every chunk keeps its character offsets (start, end) in the source document, so we
can later check whether a chunk contains a piece of gold evidence, whatever the chunker."""
from __future__ import annotations

import re

import numpy as np

_WORD = re.compile(r"\S+")
_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+[\"')\]]*|\n+|$)")


def _chunk(doc_id, text, start, end):
    return {"doc_id": doc_id, "start": start, "end": end, "text": text[start:end].strip()}


def fixed_size(text: str, doc_id=0, size: int = 200, overlap: int = 0) -> list[dict]:
    """Windows of `size` words, each overlapping the previous one by `overlap` words."""
    words = [m.span() for m in _WORD.finditer(text)]
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(words), step):
        w = words[i: i + size]
        out.append(_chunk(doc_id, text, w[0][0], w[-1][1]))
        if i + size >= len(words):
            break
    return out


def sentences(text: str) -> list[tuple[int, int]]:
    """Rough sentence splitter returning (start, end) offsets."""
    return [m.span() for m in _SENT.finditer(text) if m.group().strip()]


def sentence_packed(text: str, doc_id=0, size: int = 200, overlap_sentences: int = 0) -> list[dict]:
    """Pack whole sentences into chunks of at most ~`size` words (never cuts a sentence)."""
    sents = sentences(text)
    lens = [len(text[a:b].split()) for a, b in sents]
    out, i = [], 0
    while i < len(sents):
        j, n = i, 0
        while j < len(sents) and (n + lens[j] <= size or j == i):
            n += lens[j]
            j += 1
        out.append(_chunk(doc_id, text, sents[i][0], sents[j - 1][1]))
        if j >= len(sents):
            break
        i = max(i + 1, j - overlap_sentences)
    return out


def recursive(text: str, doc_id=0, size: int = 200, seps=("\n\n", "\n", ". ", " ")) -> list[dict]:
    """Split on the coarsest separator first (paragraphs), and only split further where a piece is
    still too long; then merge neighbouring small pieces back up to `size` words."""
    def split(a, b, level):
        if len(text[a:b].split()) <= size or level >= len(seps):
            return [(a, b)]
        sep, pieces, s = seps[level], [], a
        while True:
            k = text.find(sep, s, b)
            if k < 0:
                pieces.append((s, b))
                break
            pieces.append((s, k + len(sep)))
            s = k + len(sep)
        return [p for (x, y) in pieces if text[x:y].strip() for p in split(x, y, level + 1)]

    pieces = split(0, len(text), 0)
    out, cur, n = [], None, 0
    for a, b in pieces:
        m = len(text[a:b].split())
        if cur and n + m <= size:
            cur, n = (cur[0], b), n + m
        else:
            if cur:
                out.append(_chunk(doc_id, text, *cur))
            cur, n = (a, b), m
    if cur:
        out.append(_chunk(doc_id, text, *cur))
    return out


def semantic(text: str, model, doc_id=0, percentile: float = 90, max_size: int = 400) -> list[dict]:
    """Embed every sentence; start a new chunk where the similarity between neighbouring sentences
    drops sharply (distance above the given percentile), or when the chunk gets too long."""
    sents = sentences(text)
    if len(sents) < 2:
        return [_chunk(doc_id, text, 0, len(text))]
    emb = model.encode([text[a:b] for a, b in sents], normalize_embeddings=True)
    dist = 1 - (emb[:-1] * emb[1:]).sum(1)
    cut = np.percentile(dist, percentile)
    out, s, n = [], 0, 0
    for i in range(len(sents)):
        n += len(text[sents[i][0]:sents[i][1]].split())
        last = i == len(sents) - 1
        if last or dist[i] > cut or n >= max_size:
            out.append(_chunk(doc_id, text, sents[s][0], sents[i][1]))
            s, n = i + 1, 0
    return out


def chunk_corpus(docs, chunker, **kw) -> list[dict]:
    """Apply a chunker to a list of (doc_id, text) pairs."""
    return [c for doc_id, text in docs for c in chunker(text, doc_id=doc_id, **kw)]


def contains_evidence(chunk: dict, spans, min_overlap: float = 0.5) -> bool:
    """True if the chunk covers at least `min_overlap` of any gold evidence span (doc_id, start, end)."""
    for d, a, b in spans:
        if d == chunk["doc_id"]:
            ov = min(b, chunk["end"]) - max(a, chunk["start"])
            if ov >= min_overlap * (b - a):
                return True
    return False
