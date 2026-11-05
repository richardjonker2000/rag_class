"""Generation metrics, reference-based: exact match, token F1, ROUGE-L, embedding similarity.
LLM-as-judge prompts live in the notebook on purpose, so you can read and change them."""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

from .chunking import sentences


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    """Word overlap between prediction and reference (SQuAD-style F1)."""
    p, r = normalize(pred).split(), normalize(ref).split()
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def _rouge_tokens(s: str) -> list[str]:
    """Tokenisation as in the standard `rouge_score` package: lowercase, alphanumeric words only."""
    return re.findall(r"[a-z0-9]+", s.lower())


def rouge_l(pred: str, ref: str) -> float:
    """F-measure of the longest common subsequence of words (order matters, gaps allowed)."""
    p, r = _rouge_tokens(pred), _rouge_tokens(ref)
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r, start=1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def embedding_similarity(preds: list[str], refs: list[str], model) -> np.ndarray:
    """Cosine similarity of sentence embeddings (a cheap cousin of BERTScore)."""
    a = model.encode(preds, normalize_embeddings=True)
    b = model.encode(refs, normalize_embeddings=True)
    return (a * b).sum(1)
