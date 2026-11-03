"""Generation metrics: reference-based (exact match, token F1, ROUGE-L, embedding similarity) and
reference-free (NLI entailment against the retrieved context). LLM-as-judge prompts live in the
notebook on purpose, so you can read and change them."""
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


class NLIChecker:
    """Does the context *entail* each sentence of the answer? Uses a cross-encoder trained on
    natural language inference (labels: contradiction, entailment, neutral)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        labels = getattr(self.model.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
        self.ent = [i for i, l in labels.items() if l.lower().startswith("entail")][0]

    def sentence_scores(self, context: str, answer: str, window: int = 350) -> list[tuple[str, float]]:
        """Entailment probability of every answer sentence, given the best-matching context window."""
        words = context.split()
        wins = [" ".join(words[i:i + window]) for i in range(0, max(1, len(words)), window // 2)] or [context]
        sents = [answer[a:b].strip() for a, b in sentences(answer) if len(answer[a:b].split()) >= 3]
        if not sents:
            return []
        pairs = [(w, s) for s in sents for w in wins]
        logits = np.asarray(self.model.predict(pairs, show_progress_bar=False))
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        ent = probs[:, self.ent].reshape(len(sents), len(wins)).max(1)
        return list(zip(sents, ent.tolist()))

    def faithfulness(self, context: str, answer: str) -> float:
        """Score of the *least* supported sentence: one unsupported claim is enough to fail."""
        s = self.sentence_scores(context, answer)
        return min(p for _, p in s) if s else 1.0
