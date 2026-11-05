"""Dataset downloader and loaders, built for Colab.

* Every file is downloaded once into a cache folder and reused afterwards.
* Downloads are streamed with a progress bar, retried with backoff, written atomically
  (no half-written files after a disconnect) and checked against expected row counts.
* Each file has a list of mirrors; if Hugging Face or GitHub is slow or blocked, the next
  mirror is tried.

Cache folder: $RAGCLASS_CACHE, else /content/ragclass_cache on Colab, else ~/.cache/ragclass.
Tip: mount Google Drive and set RAGCLASS_CACHE to a Drive folder to survive runtime resets.

    from ragclass.data import prepare_all, load_bioasq, load_multihop
    prepare_all()          # download everything up front (datasets + models)
"""
from __future__ import annotations

import ast
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
GH_RAW = "https://raw.githubusercontent.com/{repo}/main/{path}"
# Optional mirror: a GitHub release of this course repo with the same file names.
COURSE_MIRROR = os.environ.get(
    "RAGCLASS_MIRROR", "https://github.com/richardjonker2000/rag_class/releases/download/data-v1/{name}"
)


# Quick mode for testing the notebooks fast (small samples); never needed in class.
QUICK = bool(os.environ.get("RAGCLASS_QUICK"))


def cache_dir() -> Path:
    if os.environ.get("RAGCLASS_CACHE"):
        d = Path(os.environ["RAGCLASS_CACHE"])
    elif Path("/content").exists():
        d = Path("/content/ragclass_cache")
    else:
        d = Path.home() / ".cache" / "ragclass"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class RemoteFile:
    name: str                      # local file name in the cache
    urls: list[str] = field(default_factory=list)


FILES = {
    "bioasq_passages": RemoteFile("bioasq_passages.parquet", [
        HF.format(repo="rag-datasets/rag-mini-bioasq", path="data/passages.parquet/part.0.parquet"),
        COURSE_MIRROR.format(name="bioasq_passages.parquet")]),
    "bioasq_questions": RemoteFile("bioasq_questions.parquet", [
        HF.format(repo="rag-datasets/rag-mini-bioasq", path="data/test.parquet/part.0.parquet"),
        COURSE_MIRROR.format(name="bioasq_questions.parquet")]),
    "multihop_corpus": RemoteFile("multihop_corpus.json", [
        HF.format(repo="yixuantt/MultiHopRAG", path="corpus.json"),
        COURSE_MIRROR.format(name="multihop_corpus.json")]),
    "multihop_queries": RemoteFile("multihop_queries.json", [
        HF.format(repo="yixuantt/MultiHopRAG", path="MultiHopRAG.json"),
        COURSE_MIRROR.format(name="multihop_queries.json")]),
}


def _download(url: str, dest: Path, retries: int = 4, quiet: bool = False) -> None:
    from tqdm.auto import tqdm

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ragclass/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0)
                with open(tmp, "wb") as f, tqdm(total=total or None, unit="B", unit_scale=True,
                                                desc=dest.name, disable=quiet, leave=False) as bar:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.replace(dest)                # atomic: the file exists only once complete
            return
        except Exception as e:               # noqa: BLE001 — network errors come in many types
            wait = 2 ** attempt
            if attempt < retries - 1:
                print(f"  download failed ({type(e).__name__}: {e}); retrying in {wait}s")
                time.sleep(wait)
            else:
                tmp.unlink(missing_ok=True)
                raise


def fetch(key: str, quiet: bool = False) -> Path:
    """Return the local path of a dataset file, downloading it if needed (tries all mirrors)."""
    rf = FILES[key]
    dest = cache_dir() / rf.name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    errors = []
    for url in rf.urls:
        try:
            if not quiet:
                print(f"Downloading {rf.name} from {url.split('/')[2]} ...")
            _download(url, dest, quiet=quiet)
            return dest
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {e}")
    raise RuntimeError(f"Could not download {rf.name}. Tried:\n  " + "\n  ".join(errors))


def _check(name: str, n: int, expected: int) -> None:
    if n != expected:
        print(f"WARNING: {name} has {n} rows, expected {expected}. The upstream dataset may have changed.")


# --------------------------------------------------------------------------- BioASQ (mini)

@dataclass
class BioASQ:
    """40k PubMed abstracts + expert questions with relevant-document labels (BioASQ 11b)."""
    doc_ids: list[int]             # PubMed IDs
    docs: list[str]                # abstract text
    questions: pd.DataFrame        # columns: qid, question, answer, relevant (set of PubMed IDs)

    def qrels(self, qids=None) -> dict:
        q = self.questions if qids is None else self.questions[self.questions.qid.isin(qids)]
        return {r.qid: set(r.relevant) for r in q.itertuples()}


def load_bioasq(n_questions: int | None = 300, seed: int = 0, max_docs: int | None = None) -> BioASQ:
    """Load the corpus and a fixed random sample of `n_questions` questions (None = all 4,719).

    `max_docs` shrinks the corpus for CPU-only runtimes: all documents relevant to the sampled
    questions are kept, plus random other abstracts up to `max_docs`.
    """
    if QUICK:
        n_questions = min(n_questions or 40, 40)
        max_docs = min(max_docs or 3000, 3000)
    p = pd.read_parquet(fetch("bioasq_passages")).reset_index()   # the id is stored as the index
    q = pd.read_parquet(fetch("bioasq_questions")).reset_index()
    _check("bioasq passages", len(p), 40221)
    _check("bioasq questions", len(q), 4719)
    q = q.rename(columns={"id": "qid"})
    q["relevant"] = q["relevant_passage_ids"].map(lambda s: sorted(set(ast.literal_eval(s))))
    q = q[q.relevant.map(len) > 0].drop(columns=["relevant_passage_ids"])
    if n_questions is not None:
        q = q.sample(n=min(n_questions, len(q)), random_state=seed)
    q = q.reset_index(drop=True)
    p["passage"] = p["passage"].str.replace(r"\s*\n\s*", " ", regex=True).str.strip()
    p = p[p.passage.str.len() > 0]
    if max_docs is not None and max_docs < len(p):
        needed = set().union(*q.relevant)
        keep = p[p.id.isin(needed)]
        rest = p[~p.id.isin(needed)].sample(n=max(0, max_docs - len(keep)), random_state=seed)
        p = pd.concat([keep, rest]).sort_values("id")
    return BioASQ(doc_ids=p.id.tolist(), docs=p.passage.tolist(), questions=q)


# --------------------------------------------------------------------------- MultiHop-RAG

@dataclass
class MultiHop:
    """609 news articles (Sep–Dec 2023) + 2,556 multi-hop questions with evidence facts."""
    articles: pd.DataFrame         # columns: doc_id, title, source, category, published_at, url, body
    queries: pd.DataFrame          # columns: qid, query, answer, question_type, evidence (list of dicts)

    def evidence_spans(self, qid: int) -> list[tuple[int, int, int]]:
        """(doc_id, start_char, end_char) of every evidence fact of a query, found in the article body."""
        row = self.queries.loc[self.queries.qid == qid].iloc[0]
        spans = []
        for ev in row.evidence:
            art = self.articles.loc[self.articles.title == ev["title"]]
            if art.empty:
                continue
            body = art.iloc[0].body
            start = body.find(ev["fact"])
            if start >= 0:
                spans.append((int(art.iloc[0].doc_id), start, start + len(ev["fact"])))
        return spans


def load_multihop() -> MultiHop:
    corpus = json.loads(fetch("multihop_corpus").read_text())
    queries = json.loads(fetch("multihop_queries").read_text())
    if not QUICK:
        _check("multihop corpus", len(corpus), 609)
        _check("multihop queries", len(queries), 2556)
    if QUICK:   # keep the articles needed by the first 300 questions (+ null questions), drop the rest
        queries = queries[:300] + [x for x in queries if x["question_type"] == "null_query"][:20]
        need = {e["title"] for x in queries for e in x["evidence_list"]}
        corpus = [c for c in corpus if c["title"] in need][:150]
        kept = {c["title"] for c in corpus}
        queries = [x for x in queries if all(e["title"] in kept for e in x["evidence_list"])]
    a = pd.DataFrame(corpus)
    a.insert(0, "doc_id", range(len(a)))
    q = pd.DataFrame(queries).rename(columns={"evidence_list": "evidence"})
    q.insert(0, "qid", range(len(q)))
    q = q[["qid", "query", "answer", "question_type", "evidence"]]
    return MultiHop(articles=a[["doc_id", "title", "source", "category", "published_at", "url", "body"]],
                    queries=q)


# --------------------------------------------------------------------------- models

MODELS = {
    "embed_small": "sentence-transformers/all-MiniLM-L6-v2",
    "embed_bge": "BAAI/bge-small-en-v1.5",
    "embed_bio": "NeuML/pubmedbert-base-embeddings",
    "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2",
}


def prepare_all(models: bool = True) -> None:
    """Download every dataset (and optionally every model) used in the course, up front."""
    t = time.time()
    for key in FILES:
        fetch(key)
    print("Datasets ready in", cache_dir())
    if models:
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for key, name in MODELS.items():
            print("Downloading model", name)
            for attempt in range(3):
                try:
                    (CrossEncoder if key == "reranker" else SentenceTransformer)(name, device=device)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  failed ({e}); retrying")
                    time.sleep(2 ** attempt)
    print(f"All done in {time.time() - t:.0f}s")
