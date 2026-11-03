"""Compute the numbers shown on the slides, on the FULL BioASQ corpus (40,221 abstracts), with real Gemini.

    GEMINI_API_KEY=... python scripts/slide_numbers.py   -> slides/data/results.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragclass.data import load_bioasq
from ragclass.retrieval import BM25, Dense, Reranker, rrf
from ragclass import metrics as M

OUT = Path(__file__).resolve().parents[1] / "slides" / "data" / "results.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {}
def save():
    OUT.write_text(json.dumps(res, indent=1))

t0 = time.time()
bio = load_bioasq(n_questions=300)
ids = np.array(bio.doc_ids)
qrels = bio.qrels()
qs, qids = bio.questions.question.tolist(), bio.questions.qid.tolist()
print("docs", len(bio.docs), flush=True)
bm25 = BM25(bio.docs)

# 1) vocabulary mismatch on the full corpus
def counts(idx):
    t = [bio.docs[i].lower() for i in idx]
    return {"mentions_mi": sum("myocardial infarction" in x for x in t), "mentions_heart_attack": sum("heart attack" in x for x in t)}
res["bm25_heart"] = {q: counts(bm25.search(q, 10)[0]) for q in ["heart attack treatment", "myocardial infarction treatment"]}
print(res["bm25_heart"], flush=True); save()

# 2) dense models on a heart-related pool of the full corpus
from sentence_transformers import SentenceTransformer
pool = [i for i, t in enumerate(bio.docs) if any(w in t.lower() for w in ("cardi", "heart", "myocard", "coronary"))]
res["pool_size"] = len(pool)
res["pool_mi"] = {}
for name, path in [("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"), ("BGE-small", "BAAI/bge-small-en-v1.5"),
                   ("PubMedBERT", "NeuML/pubmedbert-base-embeddings")]:
    m = SentenceTransformer(path, device="cpu")
    E = m.encode([bio.docs[i] for i in pool], normalize_embeddings=True, batch_size=64)
    qv = m.encode(["heart attack treatment"], normalize_embeddings=True)[0]
    top = [pool[j] for j in np.argsort(-(E @ qv))[:10]]
    res["pool_mi"][name] = counts(top)
    print(name, res["pool_mi"][name], round(time.time() - t0), flush=True); save()
bp = bm25.scores("heart attack treatment")[pool]
res["pool_mi"]["BM25"] = counts([pool[j] for j in np.argsort(-bp)[:10]]); save()

# 3) leaderboard on 300 questions, full corpus
K = 100
runs = {"BM25": {q: ids[bm25.search(t, K)[0]].tolist() for q, t in zip(qids, qs)}}
dense = {}
for name, path in [("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"), ("BGE-small", "BAAI/bge-small-en-v1.5")]:
    dense[name] = Dense(bio.docs, path, show_progress=False)
    runs[name] = {q: ids[i].tolist() for q, (i, _) in zip(qids, dense[name].search_many(qs, K))}
    print("embedded", name, round(time.time() - t0), flush=True)
hyb = {}
for q, t, (di, _) in zip(qids, qs, dense["BGE-small"].search_many(qs, K)):
    hyb[q] = ids[rrf([bm25.search(t, K)[0], di], top=K)[0]].tolist()
runs["Hybrid (BM25 + BGE)"] = hyb
rr = Reranker()
id2idx = {d: i for i, d in enumerate(ids)}
runs["Hybrid + rerank"] = {}
for q, t in zip(qids, qs):
    top, _ = rr.rerank(t, [id2idx[d] for d in hyb[q][:50]], bio.docs)
    runs["Hybrid + rerank"][q] = ids[top].tolist() + hyb[q][50:]
res["leaderboard"] = {n: {k: round(float(v), 3) for k, v in M.evaluate(r, qrels).items() if k in ("nDCG@10", "MRR@10", "R@10", "R@100")}
                      for n, r in runs.items()}
print(res["leaderboard"], round(time.time() - t0), flush=True); save()

# 4) Lab 3 grid: {none, weak, good} x {small, big}, judged against the expert answer
from ragclass import llm
llm.setup("gemini")
def weak(q): return dense["MiniLM"].search(q, 5)[0]
def good(q):
    cand, _ = rrf([bm25.search(q, 50)[0], dense["BGE-small"].search(q, 50)[0]], top=50)
    return rr.rerank(q, cand, bio.docs, top=5)[0]
SYSTEM = ("You are a biomedical assistant. Answer the question using ONLY the numbered sources. Cite sources like [1]. "
          "If the sources do not contain the answer, reply exactly: \"I don't know based on the provided sources.\" "
          "Answer in at most 3 sentences.")
picked = []
for r in bio.questions.itertuples():
    rel = set(r.relevant)
    if not rel & set(ids[weak(r.question)]) and len(rel & set(ids[good(r.question)])) >= 2:
        picked.append(r)
    if len(picked) == 4:
        break
JUDGE = """Question: {q}\nExpert answer: {ref}\nCandidate answer: {a}\nDoes the candidate convey the expert's key facts without contradicting them? Reply only with JSON: {{"correct": true/false}}"""
grid = []
for r in picked:
    for mname, model in [("small", llm.small()), ("big", llm.big())]:
        for cond, docs in [("no retrieval", None), ("weak retrieval", weak(r.question)), ("good retrieval", good(r.question))]:
            if docs is None:
                a = llm.ask(r.question + "\nAnswer in at most 3 sentences.", model=model)
            else:
                src = "\n\n".join(f"[{n}] {bio.docs[i]}" for n, i in enumerate(docs, 1))
                a = llm.ask(f"SOURCES:\n{src}\n\nQUESTION: {r.question}", system=SYSTEM, model=model)
            j = llm.ask(JUDGE.format(q=r.question, ref=r.answer[:800], a=a))
            grid.append(dict(question=r.question, model=mname, condition=cond, answer=a,
                             abstained="don't know" in a.lower(), correct='"correct": true' in j.replace(" ", "").replace('"correct":true', '"correct": true')))
            print(mname, cond, grid[-1]["abstained"], grid[-1]["correct"], flush=True)
res["lab3_grid"] = grid; save()
print("done", round(time.time() - t0))
