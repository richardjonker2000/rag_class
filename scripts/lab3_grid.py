"""Small, fast version of the Lab 3 experiment for the results slide (5k-abstract subset, real Gemini).

    GEMINI_API_KEY=... python scripts/lab3_grid.py   -> adds "lab3_grid" to slides/data/results.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragclass.data import load_bioasq
from ragclass.retrieval import BM25, Dense, Reranker, rrf
from ragclass import llm

OUT = Path(__file__).resolve().parents[1] / "slides" / "data" / "results.json"
res = json.loads(OUT.read_text())
bio = load_bioasq(n_questions=300, max_docs=5000)
ids = np.array(bio.doc_ids)
bm25 = BM25(bio.docs)
minilm = Dense(bio.docs, "sentence-transformers/all-MiniLM-L6-v2", show_progress=False)
bge = Dense(bio.docs, "BAAI/bge-small-en-v1.5", show_progress=False)
rr = Reranker()
def weak(q): return minilm.search(q, 5)[0]
def good(q):
    cand, _ = rrf([bm25.search(q, 50)[0], bge.search(q, 50)[0]], top=50)
    return rr.rerank(q, cand, bio.docs, top=5)[0]
picked = []
for r in bio.questions.itertuples():
    rel = set(r.relevant)
    if not rel & set(ids[weak(r.question)]) and len(rel & set(ids[good(r.question)])) >= 2:
        picked.append(r)
    if len(picked) == 4:
        break
print("picked", len(picked), flush=True)
llm.setup("gemini")
SYSTEM = ("You are a biomedical assistant. Answer the question using ONLY the numbered sources. Cite sources like [1]. "
          "If the sources do not contain the answer, reply exactly: \"I don't know based on the provided sources.\" "
          "Answer in at most 3 sentences.")
JUDGE = ("Question: {q}\nExpert answer: {ref}\nCandidate answer: {a}\nDoes the candidate convey the expert's key facts "
         "without contradicting them? An \"I don't know\" answer is not correct. Reply only with JSON: {{\"correct\": true}} or {{\"correct\": false}}")
grid = []
for r in picked:
    for mname, model in [("small", llm.small()), ("big", llm.big())]:
        for cond, docs in [("no retrieval", None), ("weak retrieval", weak(r.question)), ("good retrieval", good(r.question))]:
            if docs is None:
                a = llm.ask(r.question + "\nAnswer in at most 3 sentences.", model=model)
            else:
                src = "\n\n".join(f"[{n}] {bio.docs[i]}" for n, i in enumerate(docs, 1))
                a = llm.ask(f"SOURCES:\n{src}\n\nQUESTION: {r.question}", system=SYSTEM, model=model)
            j = llm.ask(JUDGE.format(q=r.question, ref=r.answer[:800], a=a)).replace(" ", "").lower()
            grid.append(dict(question=r.question, model=mname, condition=cond, answer=a,
                             abstained="don't know" in a.lower(), correct='"correct":true' in j))
            print(mname, cond, grid[-1]["abstained"], grid[-1]["correct"], flush=True)
res["lab3_grid"] = grid
res["lab3_note"] = f"{len(picked)} BioASQ questions, 5,000-abstract subset, Gemini 3.5 Flash-Lite vs 3.6 Flash, 2 Nov 2026"
OUT.write_text(json.dumps(res, indent=1))
print("done")
