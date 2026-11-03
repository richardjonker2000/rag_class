# %% [markdown]
# # Lab 2 — Measuring retrieval: metrics, hybrid search and reranking
#
# **Advanced LLM Applications: RAG, Agents, and Evaluation** · ~30 min
#
# In Lab 1 we compared retrievers by *looking* at results. Now we **measure** them on 300 BioASQ
# questions with expert relevance labels, using the metrics used in IR papers and in industry:
#
# 1. **The metrics, by hand:** P@k, Recall@k, MRR, MAP, nDCG on one real query.
# 2. **The leaderboard:** BM25 vs several embedding models.
# 3. **Hybrid search:** combine BM25 and dense with reciprocal rank fusion.
# 4. **Reranking:** a cross-encoder re-scores the top 50.
# 5. **Are the labels right?** The incomplete-judgement problem.
# 6. *(Stretch)* **Chunk size, measured:** back to the news articles.
#
# No API key is needed. **Runtime → Change runtime type → T4 GPU.**

# %%
# @title Setup (run me first): gets the course code and data, builds BM25
import os, subprocess, sys
if os.path.exists("../ragclass"):
    sys.path.insert(0, "..")
else:
    if not os.path.exists("rag_class"):
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/richardjonker2000/rag_class.git"], check=True)
    else:                                    # already cloned: get the latest version of the course code
        subprocess.run(["git", "-C", "rag_class", "pull", "-q"], check=False)
    sys.path.insert(0, "rag_class")
try:
    import sentence_transformers, openai  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers>=3", "openai"], check=True)

import time, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 160)
GPU = torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0) if GPU else "none: running a smaller version on CPU")

from ragclass.data import load_bioasq
from ragclass.retrieval import BM25, Dense, Reranker, rrf
from ragclass import metrics as M
from ragclass.show import show_code, show_html, highlight

bio = load_bioasq(n_questions=300, max_docs=None if GPU else 10_000)
ids = np.array(bio.doc_ids)
qrels = bio.qrels()
questions = bio.questions.question.tolist()
qids = bio.questions.qid.tolist()
bm25 = BM25(bio.docs)
print(f"{len(bio.docs):,} abstracts, {len(qids)} labelled questions")

# %% [markdown]
# ---
# ## Part 1 — The metrics, by hand
#
# One query, BM25's top 10, and the expert labels:

# %%
# pick an instructive question: the first hit is not at #1, and the top 5 mixes hits and misses
def _pick(strict):
    for q in bio.questions.itertuples():
        rel = set(q.relevant)
        ranked = ids[bm25.search(q.question, 100)[0]].tolist()
        top5 = [d in rel for d in ranked[:5]]
        if (not top5[0] and 2 <= sum(top5) <= 3 and len(rel) >= 6) if strict else (1 <= sum(top5) <= 4):
            return q, rel, ranked
q, rel, ranked = _pick(strict=True) or _pick(strict=False)
print("Q:", q.question, f"\n{len(rel)} relevant documents in total\n")
for r, d in enumerate(ranked[:10], 1):
    print(f"{r:2d}. {'✅ relevant' if d in rel else '   -'}")

# %% [markdown]
# 🤔 **Question 1.1.** Before running the next cell, compute **P@5**, **R@5** and the **reciprocal rank** for this ranking in your head.

# %%
for name, fn in [("P@5", lambda: M.precision_at_k(ranked, rel, 5)),
                 ("R@5", lambda: M.recall_at_k(ranked, rel, 5)),
                 ("R@10", lambda: M.recall_at_k(ranked, rel, 10)),
                 ("Hit@10", lambda: M.hit_at_k(ranked, rel, 10)),
                 ("RR (first relevant)", lambda: M.reciprocal_rank(ranked, rel)),
                 ("AP@100", lambda: M.average_precision(ranked, rel, 100)),
                 ("nDCG@10", lambda: M.ndcg_at_k(ranked, rel, 10))]:
    print(f"{name:20s} {fn():.3f}")
print(f"\nNote: with {len(rel)} relevant documents, even a perfect ranking only reaches "
      f"R@5 = {min(5, len(rel)) / len(rel):.3f} and R@10 = {min(10, len(rel)) / len(rel):.3f}.")

# %% [markdown]
# Each metric is a few lines of code. Read nDCG and average precision:

# %%
show_code(M.ndcg_at_k, M.average_precision)

# %% [markdown]
# | metric | question it answers | when to use it |
# |---|---|---|
# | **P@k** | how much of what I show is relevant? | the user (or the LLM) sees exactly k results |
# | **Recall@k** | how much of what exists did I find? | RAG: did the evidence make it into the context? |
# | **Hit@k** | did I find *anything* relevant? | factoid QA, where one good document is enough |
# | **MRR** | how far down is the first relevant result? | search boxes, "I'm feeling lucky" |
# | **MAP** | precision at every relevant result, averaged | classic TREC-style ranking evaluation |
# | **nDCG@k** | are the relevant results near the top? (supports graded labels) | **the industry and benchmark standard** (BEIR, MTEB) |
#
# 🤔 **Question 1.2.** A question has 40 relevant documents. What is the best possible **Recall@10**? And **P@10**? What does that tell you about comparing these metrics across queries?
#
# <details><summary>Answer</summary>
# Best R@10 = 10/40 = 0.25, while the best P@10 = 1.0. Recall@k is bounded by k/|relevant|, so
# queries with many relevant documents always look bad on recall. nDCG normalises by the
# <i>ideal</i> ranking, so a perfect top 10 scores 1.0 whatever the number of relevant
# documents. That is one reason nDCG@10 is the default leaderboard metric.
# </details>
#
# 🤔 **Question 1.3.** These are *offline* metrics with expert labels. What do companies like Google, Amazon or Spotify measure *online*?
#
# <details><summary>Answer</summary>
#
# Behaviour of real users, usually through <b>A/B tests</b>: click-through rate, clicks at the
# first position, time to first click, abandonment (the user leaves without clicking),
# reformulation rate (the user rephrases the query), conversions or purchases, and dwell time.
# Offline nDCG/recall on a labelled set is used to <i>decide what is worth an A/B test</i>.
# Online metrics decide what ships. For RAG products the online signals are thumbs up/down,
# copy-paste of the answer, follow-up questions and escalation to a human.
# </details>

# %% [markdown]
# ---
# ## Part 2 — The leaderboard: BM25 vs embedding models
#
# `run` = for every question, the ranked list of document ids. `M.evaluate` averages every metric over all 300 questions.

# %%
K = 100
runs = {}
runs["BM25"] = {qid: ids[bm25.search(q, K)[0]].tolist() for qid, q in zip(qids, questions)}

dense_models = {"MiniLM (general, small)": "sentence-transformers/all-MiniLM-L6-v2",
                "BGE-small (general, stronger)": "BAAI/bge-small-en-v1.5"}
if GPU:   # ~2-3 min on a T4; skipped on CPU
    dense_models["PubMedBERT (biomedical)"] = "NeuML/pubmedbert-base-embeddings"

dense = {}
for name, path in dense_models.items():
    t = time.time()
    dense[name] = Dense(bio.docs, path, show_progress=False)
    res = dense[name].search_many(questions, K)
    runs[name] = {qid: ids[i].tolist() for qid, (i, _) in zip(qids, res)}
    print(f"{name}: embedded in {time.time()-t:.0f}s")

def leaderboard(runs):
    """Score every system on the questions that *all* systems were run on (a fair comparison), best first."""
    common = set(qrels).intersection(*[set(r) for r in runs.values()])
    q = {k: v for k, v in qrels.items() if k in common}
    lb = pd.DataFrame({n: M.evaluate(r, q) for n, r in runs.items()}).T[
        ["nDCG@10", "MRR@10", "P@5", "R@10", "R@100", "MAP@100"]].round(3)
    if "BM25" in lb.index:
        lb.insert(1, "Δ nDCG vs BM25", (lb["nDCG@10"] - lb.loc["BM25", "nDCG@10"]).round(3))
    lb = lb.sort_values("nDCG@10", ascending=False)
    lb.index.name = f"system ({len(q)} questions)"
    return lb

def show_lb(lb):
    """Display a leaderboard: best value of each column in bold green."""
    from IPython.display import display
    metrics = [c for c in lb.columns if not c.startswith("Δ")]
    display(lb.style.format(precision=3, subset=metrics).format("{:+.3f}", subset=[c for c in lb.columns if c.startswith("Δ")])
            .highlight_max(subset=metrics, props="font-weight:bold;background-color:#D9F0E3;color:#111")
            .set_caption("Best first. Bold green = best in that column. Δ = difference in nDCG@10 from BM25 "
                         "(positive = better than BM25)."))

show_lb(leaderboard(runs))

# %% [markdown]
# 🤔 **Question 2.1.** Rank the systems. Does a general embedding model beat BM25? Does a *domain-specific* one? What would you try next if this were your company's data?
#
# <details><summary>Answer</summary>
# In our full-corpus run (T4, 40k abstracts) <b>BM25 beat every embedding model</b> (nDCG@10 0.59
# vs 0.57 BGE-small, 0.47 MiniLM, 0.44 PubMedBERT). Why?
#
# * BioASQ questions use precise terms (genes, drugs, diseases), which is exactly what keyword match rewards.
# * <b>Label bias:</b> the experts found their relevant abstracts with PubMed <i>keyword</i> searches. The gold
#   standard is built from what keyword search finds, so a dense model that finds other relevant abstracts gets no credit.
# * The "biomedical" model was the weakest: being trained on PubMed text (matching titles to abstracts) is
#   not the same as being trained for <i>question → document</i> search. Domain ≠ task.
#
# On your own data: build a small labelled set first (even 50–100 real queries), then compare BM25, a few
# retrieval models from the MTEB leaderboard (look at the <i>retrieval</i> column), and hybrids.
# </details>
#
# 🤔 **Question 2.2.** Is a difference of 0.01 nDCG@10 real, or noise from *which* 300 questions we happened to pick?

# %%
per = {n: M.per_query(r, qrels).set_index("qid")["nDCG@10"] for n, r in runs.items()}
a, b = "BM25", list(dense_models)[-1]
print(f"{b} - {a}: mean diff = {(per[b] - per[a]).mean():+.3f}, "
      f"bootstrap p ≈ {M.paired_bootstrap(per[a].values, per[b].values):.3f}")
print(f"{b} better on {(per[b] > per[a]).sum()} questions, worse on {(per[b] < per[a]).sum()}")

# %% [markdown]
# <details><summary>Answer</summary>
# A <b>paired bootstrap</b> resamples the questions many times and checks how often the
# difference disappears. A small p (say &lt; 0.05) means the gain is unlikely to be luck.
# Also look at the per-query wins and losses: an "average improvement" can hide a system that is
# much worse on a whole category of questions.
# </details>

# %% [markdown]
# ---
# ## Part 3 — Hybrid search with reciprocal rank fusion (RRF)
#
# BM25 and dense fail on *different* questions (Lab 1). RRF merges ranked lists using only the ranks:
#
# $$\text{RRF}(d) = \sum_{\text{systems}} \frac{1}{k + \text{rank}(d)}, \qquad k = 60$$

# %%
best_dense = max(dense_models, key=lambda n: M.evaluate(runs[n], qrels)["nDCG@10"])   # the best dense model
print("best dense model:", best_dense)
def hybrid_run(dense_name, k_rrf=60, weights=(1, 1)):
    res = dense[dense_name].search_many(questions, K)
    out = {}
    for qid, q, (di, _) in zip(qids, questions, res):
        bi = bm25.search(q, K)[0]
        rankings = [bi] * weights[0] + [di] * weights[1]          # crude weighting: repeat a list
        out[qid] = ids[rrf(rankings, k=k_rrf, top=K)[0]].tolist()
    return out

runs[f"Hybrid: BM25 + {best_dense}"] = hybrid_run(best_dense)
runs[f"Hybrid: 2×BM25 + {best_dense}"] = hybrid_run(best_dense, weights=(2, 1))   # trust BM25 more
runs["Hybrid: BM25 + MiniLM"] = hybrid_run("MiniLM (general, small)")
show_lb(leaderboard(runs))

# %% [markdown]
# **RRF step by step on one real question.** For every document in either top 10: its rank in each
# list, its fused score, and whether the experts marked it relevant.

# %%
show_code(rrf)
# pick an informative question: where BM25 and dense disagree the most about the top 10
def hits10(run, qid): return len(set(run[qid][:10]) & qrels[qid])
gap = {qid: abs(hits10(runs["BM25"], qid) - hits10(runs[best_dense], qid)) for qid in qids}
q = bio.questions.set_index("qid").loc[max(gap, key=gap.get)]
b_rank = {int(d): r for r, d in enumerate(bm25.search(q.question, K)[0], 1)}
d_rank = {int(d): r for r, d in enumerate(dense[best_dense].search(q.question, K)[0], 1)}
fused_idx, fused_sc = rrf([list(b_rank), list(d_rank)], top=K)
cand = {d for d in list(b_rank)[:10] + list(d_rank)[:10]}
walk = pd.DataFrame([{"doc": ids[d], "BM25 rank": b_rank.get(d, "–"), "dense rank": d_rank.get(d, "–"),
                      "RRF score": round(sum(1 / (60 + r[d]) for r in (b_rank, d_rank) if d in r), 4),
                      "relevant": "✅" if ids[d] in qrels[q.name] else "", "start of abstract": bio.docs[d][:70]}
                     for d in cand]).sort_values("RRF score", ascending=False).reset_index(drop=True)
walk.index += 1
print("Q:", q.question)
walk

# %% [markdown]
# 🤔 Which documents rose in the fused ranking, and why? Find one that was mediocre in *both* lists but ends up high.
#
# **Per question, not just on average.** For how many questions did hybrid beat BM25, and by how much?

# %%
ph = M.per_query(runs[f"Hybrid: BM25 + {best_dense}"], qrels).set_index("qid")["nDCG@10"]
pb = M.per_query(runs["BM25"], qrels).set_index("qid")["nDCG@10"]
diff = (ph - pb)
print(f"hybrid better on {(diff > 0).sum()}, worse on {(diff < 0).sum()}, equal on {(diff == 0).sum()} questions")
diff.plot.hist(bins=30, figsize=(7, 3), title="nDCG@10 (hybrid) − nDCG@10 (BM25), per question"); plt.show()
print("Biggest loss for hybrid:", bio.questions.set_index("qid").question[diff.idxmin()])

# %% [markdown]
# 🤔 **Question 3.1.** Why fuse *ranks* instead of simply adding the BM25 score and the cosine similarity?
#
# <details><summary>Answer</summary>
# The scores live on different scales: BM25 is unbounded (5, 30, 80…) and depends on the query
# length, while cosine similarity sits in a narrow band (0.3–0.8). Adding them lets one system
# dominate. Score fusion needs per-query normalisation (min-max, z-score) and weight tuning. RRF
# needs neither, which is why most search engines offer it (Elasticsearch, OpenSearch, Weaviate; Azure AI
# Search uses it for every hybrid query), next to normalised score fusion.
# </details>
#
# 🤔 **Question 3.2.** Did hybrid beat BM25? Compare nDCG@10 (the top of the ranking) with R@100 (the long list). When does fusion help, and when not?
#
# <details><summary>Answer</summary>
#
# In our full-corpus run, equal-weight hybrid did <b>not</b> beat BM25 on nDCG@10 (0.584 vs 0.589), but it
# <b>raised R@100</b> (0.636 vs 0.619): fusion brings in relevant documents that only one system finds,
# while at the very top a much weaker partner adds noise.
#
# * Fusion helps most when both systems are good and make <i>different</i> mistakes (ensembles).
# * If one system is much weaker, give it less weight (try `weights=(2, 1)`) or leave it out.
# * Higher R@100 still matters: the reranker in Part 4 can only reorder what the first stage found.
# * And remember the label bias: the gold labels came from keyword search, which flatters BM25.
#
# "Hybrid always wins" is a rule of thumb, not a law. Measure it on your data.
# </details>

# %% [markdown]
# ---
# ## Part 4 — Reranking with a cross-encoder
#
# A **bi-encoder** embeds query and document *separately*, so it is fast and the documents can be
# embedded in advance. A **cross-encoder** reads `[query, document]` *together* and outputs one
# relevance score. It is much more accurate and much slower, so we only use it on the top 50 candidates.

# %%
reranker = Reranker("cross-encoder/ms-marco-MiniLM-L6-v2")
id2idx = {d: i for i, d in enumerate(ids)}
hyb = runs[f"Hybrid: BM25 + {best_dense}"]
n_q = len(qids) if GPU else 60                       # CPU: rerank a subset only
t = time.time()
reranked = {}
for qid, q in list(zip(qids, questions))[:n_q]:
    cand = [id2idx[d] for d in hyb[qid][:50]]
    top, _ = reranker.rerank(q, cand, bio.docs)
    reranked[qid] = ids[top].tolist() + hyb[qid][50:]
print(f"reranked {n_q} queries x 50 docs in {time.time()-t:.0f}s")
runs[f"Hybrid + rerank (top 50 of BM25 + {best_dense})"] = reranked
lb = leaderboard(runs)          # on CPU only the reranked subset of questions is compared
show_lb(lb)

# %% [markdown]
# **What did the reranker actually do?** One question: the hybrid top 10, and where each document
# ends up after the cross-encoder has read it together with the question.

# %%
show_code(Reranker.rerank)
# pick a question whose hybrid top 10 mixes relevant and irrelevant documents
mixed = [qid for qid in reranked if 3 <= len(set(hyb[qid][:10]) & qrels[qid]) <= 7] or list(reranked)
q = bio.questions.set_index("qid").loc[mixed[0]]
cand = [id2idx[d] for d in hyb[q.name][:50]]
new, ce = reranker.rerank(q.question, cand, bio.docs)
new_rank = {int(d): r for r, d in enumerate(new, 1)}
ce_score = {int(d): float(s_) for d, s_ in zip(new, ce)}
before_after = pd.DataFrame([{"hybrid rank": r, "reranked": new_rank[d], "cross-encoder score": round(ce_score[d], 2),
                              "relevant": "✅" if ids[d] in qrels[q.name] else "", "start of abstract": bio.docs[d][:70]}
                             for r, d in enumerate(cand[:10], 1)])
print("Q:", q.question)
before_after

# %%
lb["nDCG@10"][::-1].plot.barh(figsize=(7, 3.5), title="nDCG@10 on BioASQ questions (best at the top)", xlim=(0, 1)); plt.show()

# %% [markdown]
# 🤔 **Question 4.1.** Did reranking help? Note that our cross-encoder was trained on *web search* queries (MS MARCO), not biomedicine. What does that suggest?
#
# <details><summary>Answer</summary>
# In our full-corpus run (T4, 300 questions) reranking the hybrid's top 50 gave the <b>best result
# of the whole lab</b>: nDCG@10 0.633 vs 0.589 for BM25, although the cross-encoder was trained on web
# search, not biomedicine. It reads query and document <i>together</i>, so it can judge relevance
# much better than comparing two separate vectors. Rerankers are domain-sensitive, though, so treat
# every component as a hypothesis to measure on your data. Stronger options: BAAI/bge-reranker-v2-m3,
# biomedical rerankers, or an LLM as the reranker (expensive).
# </details>
#
# 🤔 **Question 4.2.** Reranking can only reorder the 50 candidates. Which metric of the *first stage* therefore matters most?
#
# <details><summary>Answer</summary>
# <b>Recall@50 (or @100).</b> The first stage must <i>find</i> the documents, and the reranker
# puts them in order. That is why a typical pipeline is high-recall first stage (hybrid) →
# high-precision reranker → top 5 into the prompt.
# </details>

# %% [markdown]
# ---
# ## Part 5 — Can we trust the labels?
#
# Remember Question 0.0 in Lab 1: the gold standard is never complete. Here is what that looks like.
#
# Let's look at documents our best system ranks in the **top 3** that the experts did **not** mark as relevant.

# %%
best = reranked
doc_text = dict(zip(ids, bio.docs))
shown = 0
for qid, row in zip(qids, bio.questions.itertuples()):
    fp = [d for d in best.get(qid, [])[:3] if d not in qrels[qid]]
    if fp and shown < 3:
        shown += 1
        print(f"Q: {row.question}\nEXPERT ANSWER: {row.answer[:200]}\n"
              f"UNLABELLED TOP RESULT: {doc_text[fp[0]][:500]}\n" + "-" * 100)

# %% [markdown]
# 🤔 **Question 5.1.** Are these really irrelevant? What does this mean for the scores in our leaderboard?
#
# <details><summary>Answer</summary>
# Read them critically: some are clearly off-topic (a real retrieval error), but quite often an
# unlabelled document <i>does</i> answer the question. Experts only judged the documents they
# found with their own searches, so a document with no label is <b>unjudged</b>, not
# necessarily irrelevant. Consequences:
#
# * Absolute scores are underestimates, especially for systems that differ from the ones used
#   to build the labels. A new, better retriever can be <i>penalised</i> for finding new
#   relevant documents.
# * Comparisons are fairer if you judge the new documents too (TREC pooling), or report
#   metrics that ignore unjudged documents (e.g. condensed nDCG / bpref).
# * In industry, labels come from human raters following guidelines, from clicks, or
#   increasingly from LLM judges. Each has its own biases.
# </details>

# %% [markdown]
# ---
# ## Leaderboard challenge (5 min)
#
# Try to get the best **nDCG@10** on the 300 questions. Ideas: the `k` in RRF, weighting
# (`weights=(2, 1)`), a different dense model, reranking more or fewer candidates, BM25 `k1`/`b`.
# Post your best score and config in the class chat.

# %%
runs["my experiment"] = hybrid_run(best_dense, k_rrf=20, weights=(1, 2))      # <- change me
show_lb(leaderboard(runs))

# %% [markdown]
# ---
# ## Stretch — Chunk size, measured (MultiHop-RAG news)
#
# In Lab 1 we counted how many evidence facts stay intact. Now we measure **retrieval**: for 300
# news questions, does the top of the ranking contain the evidence facts? To be fair we give
# every configuration **the same context budget** of about 1,000 words. Small chunks get a larger k,
# big chunks a smaller k.

# %%
from ragclass.data import load_multihop
from ragclass import chunking as C
mh = load_multihop()
docs = list(zip(mh.articles.doc_id, mh.articles.body))
mq = mh.queries[mh.queries.question_type != "null_query"]
mq = mq.sample(min(len(mq), 300 if GPU else 60), random_state=0)
emb_model = dense["BGE-small (general, stronger)"].model

def evidence_recall(chunks, budget_words=1000):
    d = Dense([c["text"] for c in chunks], "BAAI/bge-small-en-v1.5", model=emb_model, show_progress=False)
    avg = np.mean([len(c["text"].split()) for c in chunks])
    k = max(1, int(budget_words // avg))
    res = d.search_many(mq["query"].tolist(), k)
    rec = []
    for qid, (top, _) in zip(mq.qid, res):
        spans = mh.evidence_spans(qid)
        found = sum(any(C.contains_evidence(chunks[i], [s]) for i in top) for s in spans)
        rec.append(found / max(1, len(spans)))
    return k, np.mean(rec)

rows = []
for size in [50, 100, 200, 400, 800]:
    ch = C.chunk_corpus(docs, C.sentence_packed, size=size)
    k, r = evidence_recall(ch)
    rows.append(dict(chunk_words=size, n_chunks=len(ch), k_for_1000_words=k, evidence_recall=round(r, 3)))
pd.DataFrame(rows)

# %% [markdown]
# 🤔 **Question S.1.** Which chunk size wins at equal budget? Does that mean you should always use tiny chunks?
#
# <details><summary>Answer</summary>
# In our test runs, small chunks (50–100 words) won clearly and 800-word chunks were worst. With
# the same budget, many small chunks can cover many different articles, and the evidence here is
# single sentences. But this metric only asks "is the evidence sentence somewhere in the
# context?" Tiny chunks lose the surrounding context the <i>generator</i> needs (who "he" is,
# which product, which date), and multiply the number of vectors. A common compromise is to
# <b>retrieve small, then expand</b>: search over small chunks, then give the LLM the
# surrounding paragraph or a parent chunk. The right answer depends on your documents and
# questions, so measure it, exactly like this.
# </details>
