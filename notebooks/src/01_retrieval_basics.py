# %% [markdown]
# # Lab 1 — Retrieval basics: chunking, sparse and dense search
#
# **Advanced LLM Applications: RAG, Agents, and Evaluation** · ~25 min
#
# A RAG system can only be as good as what it retrieves. In this lab you take apart the first
# half of the pipeline:
#
# 0. **The task and the data.** What are we searching, and for what?
# 1. **Chunking.** How do we cut long documents into searchable pieces? *(news articles, MultiHop-RAG)*
# 2. **Sparse retrieval (BM25).** Match the words. *(40k PubMed abstracts, BioASQ)*
# 3. **Dense retrieval (embeddings).** Match the meaning. *(same corpus)*
# 4. **Side by side.** Where does each one win, and why?
#
# Everything runs locally on the Colab GPU. No API key is needed for this lab.
#
# > **How to use this notebook:** run the cells in order. 🤔 marks an open question.
# > Discuss it with your neighbour *before* opening the answer.
#
# **Runtime → Change runtime type → T4 GPU** before you start.

# %%
# @title Setup (run me first): gets the course code and data
import os, subprocess, sys
if os.path.exists("../ragclass"):            # running from a local clone of the repo
    sys.path.insert(0, "..")
else:                                        # on Colab: clone the course repo once
    if not os.path.exists("rag_class"):
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/richardjonker2000/rag_class.git"], check=True)
    else:                                    # already cloned: get the latest version of the course code
        subprocess.run(["git", "-C", "rag_class", "pull", "-q"], check=False)
    sys.path.insert(0, "rag_class")
try:
    import sentence_transformers, openai  # noqa: F401  (preinstalled on Colab)
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers>=3", "openai"], check=True)

import numpy as np, pandas as pd, torch
pd.set_option("display.max_colwidth", 200)
GPU = torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0) if GPU else "none (switch the runtime to T4 GPU if you can)")

from ragclass.data import load_bioasq, load_multihop
mh = load_multihop()                                   # 609 news articles
bio = load_bioasq(n_questions=300, max_docs=None if GPU else 10_000)   # PubMed abstracts
print(f"MultiHop-RAG: {len(mh.articles)} articles | BioASQ: {len(bio.docs):,} abstracts, "
      f"{len(bio.questions)} questions")

# %% [markdown]
# ---
# ## Part 0 — The task and the data
#
# **The task.** A user asks a question; somewhere in a large collection of documents is the answer. Before any LLM
# writes anything, we must **find the right documents** (or pieces of documents). That is *retrieval*, step 2
# of the RAG pipeline, and it is what this lab and the next are about:
#
# > question → **search the collection** → top-k documents → (later) LLM writes an answer from them
#
# We use two real datasets, each where it is strongest:
#
# | | BioASQ (PubMed) | MultiHop-RAG (news) |
# |---|---|---|
# | documents | 40,221 short biomedical abstracts | 609 long news articles (Sep–Dec 2023) |
# | questions | real questions written by biomedical experts | questions whose answer is spread over 2–4 articles |
# | labels | which abstracts the experts marked relevant | the exact evidence sentences in each article |
# | used for | searching and measuring search | chunking (long documents), and agents in Lab 5 |
#
# Let's look at both before doing anything with them.

# %% [markdown]
# ### BioASQ: questions written by experts
#
# Each question comes with an expert answer and a list of abstracts the experts judged relevant.

# %%
import matplotlib.pyplot as plt
ex = bio.questions.sample(5, random_state=4)
for r in ex.itertuples():
    print(f"Q: {r.question}\n   expert answer: {r.answer[:160]}{'…' if len(r.answer) > 160 else ''}"
          f"\n   relevant abstracts: {len(r.relevant)}\n")

# %% [markdown]
# What does a relevant abstract look like?

# %%
r = ex.iloc[0]
abstract = dict(zip(bio.doc_ids, bio.docs))[r.relevant[0]]
print("Q:", r.question, f"\n\nPubMed ID {r.relevant[0]}:\n{abstract}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
pd.Series([len(d.split()) for d in bio.docs]).plot.hist(bins=50, ax=ax[0], title=f"{len(bio.docs):,} abstracts: length in words")
bio.questions.relevant.map(len).clip(upper=40).plot.hist(bins=40, ax=ax[1],
    title="relevant abstracts per question (capped at 40)")
plt.tight_layout(); plt.show()
print("median abstract length:", int(np.median([len(d.split()) for d in bio.docs])), "words")
print("relevant abstracts per question: median", int(bio.questions.relevant.map(len).median()),
      "| max", bio.questions.relevant.map(len).max())

# %% [markdown]
# ### MultiHop-RAG: long news articles, questions that need several of them

# %%
art = mh.articles
words = art.body.str.split().str.len()
fig, ax = plt.subplots(1, 3, figsize=(14, 3))
words.clip(upper=6000).plot.hist(bins=40, ax=ax[0], title="article length in words (capped at 6,000)")
art.category.value_counts().plot.barh(ax=ax[1], title="articles per category")
art.source.value_counts().head(8).plot.barh(ax=ax[2], title="top 8 sources")
plt.tight_layout(); plt.show()
print(f"{len(art)} articles, {art.published_at.min()[:10]} to {art.published_at.max()[:10]}; "
      f"median {int(words.median())} words, longest {words.max():,}")

# %% [markdown]
# Four kinds of questions. The evidence for one answer usually sits in **several different articles**:

# %%
print(mh.queries.question_type.value_counts().to_string(), "\n")
for qt in ["inference_query", "comparison_query", "temporal_query", "null_query"]:
    r = mh.queries[mh.queries.question_type == qt].iloc[0]
    n_art = len({e["title"] for e in r.evidence})
    print(f"[{qt}]  ({n_art} articles of evidence)\nQ: {r.query}\nA: {r.answer}\n")
print("articles of evidence per question:",
      mh.queries.evidence.map(lambda ev: len({e['title'] for e in ev})).value_counts().sort_index().to_dict())

# %% [markdown]
# 🤔 **Question 0.0.** These labels are what we will measure retrieval against: our "gold standard". Are they the complete truth?
#
# <details><summary>Answer</summary>
#
# No. A gold standard is never truly gold. Nobody reads all 40,221 abstracts for every question: the
# BioASQ experts judged the abstracts *their own searches* found. Other abstracts may answer the
# question just as well, and some labelled ones are borderline. Likewise, MultiHop-RAG's evidence
# sentences are the ones the dataset builders picked; another sentence or article may support the
# same answer. So **unlabelled ≠ irrelevant**: a retriever that finds a good document nobody labelled
# gets *no credit*. Keep this in mind whenever you read a metric. In Lab 2 (Part 5) you will look at
# such "false positives" yourself.
# </details>

# %% [markdown]
# 🤔 **Question 0.1.** Look at the questions above. What will make them hard to search for?
#
# <details><summary>Answer</summary>
#
# * **BioASQ:** very specific technical terms (gene, drug and protein names, abbreviations) and synonyms
#   ("heart attack" vs "myocardial infarction"). Many questions have dozens of relevant abstracts.
# * **MultiHop-RAG:** long, wordy questions that mention several sources and dates; the evidence is
#   spread over 2–4 different articles, so one search result is rarely enough; and some questions
#   (null queries) have **no** answer in the collection at all.
# * **Both:** the documents never use the question's exact wording.
# </details>

# %% [markdown]
# ---
# ## Part A — Chunking (MultiHop-RAG news articles)
#
# The news articles are long (median ~1,300 words, some over 12,000). Search engines and embedding
# models work on pieces of text, so first we have to cut them up.

# %% [markdown]
# **Why not just embed each whole article?** Let's measure it. For a real question we embed (1) the
# evidence sentence that answers it, (2) the 200-word chunk around it, and (3) the whole article, and
# compare each with the question by cosine similarity.

# %%
from sentence_transformers import SentenceTransformer
from ragclass import chunking as C
emb_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda" if GPU else "cpu")
PREFIX = "Represent this sentence for searching relevant passages: "      # BGE's instruction for queries
print("This embedding model reads at most", emb_model.max_seq_length, "tokens (~380 words). Longer texts are cut off.\n")

def three_way(qrow):
    d, a, b = mh.evidence_spans(qrow.qid)[0]
    body = mh.articles.body[d]
    chunk = next(c["text"] for c in C.fixed_size(body, doc_id=d, size=200) if c["start"] <= a < c["end"])
    e = emb_model.encode([PREFIX + qrow.query, body[a:b], chunk, body], normalize_embeddings=True)
    return dict(sentence=float(e[0] @ e[1]), chunk_200=float(e[0] @ e[2]), whole_article=float(e[0] @ e[3]),
                evidence_at_word=len(body[:a].split()), article_words=len(body.split()), fact=body[a:b])

q0 = mh.queries.loc[mh.queries.qid == 36].iloc[0]        # a "best sportsbooks" guide: many topics in one article
if not len(mh.evidence_spans(36)):                        # (quick-test mode may not contain this article)
    q0 = mh.queries[mh.queries.question_type != "null_query"].iloc[2]
r0 = three_way(q0)
print("QUESTION:", q0.query, "\nEVIDENCE:", r0["fact"])
print(f"(evidence starts at word {r0['evidence_at_word']} of {r0['article_words']})\n")
for k in ["sentence", "chunk_200", "whole_article"]:
    print(f"  similarity question ↔ {k:14s} {r0[k]:.3f}")

# %% [markdown]
# Here the evidence is one sentence in an article that reviews many different sportsbooks, and the
# whole-article vector is clearly further from the question. But one example proves nothing.
# Averaged over 60 questions, split by **where** in the article the evidence sits (inside the first
# ~380 words the model reads, or further down):

# %%
sample = mh.queries[mh.queries.question_type != "null_query"].head(60)
res3 = pd.DataFrame([three_way(r) for r in sample.itertuples()])
res3["evidence position"] = np.where(res3.evidence_at_word < 380, "in the first 380 words", "after word 380")
res3.groupby("evidence position")[["sentence", "chunk_200", "whole_article"]].mean().round(3)

# %% [markdown]
# 🤔 **Question A1.** Why not embed every article as one vector, or paste whole articles into the prompt?
#
# <details><summary>Answer</summary>
#
# Look at the average table honestly: for most news articles the whole article is about as similar to
# the question as the chunk. A news article is usually about **one story**, and these long questions
# describe the story, so the article's opening already matches. And the lone sentence scores *lowest*:
# without its context it is hard to match ("who is *he*?"). So why chunk at all?
#
# * **Multi-topic documents get diluted**, like the sportsbook guide above (and manuals, reports,
#   contracts, wikis): one vector for many topics is close to none of them.
# * **Truncation:** the model reads only the first 512 tokens. Evidence further down is never seen;
#   the "whole article" score is really a score of the article's opening.
# * **The prompt budget:** we can put ~5 chunks in front of the LLM, not 5 whole articles of 2,000+
#   words. Retrieval must deliver the *passage*, not just the right document.
# * In the prompt, whole articles cost tokens and money, add latency, and the relevant sentence
#   gets lost among irrelevant text.
#
# So we cut documents into **chunks** and retrieve chunks.
# </details>

# %% [markdown]
# Every MultiHop-RAG question comes with the exact **evidence sentences** (facts) that answer it.
# That lets us check something important: *does a chunker keep each fact in one piece?*

# %%
from ragclass import chunking as C

q = mh.queries.iloc[0]
print("QUESTION:", q.query, "\nANSWER:", q.answer, "\n")
for ev in q.evidence:
    print(f"- [{ev['source']}, {ev['published_at'][:10]}] {ev['fact']}")

# %% [markdown]
# Let's chunk all 609 articles in different ways and count how often an evidence fact
# survives **intact inside a single chunk**.

# %%
docs = list(zip(mh.articles.doc_id, mh.articles.body))
spans = [s for qid in mh.queries.qid[:500] for s in mh.evidence_spans(qid)]   # gold fact positions

def intact_rate(chunks):
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    ok = sum(any(c["start"] <= a and c["end"] >= b for c in by_doc.get(d, [])) for d, a, b in spans)
    return ok / len(spans)

rows = []
for name, fn, kw in [
    ("fixed 50 words",              C.fixed_size,      dict(size=50)),
    ("fixed 200 words",             C.fixed_size,      dict(size=200)),
    ("fixed 200 + 50 overlap",      C.fixed_size,      dict(size=200, overlap=50)),
    ("sentence-packed 200",         C.sentence_packed, dict(size=200)),
    ("recursive (paragraphs) 200",  C.recursive,       dict(size=200)),
    ("fixed 800 words",             C.fixed_size,      dict(size=800)),
]:
    ch = C.chunk_corpus(docs, fn, **kw)
    rows.append(dict(chunker=name, n_chunks=len(ch),
                     avg_words=np.mean([len(c["text"].split()) for c in ch]).round(0),
                     facts_intact=round(intact_rate(ch), 3)))
pd.DataFrame(rows)

# %% [markdown]
# The chunkers are short. Read two of them: every chunk keeps its character offsets, which is how
# we can check whether it contains a gold fact.

# %%
from ragclass.show import show_code, show_html, highlight

# %% [markdown]
# **Look at real chunks.** The opening of one article, cut by each chunker (small sizes so you can see
# several chunks). Each colour is one chunk; ¶ marks a paragraph break.

# %%
from html import escape
COLORS = ["#FCE3A6", "#C9DDF8", "#D9F0E3", "#E6DCF6", "#F8D0CE"]
art_id = 5
words_pos = [m.end() for m in __import__("re").finditer(r"\S+", mh.articles.body[art_id])]
excerpt = mh.articles.body[art_id][:words_pos[min(220, len(words_pos) - 1)]]
print("ARTICLE:", mh.articles.title[art_id], "— first 220 words\n")

def show_chunks(name, chunks):
    spans = "".join(f'<span style="background:{COLORS[i % 5]};color:#111;padding:1px 2px">'
                    f'{escape(excerpt[c["start"]:c["end"]])}</span> ' for i, c in enumerate(chunks))
    show_html(f"<p><b>{name}</b> — {len(chunks)} chunks<br>{spans.replace(chr(10) * 2, ' ¶ ')}</p>")

show_chunks("Fixed size: every 40 words (cuts anywhere)", C.fixed_size(excerpt, size=40))
show_chunks("Sentence-packed: whole sentences up to 60 words", C.sentence_packed(excerpt, size=60))
show_chunks("Recursive: paragraphs first, up to 80 words", C.recursive(excerpt, size=80))

# %% [markdown]
# **Overlap (sliding windows):** each window repeats the last 10 words of the previous one (in bold), so a
# sentence cut by one window is complete in the next.

# %%
rows = []
for i, c in enumerate(C.fixed_size(excerpt, size=40, overlap=10)[:4]):
    w = excerpt[c["start"]:c["end"]].split()
    shared, rest = (w[:10], w[10:]) if i else ([], w)
    rows.append(f'<span style="background:{COLORS[i % 5]};color:#111;padding:1px 2px">window {i + 1}: <b>{escape(" ".join(shared))}</b> '
                f'{escape(" ".join(rest))}</span>')
show_html("<br>".join(rows))

# %% [markdown]
# Semantic chunking (cut where the topic changes) works the same way but needs an embedding of every
# sentence; try `C.semantic(excerpt, emb_model)` in the stretch tasks. The code of the chunkers is short:

# %%
show_code(C.fixed_size, C.recursive)

# A second, quieter problem: chunks that **start in the middle of a thought**. A chunk that begins
# with "He said…" or "This means…" is meaningless on its own, both for the embedding and for the LLM.
# How often does that happen?

# %%
ORPHANS = {"he", "she", "it", "they", "this", "these", "that", "those", "his", "her", "their", "its", "but", "and"}
def orphan_rate(chunks):
    first = [c["text"].split()[0].strip("\"'“‘(").lower() for c in chunks if c["text"].split()]
    return np.mean([w in ORPHANS for w in first])

for name, fn, kw in [("fixed 50 words", C.fixed_size, dict(size=50)),
                     ("fixed 200 words", C.fixed_size, dict(size=200)),
                     ("sentence-packed 200", C.sentence_packed, dict(size=200)),
                     ("recursive (paragraphs) 200", C.recursive, dict(size=200))]:
    ch = C.chunk_corpus(docs, fn, **kw)
    ex = next(c["text"][:110] for c in ch if c["text"].split()[0].lower() in ORPHANS)
    print(f"{name:28s} {orphan_rate(ch):5.1%} of chunks start with a pronoun/connector   e.g. «{ex}…»")

# %% [markdown]
# 🤔 **Question A1b.** Did respecting sentence boundaries reduce the number of chunks that start with "It…", "But…", "They…"?
#
# <details><summary>Answer</summary>
# No. In our test runs, sentence-packed chunks started with a pronoun or connector <i>more</i> often (~12%) than
# fixed-size ones (~7%): real sentences often start with "It" or "But", while fixed windows start at
# a random word. Clean boundaries keep <i>facts</i> intact but do not restore <i>context</i>: "It worked
# like a charm" still doesn't say what "it" is. Fixes used in practice: prepend the document title
# or section heading to every chunk ("contextual chunk headers"), add overlap, or retrieve small
# chunks and pass the surrounding parent chunk to the LLM.
# </details>
#
# Here is what a split fact looks like in practice: a real evidence sentence cut by the 200-word fixed chunker.

# %%
fixed = C.chunk_corpus(docs, C.fixed_size, size=200)
for d, a, b in spans:
    parts = [c for c in fixed if c["doc_id"] == d and c["start"] < b and c["end"] > a]
    if len(parts) > 1:
        body = mh.articles.body[d]
        print("FACT:", body[a:b], "\n")
        for c in parts:
            print(f"--- chunk ends/starts here ---\n...{body[max(c['start'], a - 80):min(c['end'], b + 80)]}...\n")
        break

# %% [markdown]
# 🤔 **Question A2.** Bigger chunks keep more facts intact. So why doesn't everybody use 800-word chunks?
#
# <details><summary>Answer</summary>
#
# Chunk size is a trade-off:
#
# | small chunks | large chunks |
# |---|---|
# | precise embeddings, cheap to put in the prompt | facts stay intact, more surrounding context |
# | facts get split; pronouns lose their referent ("*he* said…") | embeddings get diluted; retrieval gets less precise |
# | many more vectors to store and search | fewer, but each costs more tokens in the prompt |
#
# Overlap and sentence or paragraph boundaries are cheap ways to avoid cutting facts without
# making chunks huge. In Lab 2 we *measure* which chunk size retrieves best instead of guessing.
# </details>
#
# 🤔 **Question A3.** BioASQ abstracts are ~150 words each. Should we chunk them?
#
# <details><summary>Answer</summary>
# No. One abstract is already about one chunk. Chunking strategy matters for long, multi-topic
# documents (manuals, contracts, news, papers), not for short, focused ones. From here on we use
# whole abstracts as our retrieval units.
# </details>

# %% [markdown]
# **Stretch:** try `C.semantic(text, model)`. It starts a new chunk where neighbouring sentences
# stop being similar in meaning. Is it better than paragraphs on these articles? Is it worth the
# extra embedding cost?

# %% [markdown]
# ---
# ## Part B — Sparse retrieval with BM25 (BioASQ)
#
# A real question from the BioASQ benchmark, and the abstracts biomedical experts marked as relevant:

# %%
row = bio.questions.iloc[3]
doc_text = dict(zip(bio.doc_ids, bio.docs))
print("QUESTION:", row.question)
print("EXPERT ANSWER:", row.answer[:300])
print(f"\n{len(row.relevant)} relevant abstracts, e.g.:\n", doc_text[row.relevant[0]][:400], "...")

# %% [markdown]
# BM25 scores a document by the query words it contains, weighted by how **rare** each word is (IDF),
# with **diminishing returns** for repeated words (`k1`) and a penalty for **long documents** (`b`).
# Here it is from scratch: it really is this small.

# %%
from collections import Counter
from ragclass.retrieval import tokenize

def bm25_scores_from_scratch(query, docs_tokens, k1=1.5, b=0.75):
    N = len(docs_tokens)
    avg_len = np.mean([len(d) for d in docs_tokens])
    df = Counter(t for d in docs_tokens for t in set(d))                 # in how many docs does t appear?
    scores = np.zeros(N)
    for t in set(tokenize(query)):
        idf = np.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))             # rare words weigh more
        for i, d in enumerate(docs_tokens):
            tf = d.count(t)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    return scores

small = [tokenize(t) for t in bio.docs[:2000]]                            # slow loop: only 2k docs
s = bm25_scores_from_scratch(row.question, small)
print("top-3 docs in the first 2,000:", np.argsort(-s)[:3], np.sort(s)[::-1][:3].round(2))

# %% [markdown]
# The library version (`ragclass.retrieval.BM25`) computes exactly the same formula with a sparse
# matrix, which makes searching all 40k abstracts take milliseconds.

# %%
from ragclass.retrieval import BM25
import time
t = time.time(); bm25 = BM25(bio.docs); print(f"indexed {len(bio.docs):,} abstracts in {time.time()-t:.1f}s")

ids = np.array(bio.doc_ids)
def show(query, idx, scores, relevant=(), n=5, width=160):
    rel = set(relevant)
    for r, (i, s) in enumerate(zip(idx[:n], scores[:n]), 1):
        mark = "✅" if ids[i] in rel else "  "
        print(f"{r}. {mark} [{s:6.2f}] {bio.docs[i][:width]}")

idx, sc = bm25.search(row.question, 10)
show(row.question, idx, sc, row.relevant)

# %% [markdown]
# **What BM25 actually sees.** The question becomes a bag of tokens: lowercased, stop words removed,
# no order, no meaning. Each token gets an IDF weight from how many of the abstracts contain it:

# %%
q_tokens = tokenize(row.question)
print("question :", row.question)
print("tokens   :", q_tokens)
df_counts = np.asarray((bm25.weights > 0).sum(0)).ravel()
pd.DataFrame([{"token": t, "in # abstracts": int(df_counts[bm25.vocab[t]]) if t in bm25.vocab else 0,
               "IDF": round(float(bm25.idf[bm25.vocab[t]]), 2) if t in bm25.vocab else None} for t in q_tokens]
             ).sort_values("IDF", ascending=False)

# %% [markdown]
# And the top 3 documents with the matched tokens highlighted. ✅ = marked relevant by the experts.

# %%
for r, i in enumerate(idx[:3], 1):
    mark = "✅" if ids[i] in set(row.relevant) else "❌"
    show_html(f"<b>{r}. {mark}</b> " + highlight(bio.docs[i], q_tokens, width=700))

# %% [markdown]
# How much did each token contribute to the top document's score?

# %%
pd.Series(bm25.explain(row.question, idx[0])).sort_values(ascending=False).round(2)

# %% [markdown]
# 🤔 **Question B1.** Look at the contributions. Which kind of word dominates, and which words count for almost nothing? Why?
#
# <details><summary>Answer</summary>
# Rare, specific terms (gene names, drug names, diseases) carry most of the score because their
# IDF is high. Common words like "patients" or "role" appear in thousands of abstracts, so their
# IDF is close to zero. Stop words are removed before scoring.
# </details>
#
# 🤔 **Question B2.** The query says *"heart attack"*, the abstract says *"myocardial infarction"*. What does BM25 do?
#
# <details><summary>Answer</summary>
# Nothing. There is no word overlap, so the score is 0. This is the <b>vocabulary mismatch</b>
# problem. The classic fixes are synonym lists, query expansion and stemming. The modern answer is
# dense retrieval, which, as we'll see, only partly fixes it.
# </details>

# %% [markdown]
# Let's quantify it: of the top 10 BM25 results, how many abstracts even mention "myocardial infarction"?

# %%
def about_mi(i):
    return "myocardial infarction" in bio.docs[i].lower()
for q in ["heart attack treatment", "myocardial infarction treatment"]:
    i, s = bm25.search(q, 10)
    print(f"\n>>> {q!r}: {sum(about_mi(j) for j in i)}/10 of the top 10 mention myocardial infarction")
    show(q, i, s, width=110, n=3)

# %% [markdown]
# ---
# ## Part C — Dense retrieval (embeddings)
#
# A **bi-encoder** maps any text to a vector so that texts with similar *meaning* end up close
# together. We embed every abstract once (offline), then embed the query and take the nearest
# vectors by cosine similarity. We compare three real models:
#
# | model | trained on | size |
# |---|---|---|
# | `all-MiniLM-L6-v2` | general web text (Reddit, Q&A sites, Wikipedia…) | 22M parameters, 384 dims |
# | `bge-small-en-v1.5` | general text, stronger training recipe | 33M, 384 dims |
# | `pubmedbert-base-embeddings` | **biomedical** (PubMed) | 110M, 768 dims |

# %%
from sentence_transformers import SentenceTransformer
dev = "cuda" if GPU else "cpu"
models = {"MiniLM (general)": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev),
          "BGE-small (general)": emb_model,                      # loaded in Part A
          "PubMedBERT (biomedical)": SentenceTransformer("NeuML/pubmedbert-base-embeddings", device=dev)}
model = models["MiniLM (general)"]

# %% [markdown]
# **First, what does the model even read?** Before embedding, text is split into *word pieces* from a
# fixed vocabulary. Words the vocabulary doesn't know are chopped into fragments:

# %%
for w in ["AtxA", "BRCA1", "metformin", "myocardial infarction", "thyroxine", "heart attack"]:
    print(f"{w:22s}", " | ".join(f"{n.split()[0]}: {m.tokenizer.tokenize(w)}" for n, m in models.items()))

# %% [markdown]
# Now: which pairs does each model consider similar? The first rows are short phrases; the last three
# compare a *question* with *answer-like sentences*, which is closer to what retrieval does.

# %%
pairs = [("heart attack", "myocardial infarction"), ("heart attack", "heart failure"),
         ("heart attack", "panic attack"), ("BRCA1 mutation", "BRCA2 mutation"),
         ("BRCA1 mutation", "a mutation in BRCA1"), ("drug X is effective", "drug X is not effective"),
         ("What causes a heart attack?", "Myocardial infarction results from occlusion of a coronary artery."),
         ("What causes a heart attack?", "Heart failure is a chronic condition in which the heart cannot pump enough blood."),
         ("What causes a heart attack?", "Panic attacks are sudden episodes of intense fear.")]
sims = pd.DataFrame({"text A": [a for a, _ in pairs], "text B": [b[:45] for _, b in pairs]})
for name, m in models.items():
    e = m.encode([t for p in pairs for t in p], normalize_embeddings=True)
    sims[name] = [round(float(e[2*i] @ e[2*i+1]), 2) for i in range(len(pairs))]
sims

# %% [markdown]
# Three pairs is a small test. Let's be fairer: rank **30 medical terms** by similarity to "heart attack"
# and see where the true synonym lands for each model.

# %%
terms = ["myocardial infarction", "heart failure", "panic attack", "cardiac arrest", "angina pectoris",
         "atrial fibrillation", "stroke", "coronary artery disease", "hypertension", "arrhythmia", "cardiomyopathy",
         "pericarditis", "pulmonary embolism", "asthma attack", "migraine", "heart transplant", "heartburn",
         "anxiety disorder", "myocarditis", "aortic stenosis", "deep vein thrombosis", "heart murmur", "tachycardia",
         "shark attack", "terrorist attack", "seizure", "diabetes", "pneumonia", "sepsis", "myocardial ischemia"]
for name, m in models.items():
    e = m.encode(["heart attack"] + terms, normalize_embeddings=True)
    sim = e[1:] @ e[0]
    order = list(np.argsort(-sim))
    top = ", ".join(f"{terms[i]} ({sim[i]:.2f})" for i in order[:5])
    print(f"{name:26s} 'myocardial infarction' ranks #{order.index(0) + 1} of {len(terms)}   top 5: {top}")

# %% [markdown]
# 🤔 **Question C1.** Where did the synonym land? Which terms beat it, and what do they have in common? Which similarities in the pair table are *dangerous* for search?
#
# <details><summary>Answer</summary>
#
# * **No model reliably maps "heart attack" to "myocardial infarction".** In our test runs the synonym ranked
#   7th (MiniLM), 4th (BGE) and 3rd (PubMedBERT) out of 30. The domain model is better, not solved.
# * **Shared surface words win:** heart failure, heartburn, panic attack, asthma attack. Embeddings
#   pick up words and topic at least as much as meaning, especially for short phrases.
# * **Dangerous for every model:** BRCA1 vs BRCA2 (different genes) and "effective" vs "not
#   effective" (opposite claims) score around 0.9. Identifiers, numbers and negation are weak spots
#   of embeddings, exactly where BM25's literal matching shines.
# * Raw cosine values cannot be compared across models (BGE puts almost everything at 0.7–0.9).
#   Compare **rankings within one model**.
#
# Similarity ≠ relevance: "similar" is whatever the training data made similar. Phrase-level
# tests are also only a proxy. What matters is retrieval on real queries, so let's test that.
# </details>

# %% [markdown]
# **Does it matter for retrieval?** Take the ~2,000 cardiology-ish abstracts (anything mentioning the
# heart) and search them for *"heart attack treatment"* with each model. How many of the top 10 mention
# myocardial infarction?

# %%
pool = [i for i, t in enumerate(bio.docs) if any(w in t.lower() for w in ("cardi", "heart", "myocard", "coronary"))]
pool_txt = [bio.docs[i] for i in pool]
print(f"pool: {len(pool):,} abstracts; BM25 on the same query is in the cell above")
for name, m in models.items():
    E = m.encode(pool_txt, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    qv = m.encode(["heart attack treatment"], normalize_embeddings=True)[0]
    top = np.argsort(-(E @ qv))[:10]
    print(f"{name:26s} {sum(about_mi(pool[j]) for j in top)}/10 mention myocardial infarction")

# %% [markdown]
# Now let's embed the **whole** corpus with the small general model, the most common default in
# tutorials (about 30 s on a T4 GPU, several minutes on CPU). Lab 2 compares all three properly.

# %%
from ragclass.retrieval import Dense
t = time.time()
dense = Dense(bio.docs, "sentence-transformers/all-MiniLM-L6-v2", model=model)
print(f"embedded {len(bio.docs):,} abstracts in {time.time()-t:.0f}s -> matrix {dense.emb.shape}")

idx, sc = dense.search(row.question, 10)
show(row.question, idx, sc, row.relevant)

# %%
show_code(Dense.search, Dense.encode_queries)
for q in ["heart attack treatment", "myocardial infarction treatment"]:
    i, s = dense.search(q, 10)
    print(f"\n>>> {q!r}: {sum(about_mi(j) for j in i)}/10 mention myocardial infarction")
    show(q, i, s, width=110, n=3)

# %% [markdown]
# 🤔 **Question C2.** Searching here is a full matrix multiplication over all 40k vectors. What happens with 100 million documents?
#
# <details><summary>Answer</summary>
# Exact search becomes too slow and the vectors no longer fit in memory. Real systems use
# <b>approximate nearest neighbour (ANN)</b> indexes, such as HNSW graphs or IVF/PQ in FAISS, or a
# vector database. They trade a little recall for orders of magnitude more speed. See the stretch
# task below.
# </details>

# %% [markdown]
# ---
# ## Part D — Side by side: where does each method win?
#
# Instead of hand-picking examples, let's run both systems on all 300 questions and find the
# questions where they **disagree the most** (measured by how many of the top 10 are relevant).

# %%
qs = bio.questions
b_runs = [ids[bm25.search(q, 10)[0]] for q in qs.question]
d_runs = [ids[i] for i, _ in dense.search_many(qs.question.tolist(), 10)]
hits = lambda run, rel: len(set(run) & set(rel))
qs = qs.assign(bm25_hits=[hits(r, rel) for r, rel in zip(b_runs, qs.relevant)],
               dense_hits=[hits(r, rel) for r, rel in zip(d_runs, qs.relevant)])
qs["diff"] = qs.dense_hits - qs.bm25_hits
print("relevant docs in top-10, averaged:  BM25 =", qs.bm25_hits.mean().round(2),
      "| dense =", qs.dense_hits.mean().round(2))
print("\nBM25 wins most clearly on:"); print(qs.nsmallest(4, "diff")[["question", "bm25_hits", "dense_hits"]].to_string())
print("\nDense wins most clearly on:"); print(qs.nlargest(4, "diff")[["question", "bm25_hits", "dense_hits"]].to_string())

# %% [markdown]
# Pick one question from each list and look at what each system returned:

# %%
pick = qs.nsmallest(1, "diff").iloc[0]          # change to qs.nlargest(1, "diff").iloc[0], or any row
print("Q:", pick.question, "\n\n--- BM25 ---"); i, s = bm25.search(pick.question, 5); show(pick.question, i, s, pick.relevant)
print("\n--- Dense ---"); i, s = dense.search(pick.question, 5); show(pick.question, i, s, pick.relevant)

# %% [markdown]
# 🤔 **Question D1.** On this biomedical corpus, the general-purpose embedding model does *not* beat 40-year-old BM25. Why might that be?
#
# <details><summary>Answer</summary>
#
# * **Domain shift.** all-MiniLM-L6-v2 was trained on general web text (Reddit, Q&A sites,
#   Wikipedia). Gene, protein and drug names are rare or unseen, so they get split into
#   meaningless sub-word pieces.
# * **Exact terms matter.** BioASQ questions are full of precise entities, where literal
#   matching is exactly what you want.
# * **Small model.** 6 layers and 384 dimensions, optimised for speed.
#
# Lesson: *do not assume embeddings beat keywords.* Measure on your own data. In Lab 2 we try
# stronger and domain-specific embedding models and combine both methods.
# </details>
#
# 🤔 **Question D2.** "Dense won on 4 questions and BM25 won on 4" is not a conclusion. How *should* we decide which retriever is better?
#
# <details><summary>Answer</summary>
# With <b>labelled queries and metrics</b> computed over all of them: precision, recall, MRR,
# nDCG. That is the next block.
# </details>

# %% [markdown]
# ---
# ## Stretch tasks (optional / homework)
#
# 1. **BM25 knobs.** Rebuild with `BM25(bio.docs, k1=0.5, b=0.3)` and other values. How do the
#    results for the questions above change?
# 2. **Another embedding model.** Try `"BAAI/bge-small-en-v1.5"` (stronger general model) or
#    `"NeuML/pubmedbert-base-embeddings"` (biomedical) in `Dense(...)`. Does dense now win?
# 3. **Approximate search.** Build a FAISS HNSW index and compare speed and results with exact search:
#
# ```python
# !pip -q install faiss-cpu
# import faiss
# index = faiss.IndexHNSWFlat(dense.emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
# index.add(dense.emb)
# scores, idx = index.search(dense.encode_queries([row.question]), 10)
# ```
# 4. **Break it.** Write a question that BM25 answers perfectly and dense gets completely wrong, and the reverse.
