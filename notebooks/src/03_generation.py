# %% [markdown]
# # Lab 3 — Generation: from retrieved documents to an answer
#
# **Advanced LLM Applications: RAG, Agents, and Evaluation** · ~15 min
#
# We now put an LLM on top of the retriever and test one claim:
#
# > **The answer can only be as good as the retrieved documents.**
#
# 1. A grounded prompt: numbered sources, citations, permission to say "I don't know".
# 2. The experiment: the same model and questions with **no retrieval, weak retrieval and good retrieval**.
# 3. *(Stretch)* Lost in the middle: does the *position* of the evidence matter?
#
# Needs an LLM. Default: **Gemma 4 E2B**, a small open model running on your Colab GPU via Ollama,
# no key needed. Optional: `PROVIDER = "gemini"` with a free key from
# https://aistudio.google.com/apikey in Colab 🔑 *Secrets* as `GEMINI_API_KEY`.

# %%
# @title Setup (run me first): code, data, LLM, retriever (~1-2 min)
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

import numpy as np, pandas as pd, torch
pd.set_option("display.max_colwidth", 400)
GPU = torch.cuda.is_available()

PROVIDER = "ollama"  # @param ["ollama", "gemini", "openai"]
from ragclass import llm
llm.setup(os.environ.get("RAGCLASS_LLM", PROVIDER))

from ragclass.data import load_bioasq
from ragclass.retrieval import BM25, Dense, Reranker, rrf
bio = load_bioasq(n_questions=300, max_docs=None if GPU else 10_000)
ids = np.array(bio.doc_ids)
bm25 = BM25(bio.docs)
bge = Dense(bio.docs, "BAAI/bge-small-en-v1.5", show_progress=False)        # good dense model
minilm = Dense(bio.docs, "sentence-transformers/all-MiniLM-L6-v2", show_progress=False)  # weak one
reranker = Reranker()

def retrieve_good(q, k=5):
    """Hybrid (BM25 + BGE, fused with RRF) -> cross-encoder rerank of the top 50."""
    cand, _ = rrf([bm25.search(q, k=50)[0], bge.search(q, k=50)[0]], top=50)
    return reranker.rerank(q, cand, bio.docs, top=k)[0]

def retrieve_weak(q, k=5):
    """Small general-purpose embedding model only."""
    return minilm.search(q, k=k)[0]
print("ready")

# %% [markdown]
# ---
# ## Part 1 — A grounded prompt
#
# Read the prompt carefully. Every line is there for a reason.

# %%
SYSTEM = """You are a biomedical assistant. Answer the question using ONLY the numbered sources.
Cite the sources that support each statement, like [1] or [2][3].
If the sources do not contain the answer, reply exactly: "I don't know based on the provided sources."
Answer in at most 3 sentences."""

def build_prompt(question, doc_idx):
    sources = "\n\n".join(f"[{n}] {bio.docs[i]}" for n, i in enumerate(doc_idx, 1))
    return f"SOURCES:\n{sources}\n\nQUESTION: {question}"

def rag_answer(question, doc_idx):
    return llm.ask(build_prompt(question, doc_idx), system=SYSTEM)

def closed_book(question):
    return llm.ask(f"{question}\nAnswer in at most 3 sentences.")

row = bio.questions.iloc[0]
docs = retrieve_good(row.question)
print(build_prompt(row.question, docs)[:1500], "...\n")
u0 = llm.usage()
print("ANSWER:", rag_answer(row.question, docs))
u1 = llm.usage()
print("\nEXPERT:", row.answer)
print(f"\nThis one call: {u1['prompt_tokens'] - u0['prompt_tokens']} prompt tokens in, "
      f"{u1['completion_tokens'] - u0['completion_tokens']} tokens out. "
      f"Closed-book the prompt would be ~{len(row.question.split()) * 2} tokens: retrieval is what costs.")

# %% [markdown]
# This is exactly what is sent over the wire: a list of messages. No magic, just text.

# %%
import json
print(json.dumps([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_prompt(row.question, docs)[:600] + " …"}], indent=2))

# %% [markdown]
# 🤔 **Question 1.1.** Why number the sources and ask for citations? Why explicitly allow "I don't know"?
#
# <details><summary>Answer</summary>
#
# * **Citations** make answers *verifiable*: a user (or an automatic checker) can open [2] and
#   see whether it really says that. They also nudge the model to stay close to the sources.
# * **"I don't know"**: models are trained to be helpful, so without explicit permission they
#   fill gaps from memory or invent things. An honest abstention is better than a confident
#   wrong answer, especially in medicine.
# * **"ONLY the sources"** separates *grounded* answers from the model's parametric memory, which
#   may be outdated or wrong.
# </details>

# %% [markdown]
# ---
# ## Part 2 — The experiment: how much does retrieval matter?
#
# We pick real questions where the **weak** retriever found *no* expert-relevant document in its
# top 5 while the **good** retriever found at least two. For each question we compare three answers:

# %%
picked = []
for r in bio.questions.itertuples():
    rel = set(r.relevant)
    weak_hits = len(rel & set(ids[retrieve_weak(r.question)]))
    if weak_hits == 0:
        good_hits = len(rel & set(ids[retrieve_good(r.question)]))
        if good_hits >= 2:
            picked.append(r)
    if len(picked) == 4:
        break
print(f"{len(picked)} questions selected")

results = []
for r in picked:
    weak, good = retrieve_weak(r.question), retrieve_good(r.question)
    results.append(dict(question=r.question, condition="1 no retrieval", answer=closed_book(r.question)))
    results.append(dict(question=r.question, condition="2 weak retrieval", answer=rag_answer(r.question, weak)))
    results.append(dict(question=r.question, condition="3 good retrieval", answer=rag_answer(r.question, good)))
res = pd.DataFrame(results)
print("LLM calls so far:", llm.usage()["calls"])

# %% [markdown]
# Read the answers question by question and compare them with the expert answer.

# %%
from IPython.display import display, Markdown
for r in picked:
    display(Markdown(f"### {r.question}\n**Expert answer:** {r.answer[:600]}"))
    display(res[res.question == r.question].set_index("condition")[["answer"]])

# %% [markdown]
# 🤔 **Question 2.1.** How did the answers change from weak to good retrieval? What does the model do when retrieval is weak?
#
# <details><summary>Answer</summary>
# With weak retrieval the model either abstains ("I don't know"), which is good behaviour, or
# answers something only loosely related to the documents. With good retrieval it is usually close
# to the expert answer. The model did not change, only its input did: the generator cannot fix
# missing evidence, it just phrases the miss fluently. <b>Garbage in, eloquent garbage out.</b>
# A bigger model would not change this picture much; better retrieval does.
# </details>
#
# 🤔 **Question 2.2.** Look at the *no retrieval* answers. The model clearly knows a lot of biomedicine already. So what is RAG for?
#
# <details><summary>Answer</summary>
#
# * **Verifiability:** a closed-book answer may be right, but you cannot check it, and in
#   medicine or law you must be able to.
# * **Freshness:** model knowledge stops at the training cutoff.
# * **Private data:** your company's documents were never in the training data.
# * **Control:** you decide which sources are trusted.
#
# Note also that judging whether these answers are "right" was hard and slow. That is
# exactly the problem of the next block, evaluation.
# </details>

# %% [markdown]
# ---
# ## Stretch — Lost in the middle
#
# We hide the single most relevant document among 19 others and move it from the first position
# to the middle and to the end. Does the answer change? (Liu et al., 2023, *Lost in the Middle*)

# %%
r = picked[0] if picked else bio.questions.iloc[0]
good = retrieve_good(r.question, k=1)
distractors = [i for i in bm25.search(r.question, k=60)[0] if ids[i] not in set(r.relevant)][:19]
for pos in [0, 10, 19]:
    ctx = list(distractors)
    ctx.insert(pos, good[0])
    print(f"--- evidence at position {pos + 1}/20 ---\n{rag_answer(r.question, ctx)}\n")

# %% [markdown]
# 🤔 **Did the position matter?** In our test run with Gemini Flash-Lite it did *not*: the answer
# was the same at positions 1, 11 and 20. Current models are much more robust than those in the
# 2023 paper, at least for 20 short abstracts. The effect comes back with long contexts (hundreds
# of chunks, 50k+ tokens) and with weaker models. Try `distractors` of 100 documents. And note
# that one run per position proves nothing either way.
#
# **More to try:**
# * Remove the "I don't know" sentence from `SYSTEM` and rerun Part 2. What happens with weak retrieval?
# * Ask a question the corpus cannot answer (e.g. *"What is the capital of Portugal?"*) with `retrieve_good`.
# * Put `k=15` sources in the prompt instead of 5. Better or worse?
