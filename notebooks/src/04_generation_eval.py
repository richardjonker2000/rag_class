# %% [markdown]
# # Lab 4 — Evaluating generated answers
#
# **Advanced LLM Applications: RAG, Agents, and Evaluation** · ~25 min
#
# Retrieval had clean metrics because "is this document relevant?" has a fairly clear answer.
# For generated text it is much harder. In this lab you put the popular metrics to the test:
# exact match, token F1, ROUGE-L, embedding similarity and an LLM judge, scored against BioASQ
# expert answers, plus a **stress test** that shows what each metric really measures.
#
# Needs an LLM (default: Gemma 4 E2B on your Colab GPU, same as Lab 3). About 20 LLM calls in total.

# %%
# @title Setup (run me first)
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

import json, re, numpy as np, pandas as pd, torch
from IPython.display import display, Markdown
pd.set_option("display.max_colwidth", 300)
GPU = torch.cuda.is_available()

PROVIDER = "ollama"  # @param ["ollama", "gemini", "openai"]
from ragclass import llm
llm.setup(os.environ.get("RAGCLASS_LLM", PROVIDER))

from ragclass.data import load_bioasq
from ragclass.retrieval import BM25, Reranker, rrf
from ragclass import genmetrics as G
from sentence_transformers import SentenceTransformer

bio = load_bioasq(n_questions=300)
bm25 = BM25(bio.docs)
reranker = Reranker()
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if GPU else "cpu")

def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}
print("ready")

# %% [markdown]
# ---
# ## Part 1 — Reference-based metrics (BioASQ expert answers)
#
# First we generate answers for 6 questions with a simple RAG pipeline (BM25 + rerank, to keep
# setup fast), then score each answer against the expert answer in five ways.

# %%
SYSTEM = ("Answer the biomedical question using only the numbered sources, citing them like [1]. "
          "At most 3 sentences.")
def rag(q):
    cand = bm25.search(q, k=50)[0]
    top = reranker.rerank(q, cand, bio.docs, top=5)[0]
    src = "\n\n".join(f"[{n}] {bio.docs[i]}" for n, i in enumerate(top, 1))
    return llm.ask(f"SOURCES:\n{src}\n\nQUESTION: {q}", system=SYSTEM)

JUDGE_CORRECT = """You are grading an answer to a biomedical question against an expert reference answer.
QUESTION: {q}
REFERENCE ANSWER: {ref}
CANDIDATE ANSWER: {pred}
Does the candidate convey the same key facts as the reference, without contradicting it?
Reply only with JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

def judge_correct(q, ref, pred):
    return parse_json(llm.ask(JUDGE_CORRECT.format(q=q, ref=ref, pred=pred))).get("score", np.nan)

def score_all(q, ref, pred):
    return dict(EM=G.exact_match(pred, ref), F1=G.token_f1(pred, ref), ROUGE_L=G.rouge_l(pred, ref),
                EmbSim=float(G.embedding_similarity([pred], [ref], emb)[0]),
                LLM_judge=judge_correct(q, ref, pred) / 5)

sample = bio.questions.sample(6, random_state=1)
rows = []
for r in sample.itertuples():
    pred = rag(r.question)
    rows.append(dict(question=r.question, expert=r.answer, generated=pred, **score_all(r.question, r.answer, pred)))
part1 = pd.DataFrame(rows)
part1[["question", "EM", "F1", "ROUGE_L", "EmbSim", "LLM_judge"]].round(2)

# %%
for r in part1.itertuples():
    display(Markdown(f"**Q:** {r.question}  \n**Expert:** {r.expert[:400]}  \n**Generated:** {r.generated}  \n"
                     f"F1={r.F1:.2f} · ROUGE-L={r.ROUGE_L:.2f} · EmbSim={r.EmbSim:.2f} · judge={r.LLM_judge:.2f}"))

# %% [markdown]
# 🤔 **Question 1.1.** Read the pairs. Do the metric values match *your* sense of which answers are good? Why is exact match useless here?
#
# <details><summary>Answer</summary>
# Exact match only works for short factoid answers ("BRCA1", "yes"). Free-text answers that are
# fully correct get 0. F1 and ROUGE reward <i>word overlap</i>, so they punish correct answers
# phrased differently from the expert and are sensitive to answer length. Embedding similarity
# mostly measures "same topic", and the LLM judge comes closest to meaning but has its own
# biases. Each number measures something different.
# </details>

# %% [markdown]
# ### Stress test: what do the metrics actually measure?
#
# We take real expert answers and create controlled variants of them:
# * **paraphrase**: same meaning, different words → *should score high*
# * **negated**: main claim flipped → *should score low*
# * **plus unsupported claim**: correct answer + one extra invented sentence → *should be penalised*

# %%
def variant(kind, ref):
    instr = {"paraphrase": "Rewrite this answer with different wording but exactly the same meaning.",
             "negated": "Rewrite this answer so that its main claim is reversed (true becomes false), "
                        "changing as few words as possible.",
             "plus_unsupported": "Repeat this answer exactly, then add one plausible-sounding extra sentence "
                                 "with a specific claim (a number, a drug or a gene) that is NOT in the answer."}[kind]
    return llm.ask(f"{instr}\nOnly output the rewritten answer.\n\nANSWER: {ref}")

rows = []
for r in bio.questions.sample(3, random_state=2).itertuples():
    ref = r.answer[:500]
    for kind in ["identical", "paraphrase", "negated", "plus_unsupported"]:
        pred = ref if kind == "identical" else variant(kind, ref)
        rows.append(dict(question=r.question[:60], variant=kind, text=pred, **score_all(r.question, ref, pred)))
stress = pd.DataFrame(rows)
stress.groupby("variant", sort=False)[["F1", "ROUGE_L", "EmbSim", "LLM_judge"]].mean().round(2)

# %%
stress[["question", "variant", "text"]]

# %% [markdown]
# 🤔 **Question 1.2.** Which metrics rank *negated* above *paraphrase*? Which metric would you trust to catch an extra invented claim?
#
# <details><summary>Answer</summary>
# Overlap metrics (F1, ROUGE-L) typically score the <b>negated</b> answer higher than the
# <b>paraphrase</b>, because changing "is" to "is not" keeps almost all words. Embedding
# similarity also barely notices negation. None of the reference-based metrics is designed to
# notice an <i>added</i> unsupported claim: the reference is still fully covered. The LLM judge
# does best but is not perfect, and it can be inconsistent between runs or prompt wordings.
# </details>

# %% [markdown]
# ---
# ## Take-away
#
# | metric | what it really measures | blind spots |
# |---|---|---|
# | Exact match | identical string | everything that is phrased differently |
# | Token F1 / ROUGE | word overlap with a reference | negation, paraphrase, extra claims, length |
# | Embedding similarity | same topic | negation, numbers, extra claims |
# | LLM-as-judge | an LLM's opinion, given your prompt | prompt-sensitive, verbosity and self-preference bias, inconsistent, costs money |
# | Human judgement | what you actually care about | slow, expensive, and humans disagree too |
#
# **Every automatic metric measures a different thing, and none of them is good enough on its
# own.** Use them to *monitor* and *triage* at scale, but calibrate and validate them against
# careful **human evaluation** on your own data, and keep humans reading real outputs.
