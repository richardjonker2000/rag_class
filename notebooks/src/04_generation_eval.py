# %% [markdown]
# # Lab 4 — Evaluating generated answers
#
# **Advanced LLM Applications: RAG, Agents, and Evaluation** · ~25 min
#
# Retrieval had clean metrics because "is this document relevant?" has a fairly clear answer.
# For generated text it is much harder. In this lab you put the popular metrics to the test:
#
# 1. **Reference-based** metrics (exact match, token F1, ROUGE-L, embedding similarity,
#    LLM-judge correctness) against BioASQ expert answers, plus a **stress test**.
# 2. **Reference-free** hallucination detection (lexical overlap, NLI, LLM-as-judge) against
#    **human labels** on real LLM answers (RAGTruth).
# 3. **You are the judge:** label answers yourself and compare with the human annotators and with your neighbour.
#
# Needs an LLM (default: Gemma 4 E2B on your Colab GPU, same as Lab 3). About 60 LLM calls in total.

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

from ragclass.data import load_bioasq, load_ragtruth_qa
from ragclass.retrieval import BM25, Reranker, rrf
from ragclass import genmetrics as G
from ragclass.show import show_code, show_html, highlight
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
    cand = bm25.search(q, 50)[0]
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
# ## Part 2 — Hallucination detection vs human labels (RAGTruth)
#
# In production there is usually **no reference answer**. Instead we ask: *is the answer
# supported by the retrieved context?* (faithfulness / groundedness).
#
# **RAGTruth** has real answers from GPT-3.5/4, Llama-2 and Mistral to real search questions
# (MS MARCO), and **human annotators marked every hallucinated span**. We take a balanced sample
# of 40 (half contain a hallucination) and test three automatic detectors.

# %%
rt = load_ragtruth_qa(n=40, seed=0)
print(rt.model.value_counts().to_dict())
ex = rt[rt.hallucinated].iloc[0]
print("QUESTION:", ex.question, "\n\nCONTEXT:", ex.context[:1200], "...\n\nRESPONSE:", ex.response)
print("\nHUMAN-LABELLED HALLUCINATIONS:", [(s["text"], s["label_type"]) for s in ex.spans])

# %% [markdown]
# Three detectors, from cheap to expensive:
# 1. **Lexical overlap**: share of the answer's content words that appear in the context.
# 2. **NLI**: a small local model checks whether the context *entails* each answer sentence.
# 3. **LLM-as-judge**: we ask the LLM.

# %%
from ragclass.retrieval import tokenize
def lexical_support(context, answer):
    a, c = set(tokenize(answer)), set(tokenize(context))
    return len(a & c) / max(1, len(a))

nli = G.NLIChecker()

JUDGE_FAITH = """Check the ANSWER for hallucinations against the CONTEXT.
A hallucination is any claim in the answer that is not supported by the context, or that contradicts it.
CONTEXT:
{context}

QUESTION: {question}
ANSWER:
{answer}

Reply only with JSON: {{"verdict": "supported" or "hallucinated", "unsupported_claims": ["..."]}}"""

def judge_faith(r):
    out = parse_json(llm.ask(JUDGE_FAITH.format(context=r.context, question=r.question, answer=r.response)))
    return out.get("verdict", "").lower().startswith("halluc"), out.get("unsupported_claims", [])

LEX_T, NLI_T = 0.8, 0.5        # 🔧 decision thresholds, used below; try other values
rt["lexical"] = [lexical_support(r.context, r.response) for r in rt.itertuples()]
rt["nli"] = [nli.faithfulness(r.context, r.response) for r in rt.itertuples()]
judged = [judge_faith(r) for r in rt.itertuples()]
rt["judge_halluc"] = [j[0] for j in judged]
rt["judge_claims"] = [j[1] for j in judged]
print("LLM calls so far:", llm.usage()["calls"])

# %% [markdown]
# **Look inside the detectors** on two responses that humans marked as hallucinated. Human-marked
# spans are highlighted in red; each sentence gets the NLI entailment probability; below is the LLM
# judge's raw output.

# %%
show_code(G.NLIChecker.faithfulness)
for r in rt[rt.hallucinated].head(2).itertuples():
    marked = highlight(r.response, [sp["text"] for sp in r.spans], color="#F8B4B4")
    show_html(f"<b>Q:</b> {r.question}<br><b>Response ({r.model}), human spans in red:</b><br>{marked}")
    for sent, p_ in nli.sentence_scores(r.context, r.response):
        print(f"   NLI {p_:4.2f}  {'⚠️' if p_ < NLI_T else '  '}  {sent[:120]}")
    print("   LLM judge:", json.dumps({"hallucinated": r.judge_halluc, "unsupported_claims": r.judge_claims})[:400], "\n")

# %% [markdown]
# The lexical and NLI scores are continuous, so we need a threshold to turn them into "hallucinated?" decisions:

# %%
def report(pred, name):
    y = rt.hallucinated.values
    tp, fp = (pred & y).sum(), (pred & ~y).sum()
    fn, tn = (~pred & y).sum(), (~pred & ~y).sum()
    return dict(detector=name, accuracy=(tp + tn) / len(y), precision=tp / max(1, tp + fp),
                recall=tp / max(1, tp + fn), flagged=int(pred.sum()), missed=int(fn), false_alarms=int(fp))

summary = pd.DataFrame([report(rt.lexical.values < LEX_T, f"lexical < {LEX_T}"),
                        report(rt.nli.values < NLI_T, f"NLI < {NLI_T}"),
                        report(rt.judge_halluc.values, "LLM judge")]).set_index("detector").round(2)
summary

# %% [markdown]
# The threshold is a choice. Sweep it and watch precision and recall trade off:

# %%
sweep = []
for t in [0.3, 0.5, 0.7, 0.8, 0.9]:
    sweep.append({"threshold": t, **{f"lexical {k}": v for k, v in report(rt.lexical.values < t, "").items()
                                     if k in ("precision", "recall")},
                  **{f"NLI {k}": v for k, v in report(rt.nli.values < t, "").items() if k in ("precision", "recall")}})
pd.DataFrame(sweep).set_index("threshold").round(2)

# %% [markdown]
# 🤔 **Question 2.0.** Is there a threshold that catches most hallucinations *without* flagging most faithful answers?
#
# <details><summary>Answer</summary>
# Usually not with these cheap detectors. Faithful answers also paraphrase, add connecting words, and
# combine sentences, so "not literally in the context" is common even without hallucination. A high
# recall comes with many false alarms. Precision and recall are the honest way to report a
# detector, and with 40 items each number is still noisy.
# </details>

# %% [markdown]
# Now the interesting part: **where do the detectors disagree with the humans?**

# %%
wrong = rt[rt.judge_halluc != rt.hallucinated]
for r in wrong.head(3).itertuples():
    display(Markdown(
        f"**Human label:** {'HALLUCINATED' if r.hallucinated else 'faithful'} · "
        f"**LLM judge:** {'HALLUCINATED' if r.judge_halluc else 'faithful'} · NLI={r.nli:.2f} · lexical={r.lexical:.2f}  \n"
        f"**Q:** {r.question}  \n**Response ({r.model}):** {r.response}  \n"
        f"**Human-marked spans:** {[s['text'] for s in r.spans]}  \n**Judge's unsupported claims:** {r.judge_claims}"))
    print("CONTEXT:", r.context[:1500], "\n" + "-" * 100)

# %% [markdown]
# 🤔 **Question 2.1.** Read the disagreements. Is the detector wrong, or the human? Is there always a right answer?
#
# <details><summary>Answer</summary>
#
# All three happen:
# * The detector misses a subtle unsupported detail (a number or a date) that the human caught.
# * The detector flags common knowledge that is true but not in the context. RAGTruth humans
#   sometimes mark this as a hallucination and sometimes not (see the <code>implicit_true</code> field).
# * The human label itself is debatable.
#
# "Faithful to the context" and "true in the world" are different properties, and annotation
# guidelines have to choose.
# </details>
#
# 🤔 **Question 2.2.** Your detector flags 30% of production answers. Who decides the threshold, and how?
#
# <details><summary>Answer</summary>
# You need a <b>human-labelled sample</b> of your own traffic to choose the threshold:
# trade false alarms (annoying, costly reviews) against misses (wrong answers reaching users).
# The right trade-off depends on the domain, e.g. medical vs a shopping assistant. Without
# human labels, the threshold is a guess.
# </details>

# %% [markdown]
# ---
# ## Part 3 — You are the judge (5 min)
#
# Read 6 responses (context included) and decide: **1 = contains a hallucination, 0 = faithful**.
# Write your labels in the list *before* looking at the human labels. Then compare with your neighbour.

# %%
mine = rt.sample(min(6, len(rt)), random_state=3).reset_index(drop=True)
for i, r in mine.iterrows():
    display(Markdown(f"### Item {i}\n**Q:** {r.question}  \n**Response:** {r.response}"))
    print("CONTEXT:", r.context[:1500], "\n")

# %%
my_labels = [None, None, None, None, None, None]     # <- fill in 0 or 1 for items 0..5

# %%
if None in my_labels:
    print("Fill in all six labels first!")
else:
    mine["you"] = my_labels
    mine["human"] = mine.hallucinated.astype(int)
    mine["llm_judge"] = mine.judge_halluc.astype(int)
    display(mine[["you", "human", "llm_judge", "model"]])
    def kappa(a, b):   # Cohen's kappa: agreement corrected for chance
        a, b = np.asarray(a), np.asarray(b)
        po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"you vs human:     agreement {np.mean(mine.you == mine.human):.0%}, kappa {kappa(mine.you, mine.human):.2f}")
    print(f"LLM vs human:     agreement {np.mean(mine.llm_judge == mine.human):.0%}, kappa {kappa(mine.llm_judge, mine.human):.2f}")

# %% [markdown]
# 🤔 **Question 3.1.** How long did labelling 6 items take you? Did you agree with the original annotators? With your neighbour?
#
# <details><summary>Answer</summary>
# Careful labelling takes 1–2 minutes per item, even for short answers. Humans disagree too
# (RAGTruth reports inter-annotator agreement well below 100%). That is why serious evaluation
# needs <b>written guidelines, several annotators and agreement statistics</b>, not just "a human looked at it".
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
# | NLI faithfulness | is each sentence entailed by the context | long contexts, multi-sentence reasoning, world knowledge |
# | LLM-as-judge | an LLM's opinion, given your prompt | prompt-sensitive, verbosity and self-preference bias, inconsistent, costs money |
# | Human judgement | what you actually care about | slow, expensive, and humans disagree too |
#
# **Every automatic metric measures a different thing, and none of them is good enough on its
# own.** Use them to *monitor* and *triage* at scale, but calibrate and validate them against
# careful **human evaluation** on your own data, and keep humans reading real outputs.
