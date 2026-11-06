# Advanced LLM Applications: RAG, Agents, and Evaluation

A hands-on, half-day course (08:15–12:00). Short theory blocks with open questions, then Colab labs on **real data**:

* **BioASQ** ([rag-mini-bioasq](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq)): 40,221 PubMed abstracts, 4,719 expert questions with relevant-document labels and reference answers
* **MultiHop-RAG** ([paper](https://arxiv.org/abs/2401.15391), [data](https://huggingface.co/datasets/yixuantt/MultiHopRAG)): 609 long news articles, 2,556 multi-hop questions with evidence sentences

## Labs

**Recommended: one notebook for the whole day.**
[`rag_class_all_labs`](notebooks/rag_class_all_labs.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/richardjonker2000/rag_class/blob/main/notebooks/rag_class_all_labs.ipynb)
contains Labs 0–5 in sections, so everything runs on **one Colab machine**: datasets and models are downloaded once,
and embeddings are cached on disk and reused by later labs. The separate notebooks below are the same labs, for
jumping in directly (e.g. after a runtime reset).


| | Notebook | Topic | Needs an LLM? |
|---|---|---|---|
| 0 | [00_setup](notebooks/00_setup.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/richardjonker2000/rag_class/blob/main/notebooks/00_setup.ipynb) | GPU and key check, downloads, warm-up: can an LLM cite PubMed from memory? | yes |
| 1 | [01_retrieval_basics](notebooks/01_retrieval_basics.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/richardjonker2000/rag_class/blob/main/notebooks/01_retrieval_basics.ipynb) | chunking, BM25 from scratch, dense retrieval, where each wins | no |
| 2 | [02_hybrid_rerank_eval](notebooks/02_hybrid_rerank_eval.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/richardjonker2000/rag_class/blob/main/notebooks/02_hybrid_rerank_eval.ipynb) | IR metrics, leaderboard, hybrid (RRF), reranking, incomplete labels, chunk size | no |
| 3 | [03_generation](notebooks/03_generation.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/richardjonker2000/rag_class/blob/main/notebooks/03_generation.ipynb) | grounded prompts; model size vs retrieval quality; lost in the middle | yes |
| 4 | [04_generation_eval](notebooks/04_generation_eval.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/richardjonker2000/rag_class/blob/main/notebooks/04_generation_eval.ipynb) | reference-based metrics and an LLM judge vs expert answers, stress test | yes |
| 5 | [05_agents](notebooks/05_agents.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/richardjonker2000/rag_class/blob/main/notebooks/05_agents.ipynb) | agent loop from scratch, real tools, tool descriptions, prompt injection, agent evaluation | yes |

Use a **T4 GPU** runtime (*Runtime → Change runtime type*). Every notebook clones this repo and downloads only
what it needs. Downloads are cached, retried and verified (`ragclass/data.py`).

### LLM access

Default: **Gemma 4 E2B** (and E4B as the "bigger" model in Lab 3), open models running on the student's own Colab
GPU via **Ollama**. No account or API key needed; the first run downloads the model (~5 GB, 1–3 minutes).
Alternatives (dropdown in each notebook): `gemini` (free key from <https://aistudio.google.com/apikey>, stored in
Colab 🔑 *Secrets* as `GEMINI_API_KEY`) or `openai`. Model names are in `ragclass/llm.py` (`PROVIDERS`).

## Repository

```
notebooks/*.ipynb        the labs (open these in Colab)
notebooks/src/*.py       editable sources of the notebooks (jupytext); `python notebooks/build.py` regenerates
ragclass/                small shared library: data download, chunking, retrievers, IR metrics, LLM client, generation metrics
```
