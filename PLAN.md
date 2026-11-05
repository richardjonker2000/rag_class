# Advanced LLM Applications: RAG, Agents, and Evaluation — course plan

**Session:** 08:15–12:00 (225 min) · 3 × 10-min breaks · 195 min content · English

## Teaching approach

- **Small theory blocks (8–15 min)** built from *open question → answer* slide pairs: each
  problem is posed as a question to the room before the answer is revealed.
- **Practical-heavy:** complete, working, commented Colab notebooks. Students *run, change and
  observe* rather than write boilerplate. The focus is **how it works, what it does, and where
  it breaks**. Each lab has core questions + optional "stretch / break it" tasks.
- **Core thesis:** RAG quality is dominated by retrieval, so retrieval gets the most time.
  The model matters much less than what you put in front of it.
- **Closing message:** generation metrics measure different things and none is good enough on
  its own. **Human evaluation (and reading traces) is key.**

## Timetable

| Time | Block | Min | Material |
|---|---|---|---|
| 08:15–08:25 | Hook: LLM hallucinates about unseen company docs · key check | 10 | `00_setup` |
| **Retrieval I** | | | |
| 08:25–08:40 | Theory: chunking, sparse (TF-IDF → BM25), dense (bi-encoders, ANN) | 15 | slides |
| 08:40–09:05 | Lab 1: chunk, BM25 + dense indexes, compare results by eye on tricky queries | 25 | `01_retrieval_basics` |
| 09:05–09:15 | Theory: retrieval evaluation — qrels, P@k, Recall@k, Hit rate, MRR, MAP, nDCG@10; offline vs online (CTR, A/B) in industry | 10 | slides |
| 09:15–09:25 | **Break** | 10 | |
| **Retrieval II + Generation** | | | |
| 09:25–09:35 | Theory: hybrid (RRF), cross-encoder reranking, query rewriting | 10 | slides |
| 09:35–10:05 | Lab 2: measure BM25 vs dense vs hybrid vs +rerank; chunk-size sweep; leaderboard | 30 | `02_hybrid_rerank_eval` |
| 10:05–10:20 | Lab 3: grounded prompt (citations, abstaining); **{no, weak, good retrieval} × {small, big model}**; lost in the middle | 15 | `03_generation` |
| 10:20–10:30 | **Break** | 10 | |
| **Generation evaluation** | | | |
| 10:30–10:40 | Theory: reference-based (EM/F1, ROUGE, embedding similarity) vs reference-free (LLM-as-judge: faithfulness, relevance); judge biases | 10 | slides |
| 10:40–11:05 | Lab 4: five metrics vs BioASQ expert answers + stress test → no metric is enough, humans are key | 25 | `04_generation_eval` |
| 11:05–11:15 | **Break** | 10 | |
| **Agents (generic, retrieval as running example)** | | | |
| 11:15–11:25 | Theory: workflow vs agent, when not to use one, tool schemas = prompts, loop, failure modes (loops, side effects, prompt injection), MCP/memory/multi-agent mention | 10 | slides |
| 11:25–11:50 | Lab 5: agent loop from scratch; tools: `search_news` (their retriever over MultiHop-RAG), `wikipedia` (real), `get_weather` (real, Open-Meteo), `run_python`, `send_email` (mocked, needs human approval); bad tool descriptions, prompt injection, agent eval | 25 | `05_agents` |
| 11:50–12:00 | Wrap-up: evaluating agents (task success, trajectory, cost), human review of traces, "RAG is just a tool", Q&A | 10 | slides |

## Technical design

- **LLM access:** one `llm()` helper on the OpenAI-compatible API, so the same code works for
  - **Gemini (default)**, using the free AI Studio key via Gemini's OpenAI-compatible endpoint
    with a Flash-Lite-class model, plus retry/backoff for rate limits;
  - **a local model in Colab**: Ollama + a ~4B open model on the T4, switched with one line;
  - any other OpenAI-compatible provider/key.
- **Retrieval runs fully locally:** sentence-transformers (embeddings, cross-encoder),
  rank_bm25 and numpy/FAISS. Labs 1–2 need **no API calls**.
- **Data (all real):**
  - **BioASQ** (rag-mini-bioasq): 40,221 PubMed abstracts and 4,719 expert questions with relevant-document labels
    (median 6 per question) and expert answers. Used for retrieval, IR metrics, generation and reference-based eval.
  - **MultiHop-RAG**: 609 long news articles (Sep–Dec 2023), 2,556 questions (inference, comparison, temporal,
    null) with evidence sentences. Used for chunking and agents.
- **Chunk-level labels are chunker-independent:** MultiHop-RAG evidence sentences are located in the
  articles; a chunk is relevant if it covers a gold span. Any chunking strategy can be scored.
- **Self-contained notebooks:** each one clones the repo and rebuilds its index in under a
  minute, so a Colab disconnect costs nothing. Shared plumbing lives in `ragclass/`; the ideas
  being taught stay visible in the notebook.
- **Answers to the open questions:** in collapsible blocks in the notebooks and in the slide speaker notes;
  logistics in `instructor/notes.md`.
- **Slides:** DeckCraft (Python build script → `.deckcraft` → PPTX/PDF), one slide per question
  followed by an answer slide.

## Repo layout

```
ragclass/                 shared helpers (data download, chunking, retrievers, metrics, llm client)
notebooks/00..05_*.ipynb  Colab labs (built from notebooks/src/*.py)
slides/                   DeckCraft build script + exported PPTX/PDF
instructor/notes.md       checklist, timing, expected results, fallbacks
```

## Before class

- The repo must be **public** for the Colab badges and `git clone` to work.
- Students create a free Gemini key in Google AI Studio beforehand; `00_setup` checks it.
- Check the free-tier limits in AI Studio for the chosen model (rate-limit page) the day before.
