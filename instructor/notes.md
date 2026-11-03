# Instructor notes

The answers to every open question are in the notebooks themselves, in collapsible *Answer* blocks, and in
the slide speaker notes. This file covers logistics, timing and risks.

## One notebook for the day

Students use **`notebooks/rag_class_all_labs.ipynb`**: all labs in one notebook, so they share one Colab machine.
Data, models and (cached) embeddings from earlier labs are reused, so Labs 2–5 start in seconds instead of minutes.
If a student's runtime dies, they can open the single-lab notebook (`0N_*.ipynb`) and continue from there.

## Before class (checklist)

- [ ] **Make the GitHub repo public** (Colab badges and `git clone` need it). Push `notebooks/*.ipynb`.
- [ ] Send students the setup instructions the day before: Google account, free Gemini key
      (<https://aistudio.google.com/apikey>), saved in Colab Secrets as `GEMINI_API_KEY`.
- [ ] **The day before, run every notebook once in Colab** (T4 runtime) with a fresh key. Tested on 2 Nov 2026 with
      a real key (see "Model findings" below).
- [ ] Optional mirror: if Hugging Face or GitHub raw might be blocked on the university network, upload the
      six files from your cache folder (`~/.cache/ragclass` after a local run) as assets of a GitHub release
      tagged `data-v1`. `ragclass/data.py` already falls back to
      `github.com/richardjonker2000/rag_class/releases/download/data-v1/<file>`.
- [ ] Rebuild the slides if you edit them: `make -C slides` (downloads DeckCraft 0.12.0 into `slides/tools/`).

## LLM: Gemma 4 on Ollama (default since 3 Nov 2026)

The notebooks default to `PROVIDER = "ollama"` with `gemma4:e2b` (small) and `gemma4:e4b` (big, Lab 3 only, pulled on first
use). No keys, no rate limits; everything runs on the student's T4. **Not yet tested end-to-end** (no Ollama on the build
machine): check in Colab that the install cell works, how long the model download takes, answer speed, and that tool calling
in Lab 5 works with E2B. If E2B's tool calling is too weak, set the Lab 5 model to `gemma4:e4b`. The Gemini option below
still works (tested).

## Model findings: Gemini (tested 2 Nov 2026, free tier)

Free-tier limits per Google Cloud project (AI Studio → Rate Limit). Every student creates their own key and
project, so these apply per student:

| model | role here | RPM | requests/day |
|---|---|---|---|
| `gemini-3.5-flash-lite` | **default for everything** | 15 | 500 |
| `gemini-3.6-flash` | "big" model, only in Lab 3 (`USE_BIG_MODEL`) | 5 | 20 |
| `gemini-3.8-flash` | not used: requests hung for 90 s twice in testing | 5 | 20 |

- Flash-Lite answers in about 1 s. **Tool calling works** through the OpenAI-compatible endpoint, including
  parallel tool calls and Gemini's thought signatures (`extra_content`, kept by `model_dump()`).
- The client throttles to 14 RPM for Flash-Lite and 4 RPM for the big model, retries on 429/5xx/timeouts, and gives
  up a request after 90 s.

## LLM budget per student (approximate)

| Lab | Flash-Lite calls | big-model calls |
|---|---|---|
| 0 | ~2 | 0 |
| 1, 2 | 0 | 0 |
| 3 | ~20 | 12 (switch off with `USE_BIG_MODEL = False` to rerun freely) |
| 4 | ~60 | 0 |
| 5 | ~60–80 | 0 |

About 150–170 Flash-Lite calls per student, about a third of the daily 500, so there is room to play.
At 14 RPM the waiting is noticeable in Lab 4 (~4 min) and Lab 5 Part 6 (~5 min).

## Timing and facilitation

| Time | What | Notes |
|---|---|---|
| 08:15 | Warm-up (slides 1–4) | Students open Lab 0 immediately; live citation demo on openrouter.ai/chat. |
| 08:25 | Retrieval (slides 5–24) | Question slides: 20–40 s for the room, then reveal. |
| 08:40 | Lab 1 (25 min) | Everyone should reach Part D (side by side). |
| 09:05 | Measuring retrieval (15 min) | The two-engines example: let neighbours compute P@5 / R@5 first. |
| 09:20 | Break | |
| 09:30 | Better retrieval (10 min) | |
| 09:40 | Lab 2 (25 min) | Leaderboard challenge in the last 5 min; show the "Lab 2 results" slide afterwards. |
| 10:05 | Generation (5 min slides + Lab 3, 15 min) | Vote on "bigger model or better retrieval" before the lab; results slide after. |
| 10:25 | Break | |
| 10:35 | Evaluating answers (10 min) | |
| 10:45 | Lab 4 (20 min) | Core: Parts 1–2; make sure everyone gets to Part 3 (you are the judge). Then the "which metric" slide. |
| 11:05 | Break | |
| 11:15 | Agents (10 min) | |
| 11:25 | Lab 5 (25 min) | Everyone reads one full trace and runs the injection demo. |
| 11:50 | Wrap-up | Agent evaluation → five take-aways → questions. Backup slides after "Questions?". |

## Results to expect (from the test runs)

- **Chunking (Lab 1):** fixed 50 words keeps 45% of evidence facts intact; fixed 200 → 90%; 200 + 50 overlap → 99.9%;
  paragraph-recursive → 100%.
- **Embedding similarity (Lab 1, MiniLM):** heart attack ↔ myocardial infarction 0.59 < panic attack 0.62 < heart
  failure 0.72; BRCA1 ↔ BRCA2 mutation 0.94; effective ↔ not effective 0.91.
- **Leaderboard (Lab 2, 300 questions, 10k-abstract CPU subset), nDCG@10:** MiniLM 0.574 · BGE-small 0.663 ·
  BM25 0.666 · hybrid BM25+MiniLM 0.672 · hybrid BM25+BGE 0.693. BGE vs BM25 is a coin flip (84 wins / 86
  losses, bootstrap p ≈ 0.6), a nice moment for "is a difference real?". On the full 40k corpus (GPU) absolute
  numbers are lower; the ordering should hold. PubMedBERT runs only on GPU, so we have no number for it yet.
- **Where each wins (Lab 1):** BM25: "Which metabolite activates AtxA?" (8 vs 0 relevant in top 10), "ROCKET AF
  Trial". Dense: "mindfulness interventions in prostate cancer patients" (9 vs 3), "supplemental antioxidant in athletes".
- **Chunk size at an equal 1,000-word budget (Lab 2 stretch), evidence recall:** 50 words 0.42 · 100 → 0.44 ·
  200 → 0.35 · 400 → 0.21 · 800 → 0.11. Small chunks win on this metric; discuss "retrieve small, expand for the
  generator".
- **Similarity (Lab 1):** among 30 medical terms, "myocardial infarction" ranks 7th (MiniLM), 4th (BGE) and 3rd
  (PubMedBERT) as the nearest neighbour of "heart attack". Shared surface words win (heart failure, heartburn, panic/asthma
  attack). Searching "heart attack treatment" among ~heart-related abstracts of the full corpus: PubMedBERT 6/10 mention MI, MiniLM and BGE 3/10, BM25 0/10 (full corpus).
- **Stress test (Lab 4, real Gemini, 3 BioASQ answers):** negated answer F1 0.97 / embedding 0.99 / LLM judge 0.2;
  answer + invented claim F1 0.78 / embedding 0.90 / LLM judge **1.0**. The judge catches negation but not an added claim.
- **RAGTruth detectors (Lab 4):** at default thresholds lexical and NLI flag almost everything (recall 1.0); the sweep
  cell shows the trade-off. Quick run (12 items): LLM judge accuracy 0.75, lexical 0.75, NLI 0.58.
- **Agents with Gemma 4 E2B (your Colab run):** closed-book 0.00, workflow 0.62, agent 0.75 (~2 LLM calls per question).
  Gemma doesn't know the 2023 stories (or says it can't access them), so the tools clearly matter.
- **Agents (Lab 5, real Gemini):** multi-step tool use works (weather + Python, Wikipedia + Python). On a null question
  the agent searched 8 times and hit the step limit, a nice live example of looping. Injection: the obvious hidden-comment attack was ignored every time (Gemini and Gemma),
  so Part 5 now contrasts it with a plausible spoofed e-mail; with the "autonomous" system prompt Gemma 4 E2B sent the
  summary to the attacker's address 3/3 (default prompt: 0/3). Tell students to answer N at the approval prompt. Evaluation on 8 questions: closed-book 0.75, agent 0.62, workflow 0.50; the agent loops on
  unanswerable questions, while the workflow says "insufficient". Discuss Question 6.1 (the model already knows 2023 news).

## If things go wrong

- **No GPU available** (Colab quota): notebooks detect it and use a 10k-abstract subset and fewer reranked
  queries. Slower but complete.
- **Rate limited (429):** the client waits and retries automatically. If a whole room hits the limit, pair students
  up (one key per pair) or switch some to `ollama`.
- **No Google account / key:** `PROVIDER = "ollama"` installs Ollama and pulls a ~4B open model on the Colab
  GPU (2–4 min the first time). Tool calling in Lab 5 is weaker with small models, which is itself worth
  discussing.
- **Colab disconnects:** rerun the setup cell. Data is cached on the VM, or in Drive if `USE_DRIVE` was ticked in Lab 0.
