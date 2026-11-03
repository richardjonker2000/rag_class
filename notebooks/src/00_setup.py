# %% [markdown]
# # Lab 0 — Setup and a warm-up experiment
#
# **Advanced LLM Applications: RAG, Agents, and Evaluation**
#
# Run this at the start of class (about 2 minutes). It
# 1. checks that you have a **GPU** runtime,
# 2. starts the **LLM**: Gemma 4 (E2B), an open model running on your own Colab GPU,
# 3. downloads all **datasets and models** used today, so later labs start fast,
# 4. runs a small experiment: *can an LLM cite its sources from memory?*
#
# **Before you start:** *Runtime → Change runtime type → T4 GPU.*
#
# **The LLM:** by default we run **Gemma 4 E2B**, an open model, on your Colab GPU with Ollama.
# No account or API key needed. The first run downloads the model (about 5 GB, 1–3 minutes).
# Optional: choose `PROVIDER = "gemini"` to use Google's hosted models instead (needs a free key from
# https://aistudio.google.com/apikey, stored in Colab 🔑 *Secrets* as `GEMINI_API_KEY`).

# %%
# @title 1 · Course code and GPU check
import os, subprocess, sys
if os.path.exists("../ragclass"):
    sys.path.insert(0, "..")
else:
    if not os.path.exists("rag_class"):
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/richardjonker2000/rag_class.git"], check=True)
    sys.path.insert(0, "rag_class")
try:
    import sentence_transformers, openai  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers>=3", "openai"], check=True)
import torch
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️  No GPU. Runtime → Change runtime type → T4 GPU (the labs still run on CPU, but smaller and slower).")

# %%
# @title 2 · Optional: keep the downloads in Google Drive (survives disconnects)
USE_DRIVE = False  # @param {type:"boolean"}
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["RAGCLASS_CACHE"] = "/content/drive/MyDrive/ragclass_cache"
    print("Cache:", os.environ["RAGCLASS_CACHE"])

# %%
# @title 3 · Start the LLM (first time: downloads Gemma 4 E2B, 1-3 min)
PROVIDER = "ollama"  # @param ["ollama", "gemini", "openai"]
from ragclass import llm
llm.setup(os.environ.get("RAGCLASS_LLM", PROVIDER))
print("model says:", llm.ask("Reply with exactly: ready"))
# (the bigger model, Gemma 4 E4B, is only used in Lab 3 and downloads when first needed)

# %%
# @title 4 · Download all datasets and models (~1-2 min)
from ragclass.data import prepare_all
prepare_all()

# %% [markdown]
# ---
# ## Warm-up: can an LLM cite its sources from memory?
#
# We ask the model for PubMed papers on a niche biomedical question (a real BioASQ question you will
# meet again in Lab 1), then **check every PMID against the real PubMed database** (NCBI E-utilities, free, no key).

# %%
import json, re, urllib.request

PROMPT = ("List 8 PubMed articles about which metabolite activates the AtxA regulator in Bacillus anthracis. "
          "For each give the PMID and the exact title, one per line as: PMID | title")   # 🔧 try your own topic
reply = llm.ask(PROMPT)
print(reply)

def pubmed_title(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id={pmid}"
    with urllib.request.urlopen(url, timeout=15) as r:
        res = json.loads(r.read())["result"]
    return res.get(pmid, {}).get("title", "<no such PMID>")

print("\nWhat PubMed actually says:")
for pmid, title in re.findall(r"(\d{6,9})\s*\|\s*(.+)", reply):
    real = pubmed_title(pmid)
    same = set(re.findall(r"\w+", title.lower())) >= set(re.findall(r"\w+", real.lower())[:5])
    print(f"{'✅' if same else '❌'} {pmid}\n   model: {title.strip()[:110]}\n   real:  {real[:110]}")

# %% [markdown]
# 🤔 How many citations were real *and* matched the title? The model knows the field very well,
# yet it cannot reliably tell you **where** its knowledge comes from. That is the problem that
# retrieval-augmented generation solves, and the reason we start with retrieval.
