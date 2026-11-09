"""One small LLM client for the whole course, using the OpenAI-compatible chat API, so the same
code works with Gemini (default, free key from Google AI Studio), a local model served by Ollama
inside Colab, or any other OpenAI-compatible provider.

    from ragclass import llm
    llm.setup()                       # default: Gemma 4 E2B on the Colab GPU via Ollama (no key needed)
    llm.setup("gemini")               # Gemini; reads GEMINI_API_KEY from Colab secrets / env / prompt
    llm.ask("Say hi")                 # -> str
    llm.chat(messages, tools=[...])   # -> full assistant message (for tool calling)
"""
from __future__ import annotations

import os
import subprocess
import time

PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "model": "gemini-3.5-flash-lite", "key": "GEMINI_API_KEY",
               # free tier per project (Nov 2026): flash-lite 15 RPM / 500 per day; flash 5 RPM / 20 per day
               "rpm": 14},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "gemma4:e2b",
               "key": None},
    "openai": {"base_url": None, "model": "gpt-5-mini", "key": "OPENAI_API_KEY"},
}

_state = {"client": None, "provider": None, "model": None, "min_interval": 0.0,
          "last_call": {}, "pulled": set(), "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}}


def _get_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata  # Colab "Secrets" panel (key icon on the left)
        v = userdata.get(name)
        if v:
            return v
    except Exception:  # noqa: BLE001 — not on Colab, or secret not shared with this notebook
        pass
    from getpass import getpass
    return getpass(f"Paste your {name} (input hidden): ").strip()


def setup(provider: str = "ollama", model: str | None = None,
          api_key: str | None = None, base_url: str | None = None, rpm: float | None = None) -> None:
    """Choose the provider. `rpm` spaces out calls to stay under a requests-per-minute limit."""
    from openai import OpenAI

    if provider == "fake":            # offline testing only: no model, canned replies
        _state.update(client=_FakeClient(), provider="fake", model="fake", min_interval=0.0)
        print("LLM ready: FAKE provider (testing only)")
        return
    cfg = PROVIDERS[provider]
    if provider == "ollama":
        start_ollama([model or cfg["model"]])
        api_key = "ollama"
    key = api_key or _get_key(cfg["key"])
    _state.update(client=OpenAI(api_key=key, base_url=base_url or cfg["base_url"], max_retries=0, timeout=90),
                  provider=provider, model=model or cfg["model"])
    rpm = rpm if rpm is not None else cfg.get("rpm", 0)        # stay under the free-tier rate limits
    _state["min_interval"] = 60.0 / rpm if rpm else 0.0
    print(f"LLM ready: provider={provider}, model={_state['model']}")


def _client():
    if _state["client"] is None:
        setup()
    return _state["client"]


def chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
         temperature: float = 0.0, max_tokens: int | None = None, retries: int = 6, **kw):
    """Send a chat request and return the assistant message. Retries on rate limits and server errors."""
    import openai

    client = _client()
    args = dict(model=model or _state["model"], messages=messages, temperature=temperature, **kw)
    if tools:
        args["tools"] = tools
    if max_tokens:
        args["max_tokens"] = max_tokens
    m = args["model"]
    if _state["provider"] == "ollama" and m not in _state["pulled"]:
        start_ollama([m])                 # another model: downloaded the first time it is used
    interval = _state["min_interval"]
    for attempt in range(retries):
        wait = interval - (time.time() - _state["last_call"].get(m, 0.0))
        if wait > 0:
            time.sleep(wait)
        _state["last_call"][m] = time.time()
        try:
            r = client.chat.completions.create(**args)
            u = _state["usage"]
            u["calls"] += 1
            if r.usage:
                u["prompt_tokens"] += r.usage.prompt_tokens or 0
                u["completion_tokens"] += r.usage.completion_tokens or 0
            return r.choices[0].message
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as e:
            if attempt == retries - 1:
                raise
            delay = min(60, 5 * 2 ** attempt)
            print(f"  [{type(e).__name__}] waiting {delay}s and retrying ({attempt + 1}/{retries - 1})")
            time.sleep(delay)


def ask(prompt: str, system: str | None = None, model: str | None = None, **kw) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return (chat(msgs, model=model, **kw).content or "").strip()


def model() -> str:
    """Name of the model in use."""
    return _state["model"]


def usage() -> dict:
    return dict(_state["usage"])


def start_ollama(models: list[str]) -> None:
    """Install Ollama (if needed), start the server in the background and pull the models."""
    import shutil
    import urllib.request

    if not shutil.which("ollama"):
        print("Installing Ollama (about 1 minute)...")
        if not shutil.which("zstd"):          # the Ollama download is a .tar.zst archive
            subprocess.run("apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq zstd > /dev/null 2>&1",
                           shell=True)
        r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True,
                           capture_output=True, text=True)
        if r.returncode != 0 or not shutil.which("ollama"):
            print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
            raise RuntimeError("Ollama install failed (output above). Plan B: set PROVIDER = 'gemini'.")
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
    except Exception:  # noqa: BLE001
        # one model in GPU memory at a time: the T4 also has to hold the embedding / reranker models
        env = {**os.environ, "OLLAMA_MAX_LOADED_MODELS": "1"}
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        for _ in range(30):
            try:
                urllib.request.urlopen("http://localhost:11434", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
    for m in models:
        if m in _state["pulled"]:
            continue
        print(f"Getting model {m} (first time: a few GB, 1-3 minutes on Colab)...")
        subprocess.run(["ollama", "pull", m], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _state["pulled"].add(m)


class _FakeClient:
    """Stands in for the OpenAI client when testing notebooks without a key."""

    class _Msg:
        def __init__(self, content):
            self.content, self.tool_calls, self.role = content, None, "assistant"

        def model_dump(self, **_):
            return {"role": "assistant", "content": self.content}

    def __init__(self):
        outer = self

        class _Completions:
            def create(self, **kw):
                last = kw["messages"][-1]["content"] or ""
                text = '{"verdict": "supported", "score": 4, "reason": "fake"}' if "JSON" in str(last) else \
                    "I don't know based on the provided sources. [1]"
                r = type("R", (), {})()
                r.choices = [type("C", (), {"message": outer._Msg(text)})()]
                r.usage = None
                return r

        self.chat = type("Chat", (), {"completions": _Completions()})()
