"""Small display helpers so notebooks can show the library code and intermediate results."""
from __future__ import annotations

import html
import inspect
import re


def show_code(*objs) -> None:
    """Display the source code of functions/classes from the course library, so nothing is hidden."""
    from IPython.display import Markdown, display

    for obj in objs:
        src = inspect.getsource(obj)
        mod = obj.__module__
        display(Markdown(f"**`{mod}.{obj.__qualname__}`**\n```python\n{src}\n```"))


def highlight(text: str, terms, color: str = "#FFE58F", width: int | None = None) -> str:
    """HTML with every occurrence of the given terms highlighted (case-insensitive, whole words)."""
    t = html.escape(text if width is None else text[:width] + ("…" if len(text) > width else ""))
    terms = sorted({x for x in terms if x}, key=len, reverse=True)
    if not terms:
        return t
    pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(html.escape(x)) for x in terms) + r")(?![\w-])", re.I)
    return pat.sub(lambda m: f'<mark style="background:{color}">{m.group(0)}</mark>', t)


def show_html(s: str) -> None:
    from IPython.display import HTML, display

    display(HTML(f'<div style="font-family:sans-serif;line-height:1.45">{s}</div>'))
