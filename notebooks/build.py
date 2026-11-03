"""Build the Colab notebooks from the jupytext sources in notebooks/src/.

    python notebooks/build.py              # convert src/*.py -> notebooks/*.ipynb (no outputs)
    python notebooks/build.py --execute 01 # also run the notebook(s) whose name starts with 01 (testing)
"""
import sys
from pathlib import Path

import jupytext
import nbformat

HERE = Path(__file__).parent
COLAB = "https://colab.research.google.com/github/richardjonker2000/rag_class/blob/main/notebooks/{name}"


def build(src: Path) -> Path:
    nb = jupytext.read(src, fmt="py:percent")
    name = src.stem + ".ipynb"
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB.format(name=name)})"
    nb.cells.insert(0, nbformat.v4.new_markdown_cell(badge))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    for c in nb.cells:                          # setup cells: collapse the code in Colab forms
        if c.cell_type == "code" and c.source.startswith("# @title"):
            c.metadata["cellView"] = "form"
    out = HERE / name
    nbformat.write(nb, out)
    return out


LABS = [("00_setup", "Lab 0 · Setup and warm-up", "08:15"),
        ("01_retrieval_basics", "Lab 1 · Retrieval basics: chunking, BM25, embeddings", "08:40"),
        ("02_hybrid_rerank_eval", "Lab 2 · Measuring and improving retrieval", "09:40"),
        ("03_generation", "Lab 3 · Generation: model size vs retrieval", "10:10"),
        ("04_generation_eval", "Lab 4 · Evaluating generated answers", "10:45"),
        ("05_agents", "Lab 5 · Agents from scratch", "11:25")]


def build_combined() -> Path:
    """All labs in ONE notebook = one Colab machine: data, models and cached embeddings are shared."""
    name = "rag_class_all_labs.ipynb"
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB.format(name=name)})"
    toc = "\n".join(f"{i}. **{t}** · {when}" for i, (_, t, when) in enumerate(LABS))
    cells = [nbformat.v4.new_markdown_cell(
        f"{badge}\n\n# Advanced LLM Applications: RAG, Agents, and Evaluation — all labs\n\n"
        "All six labs in **one notebook**, so they run on **one Colab machine**: datasets, models and embeddings are "
        "downloaded or computed once and reused by every later lab. Embeddings are cached on disk, so re-running a "
        "lab after an error takes seconds.\n\n"
        "**Runtime → Change runtime type → T4 GPU.** Use the table of contents (left sidebar) to jump between labs. "
        "Each lab starts with its own setup cell: run it when you start that lab (it is quick if earlier labs ran).\n\n"
        + toc)]
    for stem, title, when in LABS:
        nb = jupytext.read(HERE / "src" / f"{stem}.py", fmt="py:percent")
        cells.append(nbformat.v4.new_markdown_cell(f"---\n# {title}\n*{when}*"))
        for c in nb.cells:
            if c.cell_type == "markdown" and c.source.lstrip().startswith("# Lab"):
                c.source = c.source.replace("# Lab", "## Lab", 1)        # demote the lab title under the section
            if c.cell_type == "code" and c.source.startswith("# @title Setup"):
                c.source = c.source.replace("# @title Setup", "# @title Setup for this lab (quick if you ran the earlier labs)", 1)
            if c.cell_type == "code" and c.source.startswith("# @title"):
                c.metadata["cellView"] = "form"
            cells.append(c)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "T4", "toc_visible": True}
    out = HERE / name
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    pats = [a for a in sys.argv[1:] if not a.startswith("--")]
    for src in sorted((HERE / "src").glob("*.py")):
        out = build(src)
        print("built", out.name)
        if execute and (not pats or any(src.stem.startswith(p) for p in pats)):
            from nbclient import NotebookClient
            nb = nbformat.read(out, as_version=4)
            NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
            test_out = HERE / "_executed" / out.name
            test_out.parent.mkdir(exist_ok=True)
            nbformat.write(nb, test_out)
            print("executed ->", test_out)
    out = build_combined()
    print("built", out.name)
    if execute and "all" in pats:
        from nbclient import NotebookClient
        nb = nbformat.read(out, as_version=4)
        NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        (HERE / "_executed").mkdir(exist_ok=True)
        nbformat.write(nb, HERE / "_executed" / out.name)
        print("executed ->", out.name)
