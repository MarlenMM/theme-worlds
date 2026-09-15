"""Load definitions straight out of the notebook.

The deliverable had to be one self-contained ``.ipynb``, so the sampler lives
in the notebook's cells and nowhere else. Keeping a second copy of it in a
package would mean testing code that was never submitted, and the two would
drift. Instead, the tests and scripts in this repository pull the definitions
they need out of the notebook's own source and execute those: what is tested
is exactly what ran on Colab.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebook" / "theme_worlds.ipynb"

# IPython line magics (`%pip install ...`, `!nvidia-smi`) are not Python.
_MAGIC = re.compile(r"^(\s*)([%!].*)$", re.MULTILINE)


def read_notebook(path: Path = NOTEBOOK) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_cells(path: Path = NOTEBOOK) -> list[str]:
    """Source of every code cell, in order, with magics neutralised."""
    return [
        _MAGIC.sub(r"\1pass  # \2", "".join(cell["source"]))
        for cell in read_notebook(path)["cells"]
        if cell["cell_type"] == "code"
    ]


def definitions(path: Path = NOTEBOOK) -> dict[str, str]:
    """Top-level functions, classes and assignments, keyed by name.

    Values are source text including decorators. Only the first definition of a
    name is kept, and the dict preserves notebook order, so executing the
    values in order reproduces the notebook's own definition order.
    """
    found: dict[str, str] = {}
    for source in code_cells(path):
        lines = source.splitlines()
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            else:
                continue
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            text = "\n".join(lines[first - 1 : node.end_lineno])
            for name in names:
                found.setdefault(name, text)
    return found


def load(*names: str, path: Path = NOTEBOOK, **namespace) -> dict:
    """Execute the named notebook definitions into a fresh namespace.

    ``namespace`` supplies whatever those definitions expect to find already
    defined by earlier cells (``torch``, ``np``, ``pipe``, ``DEVICE`` ...).
    """
    available = definitions(path)
    missing = [n for n in names if n not in available]
    if missing:
        raise KeyError(f"not defined at top level in {Path(path).name}: {missing}")
    wanted = set(names)
    executed: set[str] = set()
    for name, text in available.items():
        if name in wanted and text not in executed:
            exec(compile(text, f"{Path(path).name}:{name}", "exec"), namespace)
            executed.add(text)
    return namespace
