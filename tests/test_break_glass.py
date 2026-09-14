"""Every break-glass appendix must actually work: extract its fenced code and run it."""

import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def notebooks_with_appendix():
    found = []
    for path in sorted(REPO.glob("*/*.ipynb")):
        if "-mywork" in path.name or any(part.startswith(".") for part in path.relative_to(REPO).parts):
            continue
        cells = json.loads(path.read_text())["cells"]
        if any("break-glass" in c.get("metadata", {}).get("tags", []) for c in cells):
            found.append(path)
    return found


def fenced_blocks(notebook: Path):
    for cell in json.loads(notebook.read_text())["cells"]:
        if "break-glass" in cell.get("metadata", {}).get("tags", []):
            yield from FENCE.findall("".join(cell["source"]))


@pytest.mark.parametrize("notebook", notebooks_with_appendix(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_break_glass_solutions_run(notebook, ground_truth):
    """The appendix code must execute against the shipped data, from the notebook's own folder.

    Whether it produces the *right* number is asserted instructor-side, in
    data/scripts/verify_answers.py, because the answer must not ship to students.
    """
    blocks = list(fenced_blocks(notebook))
    assert blocks, f"{notebook.name} is tagged break-glass but has no fenced python"
    cwd = os.getcwd()
    os.chdir(notebook.parent)
    try:
        for block in blocks:
            namespace: dict = {}
            exec(block, namespace)  # noqa: S102
            if "answer" in namespace:
                assert isinstance(namespace["answer"], int)
                assert 0 < namespace["answer"] < ground_truth["counts"]["food"]
    finally:
        os.chdir(cwd)
