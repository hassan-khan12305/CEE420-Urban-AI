"""The P03b notebooks keep the shape they were rewritten into in October 2026.

Students run and read these notebooks; they do not write code. So every code cell is short and calls the
toolbox `sfm_tools.py` instead of defining anything, every text cell is short and plain, each notebook has
exactly one "change me" value and no blanks to fill, and nothing reaches outside the kit's folder. The
toolbox itself stays readable and self-contained.
"""

import ast
import json
import re
from importlib.util import module_from_spec, spec_from_file_location

import pytest

from conftest import P03B_KIT

NOTEBOOKS = sorted(p for p in P03B_KIT.glob("03b_*.ipynb") if "-mywork" not in p.name)
IDS = [p.stem for p in NOTEBOOKS]
MODULE = P03B_KIT / "sfm_tools.py"
ALLOWED_IMPORTS = {"json", "math", "time", "numpy", "scipy", "rasterio", "PIL", "matplotlib", "shapely", "inspect"}
MAX_CODE_LINES = 12
MAX_LINE_CHARS = 100
MAX_MD_WORDS = 120
MARKER = "# change me:"


def cells(notebook, kind):
    for index, cell in enumerate(json.loads(notebook.read_text())["cells"]):
        if cell["cell_type"] == kind:
            yield index, "".join(cell["source"])


def code_lines(text):
    return [line for line in text.splitlines() if line.strip()]


def test_the_seven_notebooks_carry_plain_names():
    expected = ["03b_01_photographs_and_metadata", "03b_02_feature_detection", "03b_03_feature_matching",
                "03b_04_bundle_adjustment", "03b_05_dense_matching", "03b_06_mesh_and_elevation_models",
                "03b_07_orthomosaic"]
    assert IDS == expected, f"the kit holds {IDS}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_no_notebook_defines_code_or_chains_statements(notebook):
    """Students call the toolbox; a notebook never defines a function or a class, and one line holds one statement."""
    for index, text in cells(notebook, "code"):
        tree = ast.parse(text)
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)), (
                f"{notebook.name} cell {index} defines a function, a class or a lambda")
        starts = [node.lineno for node in tree.body]
        assert len(starts) == len(set(starts)), f"{notebook.name} cell {index} puts two statements on one line"
        for line in code_lines(text):
            assert len(line) <= MAX_LINE_CHARS, f"{notebook.name} cell {index}: a line of {len(line)} characters"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_every_cell_is_short(notebook):
    """At most twelve lines of code per cell (the setup cell excepted), at most 120 words per text cell
    (the Further resources cell excepted)."""
    code = list(cells(notebook, "code"))
    for index, text in code[1:]:
        n = len(code_lines(text))
        assert n <= MAX_CODE_LINES, f"{notebook.name} cell {index} has {n} code lines"
    for index, text in cells(notebook, "markdown"):
        if text.lstrip().startswith("## Further resources"):
            continue
        words = len(re.sub(r"http\S+", "", text).split())
        assert words <= MAX_MD_WORDS, f"{notebook.name} cell {index} has {words} words"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_the_notebook_uses_the_toolbox_and_one_change_me(notebook):
    code = [text for _, text in cells(notebook, "code")]
    assert any("import sfm_tools as sfm" in text for text in code), f"{notebook.name} does not import the toolbox"
    marked = [text for text in code if MARKER in text]
    assert len(marked) == 1, f"{notebook.name} has {len(marked)} cells with a change-me value, one is required"
    assert sum(line.count(MARKER) for line in marked[0].splitlines()) == 1
    for text in code:
        assert "___" not in text, f"{notebook.name} still has a blank to fill"
    tags = [tag for cell in json.loads(notebook.read_text())["cells"] for tag in cell.get("metadata", {}).get("tags", [])]
    assert "break-glass" not in tags, f"{notebook.name} still carries an appendix"
    headings = [text.splitlines()[0] for _, text in cells(notebook, "markdown") if text.startswith("## ")]
    for required in ("## Your turn", "## Check your understanding", "## Where we are", "## Further resources"):
        assert required in headings, f"{notebook.name} lacks {required}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=IDS)
def test_every_data_path_in_code_exists(notebook):
    """The loaders hide the open() call from the style test, so every data path literal is checked here."""
    for index, text in cells(notebook, "code"):
        for quoted in re.findall(r"""["'](data/[^"'\n]+)["']""", text):
            assert (notebook.parent / quoted).exists(), f"{notebook.name} cell {index} names {quoted}, which is not there"


def test_the_toolbox_is_readable_and_self_contained():
    assert MODULE.exists(), "the kit ships sfm_tools.py"
    source = MODULE.read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= ALLOWED_IMPORTS, f"sfm_tools imports {sorted(imported - ALLOWED_IMPORTS)}"
    for forbidden in ("/Users", "data/scripts", "flight-20"):
        assert forbidden not in source, f"sfm_tools.py mentions {forbidden}"
    for number, line in enumerate(source.splitlines(), 1):
        assert len(line) <= MAX_LINE_CHARS, f"sfm_tools.py line {number} is {len(line)} characters"
    spec = spec_from_file_location("sfm_tools_under_test", MODULE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__doc__ and len(module.__all__) >= 30
    for name in module.__all__:
        function = getattr(module, name)
        assert callable(function), f"{name} is not callable"
        assert function.__doc__, f"{name} has no docstring"
