"""The end-of-class showcase script must still run, and still make its point.

It is the one piece of code in the repository that is demonstrated live rather than
worked through, so a silent breakage would be discovered in front of the class.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "P01" / "showcase_walking_distance.py"
OUT = REPO / "P01" / "showcase_output"


@pytest.fixture(scope="module")
def showcase():
    """Run the script once and hand every test the same output and artefacts."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO.parent,  # deliberately not the script's own folder, it must not care
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"showcase failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    return result.stdout, json.loads((OUT / "summary.json").read_text())


def test_it_runs_and_writes_its_outputs(showcase):
    output, _ = showcase
    assert "audit passed" in output, "the built-in routing audit did not report"
    for name in (
        "walkshed.png",
        "walkshed_400m.geojson",
        "within_400m_walk.geojson",
        "circle_400m_from_building.geojson",
        "summary.json",
    ):
        assert (OUT / name).exists(), f"showcase did not write {name}"


def test_the_walk_is_shorter_than_the_circle_claims(showcase):
    """The whole point of the demonstration, asserted rather than assumed.

    If a future data refresh made these two counts equal, the script would still run and
    still print a table, and the lesson it exists to teach would quietly be gone.
    """
    _, summary = showcase
    walk = summary["counts"]["network_walk_from_edge"]
    circle = summary["counts"]["straight_line_from_edge"]
    assert walk < circle, "the circle and the walk now agree, so the showcase shows nothing"


def test_every_published_count_matches_what_was_printed(showcase):
    """The exported file and the screen have to say the same thing.

    An earlier version wrote the from-edge circle count under the from-centroid key, so
    summary.json claimed the two circles agreed while the script printed a seven place
    difference between them. Nothing failed, because nothing compared them.
    """
    output, summary = showcase
    counts = summary["counts"]
    printed = {
        "network_walk_from_edge": _count_in(output, "network walk, from the building edge"),
        "straight_line_from_edge": _count_in(output, "straight line, from the building edge"),
        "straight_line_from_centre": _count_in(output, "straight line, from the centroid"),
    }
    assert counts == printed, f"summary.json {counts} disagrees with the printed table {printed}"
    assert counts["straight_line_from_edge"] != counts["straight_line_from_centre"], (
        "the two circles now agree, so the paragraph about the origin mattering says nothing"
    )


def test_the_exported_circle_is_the_one_the_headline_uses(showcase):
    """The shipped circle layer must be 400 m from the building, not from its middle.

    An earlier version exported the centroid circle beside a map that drew the building
    ring, so the layer silently contradicted the picture and the headline count.
    """
    import geopandas as gpd

    _, summary = showcase
    ring = gpd.read_file(OUT / "circle_400m_from_building.geojson").to_crs("EPSG:26918")
    inside = gpd.read_file(OUT / "within_400m_walk.geojson").to_crs("EPSG:26918")
    assert ring.geometry.iloc[0].contains(inside.union_all()), (
        "a place reachable within the walk falls outside the exported straight-line ring, "
        "which cannot happen when both start at the building"
    )


def _count_in(output: str, label: str) -> int:
    for line in output.splitlines():
        if line.startswith(label):
            return int(line.split("core")[0].split()[-1])
    raise AssertionError(f"no row labelled {label!r} in the output")
