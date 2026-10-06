"""Every published P03b file is what the notebooks assume it is, and the pipeline's promises hold.

The kit ships what the software left behind between the photographs and the mosaic, cut to one window
and one neighbourhood. These tests hold the pipeline to its contract: every file carries its provenance,
nothing that identifies a person ships, the two binary readers gave consistent arrays (a point projected
through its shot lands on its observation, the depth map's camera is the block's), and the lessons the
notebooks are built on still hold on the data as published.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest
import rasterio

from conftest import P03B_DATA

PII = re.compile(r"owner_?name|reporter|e-?mail|phone|inspector|creator|editor|operator", re.I)
FILES = ["flight.json", "photos.csv", "exif_opensfm.csv", "site.gpkg", "report.json", "overlap_count.tif", "photo_a.jpg", "photo_b.jpg", "photo_c.jpg",
         "undistorted_a.jpg", "undistorted_b.jpg", "features_pair.npz", "features_group.npz", "matches_group.npz", "pairs.csv", "block.npz", "tracks.npz",
         "reconstruction_steps.csv", "depth_a.npz", "dense_window.npz", "mesh_window.npz", "dsm_window.tif", "dtm_window.tif", "ortho_window.tif", "ground_truth.json",
         "ortho_10cm.tif", "dsm_25cm.tif", "dtm_50cm.tif", "dense_sample.npz", "mesh_whole.npz"]


@pytest.fixture(scope="module")
def block():
    return np.load(P03B_DATA / "block.npz")


@pytest.fixture(scope="module")
def tracks():
    return np.load(P03B_DATA / "tracks.npz")


@pytest.mark.parametrize("name", FILES)
def test_the_files_the_notebooks_name_travel_along(name):
    """A notebook may not read outside its folder, so the kit carries every file it opens."""
    assert (P03B_DATA / name).exists()


def test_the_kit_stays_small():
    total = sum(p.stat().st_size for p in P03B_DATA.iterdir() if p.is_file())
    assert total < 35_000_000, f"the kit weighs {total / 1e6:.1f} MB; thirty laptops pull it"
    largest = max(P03B_DATA.iterdir(), key=lambda p: p.stat().st_size)
    assert largest.stat().st_size < 10_000_000, f"{largest.name} is too large for the repository"


def test_every_raster_carries_its_provenance():
    for path in sorted(P03B_DATA.glob("*.tif")):
        with rasterio.open(path) as src:
            tags = src.tags()
        assert {"source", "licence", "retrieved", "flight"} <= set(tags), f"{path.name} lacks provenance tags: {sorted(tags)}"


def test_every_array_file_carries_its_provenance():
    for path in sorted(P03B_DATA.glob("*.npz")):
        keys = set(np.load(path).files)
        assert {"source", "licence", "flight"} <= keys, f"{path.name} lacks provenance arrays: {sorted(keys)}"


def test_nothing_that_identifies_a_person_ships():
    for csv in sorted(P03B_DATA.glob("*.csv")):
        bad = [c for c in pd.read_csv(csv, nrows=1).columns if PII.search(c)]
        assert not bad, f"{csv.name} carries {bad}"
    flight = json.loads((P03B_DATA / "flight.json").read_text())
    assert not {"operator", "operator_certificate", "approval_reference", "notes_private"} & set(flight), "the operator's details never ship"
    assert flight.get("photos_checked_for_people") is True
    assert "/" not in str(flight.get("processing", {}).get("project", "")), "flight.json carries a path of the instructor's machine"
    from PIL import Image
    for name in ("photo_a.jpg", "photo_b.jpg", "photo_c.jpg"):
        path = P03B_DATA / name
        assert path.stat().st_size < 1_500_000, f"{name} is too large to ship"
        ifd = Image.open(path).getexif().get_ifd(0x8769)
        assert 42033 not in ifd and 37500 not in ifd, f"{name} still carries a serial number or the maker note"


def test_published_p03b_ground_truth_carries_no_answers():
    published = json.loads((P03B_DATA / "ground_truth.json").read_text())
    assert set(published) == {"generated", "p03b"}, f"the published P03b ground truth carries {sorted(published)}"


def test_the_choices_are_consistent(choices03b, block):
    names = [str(n) for n in block["names"]]
    for key in ("a", "b", "c"):
        p = choices03b["photos"][key]
        assert p["name"] in names and names[p["shot_index"]] == p["name"] and 0 < p["coverage"] <= 1
    assert choices03b["pair"]["a"] == choices03b["photos"]["a"]["name"] and choices03b["pair"]["b"] == choices03b["photos"]["b"]["name"]
    assert choices03b["pair"]["baseline_m"] >= 8.0, "the pair needs a baseline, else there is no depth to find"
    assert len(choices03b["group"]) == choices03b["group_size"] and choices03b["photos"]["a"]["name"] in choices03b["group"]
    assert [names.index(n) for n in choices03b["group"]] == sorted(names.index(n) for n in choices03b["group"]), "the neighbourhood is listed in capture order"


def test_a_point_projected_through_its_shot_lands_on_its_observation(block, tracks, choices03b, p03b, sfm):
    """The tracks reader and the reconstruction reader agree, in the adjustment's own frame."""
    cam = json.loads(str(block["camera"])); W, H = cam["width"], cam["height"]; size = max(W, H)
    assert tracks["shot"].max() < len(block["names"]) and tracks["point"].max() < len(block["points"])
    i = choices03b["photos"]["a"]["shot_index"]
    s = tracks["shot"] == i
    u, v, depth = sfm.project_brown(block["points"][tracks["point"][s]], block["rotation"][i], block["translation"][i], cam, W, H)
    res = np.hypot(u - (tracks["x"][s] * size + W / 2 - 0.5), v - (tracks["y"][s] * size + H / 2 - 0.5))
    assert (depth > 0).all() and np.median(res) < 1.0 and (res >= 20).mean() < 0.005
    assert round(float(np.median(res)), 3) == p03b["block"]["residual_a"]["median_px"]
    assert p03b["block"]["residual_a"]["map_frame_median_px"] > p03b["block"]["residual_a"]["median_px"], "the map frame fits worse than the adjustment's own, the lesson of 03b_04"


def test_the_pair_features_are_the_tracks_features(tracks, choices03b, block):
    feats = np.load(P03B_DATA / "features_pair.npz")
    assert str(feats["name_a"]) == choices03b["pair"]["a"] and str(feats["name_b"]) == choices03b["pair"]["b"]
    assert feats["descriptors_a"].dtype == np.uint8 and feats["descriptors_a"].shape[1] == 128
    names = [str(n) for n in block["names"]]
    for key in ("a", "b"):
        s = tracks["shot"] == names.index(choices03b["pair"][key])
        pts = feats[f"points_{key}"]
        assert tracks["feature"][s].max() < len(pts)
        assert np.abs(pts[tracks["feature"][s], :2] - np.column_stack([tracks["x"][s], tracks["y"][s]])).max() < 1e-6


def test_the_matches_index_the_group_features():
    mg = np.load(P03B_DATA / "matches_group.npz"); fg = np.load(P03B_DATA / "features_group.npz")
    names = [str(n) for n in mg["names"]]
    for key in mg.files:
        if "__" not in key:
            continue
        a, b = key.split("__")
        assert names.index(a) < names.index(b), "the earlier photograph comes first"
        arr = mg[key]
        assert arr[:, 0].max() < len(fg[f"points_{a}"]) and arr[:, 1].max() < len(fg[f"points_{b}"])


def test_the_depth_map_camera_is_the_block_camera(block, choices03b, sfm):
    d = np.load(P03B_DATA / "depth_a.npz")
    i = choices03b["photos"]["a"]["shot_index"]
    cam = json.loads(str(block["camera"]))
    R = sfm.rodrigues(block["rotation_map"][i]); C = -R.T @ block["translation_map"][i]
    assert np.abs(d["R"] - R).max() < 1e-6 and np.abs(d["C"] - C).max() < 1e-6, "the depth map lives in the map frame with photo_a's pose"
    w = int(d["depth_size"][0])
    assert abs(d["K"][0, 0] - cam["focal_x"] * w) < 1e-6 and abs(d["K"][0, 2] - (w - 1) / 2) < 1e-6
    depth = d["depth"]
    valid = depth > 0
    assert 0.5 < valid.mean() <= 1 and float(d["dmin"]) <= depth[valid].min() and depth[valid].max() <= float(d["dmax"]) + 1e-3


def test_the_window_files_share_one_ground(block, choices03b, p03b, sfm):
    with rasterio.open(P03B_DATA / "dsm_window.tif") as dsm, rasterio.open(P03B_DATA / "dtm_window.tif") as dtm:
        assert dsm.transform == dtm.transform and dsm.shape == dtm.shape and dsm.crs == dtm.crs
        assert dsm.compression is not None and dsm.compression.name != "jpeg", "heights are never JPEG compressed"
        z = dsm.read(1, masked=True); g = dtm.read(1, masked=True)
        assert np.percentile((z - g).compressed(), 5) > -0.2, "the surface sits under the terrain somewhere"
        bounds = dsm.bounds; tr = dsm.transform
    minx, miny, maxx, maxy = choices03b["window"]["bounds"]
    assert abs(bounds.left - minx) < 0.1 and abs(bounds.top - maxy) < 0.1
    dense = np.load(P03B_DATA / "dense_window.npz")
    hw, hh = dense["half_m"]
    assert dense["xyz"][:, 0].min() >= -hw - 1e-3 and dense["xyz"][:, 0].max() <= hw + 1e-3 and dense["xyz"][:, 1].min() >= -hh - 1e-3 and dense["xyz"][:, 1].max() <= hh + 1e-3
    assert np.allclose(dense["centre_local"] + dense["offset"], choices03b["window"]["centre"], atol=0.01)
    mesh = np.load(P03B_DATA / "mesh_window.npz")
    assert mesh["faces"].max() < len(mesh["vertices"]) and mesh["faces"].shape[1] == 3
    with rasterio.open(P03B_DATA / "ortho_window.tif") as o:
        assert o.count == 3 and o.dtypes[0] == "uint8" and abs(o.bounds.left - minx) < 0.1
    # the surface projected into photo_a lands inside the frame for the share flight.json records
    cam = json.loads(str(block["camera"])); W, H = cam["width"], cam["height"]
    i = choices03b["photos"]["a"]["shot_index"]
    rows, cols = z.shape
    k = 20
    r, c = np.meshgrid(np.arange(0, rows, k), np.arange(0, cols, k), indexing="ij")
    zz = z.filled(np.nan)[r, c]
    X = tr.c + (c + 0.5) * tr.a - block["offset"][0]; Y = tr.f + (r + 0.5) * tr.e - block["offset"][1]
    keep = np.isfinite(zz)
    u, v, depth = sfm.project_brown(np.column_stack([X[keep], Y[keep], zz[keep]]), block["rotation_map"][i], block["translation_map"][i], cam, W, H)
    inside = (depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    assert abs(inside.mean() - choices03b["photos"]["a"]["coverage"]) < 0.02


def test_the_lessons_still_hold(p03b):
    c, b = p03b["block"]["calibration"], p03b["block"]
    assert abs(c["focal_change_percent"]) > 0.3, "the flight no longer moved the focal length, the calibration lesson is gone"
    assert b["residual_a"]["with_prior_camera_median_px"] > 2 * b["residual_a"]["median_px"], "the prior camera fits as well as the calibrated one"
    assert b["receiver"]["mean_3d_m"] < 3.0 and b["redundancy"] > 0
    m = b["mini_bundle"]
    assert m["rms_solved_px"] < m["rms_perturbed_px"] / 5 and m["centre_recovery_median_m"] < 0.3, "the mini adjustment no longer brings the cameras back"
    pr = p03b["pair"]["ratio_0_8"]
    assert 0 < pr["inliers"] <= pr["candidates"] and pr["inlier_share"] > 0.8
    g = p03b["tracks"]["group"]
    assert abs(g["union_find_tracks"] - g["software_tracks_within_group"]) < 0.05 * g["software_tracks_within_group"], "union find and the software disagree on the tracks"
    assert p03b["depth"]["search"]["within_half_metre_share"] > 0.6, "the correlation search no longer agrees with the depth map"
    assert p03b["ortho"]["covered_share"] > 0.9 and p03b["ortho"]["correlation_with_mosaic"] > 0.8


def test_the_whole_flight_products_cover_the_window(choices03b, p03b):
    """The whole-flight rasters that open 03b_06 and 03b_07 contain the window, at the resolutions the notebooks state."""
    minx, miny, maxx, maxy = choices03b["window"]["bounds"]
    for name, res in (("ortho_10cm.tif", 0.10), ("dsm_25cm.tif", 0.25), ("dtm_50cm.tif", 0.50)):
        with rasterio.open(P03B_DATA / name) as src:
            b = src.bounds
            assert b.left <= minx and b.bottom <= miny and b.right >= maxx and b.top >= maxy, f"{name} does not cover the window"
            assert abs(src.res[0] - res) < 0.01, f"{name} is not at {res} m"
            if name.startswith("ortho"):
                assert src.count == 3 and src.dtypes[0] == "uint8"
            else:
                assert src.count == 1 and src.dtypes[0] == "float32" and src.nodata == -9999.0
                assert src.compression is None or src.compression.name != "jpeg", "heights are never JPEG compressed"
    whole = p03b["whole"]
    assert whole["window_inside_whole"] is True and whole["ortho"]["res_cm"] == 10.0


def test_the_dense_sample_and_the_whole_mesh_are_consistent(p03b):
    sample = np.load(P03B_DATA / "dense_sample.npz")
    n = int(sample["n_sample"])
    assert sample["xyz"].shape == (n, 3) and sample["rgb"].shape == (n, 3) and sample["classification"].shape == (n,)
    assert n == p03b["whole"]["dense_sample"]["points"] and int(sample["n_total"]) == p03b["whole"]["dense_sample"]["of_total"]
    assert set(np.unique(sample["classification"]).tolist()) <= {0, 1, 2, 3, 4, 5, 6, 7, 9}, "an unknown point class"
    mesh = np.load(P03B_DATA / "mesh_whole.npz")
    assert mesh["vertices"].dtype == np.float32 and mesh["faces"].dtype == np.int32 and mesh["faces"].shape[1] == 3
    assert mesh["faces"].min() >= 0 and mesh["faces"].max() < len(mesh["vertices"])
    assert len(mesh["vertices"]) == p03b["whole"]["mesh_whole"]["vertices"] and len(mesh["faces"]) == p03b["whole"]["mesh_whole"]["faces"]
    assert int(mesh["n_vertices_total"]) == p03b["whole"]["mesh_whole"]["vertices_total"] and int(mesh["thinning"]) == p03b["whole"]["mesh_whole"]["thinning"]


def test_the_roof_is_one_building_of_the_site(choices03b, sfm):
    """The kit's one roof is an OpenStreetMap building that reaches into the window, with its corner inside it."""
    import geopandas as gpd
    from shapely.geometry import Point, box

    roof = sfm.roof_polygon(choices03b)
    window = box(*choices03b["window"]["bounds"])
    assert roof.intersects(window) and roof.intersection(window).area > 100, "the roof barely touches the window"
    buildings = gpd.read_file(P03B_DATA / "site.gpkg", layer="buildings_site").to_crs(choices03b["crs"])
    assert buildings.geometry.union_all().buffer(0.1).contains(roof), "the roof polygon is not a building of the site"
    assert window.contains(Point(*sfm.roof_corner(choices03b))), "the roof's corner lies outside the window"
    assert abs(roof.area - choices03b["roof"]["area_m2"]) < 1.0
