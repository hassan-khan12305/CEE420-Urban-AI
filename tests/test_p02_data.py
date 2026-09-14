"""Every published P02 reference number is recomputed from the shipped files.

The town's own data arrives in three coordinate systems and from three authorities, so
these tests also hold the pipeline to its promises: nothing that identifies a person
ships, every layer carries its provenance, and the lessons the notebooks are built on
still hold on the data as published.
"""

import re

import geopandas as gpd
import mapclassify
import numpy as np
import pandas as pd
import pyogrio
import pytest

from conftest import P02_DATA

METRES = "EPSG:26918"
PII = re.compile(r"owner_?name|reporter|e-?mail|phone|inspector|creator|editor", re.I)
FORBIDDEN = {"OWNER_NAME", "ST_ADDRESS", "CITY_STATE", "ZIP_CODE", "Owner_Name", "Inspector_Name",
             "Facility_Name", "Comments", "Description", "Address", "Assignee_name", "Agent_name"}


def test_every_layer_count_matches(municipal, njgin_parcels, grid, census, p02):
    c = p02["counts"]
    assert len(municipal["zoning"]) == c["zoning"]
    assert len(municipal["tax_blocks"]) == c["tax_block_polygons"]
    assert municipal["tax_blocks"]["BLOCK_NO"].nunique() == c["tax_blocks"]
    assert len(municipal["parcels"]) == c["parcels_municipal"]
    assert len(njgin_parcels) == c["parcels_njgin"]
    assert len(municipal["roads"]) == c["roads"]
    assert len(municipal["deer_reports"]) == c["deer_reports"]
    assert len(municipal["stormwater_inlets"]) == c["stormwater_inlets"]
    assert len(grid) == c["zoning"] * 0 + p02["grid"]["cells"]
    assert len(census["tracts"]) == c["tracts"]
    assert len(census["block_groups"]) == c["block_groups"]


def test_every_layer_is_in_the_crs_it_was_served_in(municipal, njgin_parcels, census, p02):
    """The whole first notebook rests on the files arriving in feet. Nothing may tidy that away."""
    for name, layer in municipal.items():
        assert layer.crs.to_string() == p02["crs"][name], f"{name} is in {layer.crs}, expected {p02['crs'][name]}"
    assert njgin_parcels.crs.to_string() == p02["crs"]["njgin_parcels"]
    assert census["tracts"].crs.to_string() == p02["crs"]["census"]
    assert municipal["zoning"].crs != municipal["tax_blocks"].crs, "the two datum realisations are the lesson"


def test_the_zoning_area_in_feet_is_the_513_trap(municipal, p02):
    zoning = municipal["zoning"]
    assert round(zoning.geometry.area.sum() / 1e6, 2) == p02["boundary"]["zoning_area_million_ft2"]
    assert round(zoning.to_crs(METRES).geometry.area.sum() / 1e6, 2) == p02["boundary"]["zoning_area_km2"]


def test_the_join_lessons_still_hold(njgin_parcels, municipal, p02):
    """An intersects join must still copy straddling parcels, and a centroid join must not."""
    parcels = njgin_parcels.to_crs(METRES)
    blocks = municipal["tax_blocks"].to_crs(METRES)[["BLOCK_NO", "geometry"]]
    inter = gpd.sjoin(parcels, blocks, predicate="intersects")
    centroids = parcels.copy()
    centroids["geometry"] = centroids.geometry.centroid
    within = gpd.sjoin(centroids, blocks, predicate="within")
    assigned = within.drop_duplicates("PAMS_PIN")
    lost = parcels[~parcels["PAMS_PIN"].isin(assigned["PAMS_PIN"])]
    j = p02["join"]
    assert len(inter) == j["intersects_rows"] and len(within) == j["centroid_rows"]
    assert len(assigned) == j["centroid_distinct"] and len(within) - len(assigned) == j["centroid_doubled"]
    assert len(lost) == j["unassigned"]
    assert round(float(inter["NET_VALUE"].sum())) == j["intersects_sum_usd"]
    assert round(float(assigned["NET_VALUE"].sum())) == j["centroid_sum_usd"]
    assert j["intersects_sum_usd"] > j["centroid_sum_usd"], "intersects no longer inflates the total"
    # The books must balance to the dollar: before minus after is exactly what was lost.
    before, after = float(parcels["NET_VALUE"].sum()), float(assigned["NET_VALUE"].sum())
    assert abs(before - after - float(lost["NET_VALUE"].sum())) < 1


def test_the_lobby_map_reproduces(njgin_parcels, municipal, p02):
    """02_04 builds the blocks in one cell; every number its prose and title lean on is recomputed here."""
    parcels = njgin_parcels.to_crs(METRES)
    blocks = municipal["tax_blocks"].to_crs(METRES)
    centroids = parcels.copy()
    centroids["geometry"] = centroids.geometry.centroid
    assigned = gpd.sjoin(centroids, blocks[["BLOCK_NO", "geometry"]], predicate="within").drop_duplicates("PAMS_PIN")
    exempt = assigned["PROP_CLASS"].astype(str).str.startswith("15")
    bl = blocks.dissolve(by="BLOCK_NO")[["geometry"]]
    bl["total"] = assigned.groupby("BLOCK_NO")["NET_VALUE"].sum()
    bl["taxable"] = assigned[~exempt].groupby("BLOCK_NO")["NET_VALUE"].sum()
    bl["acres"] = bl.geometry.area / 4046.856
    bl["per_acre"], bl["taxable_per_acre"] = bl["total"] / bl["acres"], bl["taxable"] / bl["acres"]
    b = p02["blocks"]
    assert int(bl["total"].notna().sum()) == b["with_parcels"] and int(bl["taxable"].notna().sum()) == b["with_taxable_parcels"]
    assert round(float(bl["total"].corr(bl.geometry.area)), 2) == b["corr_total_area"]
    assert len(set(bl["total"].nlargest(3).index) & set(bl["per_acre"].nlargest(3).index)) == b["top3_overlap"] == 0

    def ratio(column):
        rate = bl[column].dropna()
        cls = mapclassify.Quantiles(rate.values, k=5).yb
        return round(float(rate[cls == 4].median() / rate[cls == 0].median()), 1)

    assert ratio("per_acre") == b["per_acre_ratio_top_to_bottom_class"]
    assert ratio("taxable_per_acre") == b["taxable_per_acre_ratio_top_to_bottom_class"]
    assert b["taxable_per_acre_ratio_top_to_bottom_class"] > 5, "the title's claim has gone soft"


def test_the_grid_reproduces_the_lecture(grid, p02):
    g = p02["grid"]
    assert len(grid) == g["cells"] and int(grid["n"].sum()) == g["buildings"]
    for scheme, counts in g["occupancy"].items():
        assert mapclassify.classify(grid["n"].values.astype(float), scheme, k=5).counts.tolist() == counts
    assert len({tuple(v) for v in g["occupancy"].values()}) == 3, "two schemes agree, the classification lesson is gone"
    # Moran's I, fifteen lines of numpy rather than a dependency, as in the lecture.
    nb = gpd.sjoin(grid[["geometry"]], grid[["geometry"]], predicate="touches")
    i, j = np.asarray(nb.index), np.asarray(nb["index_right"])
    z = grid["n"].values.astype(float) - grid["n"].mean()
    moran = (len(z) / len(i)) * float(np.sum(z[i] * z[j])) / float(np.sum(z ** 2))
    assert round(moran, 4) == p02["moran"]["I"]
    assert p02["moran"]["I"] > 0.5 > abs(p02["moran"]["I_shuffled"]), "real and shuffled no longer differ"


def test_the_block_neighbours_are_a_decision(njgin_parcels, municipal, p02):
    """02_06 shows that touching finds almost no neighbours among blocks that stop at the
    kerb, and that a short band across the street does. Both numbers are recomputed."""
    parcels = njgin_parcels.to_crs(METRES)
    blocks = municipal["tax_blocks"].to_crs(METRES)
    centroids = parcels.copy()
    centroids["geometry"] = centroids.geometry.centroid
    assigned = gpd.sjoin(centroids, blocks[["BLOCK_NO", "geometry"]], predicate="within").drop_duplicates("PAMS_PIN")
    bl = blocks.dissolve(by="BLOCK_NO")[["geometry"]]
    bl["total"] = assigned.groupby("BLOCK_NO")["NET_VALUE"].sum()
    bl["per_acre"] = bl["total"] / (bl.geometry.area / 4046.856)
    wp = bl.dropna(subset=["per_acre"]).reset_index(drop=True)
    m = p02["moran_blocks"]
    assert len(wp) == m["blocks"]

    def pairs(**kw):
        nb = gpd.sjoin(wp[["geometry"]], wp[["geometry"]], **kw)
        nb = nb[nb.index != nb["index_right"]]
        return np.asarray(nb.index), np.asarray(nb["index_right"])

    def moran(values, i, j):
        z = np.asarray(values, dtype=float) - np.mean(values)
        return (len(z) / len(i)) * float(np.sum(z[i] * z[j])) / float(np.sum(z ** 2))

    ti, tj = pairs(predicate="touches")
    assert len(ti) == m["touches_pairs"] and int(np.sum(np.bincount(ti, minlength=len(wp)) == 0)) == m["touches_isolated"]
    assert m["touches_isolated"] > len(wp) / 2, "touching now finds neighbours; the lesson of 02_06 is gone"
    bi, bj = pairs(predicate="dwithin", distance=m["band_m"])
    assert len(bi) == m["band_pairs"] and int(np.sum(np.bincount(bi, minlength=len(wp)) == 0)) == m["band_isolated"] == 0
    assert round(moran(wp["per_acre"].values, ti, tj), 4) == m["I_per_acre_touches"]
    assert round(moran(wp["per_acre"].values, bi, bj), 4) == m["I_per_acre"]
    assert round(moran(wp["total"].values, bi, bj), 4) == m["I_total"]


def test_the_two_authorities_disagree_as_recorded(municipal, njgin_parcels, p02):
    """02_05 executes three wrong joins and one right one; every count it prints is recomputed here."""
    m, s, a = municipal["parcels"], njgin_parcels, p02["authorities"]

    def key(block, lot, qualifier):
        q = qualifier.fillna("").astype(str).str.strip().str.upper()
        return block.astype(str).str.strip() + "_" + lot.astype(str).str.strip() + "_" + q

    mk, sk = key(m["block"], m["lot"], m["qualifier"]), key(s["PCLBLOCK"], s["PCLLOT"], s["PCLQCODE"])
    assert len(m) == a["municipal_rows"] and len(s) == a["state_rows"] and mk.nunique() == a["municipal_keys_unique"]
    assert int(pd.to_numeric(m["propid"].str[4:9]).isin(s["PCLBLOCK"]).sum()) == a["propid_block_as_int_matches"] == 0
    naive = m["block"].astype(str) + "_" + m["lot"].astype(str) + "_" + m["qualifier"].astype(str)
    assert int(naive.isin(sk).sum()) == a["naive_key_matches"] < a["key_matched_municipal"] / 4
    assert int(mk.isin(sk).sum()) == a["key_matched_municipal"] and int(sk.isin(mk).sum()) == a["key_matched_state"]
    assert a["key_matched_state"] / a["state_rows"] > 0.99, "the key no longer pairs the two files"
    address_m = m["property_location"].fillna("").str.strip().str.upper()
    address_s = s["PROP_LOC"].fillna("").str.strip().str.upper()
    assert int(address_m.isin(address_s).sum()) == a["address_matches"] > a["key_matched_municipal"], "the address join must look better than the key"
    assert int(address_s.duplicated().sum()) == a["state_addresses_shared"] > 0, "and be worse, because addresses are shared"
    only_m, only_s = m[~mk.isin(sk)], s[~sk.isin(mk)]
    assert len(only_m) == a["unmatched_municipal"] and int(only_m["qualifier"].notna().sum()) == a["unmatched_municipal_with_qualifier"]
    assert len(only_s) == a["unmatched_state"] and int(only_s["NET_VALUE"].sum()) == a["unmatched_state_value_usd"]


def test_inspections_follow_ownership(municipal, p02):
    """The quarter of inlets without an inspection is a map of who owns the road, not of neglect."""
    inlets = municipal["stormwater_inlets"]
    never = inlets["inspected"].isna()
    r = p02["inspections"]
    assert int(never.sum()) == r["never_inspected"]
    assert inlets["Owner_Type"].value_counts().to_dict() == r["by_owner"]
    assert inlets.loc[never, "Owner_Type"].value_counts().to_dict() == r["never_inspected_by_owner"]
    share = never.groupby(inlets["Owner_Type"]).mean()
    assert share["Municipality"] < 0.05 and share["State"] > 0.95 and share["County"] > 0.95, "the ownership pattern 02_05 reads is gone"
    gap = ((inlets.geometry.x - inlets["Easting"]) ** 2 + (inlets.geometry.y - inlets["Northing"]) ** 2) ** 0.5
    assert inlets.loc[gap > 1, "Owner_Type"].value_counts().to_dict() == r["disagreements_by_owner"]


def test_the_same_points_under_five_zonings(municipal, grid, boundary, p02):
    """02_07's first table: the busiest zone's share of the food places moves with the lines."""
    from shapely.geometry import box

    food = gpd.read_file(P02_DATA / "princeton_food.geojson").to_crs(METRES)
    bnd = boundary.to_crs(METRES)

    def make_grid(step, offset=0.0):
        minx, miny, maxx, maxy = bnd.total_bounds
        cells = [box(x, y, x + step, y + step) for x in np.arange(minx + offset, maxx, step) for y in np.arange(miny + offset, maxy, step)]
        g = gpd.GeoDataFrame(geometry=cells, crs=bnd.crs)
        return g[g.intersects(bnd.geometry.iloc[0])].reset_index(drop=True)

    zonings = {"grid_250m": grid, "grid_500m": make_grid(500), "grid_500m_offset_250m": make_grid(500, 250),
               "zoning_districts": municipal["zoning"].to_crs(METRES).reset_index(drop=True),
               "tax_blocks": municipal["tax_blocks"].to_crs(METRES).reset_index(drop=True)}
    if "food_by_zoning" not in p02:
        pytest.skip("held back until 02_07 ships with the your-turn answers")
    for name, zones in zonings.items():
        n = gpd.sjoin(food[["geometry"]], zones[["geometry"]], predicate="within").groupby("index_right").size().reindex(zones.index).fillna(0)
        r = p02["food_by_zoning"][name]
        assert (len(zones), int(n.sum()), int(n.max()), int((n > 0).sum())) == (r["zones"], r["placed"], r["max_in_one_zone"], r["zones_with_any"])
        assert round(float(n.max() / n.sum()), 2) == r["top_share"]
    shares = [r["top_share"] for r in p02["food_by_zoning"].values()]
    assert max(shares) > 2 * min(shares), "the zonings agree with each other; the MAUP lesson is gone"


def test_the_deer_lie_in_the_street(municipal, njgin_parcels, p02):
    """A third of the reports fall in no zoning district, and nearly all of those sit on a road."""
    if "deer" not in p02:
        pytest.skip("held back until 02_07 ships with the your-turn answers")
    deer, zoning = municipal["deer_reports"].to_crs(METRES), municipal["zoning"].to_crs(METRES)
    roads, blocks = municipal["roads"].to_crs(METRES), municipal["tax_blocks"].to_crs(METRES)
    in_zone = gpd.sjoin(deer, zoning[["ZONING_CAT", "geometry"]], predicate="within")
    out = deer[~deer.index.isin(in_zone.index)]
    d = p02["deer"]
    assert len(out) == d["unassigned_to_zoning"] > len(deer) / 4
    union = roads.union_all()
    assert int((out.geometry.distance(union) < 15).sum()) == d["unassigned_within_15m_of_road"] > 0.9 * len(out)
    assert int((deer.geometry.distance(union) < 15).sum()) == d["within_15m_of_road"]
    in_block = gpd.sjoin(deer[["geometry"]], blocks[["BLOCK_NO", "geometry"]], predicate="within")
    assert (len(in_block), in_block["BLOCK_NO"].nunique()) == (d["inside_a_block"], d["blocks_with_a_report"])
    homes = njgin_parcels.to_crs(METRES)
    homes = homes[homes["PROP_CLASS"].isin(["2", "4C"])].copy()
    homes["geometry"] = homes.geometry.centroid
    per_category = gpd.sjoin(homes[["geometry"]], zoning[["ZONING_CAT", "geometry"]], predicate="within").groupby("ZONING_CAT").size()
    assert int(per_category.min()) == d["smallest_home_denominator"] < 50, "the small-denominator lesson needs a small denominator"


def test_nothing_that_identifies_a_person_ships():
    for gpkg in sorted(P02_DATA.glob("*.gpkg")):
        for name in gpd.list_layers(gpkg)["name"]:
            columns = list(gpd.read_file(gpkg, layer=name, rows=1).columns)
            bad = [c for c in columns if c in FORBIDDEN or PII.search(c)]
            assert not bad, f"{gpkg.name}:{name} carries {bad}"


def test_every_layer_carries_its_provenance():
    for gpkg in sorted(P02_DATA.glob("*.gpkg")):
        for name in gpd.list_layers(gpkg)["name"]:
            meta = pyogrio.read_info(gpkg, layer=name).get("layer_metadata") or {}
            assert {"source", "licence", "retrieved"} <= set(meta), f"{gpkg.name}:{name} lacks provenance: {meta}"


def test_the_grid_credits_openstreetmap():
    meta = pyogrio.read_info(P02_DATA / "princeton_grid_250m.gpkg", layer="grid")["layer_metadata"]
    assert "OpenStreetMap" in meta.get("attribution", ""), "the grid is derived from OSM buildings and must say so"


def test_published_p02_ground_truth_carries_no_answers():
    import json

    text = (P02_DATA / "ground_truth.json").read_text()
    published = json.loads(text)
    leaked = [k for k in ("p02_answers", "duel", "stretch", "mystery") if k in published]
    assert not leaked, f"published P02 ground truth leaks {leaked}"
    # The deer and food-by-zoning blocks are what 02_07 prints, and 02_07 answers the 02_02 your-turn
    # question. They may be published only once that notebook is in the folder; before that, no number
    # of the deer answer may appear anywhere in the file.
    revealed = (P02_DATA.parent / "02_07_points_into_zones.ipynb").exists()
    held = [k for k in ("deer", "food_by_zoning") if k in published.get("p02", {})]
    if not revealed:
        assert not held, f"published P02 ground truth carries {held} before 02_07 has shipped"
        assert "unassigned_to_zoning" not in text and "food_by_zoning" not in text


def test_inlet_coordinates_agree_with_their_geometry(municipal, p02):
    """The typical inlet's stored Easting and Northing match its geometry to a hundredth of
    a foot, which proves the CRS. The rows that do not are counted, not hidden."""
    inlets = municipal["stormwater_inlets"]
    gap = ((inlets.geometry.x - inlets["Easting"]) ** 2 + (inlets.geometry.y - inlets["Northing"]) ** 2) ** 0.5
    assert gap.median() < 0.01
    assert int((gap > 1).sum()) == p02["inlets"]["coordinate_disagreements_over_1ft"]


def test_every_layer_lies_in_or_near_princeton(municipal, njgin_parcels, census, boundary):
    town = boundary.to_crs(METRES).geometry.iloc[0].buffer(1000)
    for name, layer in list(municipal.items()) + [("njgin", njgin_parcels)] + list(census.items()):
        outside = ~layer.to_crs(METRES).representative_point().within(town)
        assert outside.mean() < 0.02, f"{name}: {int(outside.sum())} features lie more than a kilometre outside Princeton"


@pytest.mark.parametrize("name", ["princeton_boundary.geojson", "princeton_buildings.geojson", "princeton_food.geojson", "landmarks.csv"])
def test_the_p01_files_travel_along(name):
    """A notebook may not read outside its folder, so P02 carries its own copies."""
    assert (P02_DATA / name).exists()
