"""How far is lunch, really? Walking distance from the Engineering Quadrangle.

WHAT THIS IS
============
In the precept we answered "how many places to eat are within a 400 m walk of the
E-Quad" by drawing a circle of radius 400 m and counting what fell inside. A circle is
a fast, defensible first answer, and it is also a lie in a specific and measurable way,
because nobody walks through buildings. This script answers the same question properly,
by routing along the real network of footways, paths, steps and streets that
OpenStreetMap records around the engineering school, and then reports how far wrong the
circle was.

It was written by an AI coding agent during the preparation of this course, from a
one-sentence request, and then checked line by line against its data. It is here as an
honest exhibit of what a capable agent produces when you ask it a real question, and it
is neither a toy nor magic. Every number it prints is one you could have computed with
what you already know, plus one idea, the shortest path.

Read it as a specimen. You are not expected to write this in week one. You are expected,
by the end of the semester, to be able to read it, doubt it, and check it.

SPOILER NOTICE
==============
This script answers the Duel's question, so run it after your own attempt rather than
before. The answer whose value lies in trying it first is not one to read.

It is worth being precise about how close the two really are, because noticing this sort
of gap is the entire skill the Duel is teaching. They are not the same calculation. The
Duel starts at a single point standing for E225 and counts every food place in the file.
This script starts at the whole eight-wing footprint and counts only the four categories
it calls a place to eat, leaving the pub and the bar in a separate line. So the two
numbers should NOT agree, and if you compare them straight you are comparing three
choices at once. When an answer differs from yours, the useful question is never just
whether it is bigger, it is which decision made it bigger.

WHAT IT DOES
============
  origin    the E-Quad complex, rebuilt as the union of its eight mapped wings, so the
            walk starts at the building's edge, which is where its doors are
  distance  shortest path along the OSM walkable network, in metres. A straight-line
            circle is computed alongside it, only so the two can be compared
  CRS       every measurement happens in EPSG:26918, i.e. NAD83 UTM zone 18N, whose
            unit is the metre. Nothing is ever buffered or measured in degrees

Everything it reads is pinned in data/ and it never touches the network, so it produces
the same answer in the classroom as it did the day it was written.
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

# --------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------
# Paths are relative to this file, not to wherever you happen to have started Python,
# so the script runs the same from the terminal, from VS Code, or from a notebook.
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "showcase_output"

WGS84 = "EPSG:4326"     # longitude and latitude in degrees, how the files are stored
METRES = "EPSG:26918"    # NAD83 UTM 18N, the projected CRS we measure in
WALK_M = 400             # the walking budget, in metres
NEARBY_M = 600           # a second, looser band, for the places that only just miss

# The eight buildings that make up the E-Quad. Note that searching for names CONTAINING
# "wing", case-insensitively, would also pull in "Triumph Brewing Company", a bar on
# Nassau Street, because the letters of "wing" sit inside "Brewing". Substring matching
# on names is a reliable way to be quietly wrong, and the reason this is a fixed list.
EQUAD_WINGS = ["A Wing", "B Wing", "C Wing", "D Wing", "E Wing", "F Wing", "G Wing", "J Wing"]

# Which amenities count as "a place to eat". This is a judgement, not a fact, so it is
# written down here where it can be argued with rather than buried further down.
CORE = ["restaurant", "cafe", "fast_food", "ice_cream"]
BORDERLINE = ["pub", "bar"]

# A junction this close to the building is treated as one of its doors.
DOOR_M = 25

# How far a place may sit from the nearest path and still be considered reachable from
# it. This is the length of the final step off the network, across a forecourt or a car
# park, so it has to stay short. Anything generous here quietly lets routes cut through
# buildings.
ACCESS_M = 60


def panel_name(name, width=25):
    """A version of a name that matplotlib's default font can actually draw.

    OSM records names as their owners write them, so this data contains characters
    outside the Latin alphabet. The terminal prints them correctly. The figure cannot,
    because the default font has no glyph for them, and a missing glyph is drawn as an
    empty box that also throws the monospace column out of line. Dropping what cannot be
    drawn is the honest compromise here, and the full name stays in the printed table
    and in the exported GeoJSON.
    """
    drawable = "".join(c for c in name if ord(c) < 0x250)
    drawable = " ".join(drawable.split()) or "(name not in Latin script)"
    return drawable if len(drawable) <= width else drawable[: width - 1] + "."


def heading(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# --------------------------------------------------------------------------------
# Step 1. The origin, or where does the walk start
# --------------------------------------------------------------------------------
# The E-Quad is not a point. It is a complex most of two hundred metres across, mapped as eight
# separate wings, and where you decide the walk begins changes the answer. We build the
# whole footprint by merging the wings, and we will measure from its edge, because that
# is where you actually step outside.
heading("Step 1. The origin")

buildings = gpd.read_file(DATA / "princeton_buildings.geojson")
wings = buildings[buildings["name"].isin(EQUAD_WINGS)]
assert len(wings) == len(EQUAD_WINGS), f"expected {len(EQUAD_WINGS)} wings, found {len(wings)}"

equad = gpd.GeoDataFrame(
    {"name": ["Engineering Quadrangle"]},
    geometry=[unary_union(wings.geometry.values)],
    crs=WGS84,
).to_crs(METRES)

footprint = equad.geometry.iloc[0]
centre = footprint.centroid
minx, miny, maxx, maxy = footprint.bounds
print(f"the E-Quad is {len(wings)} wings merged into one footprint")
print(f"  extent        {maxx - minx:.0f} m by {maxy - miny:.0f} m")
print(f"  area          {footprint.area:,.0f} square metres")
print(f"  centroid      {centre.x:.1f} E, {centre.y:.1f} N in {METRES}")

# --------------------------------------------------------------------------------
# Step 2. The destinations
# --------------------------------------------------------------------------------
heading("Step 2. The places to eat")

food = gpd.read_file(DATA / "princeton_food.geojson").to_crs(METRES)
print(f"food and drink places in the Princeton snapshot: {len(food)}")
print(food["amenity"].value_counts().to_string())

# Straight-line distances, for comparison later. Two of them, because "distance from the
# E-Quad" is ambiguous until you say distance from what, the middle of the complex or
# the nearest point on its wall. The second is the fair comparison to a walk.
food["crow_from_centre"] = food.geometry.distance(centre)
food["crow_from_edge"] = food.geometry.distance(footprint)

# --------------------------------------------------------------------------------
# Step 3. The walkable network
# --------------------------------------------------------------------------------
# A network is a set of points (nodes, i.e. junctions and path ends) joined by lines
# (edges, i.e. the stretches of path between them). Each edge carries a length in metres
# measured along its real curve, so a path that bends around a courtyard is charged for
# the bend. That single attribute is what separates this from a circle.
heading("Step 3. The walkable network")

nodes = gpd.read_file(DATA / "walk_network_nodes.geojson").to_crs(METRES)
edges = gpd.read_file(DATA / "walk_network_edges.geojson").to_crs(METRES)

# Build an undirected graph. Walking is symmetric, and where OSM records two parallel
# ways between the same pair of junctions we keep the shorter, which is what a person
# would take.
G = nx.Graph()
for osmid, geom in zip(nodes["osmid"], nodes.geometry):
    G.add_node(osmid, x=geom.x, y=geom.y)
for u, v, length in zip(edges["u"], edges["v"], edges["length"]):
    if u == v:
        continue  # a loop back to the same junction cannot shorten any route
    if G.has_edge(u, v) and G[u][v]["length"] <= length:
        continue
    G.add_edge(u, v, length=length)

print(f"nodes (junctions)      {G.number_of_nodes():,}")
print(f"edges (path segments)  {G.number_of_edges():,}")
print(f"total path length      {sum(d['length'] for _, _, d in G.edges(data=True)) / 1000:.1f} km")

# A spatial index, so that "which junctions are near this restaurant" is a fast lookup
# rather than a distance calculation against every junction in the network.
node_ids = list(G.nodes)
node_points = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids]
index = STRtree(node_points)


def access_points(geometry, radius):
    """Every junction within radius of a geometry, with its straight-line gap."""
    candidates = [node_ids[i] for i in index.query(geometry.buffer(radius))]
    gaps = {n: Point(G.nodes[n]["x"], G.nodes[n]["y"]).distance(geometry) for n in candidates}
    return {n: g for n, g in gaps.items() if g <= radius}


# A second index, over the path segments themselves rather than their end points.
edge_lines = list(edges.geometry)
edge_ends = list(zip(edges["u"], edges["v"]))
edge_index = STRtree(edge_lines)


def walk_to(geometry, distances, radius=ACCESS_M):
    """Shortest walking distance from wherever `distances` was measured, to a place.

    A place is not on the network, it is beside it, and the naive way to handle that is
    to snap it to the nearest junction. That is wrong often enough to matter, because
    junctions are sparse. A restaurant in the middle of a long block has its nearest
    junction at one end of that block, and snapping charges the route for walking to the
    corner even though the door is halfway along.

    So we do not snap to a junction. We find every path segment passing within radius,
    and for each one work out where along it the place actually sits. Walking to that
    point costs the distance to one end of the segment plus the run back along it, and
    the last few metres off the path are added on top. The best of those is the answer.
    """
    best = np.inf
    for i in edge_index.query(geometry.buffer(radius)):
        line = edge_lines[i]
        offset = line.distance(geometry)
        if offset > radius:
            continue
        along = line.project(geometry)
        u, v = edge_ends[i]
        for node, run in ((u, along), (v, line.length - along)):
            reached = distances.get(node)
            if reached is not None:
                best = min(best, reached + run + offset)
    return best


def reach_from(access, cutoff=2500):
    """Shortest walking distance to every junction, starting from a set of access points.

    The starting points are joined to one artificial source node, each by an edge as long
    as the gap it spans, and the search runs from there. That measures from the thing
    itself rather than from whichever junction happens to be nearby, and it lets the route
    leave by whichever access point turns out to be best.

    Note that the artificial node goes into a COPY of the network, never into the network
    itself. A node joined to twenty-one doors is a tunnel through the building for any
    later search that is allowed to pass through it, and a search that starts there is
    the only one safe from it. The first version of this script grafted the node onto the
    shared graph and a second measurement quietly routed through the tunnel.
    """
    H = G.copy()
    H.add_node(SOURCE)
    for node, gap in access.items():
        H.add_edge(SOURCE, node, length=gap)
    distances = nx.single_source_dijkstra_path_length(H, SOURCE, cutoff=cutoff, weight="length")
    distances.pop(SOURCE, None)
    return distances


SOURCE = "artificial:source"

# The doors. Rather than guess which wing has the real entrance, the walk may start from
# any junction that touches the building, and the shortest path picks the best one.
doors = access_points(footprint, DOOR_M)
assert doors, f"no network junction within {DOOR_M} m of the E-Quad footprint"
print(f"junctions within {DOOR_M} m of the building, used as doors: {len(doors)}")

# --------------------------------------------------------------------------------
# Step 4. Routing
# --------------------------------------------------------------------------------
# Dijkstra's algorithm spreads outwards from the starting junctions, always expanding the
# nearest unvisited one, and so finds the true shortest distance to every junction it can
# reach. Given many starting points it treats them as one, which is exactly the "leave by
# whichever door is best" model we want.
heading("Step 4. Shortest paths")

from_doors = reach_from(doors)
print(f"junctions reachable on foot from the building: {len(from_doors):,}")

# Each place is now measured against the path segments beside it rather than against the
# junctions near it, which is what walk_to does.
food["walk_from_edge"] = [walk_to(g, from_doors) for g in food.geometry]
food["off_network_m"] = [
    min((edge_lines[i].distance(g) for i in edge_index.query(g.buffer(ACCESS_M))), default=np.nan)
    for g in food.geometry
]
print(f"median distance from a place to the nearest path: {np.nanmedian(food['off_network_m']):.0f} m")

# A built-in audit. Walking cannot be shorter than flying, so if any row claims it is,
# the routing is wrong and every number below it is worthless. This is the check the
# course keeps asking for, i.e. state what must be true and then test it, rather than
# trusting the output because it looks tidy.
#
# The first draft of this script failed exactly here, and the failure was worth having.
# It measured the walk from a door junction that could sit 25 m clear of the wall, while
# measuring the straight line from the wall, so two places came out closer on foot than
# in a straight line. Measuring from the building itself is what fixed it.
#
# Worth knowing what this check did NOT catch. A later review found that the nearest
# junction is often not the best way in, which had inflated several distances by up to
# 200 m, and that a second measurement was tunnelling through the building. Both were
# consistent with walking being longer than flying, so the assertion sat there passing.
# An audit tells you about the errors it was written to find, and stays silent about the
# rest, which is why it is a floor and not a guarantee.
# Places with no path within reach are left as infinity rather than given a plausible
# looking number. They are the ones outside the shipped network, which covers 900 m
# around the building, comfortably more than the furthest distance reported below.
reachable = food[np.isfinite(food["walk_from_edge"])]
unreachable = food[~np.isfinite(food["walk_from_edge"])]
if len(unreachable):
    print(f"{len(unreachable)} places lie beyond the shipped network and are left unmeasured,")
    print(f"the nearest of them {unreachable['crow_from_edge'].min():.0f} m away in a straight line,")
    print(f"which is well outside the {NEARBY_M} m this script reports on.")
impossible = reachable[reachable["walk_from_edge"] < reachable["crow_from_edge"] - 0.5]
assert impossible.empty, f"{len(impossible)} places route shorter than a straight line, routing is broken"
print(f"audit passed, no route beats a straight line, checked on {len(reachable)} reachable places")

# --------------------------------------------------------------------------------
# Step 5. The answer, four ways
# --------------------------------------------------------------------------------
heading(f"Step 5. Places to eat within {WALK_M} m, measured three ways")


def count(column, categories):
    return int(((food[column] <= WALK_M) & food["amenity"].isin(categories)).sum())


rows = [
    ("network walk, from the building edge  [the honest answer]", "walk_from_edge"),
    ("straight line, from the building edge  [a circle, same origin]", "crow_from_edge"),
    ("straight line, from the centroid  [a circle round the middle]", "crow_from_centre"),
]
for label, column in rows:
    print(f"{label:63s} {count(column, CORE):3d} core   {count(column, CORE + BORDERLINE):3d} incl. bars")

honest = count("walk_from_edge", CORE)
circle = count("crow_from_edge", CORE)
centre_circle = count("crow_from_centre", CORE)
gap = circle - honest

# Compare the first two rows, not the first and the last. They share a starting point and
# differ only in how distance is measured, so whatever separates them is the cost of the
# straight line and nothing else. Changing the origin and the metric at the same time
# would leave you unable to say which one moved the answer.
print()
print("There is deliberately no fourth row for a walk measured from the centroid. The")
print("centroid falls inside a courtyard that OpenStreetMap maps no path through, so the")
print("nearest junction to it is the end of a stub, and any walking distance from there")
print("would be measuring how completely this campus has been mapped rather than how far")
print("anything is. A number you cannot defend is worse than a number you did not print.")

print()
print("Hold the starting point fixed at the building edge and change only how distance is")
print(f"measured. The straight line finds {circle} places. The walk finds {honest}.")
if gap > 0:
    print(f"So {gap} of those {circle} are inside the radius and outside the walk. A straight line")
    print(f"reaches them and a pedestrian does not, i.e. {gap / circle:.0%} of what the circle counts")
    print(f"is not actually walkable, and it offers you {circle / honest - 1:.0%} more lunch than exists.")
elif gap < 0:
    print(f"The walk reaches {-gap} more than the straight line, which should be impossible and")
    print("means something above is wrong.")
else:
    print("The two agree, which happens and is not a general result.")

print()
print(f"Now change only the starting point. From the centroid the straight line finds")
print(f"{centre_circle} rather than {circle}, because the middle of a {maxx - minx:.0f} m building is already")
print("most of a minute from its own front door.")
origin_effect, metric_effect = abs(circle - centre_circle), abs(circle - honest)
if origin_effect >= metric_effect:
    print(f"Deciding where a place begins moved the answer by {origin_effect}, and deciding how to")
    print(f"measure distance moved it by {metric_effect}. The choice nobody writes down mattered at")
    print("least as much as the one everybody argues about, and neither announces itself.")
else:
    print(f"Deciding where a place begins moved the answer by {origin_effect}, against {metric_effect} for")
    print("the choice of metric. The smaller effect is still nobody's stated assumption.")

# --------------------------------------------------------------------------------
# Step 6. The list
# --------------------------------------------------------------------------------
heading(f"The {honest} places within a {WALK_M} m walk of the E-Quad")

inside = food[(food["walk_from_edge"] <= WALK_M) & food["amenity"].isin(CORE)].copy()
inside = inside.sort_values("walk_from_edge")
# The detour ratio, how much further you walk than you would fly. 1.0 would be a straight
# clear path, and anything above about 1.4 means something is in the way.
inside["detour"] = inside["walk_from_edge"] / inside["crow_from_edge"].clip(lower=1.0)
print(
    inside[["name", "amenity", "walk_from_edge", "crow_from_edge", "detour"]]
    .rename(columns={"walk_from_edge": "walk_m", "crow_from_edge": "crow_m"})
    .round({"walk_m": 0, "crow_m": 0, "detour": 2})
    .to_string(index=False)
)
print("\nby category:")
print(inside["amenity"].value_counts().to_string())

borderline = food[(food["walk_from_edge"] <= WALK_M) & food["amenity"].isin(BORDERLINE)]
print(f"\nalso within {WALK_M} m, if a pub counts as lunch ({len(borderline)}):")
print(
    borderline[["name", "amenity", "walk_from_edge"]].rename(columns={"walk_from_edge": "walk_m"})
    .round(0).to_string(index=False)
    if len(borderline)
    else "  none"
)

just_outside = food[
    (food["walk_from_edge"] > WALK_M)
    & (food["walk_from_edge"] <= NEARBY_M)
    & food["amenity"].isin(CORE)
].sort_values("walk_from_edge")
print(f"\njust outside, between {WALK_M} m and {NEARBY_M} m ({len(just_outside)}):")
print(
    just_outside[["name", "amenity", "walk_from_edge"]].rename(columns={"walk_from_edge": "walk_m"})
    .round(0).to_string(index=False)
)
print("\nNote how close some of these are to the cut. A threshold is a decision, not a")
print("measurement, and a place at 405 m is not meaningfully further away than one at 395.")

# --------------------------------------------------------------------------------
# Step 7. The picture
# --------------------------------------------------------------------------------
heading("Step 7. Drawing it")

# How close together are these places really? The answer decides how the map has to be
# drawn, and it is measured here rather than remembered, so that a later data refresh
# cannot leave the caption asserting something that stopped being true.
coords = np.array([[g.x, g.y] for g in inside.geometry])
spread = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
np.fill_diagonal(spread, np.inf)
closest_pair_m = spread.min()
print(f"closest two places to each other: {closest_pair_m:.1f} m apart")

# The walkshed, every stretch of path you can reach within the budget. Buffering the
# reachable segments and then shrinking the result slightly turns a bundle of lines into
# one readable blob without inflating how far it appears to reach.
# The two artificial origin nodes have no position on the ground, so they are dropped
# here rather than drawn as a line from nowhere.
real = set(node_ids)
reached = {n for n, d in from_doors.items() if d <= WALK_M and n in real}
segments = [
    LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
    for u, v in G.edges()
    if u in reached and v in reached
]
walkshed = MultiLineString(segments).buffer(35).buffer(-12)

# Two panels, the map and a ranked list beside it, so the picture can be read on its
# own without the terminal output next to it.
fig = plt.figure(figsize=(15, 10))
grid = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.02)
ax = fig.add_subplot(grid[0, 0])
side = fig.add_subplot(grid[0, 1])
side.axis("off")

edges.plot(ax=ax, color="0.86", linewidth=0.6, zorder=1)

# The circle to draw is the one the comparison actually used, i.e. 400 m from the same
# building edge the walk starts at. A circle round the centroid would be a different
# claim and would not line up with the count in the title.
ring = gpd.GeoSeries([footprint.buffer(WALK_M)], crs=METRES)
ring.plot(ax=ax, facecolor="none", edgecolor="crimson", linewidth=2.0, linestyle="--", zorder=6)
gpd.GeoSeries([walkshed], crs=METRES).plot(ax=ax, color="tab:blue", alpha=0.20, zorder=2)
equad.plot(ax=ax, color="0.30", zorder=5)

beyond = food[food["amenity"].isin(CORE + BORDERLINE) & ~food.index.isin(inside.index)]
beyond.plot(ax=ax, color="0.62", markersize=22, zorder=3)
just_outside.plot(ax=ax, color="none", edgecolor="crimson", markersize=60, linewidth=1.2, zorder=4)
inside.plot(ax=ax, color="tab:orange", markersize=150, edgecolor="black", linewidth=0.6, zorder=7)

# Numbers rather than names. Fifteen restaurants on one street cannot carry fifteen text
# labels without overlapping into an unreadable smear, so the map carries the rank and
# the panel carries the name.
for rank, (_, r) in enumerate(inside.iterrows(), start=1):
    ax.annotate(str(rank), (r.geometry.x, r.geometry.y), ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white", zorder=8)

pad = WALK_M + 220
ax.set_xlim(centre.x - pad, centre.x + pad)
ax.set_ylim(centre.y - pad, centre.y + pad)
ax.set_title(
    f"Within a {WALK_M} m walk of the E-Quad: {honest} places to eat, not the {circle} a circle promises",
    fontsize=13, pad=12,
)
ax.set_xlabel(f"easting in metres ({METRES})")
ax.set_ylabel("northing in metres")
ax.set_aspect("equal")

# Almost every one of these places is on the same two blocks of Nassau Street, close
# enough that at the scale of the whole walkshed their markers sit on top of each other.
# An inset redraws that corner large enough to separate them. The crowding is the point:
# the answer to "where is lunch" is not spread evenly around the building at all.
cluster = inside[inside["walk_from_edge"] <= 250]
if len(cluster) >= 3:
    cx0, cy0, cx1, cy1 = cluster.total_bounds
    margin = 42
    zoom = ax.inset_axes([0.015, 0.66, 0.52, 0.32], facecolor="white")
    # An inset is a child of the parent axes and takes part in its z-ordering, so the
    # 400 m ring and the markers, drawn later at higher z, would otherwise be painted
    # straight across the panel. A crimson arc through the enlargement reads as a
    # boundary running down Nassau Street, which is exactly what the panel disproves.
    zoom.set_zorder(9)
    edges.plot(ax=zoom, color="0.86", linewidth=0.7, zorder=1)
    gpd.GeoSeries([walkshed], crs=METRES).plot(ax=zoom, color="tab:blue", alpha=0.20, zorder=2)
    inside.plot(ax=zoom, color="tab:orange", markersize=70, edgecolor="black", linewidth=0.5, zorder=4)

    # The closest two of these are neighbouring storefronts, metres apart, so no amount
    # of zooming separates their markers. The numbers are therefore pushed out on short
    # leader lines instead, spread around the compass by rank so that adjacent ranks
    # never land in the same direction.
    for rank, (_, r) in enumerate(inside.iterrows(), start=1):
        angle = np.deg2rad(90 + rank * 137.5)   # the golden angle, which spreads evenly
        zoom.annotate(
            str(rank), (r.geometry.x, r.geometry.y),
            xytext=(20 * np.cos(angle), 20 * np.sin(angle)), textcoords="offset points",
            ha="center", va="center", fontsize=8, fontweight="bold", color="black",
            arrowprops=dict(arrowstyle="-", linewidth=0.5, color="0.45", shrinkA=0, shrinkB=2),
            zorder=5,
        )
    zoom.set_xlim(cx0 - margin, cx1 + margin)
    zoom.set_ylim(cy0 - margin, cy1 + margin)
    zoom.set_xticks([]); zoom.set_yticks([])
    zoom.set_aspect("equal")
    zoom.set_title("Nassau Street, enlarged", fontsize=9, pad=3)
    for spine in zoom.spines.values():
        spine.set_edgecolor("0.35")
    ax.indicate_inset_zoom(zoom, edgecolor="0.35", linewidth=1.0, alpha=0.9)

handles = [
    plt.Line2D([], [], marker="o", color="none", markerfacecolor="tab:orange",
               markeredgecolor="black", markersize=11, label=f"within a {WALK_M} m walk"),
    plt.Line2D([], [], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="crimson", markersize=9, label=f"{WALK_M} to {NEARBY_M} m walk"),
    plt.Line2D([], [], marker="o", color="none", markerfacecolor="0.62",
               markeredgecolor="none", markersize=7, label="further, or not a walk"),
    plt.Line2D([], [], color="crimson", linestyle="--", linewidth=2,
               label=f"{WALK_M} m from the building, straight line"),
    plt.Line2D([], [], color="tab:blue", alpha=0.5, linewidth=9,
               label=f"where {WALK_M} m of walking actually reaches"),
]
ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.92)

# The ranked list. Distances are rounded to 5 m, because the underlying paths are traced
# by volunteers and a metre of precision here would be invented.
lines = [f"{'#':>2}  {'place':<26}{'walk':>7}{'line':>7}"]
lines.append("-" * 44)
for rank, (_, r) in enumerate(inside.iterrows(), start=1):
    lines.append(f"{rank:>2}  {panel_name(r['name']):<26}{5 * round(r['walk_from_edge'] / 5):>5.0f} m{5 * round(r['crow_from_edge'] / 5):>5.0f} m")
lines.append("")
lines.append(f"{circle - honest} more fall inside the circle")
lines.append("but not inside the walk.")
lines.append("")
lines.append("markers touch where storefronts do:")
lines.append(f"the closest two are {closest_pair_m:.0f} m apart.")
lines.append("")
lines.append(f"just outside, {WALK_M} to {NEARBY_M} m:")
for _, r in just_outside.iterrows():
    lines.append(f"    {panel_name(r['name']):<26}{5 * round(r['walk_from_edge'] / 5):>5.0f} m")
side.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=8.5,
          va="top", ha="left", transform=side.transAxes)
side.text(0.0, 0.0, "OpenStreetMap contributors, ODbL 1.0\ndistances along the walkable network",
          fontsize=7.5, color="0.4", va="bottom", ha="left", transform=side.transAxes)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "walkshed.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"map written to {OUT / 'walkshed.png'}")

# The layers, so you can open them in QGIS next week and poke at them yourself.
gpd.GeoDataFrame(geometry=[walkshed], crs=METRES).to_crs(WGS84).to_file(
    OUT / "walkshed_400m.geojson", driver="GeoJSON"
)
# The same ring the map draws and the headline compares against, i.e. 400 m from the
# building. Exporting the centroid circle here instead would have shipped a layer that
# quietly disagrees with the picture beside it.
gpd.GeoDataFrame(geometry=[footprint.buffer(WALK_M)], crs=METRES).to_crs(WGS84).to_file(
    OUT / "circle_400m_from_building.geojson", driver="GeoJSON"
)
inside.drop(columns=["detour"]).to_crs(WGS84).to_file(OUT / "within_400m_walk.geojson", driver="GeoJSON")

summary = {
    "walk_radius_m": WALK_M,
    "counts": {
        "network_walk_from_edge": honest,
        "straight_line_from_edge": circle,
        "straight_line_from_centre": centre_circle,
    },
    "network": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
    "note": "OpenStreetMap contributors, ODbL 1.0. Distances in EPSG:26918.",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"layers and summary written to {OUT}")
print("\nOpenStreetMap contributors, ODbL 1.0.")
