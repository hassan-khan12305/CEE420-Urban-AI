"""The small toolbox of Precept 03b, structure from motion by hand.

Every piece of camera geometry and every algorithm the seven notebooks redo by hand lives here,
once, with a plain docstring. A notebook imports it as ``import sfm_tools as sfm``, opens a file
with one of the ``load_`` functions, calls one function per cell and prints the number beside its
reference. The course pipeline that computes the reference numbers imports this same file, so a
notebook and its reference numbers cannot drift apart.

Three frames appear throughout. The adjustment's own frame is east, north and up in metres about
the flight's reference point; ``block["rotation"]`` and ``block["points"]`` live in it. The map
frame is the products' coordinate system (``choices["crs"]``) minus ``choices["offset"]``, so that
the numbers stay small; ``block["rotation_map"]``, the depth map, the dense clouds, the meshes
and the elevation models live in it. Pixel coordinates count from the top left corner of a
photograph, u across and v down, in pixels of whichever image is passed in; the kit's photographs
are downsized copies of the frames the software worked on.

The software's rotations are axis angle vectors (the direction is the axis, the length the angle),
a point enters a camera as R X + t, and its camera is described in normalised units, fractions of
the longer side of the frame. The functions below take those conventions as they are.
"""

import json
import math
import time

import numpy as np
import rasterio
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image
from rasterio.features import geometry_mask
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import Delaunay
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

__all__ = [
    # loading
    "load_flight", "load_block", "load_tracks", "load_features", "load_matches", "load_depth",
    "load_cloud", "load_mesh", "read_raster", "read_image",
    # geometry
    "rodrigues", "camera_centre", "camera_axis", "project_brown", "feature_pixels", "pixel_rays",
    "grid_coordinates", "grid_bounds", "surface_points", "affine_parts", "footprints",
    "overlap_shares", "project_outline", "inside",
    # algorithms
    "harris", "dog_blobs", "ratio_matches", "two_nearest", "eight_point", "sampson_distance",
    "fundamental_ransac", "epipolar_lines", "union_find_tracks", "length_histogram", "mini_bundle",
    "ncc_depth", "unproject_depth", "grid_dsm", "fill_gaps", "ground_by_opening", "delaunay_25d",
    "orthorectify", "nearest_nadir_mosaic", "cells_of_class", "density", "roof_height",
    # rasters, clouds and the roof
    "window_bounds_local", "hillshade", "roof_polygon", "roof_mask", "roof_corner", "roof_point",
    # plotting
    "show_image", "show_grid", "show_cloud", "show_mesh", "crop_at", "descriptor_bars",
]

#: the pixel pitch of the aircraft's sensor in micrometres; the footprints are computed with it
PIXEL_UM = 2.4
#: the meaning of the class codes a classified point cloud carries (the ASPRS standard)
ASPRS_CLASSES = {0: "never classified", 1: "unclassified", 2: "ground", 3: "low vegetation",
                 4: "medium vegetation", 5: "high vegetation", 6: "building", 7: "noise",
                 9: "water"}
#: how far north leans into the vertical axis of an oblique view of a cloud
OBLIQUE_NORTH = 0.8


# --------------------------------------------------------------------------- #
# loading the kit's files
# --------------------------------------------------------------------------- #

def _plain(value):
    """A 0-d array as a plain Python value; any other array unchanged."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr


def _decode(npz, json_keys=(), list_keys=()):
    """A dict of every array of an npz file, 0-d arrays as plain values, the named keys decoded
    from json text or turned into lists of str."""
    out = {}
    for key in npz.files:
        value = _plain(npz[key])
        if key in json_keys:
            value = json.loads(str(value))
        elif key in list_keys:
            value = [str(v) for v in np.atleast_1d(value)]
        out[key] = value
    return out


def _camera(block, key="camera"):
    """The camera dict of a block, whether the block was loaded here or is the raw npz file."""
    cam = block[key]
    if isinstance(cam, dict):
        return dict(cam)
    return dict(json.loads(str(cam)))


def load_flight(path):
    """The flight's metadata as a dict, and the kit's choices (the window, the photographs a, b
    and c, the pair, the neighbourhood, the roof) as a second dict, from flight.json."""
    with open(path) as fh:
        flight = json.load(fh)
    return flight, flight["p03b"]


def load_block(path):
    """The software's reconstruction as a dict: names (a list of str), rotation, translation and
    gps per shot in the adjustment's frame, the same with ``_map`` in the map frame, the points in
    both frames (points, points_map) with their colors, the calibrated camera and camera_prior as
    dicts, offset, crs, reference_lla and the affine to_map_A, to_map_b between the frames."""
    json_keys = ("camera", "camera_prior", "reference_lla")
    return _decode(np.load(path), json_keys=json_keys, list_keys=("names",))


def load_tracks(path):
    """Every observation of a reconstructed point as a dict of arrays: shot (an index into the
    block's shots), point (an index into its points), feature (an index into that shot's feature
    list), x and y (normalised image coordinates), plus track_lengths for every track of the
    flight and the counts n_records and n_tracks."""
    return _decode(np.load(path))


def load_features(path):
    """The software's features as a dict: for the pair, points_a and points_b (x, y normalised,
    scale, angle per row), descriptors_a and descriptors_b (128 small integers per row), colors,
    name_a and name_b; for the neighbourhood, points_<name> per photograph and names (a list of
    str). Both carry the frame's width and height in pixels."""
    return _decode(np.load(path), list_keys=("names",))


def load_matches(path):
    """The robust matches of the neighbourhood as a dict keyed (earlier photograph, later
    photograph), each value an array of feature index pairs into the two photographs' features."""
    npz = np.load(path)
    matches = {}
    for key in npz.files:
        if "__" in key:
            first, second = key.split("__")
            matches[(first, second)] = npz[key].astype(np.int64)
    return matches


def load_depth(path):
    """The software's depth map of one photograph as a dict: depth in metres along the camera's
    axis (zero where the software found none), confidence, normals as unit vectors, the camera's
    K, R and C in the map frame, dmin, dmax, name, image_size, depth_size (width, height) and
    neighbours, the photographs it was matched against."""
    d = _decode(np.load(path), list_keys=("neighbours",))
    d["depth"] = np.asarray(d["depth"], dtype=np.float32)
    d["confidence"] = np.asarray(d["confidence"], dtype=np.float32)
    d["normals"] = np.asarray(d["normals"], dtype=np.float32) / 127.0
    d["image_size"] = tuple(int(v) for v in d["image_size"])
    d["depth_size"] = tuple(int(v) for v in d["depth_size"])
    return d


def load_cloud(path):
    """A dense cloud as a dict: xyz in the map frame (metres east and north of the offset, and
    the height), rgb, classification (an integer code per point, zero when unknown), classes
    (what the codes mean), offset, crs and n_total; the window's file also carries views (how
    many depth maps saw each point), centre_local and half_m."""
    cloud = _decode(np.load(path), json_keys=("classes",))
    xyz = np.asarray(cloud["xyz"], dtype=float)
    if "centre_local" in cloud:
        centre = np.asarray(cloud["centre_local"], dtype=float)
        xyz = xyz + np.array([centre[0], centre[1], 0.0])
    cloud["xyz"] = xyz
    if "classification" not in cloud:
        cloud["classification"] = np.zeros(len(xyz), dtype=np.int64)
    cloud["classification"] = np.asarray(cloud["classification"]).astype(np.int64)
    classes = dict(ASPRS_CLASSES)
    classes.update({int(k): v for k, v in cloud.get("classes", {}).items()})
    cloud["classes"] = classes
    cloud["frame"] = "the map frame, the products' coordinates minus offset"
    return cloud


def load_mesh(path):
    """A mesh as (vertices, faces, meta): vertices as x, y, z in the map frame, faces as three
    vertex indices per triangle, and meta with the counts of the whole mesh (n_vertices_total and
    n_faces_total) and, for the whole flight's file, the thinning and max_edge_m used."""
    mesh = _decode(np.load(path))
    vertices = np.asarray(mesh.pop("vertices"), dtype=float)
    faces = np.asarray(mesh.pop("faces"), dtype=np.int64)
    if "centre_local" in mesh:
        centre = np.asarray(mesh["centre_local"], dtype=float)
        vertices = vertices + np.array([centre[0], centre[1], 0.0])
    mesh["frame"] = "the map frame, the products' coordinates minus offset"
    return vertices, faces, mesh


def read_raster(path):
    """A GeoTIFF as (grid, transform, crs, cell_m): one band becomes a float grid with NaN where
    the file has no data, several bands an uint8 image of shape (rows, columns, bands) with zeros
    where the file is masked. The transform maps column and row to the crs, cell_m is one cell."""
    with rasterio.open(path) as src:
        transform = src.transform
        crs = src.crs.to_string()
        cell_m = float(src.res[0])
        if src.count == 1:
            grid = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        else:
            grid = np.moveaxis(src.read(), 0, -1)
            grid[src.dataset_mask() == 0] = 0
    return grid, transform, crs, cell_m


def read_image(path, gray=False):
    """A photograph as an array of uint8 with shape (rows, columns, 3), or as float32 grey values
    of shape (rows, columns) when gray is True."""
    image = Image.open(path)
    if gray:
        return np.asarray(image.convert("L"), dtype=np.float32)
    return np.asarray(image.convert("RGB"))


# --------------------------------------------------------------------------- #
# the camera and the ground
# --------------------------------------------------------------------------- #

def rodrigues(rv):
    """The 3 by 3 rotation matrix of an axis angle vector: a turn by the vector's length about
    its direction."""
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * K @ K


def camera_centre(rotation, translation):
    """Where the lens was, as x, y, z in the frame of the pose: minus R transposed times t."""
    return -rodrigues(rotation).T @ np.asarray(translation, dtype=float)


def camera_axis(rotation):
    """The direction the camera looked, a unit vector in the frame of the pose."""
    return rodrigues(rotation).T @ np.array([0.0, 0.0, 1.0])


def project_brown(X, rotation, translation, camera, width, height):
    """The collinearity equations: world points into (u, v, depth) of an image of the given size.
    Each point goes into the camera's frame, through the perspective division, through the Brown
    distortion (k1, k2, k3 radial, p1, p2 decentring) and the focal length and principal point
    in normalised units, then into pixels; depth is the distance along the camera's axis."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    p = X @ rodrigues(rotation).T + np.asarray(translation, dtype=float)
    depth = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = p[:, 0] / depth
        y = p[:, 1] / depth
    k1 = camera.get("k1", 0.0)
    k2 = camera.get("k2", 0.0)
    k3 = camera.get("k3", 0.0)
    p1 = camera.get("p1", 0.0)
    p2 = camera.get("p2", 0.0)
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
    fx = camera.get("focal_x", camera.get("focal"))
    fy = camera.get("focal_y", fx)
    cx = camera.get("c_x", 0.0)
    cy = camera.get("c_y", 0.0)
    size = max(width, height)
    u = (fx * xd + cx) * size + width / 2.0 - 0.5
    v = (fy * yd + cy) * size + height / 2.0 - 0.5
    return u, v, depth


def feature_pixels(points, width, height):
    """The feature file's normalised x and y as (u, v) in pixels of an image of the given size."""
    size = max(width, height)
    u = points[:, 0] * size + width / 2.0 - 0.5
    v = points[:, 1] * size + height / 2.0 - 0.5
    return u, v


def pixel_rays(u, v, camera, width, height):
    """The direction, in the camera's frame, of each pixel of an undistorted image: an array of
    (x, y, 1) rows, the pinhole run backwards."""
    size = max(width, height)
    fx = camera.get("focal_x", camera.get("focal"))
    fy = camera.get("focal_y", fx)
    x = ((np.asarray(u, dtype=float) - width / 2.0 + 0.5) / size - camera.get("c_x", 0.0)) / fx
    y = ((np.asarray(v, dtype=float) - height / 2.0 + 0.5) / size - camera.get("c_y", 0.0)) / fy
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def grid_coordinates(transform, shape, offset=(0.0, 0.0)):
    """The X and Y of every cell centre of a grid, two arrays of the grid's shape, in the
    transform's coordinates minus offset."""
    rows, cols = shape[0], shape[1]
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    X = transform.c + (c + 0.5) * transform.a - offset[0]
    Y = transform.f + (r + 0.5) * transform.e - offset[1]
    return X, Y


def grid_bounds(transform, shape, offset=(0.0, 0.0)):
    """The bounds (minx, miny, maxx, maxy) of a grid, in the transform's coordinates minus offset;
    as (minx, maxx, miny, maxy) they are what ax.imshow wants as its extent."""
    rows, cols = shape[0], shape[1]
    minx = transform.c - offset[0]
    maxx = transform.c + cols * transform.a - offset[0]
    miny = transform.f + rows * transform.e - offset[1]
    maxy = transform.f - offset[1]
    return minx, miny, maxx, maxy


def surface_points(dsm, transform, offset, step=1.0):
    """The surface of a height grid sampled every step metres, as (x, y, z) rows in the map frame:
    the cell centres with their heights, cells without a height left out."""
    rows, cols = dsm.shape
    k = max(1, int(round(step / abs(transform.a))))
    r, c = np.meshgrid(np.arange(0, rows, k), np.arange(0, cols, k), indexing="ij")
    z = np.asarray(dsm, dtype=float)[r, c]
    X = transform.c + (c + 0.5) * transform.a - offset[0]
    Y = transform.f + (r + 0.5) * transform.e - offset[1]
    keep = np.isfinite(z)
    return np.column_stack([X[keep], Y[keep], z[keep]])


def affine_parts(A):
    """The horizontal scale and the turn in degrees of an affine that is nearly a similarity."""
    scale = float(math.sqrt(abs(np.linalg.det(A[:2, :2]))))
    angle = float(math.degrees(math.atan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])))
    return scale, angle


def footprints(photos, choices):
    """One shapely polygon per photograph of the photo table, its footprint on flat ground in the
    products' coordinate system: the frame's size on the ground from the height above the take
    off point, the focal length and the pixel pitch, turned by the aircraft's yaw about the
    receiver's position."""
    crs = {str(v) for v in photos["crs"]}
    message = f"the photographs are in {crs}, the products in {choices['crs']}"
    assert crs <= {choices["crs"]}, message
    pix = PIXEL_UM * 1e-6
    f_m = np.asarray(photos["focal_mm"], dtype=float) * 1e-3
    gsd = pix * np.asarray(photos["rel_alt_m"], dtype=float) / f_m
    yaw = np.nan_to_num(np.asarray(photos["flight_yaw_deg"], dtype=float), nan=0.0)
    east = np.asarray(photos["easting"], dtype=float)
    north = np.asarray(photos["northing"], dtype=float)
    widths = np.asarray(photos["width"], dtype=int)
    heights = np.asarray(photos["height"], dtype=int)
    polygons = []
    for x, y, w, h, g, heading in zip(east, north, widths, heights, gsd, yaw):
        hw = w * g / 2
        hh = h * g / 2
        t = math.radians(heading)
        c = math.cos(t)
        s = math.sin(t)
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        turned = [(x + cx * c + cy * s, y - cx * s + cy * c) for cx, cy in corners]
        polygons.append(Polygon(turned))
    return polygons


def overlap_shares(polygons):
    """The pairs of footprints that overlap, as a dict keyed (i, j) with i before j, each value
    the overlapping area as a share of the smaller footprint."""
    shares = {}
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            shared = polygons[i].intersection(polygons[j]).area
            if shared > 0:
                shares[(i, j)] = shared / min(polygons[i].area, polygons[j].area)
    return shares


def _largest(geometry):
    """The largest polygon of a multipolygon, or the geometry itself."""
    if hasattr(geometry, "geoms"):
        return max(geometry.geoms, key=lambda part: part.area)
    return geometry


def _sample_size(block, choices):
    """The width and height of the kit's photographs, choices["sample_px"] on the longer side."""
    cam = _camera(block)
    W = int(cam["width"])
    H = int(cam["height"])
    sample = int(choices["sample_px"])
    if W >= H:
        return sample, int(round(H * sample / W))
    return int(round(W * sample / H)), sample


def project_outline(block, choices, key, geometry, ground_z):
    """The (u, v) in pixels of the kit's photograph ``key`` ("a", "b" or "c") of a polygon's
    outline laid on the ground at the height ground_z of the map frame; the polygon is in the
    products' coordinate system."""
    shot = int(choices["photos"][key]["shot_index"])
    width, height = _sample_size(block, choices)
    ring = np.asarray(_largest(geometry).exterior.coords, dtype=float)
    offset = np.asarray(choices["offset"], dtype=float)
    z = np.full(len(ring), float(ground_z))
    X = np.column_stack([ring[:, 0] - offset[0], ring[:, 1] - offset[1], z])
    rotation = block["rotation_map"][shot]
    translation = block["translation_map"][shot]
    u, v, _ = project_brown(X, rotation, translation, _camera(block), width, height)
    return u, v


def inside(u_outline, v_outline, u, v):
    """True for every point (u, v) inside the outline drawn by u_outline and v_outline."""
    path = MplPath(np.column_stack([u_outline, v_outline]))
    return path.contains_points(np.column_stack([u, v]))


# --------------------------------------------------------------------------- #
# features and matches
# --------------------------------------------------------------------------- #

def harris(gray, sigma_grad=1.0, sigma_win=2.0, k=0.05, nms=7, n=5000):
    """The n strongest Harris corners of a grey image as (rows, columns, responses): smoothed
    gradients, the structure tensor averaged over a Gaussian window, the response det minus k
    times trace squared, and one peak per window of nms pixels."""
    g = np.asarray(gray, dtype=np.float32)
    ix = ndi.gaussian_filter(g, sigma_grad, order=(0, 1))
    iy = ndi.gaussian_filter(g, sigma_grad, order=(1, 0))
    sxx = ndi.gaussian_filter(ix * ix, sigma_win)
    syy = ndi.gaussian_filter(iy * iy, sigma_win)
    sxy = ndi.gaussian_filter(ix * iy, sigma_win)
    r = (sxx * syy - sxy * sxy) - k * (sxx + syy) ** 2
    peaks = (r == ndi.maximum_filter(r, size=nms)) & (r > 0)
    rows, cols = np.nonzero(peaks)
    order = np.argsort(-r[rows, cols])[:n]
    return rows[order], cols[order], r[rows[order], cols[order]]


def dog_blobs(gray, sigma0=1.6, per_octave=4, octaves=3, threshold=2.5, edge_ratio=10.0):
    """Blobs of a grey image by the difference of Gaussians, one row (column, row, sigma,
    strength) per blob: per octave a stack of blurs, their differences, the extrema over three
    by three by three neighbours, an edge test on the second derivatives, then the picture halved
    for the next octave. A blob's size is about its sigma times the square root of two."""
    blobs = []
    layer = np.asarray(gray, dtype=np.float32)
    for octave in range(octaves):
        sigmas = [sigma0 * 2 ** (i / per_octave) for i in range(per_octave + 2)]
        blurred = np.stack([ndi.gaussian_filter(layer, s) for s in sigmas])
        dog = blurred[1:] - blurred[:-1]
        strength = np.abs(dog)
        peaks = (strength == ndi.maximum_filter(strength, size=3)) & (strength > threshold)
        peaks[0] = False
        peaks[-1] = False
        for k, r, c in zip(*np.nonzero(peaks)):
            if not (1 <= r < dog.shape[1] - 1 and 1 <= c < dog.shape[2] - 1):
                continue
            d = dog[k]
            dxx = d[r, c + 1] + d[r, c - 1] - 2 * d[r, c]
            dyy = d[r + 1, c] + d[r - 1, c] - 2 * d[r, c]
            dxy = (d[r + 1, c + 1] - d[r + 1, c - 1] - d[r - 1, c + 1] + d[r - 1, c - 1]) / 4
            det = dxx * dyy - dxy * dxy
            trace = dxx + dyy
            if det <= 0 or trace * trace / det > (edge_ratio + 1) ** 2 / edge_ratio:
                continue
            scale = 2 ** octave
            blobs.append((c * scale, r * scale, sigmas[k] * scale, strength[k, r, c]))
        layer = blurred[per_octave][::2, ::2]
    return np.array(blobs)


def two_nearest(da, db, chunk=1000):
    """For every descriptor of a, the index of its nearest descriptor in b and the ratio of the
    nearest distance to the second nearest, as (nearest, ratios); the squared distances come from
    the expansion of the norm as a matrix product, a chunk of rows at a time."""
    a = np.asarray(da, dtype=np.float32)
    b = np.asarray(db, dtype=np.float32)
    bb = (b * b).sum(axis=1)
    nearest = []
    ratios = []
    for s in range(0, len(a), chunk):
        ch = a[s:s + chunk]
        d2 = (ch * ch).sum(axis=1)[:, None] + bb[None, :] - 2.0 * ch @ b.T
        two = np.argpartition(d2, 1, axis=1)[:, :2]
        d_two = np.take_along_axis(d2, two, axis=1)
        first = np.argmin(d_two, axis=1)
        rows = np.arange(len(ch))
        best = two[rows, first]
        d1 = d_two[rows, first]
        d2nd = d_two[rows, 1 - first]
        r = np.sqrt(np.maximum(d1, 0) / np.maximum(d2nd, 1e-12))
        nearest.append(best)
        ratios.append(r)
    return np.concatenate(nearest), np.concatenate(ratios)


def ratio_matches(da, db, ratio=0.8, chunk=1000):
    """Lowe's ratio test as (indices into a, indices into b, ratios): a descriptor of a keeps its
    nearest neighbour in b when that one is closer than ratio times the second nearest."""
    nearest, ratios = two_nearest(da, db, chunk)
    keep = ratios < ratio
    return np.nonzero(keep)[0], nearest[keep], ratios[keep]


def _normalise_points(p):
    """Hartley's normalisation: the points centred and scaled so that their mean distance from the
    centre is the square root of two, with the 3 by 3 matrix that did it."""
    c = p.mean(axis=0)
    s = math.sqrt(2.0) / np.sqrt(((p - c) ** 2).sum(axis=1)).mean()
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
    return (p - c) * s, T


def eight_point(pa, pb):
    """The fundamental matrix of eight or more correspondences (pixels in a, pixels in b): one
    linear equation per pair, the last singular vector, the rank two constraint, and Hartley's
    normalisation undone."""
    na, Ta = _normalise_points(pa)
    nb, Tb = _normalise_points(pb)
    A = np.column_stack([nb[:, 0] * na[:, 0], nb[:, 0] * na[:, 1], nb[:, 0],
                         nb[:, 1] * na[:, 0], nb[:, 1] * na[:, 1], nb[:, 1],
                         na[:, 0], na[:, 1], np.ones(len(na))])
    _, _, vt = np.linalg.svd(A)
    F = vt[-1].reshape(3, 3)
    u, s, vt = np.linalg.svd(F)
    s[2] = 0.0
    F = u @ np.diag(s) @ vt
    F = Tb.T @ F @ Ta
    return F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F


def sampson_distance(F, pa, pb):
    """How far each correspondence is from satisfying the epipolar constraint, in pixels, to
    first order."""
    ha = np.column_stack([pa, np.ones(len(pa))])
    hb = np.column_stack([pb, np.ones(len(pb))])
    Fa = ha @ F.T
    Ftb = hb @ F
    num = (hb * Fa).sum(axis=1) ** 2
    den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def fundamental_ransac(pa, pb, iters=2000, thresh=1.0, seed=0):
    """The fundamental matrix and the inlier mask of a set of correspondences by RANSAC: the
    eight point solution of random samples, kept when it has the most correspondences within
    thresh pixels of Sampson distance, then refitted on its inliers."""
    rng = np.random.default_rng(seed)
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    best_mask = None
    best_n = -1
    for _ in range(iters):
        pick = rng.choice(len(pa), 8, replace=False)
        try:
            F = eight_point(pa[pick], pb[pick])
        except np.linalg.LinAlgError:
            continue
        mask = sampson_distance(F, pa, pb) < thresh
        if mask.sum() > best_n:
            best_n = int(mask.sum())
            best_mask = mask
    F = eight_point(pa[best_mask], pb[best_mask])
    mask = sampson_distance(F, pa, pb) < thresh
    return F, mask


def epipolar_lines(F, points, width):
    """The epipolar line in the second photograph of every (u, v) of the first, as its two end
    points where it crosses u = 0 and u = width: an array of shape (n, 2, 2), so that
    ax.plot(line[:, 0], line[:, 1]) draws one line."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    lines = np.column_stack([pts, np.ones(len(pts))]) @ F.T
    u = np.array([0.0, float(width)])
    v = -(lines[:, :1] * u[None, :] + lines[:, 2:3]) / lines[:, 1:2]
    return np.stack([np.broadcast_to(u, v.shape), v], axis=-1)


def union_find_tracks(matches):
    """Pairwise matches joined into tracks: every (photograph, feature) is a node, every match an
    edge, a track a connected component. Returns the track id of every node as a dict and the
    length of every track as an array."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), arr in matches.items():
        for i, j in arr:
            ra = find((a, int(i)))
            rb = find((b, int(j)))
            if ra != rb:
                parent[ra] = rb
    roots = {}
    track_of = {}
    for node in list(parent):
        r = find(node)
        track_of[node] = roots.setdefault(r, len(roots))
    lengths = np.bincount(np.fromiter(track_of.values(), dtype=np.int64), minlength=len(roots))
    return track_of, lengths


def length_histogram(lengths):
    """How many tracks run through 2, 3, ... 10 photographs, and how many through 11 or more, as
    a dict keyed "2" to "10" and "11+"."""
    lengths = np.asarray(lengths)
    h = {str(k): int((lengths == k).sum()) for k in range(2, 11)}
    h["11+"] = int((lengths >= 11).sum())
    return h


# --------------------------------------------------------------------------- #
# the adjustment
# --------------------------------------------------------------------------- #

def mini_bundle(block, tracks, shot_indices, focal=None, perturb=1.0, n_points=3000,
                min_views=4, max_residual_px=4.0, seed=20260930, perturb_deg=0.3,
                perturb_m=0.5, gps_sigma_m=3.0, ftol=1e-6, max_nfev=100):
    """A bundle adjustment of a few photographs in scipy, as one dict of numbers and arrays.

    The unknowns are six per shot (three of rotation, three of translation) and three per point.
    The camera is held at the calibrated one; a number for focal (in normalised units) replaces
    its focal length, None keeps it. The points are up to n_points of those that at least
    min_views of the shots see, and observations farther than max_residual_px from the software's
    own solution are dropped first. The software's poses are then spoiled by a random turn of
    perturb times perturb_deg degrees and a random move of perturb times perturb_m metres (perturb
    0 starts from the software's poses), and the receiver's positions hold the block in place as
    weak priors with a sigma of gps_sigma_m.

    The dict holds the counts shots, points, observations, dropped_observations, unknowns,
    equations and redundancy; the RMS residuals in pixels rms_software_px (the software's own
    solution), rms_start_px, rms_perturbed_px and rms_solved_px; the camera centres' moves in
    metres centre_perturbed_median_m, centre_recovery_median_m and height_shift_median_m; the
    perturbation_deg and perturbation_m used, the solver's status, iterations and seconds, and
    focal; and the arrays before (the residuals after the perturbation, u and v alternating),
    after (the same after the solve), centres_software, centres_perturbed, centres_solved and
    poses."""
    cam = _camera(block)
    if focal is not None:
        cam["focal_x"] = float(focal)
        cam["focal_y"] = float(focal)
    W = int(cam["width"])
    H = int(cam["height"])
    shot_indices = [int(s) for s in shot_indices]
    sel = np.isin(tracks["shot"], shot_indices)
    pts = tracks["point"][sel].astype(np.int64)
    shots = tracks["shot"][sel].astype(np.int64)
    size = max(W, H)
    fu = tracks["x"][sel] * size + W / 2.0 - 0.5
    fv = tracks["y"][sel] * size + H / 2.0 - 0.5
    uniq, counts = np.unique(pts, return_counts=True)
    cand = uniq[counts >= min_views]
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(cand, min(n_points, len(cand)), replace=False))
    keep = np.isin(pts, chosen)
    pts = pts[keep]
    shots = shots[keep]
    fu = fu[keep]
    fv = fv[keep]
    pidx = np.searchsorted(chosen, pts)
    sidx = np.array([shot_indices.index(s) for s in shots])
    n_s = len(shot_indices)
    n_p = len(chosen)
    rot0 = np.asarray(block["rotation"])[shot_indices].astype(float)
    tr0 = np.asarray(block["translation"])[shot_indices].astype(float)
    X0 = np.asarray(block["points"])[chosen].astype(float)
    gps = np.asarray(block["gps"])[shot_indices].astype(float)

    def residuals(params):
        poses = params[: n_s * 6].reshape(n_s, 6)
        X = params[n_s * 6:].reshape(n_p, 3)
        out = np.empty(2 * len(pts) + 3 * n_s)
        for k in range(n_s):
            m = sidx == k
            u, v, _ = project_brown(X[pidx[m]], poses[k, :3], poses[k, 3:], cam, W, H)
            out[np.flatnonzero(m) * 2] = u - fu[m]
            out[np.flatnonzero(m) * 2 + 1] = v - fv[m]
            prior = (camera_centre(poses[k, :3], poses[k, 3:]) - gps[k]) / gps_sigma_m
            out[2 * len(pts) + 3 * k: 2 * len(pts) + 3 * k + 3] = prior
        return out

    # the outlier cut, always against the software's own solution with its calibrated camera, and
    # every point must keep at least two views; the residual there is the floor a solve can reach
    cam_soft = _camera(block)
    d = np.empty(len(pts))
    for k in range(n_s):
        m = sidx == k
        u, v, _ = project_brown(X0[pidx[m]], rot0[k], tr0[k], cam_soft, W, H)
        d[m] = np.hypot(u - fu[m], v - fv[m])
    inlier = d <= max_residual_px
    views_left = np.bincount(pidx[inlier], minlength=n_p)
    inlier &= views_left[pidx] >= 2
    pts = pts[inlier]
    sidx = sidx[inlier]
    pidx = pidx[inlier]
    fu = fu[inlier]
    fv = fv[inlier]
    used = np.unique(pidx)
    remap = np.full(n_p, -1)
    remap[used] = np.arange(len(used))
    pidx = remap[pidx]
    X0 = X0[used]
    n_p = len(used)
    rms_software = float(np.sqrt((d[inlier] ** 2).mean() / 2.0))
    x0 = np.concatenate([np.hstack([rot0, tr0]).ravel(), X0.ravel()])
    r_start = residuals(x0)[: 2 * len(pts)]
    rms_start = float(np.sqrt((r_start ** 2).mean()))
    # the perturbation: a small random turn of every camera and a small random move of its centre
    turn_deg = perturb_deg * perturb
    move_m = perturb_m * perturb
    rot1 = rot0 + rng.normal(size=rot0.shape) * math.radians(turn_deg) / math.sqrt(3)
    centres0 = np.array([camera_centre(r, t) for r, t in zip(rot0, tr0)])
    centres1 = centres0 + rng.normal(size=centres0.shape) * move_m / math.sqrt(3)
    tr1 = np.array([-rodrigues(r) @ c for r, c in zip(rot1, centres1)])
    x1 = np.concatenate([np.hstack([rot1, tr1]).ravel(), X0.ravel()])
    r1 = residuals(x1)[: 2 * len(pts)]
    rms_perturbed = float(np.sqrt((r1 ** 2).mean()))
    # which residual depends on which unknown; the solver never forms the rest of the Jacobian
    S = lil_matrix((2 * len(pts) + 3 * n_s, n_s * 6 + n_p * 3), dtype=int)
    rows = np.arange(len(pts))
    for j in range(6):
        S[2 * rows, sidx * 6 + j] = 1
        S[2 * rows + 1, sidx * 6 + j] = 1
    for j in range(3):
        S[2 * rows, n_s * 6 + pidx * 3 + j] = 1
        S[2 * rows + 1, n_s * 6 + pidx * 3 + j] = 1
    for k in range(n_s):
        for j in range(6):
            S[2 * len(pts) + 3 * k: 2 * len(pts) + 3 * k + 3, k * 6 + j] = 1
    t0 = time.time()
    fit = least_squares(residuals, x1, jac_sparsity=S, method="trf", x_scale="jac", loss="soft_l1",
                        f_scale=1.0, ftol=ftol, xtol=1e-8, max_nfev=max_nfev)
    seconds = time.time() - t0
    r2 = fit.fun[: 2 * len(pts)]
    poses = fit.x[: n_s * 6].reshape(n_s, 6)
    centres2 = np.array([camera_centre(p[:3], p[3:]) for p in poses])
    rms_solved = float(np.sqrt((r2 ** 2).mean()))
    recovery = np.linalg.norm(centres2 - centres0, axis=1)
    pushed = np.linalg.norm(centres1 - centres0, axis=1)
    return dict(
        shots=n_s, points=int(n_p), observations=int(len(pts)),
        dropped_observations=int((~inlier).sum()),
        unknowns=int(n_s * 6 + n_p * 3), equations=int(2 * len(pts)),
        redundancy=int(2 * len(pts) - n_s * 6 - n_p * 3),
        rms_software_px=round(rms_software, 3), rms_start_px=round(rms_start, 3),
        rms_perturbed_px=round(rms_perturbed, 3), rms_solved_px=round(rms_solved, 3),
        perturbation_deg=turn_deg, perturbation_m=move_m,
        centre_recovery_median_m=round(float(np.median(recovery)), 3),
        centre_perturbed_median_m=round(float(np.median(pushed)), 3),
        height_shift_median_m=round(float(np.median(centres2[:, 2] - centres0[:, 2])), 3),
        iterations=int(fit.nfev), seconds=round(seconds, 1), status=int(fit.status),
        focal=float(cam["focal_x"]),
        before=r1, after=r2, centres_software=centres0, centres_perturbed=centres1,
        centres_solved=centres2, poses=poses,
    )


# --------------------------------------------------------------------------- #
# depth, clouds, meshes and elevation models
# --------------------------------------------------------------------------- #

def ncc_depth(gray_a, gray_b, u, v, pose_a, pose_b, camera, width, height, depths, half=7):
    """The correlation search of dense matching for one pixel (u, v) of photograph a: for every
    candidate depth along its ray the point is projected into photograph b, and the normalised
    cross correlation of the patch there with the patch around (u, v) is returned, one value per
    depth (NaN where the patch leaves the frame). The patch is 2 half + 1 pixels wide."""
    ra, ta = pose_a
    rb, tb = pose_b
    Ra = rodrigues(ra)
    Ca = -Ra.T @ np.asarray(ta, dtype=float)
    ray = Ra.T @ pixel_rays(u, v, camera, width, height)
    X = Ca[None, :] + np.asarray(depths, dtype=float)[:, None] * ray[None, :]
    ub, vb, _ = project_brown(X, rb, tb, camera, width, height)
    pa = gray_a[v - half:v + half + 1, u - half:u + half + 1].astype(np.float32)
    pa = (pa - pa.mean()) / (pa.std() + 1e-6)
    out = np.full(len(depths), np.nan, dtype=np.float32)
    for k, (x, y) in enumerate(zip(np.round(ub).astype(int), np.round(vb).astype(int))):
        if half <= x < width - half and half <= y < height - half:
            pb = gray_b[y - half:y + half + 1, x - half:x + half + 1].astype(np.float32)
            pb = (pb - pb.mean()) / (pb.std() + 1e-6)
            out[k] = float((pa * pb).mean())
    return out


def unproject_depth(depth, K, R, C):
    """The valid pixels of a depth map as points of the map frame, (xyz, valid): each pixel's ray
    through K inverse, times its depth along the axis, turned by R transposed and moved to the
    camera centre C; valid is the mask of pixels that had a depth."""
    depth = np.asarray(depth)
    valid = depth > 0
    rows, cols = np.nonzero(valid)
    pixels = np.column_stack([cols, rows, np.ones(len(rows))]).astype(float)
    directions = pixels @ np.linalg.inv(K).T
    in_camera = directions * depth[rows, cols][:, None]
    xyz = in_camera @ np.asarray(R) + np.asarray(C)
    return xyz, valid


def grid_dsm(xyz, bounds, cell):
    """The highest point per cell of a grid over bounds (minx, miny, maxx, maxy), north up, the
    surface model's recipe, as (heights, counts); cells no point reaches stay NaN."""
    minx, miny, maxx, maxy = bounds
    cols = int(round((maxx - minx) / cell))
    rows = int(round((maxy - miny) / cell))
    c = np.floor((xyz[:, 0] - minx) / cell).astype(int)
    r = np.floor((maxy - xyz[:, 1]) / cell).astype(int)
    keep = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
    z = np.full((rows, cols), -np.inf, dtype=np.float32)
    n = np.zeros((rows, cols), dtype=np.int32)
    np.maximum.at(z, (r[keep], c[keep]), xyz[keep, 2].astype(np.float32))
    np.add.at(n, (r[keep], c[keep]), 1)
    z[n == 0] = np.nan
    return z, n


def fill_gaps(grid):
    """The grid with every NaN cell given the value of its nearest filled cell."""
    mask = np.isnan(grid)
    if not mask.any():
        return grid.copy()
    idx = ndi.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def ground_by_opening(dsm, cell, windows_m=(1, 2, 4, 8, 12, 18), slope=0.15, threshold=0.5):
    """The terrain under a surface model as (ground, mask), by a progressive morphological
    filter: the surface is opened with windows of growing size, and a cell stays ground while it
    never rises above the opened surface by more than slope times the window plus threshold.
    The ground grid is the surface where the cell was kept and the opened surface where it was
    not; the mask says which cells were kept."""
    z = fill_gaps(np.asarray(dsm, dtype=np.float32))
    ground = np.ones(z.shape, dtype=bool)
    opened = z.copy()
    for w in windows_m:
        size = max(3, int(round(w / cell)) | 1)
        new = ndi.grey_opening(opened, size=(size, size))
        ground &= (opened - new) <= slope * w + threshold
        opened = new
    out = np.where(ground, z, opened)
    return out, ground


def delaunay_25d(xyz, thin_m=None, max_edge_m=None):
    """A mesh over a cloud as (vertices, faces): the points thinned to one per cell of thin_m
    metres (the first point of every occupied cell), the triangles of a Delaunay triangulation in
    plan with the heights carried along, and every triangle with an edge longer than max_edge_m
    dropped. Faces are three indices into vertices per triangle."""
    points = np.asarray(xyz, dtype=float)
    if thin_m is not None:
        cell = np.floor(points[:, :2] / thin_m).astype(int)
        _, first = np.unique(cell[:, 0] * 100000 + cell[:, 1], return_index=True)
        points = points[first]
    faces = Delaunay(points[:, :2]).simplices.astype(np.int64)
    if max_edge_m is not None:
        a = points[faces[:, 0], :2]
        b = points[faces[:, 1], :2]
        c = points[faces[:, 2], :2]
        ab = np.linalg.norm(a - b, axis=1)
        bc = np.linalg.norm(b - c, axis=1)
        ca = np.linalg.norm(c - a, axis=1)
        faces = faces[np.maximum(np.maximum(ab, bc), ca) <= max_edge_m]
    return points, faces


# --------------------------------------------------------------------------- #
# the orthomosaic
# --------------------------------------------------------------------------- #

def orthorectify(dsm, transform, nodata, image, rotation, translation, camera, offset,
                 z_shift=0.0):
    """One photograph laid onto a height grid, as (picture, covered): every cell's X, Y and Z
    (the grid's coordinates minus offset, plus z_shift) is projected into the photograph with the
    collinearity equations and the colour under that pixel is taken, nearest neighbour; cells the
    photograph does not see, or whose height is nodata, stay zero and are False in covered."""
    rows, cols = dsm.shape
    X, Y = grid_coordinates(transform, (rows, cols), offset)
    Z = np.asarray(dsm, dtype=float) + z_shift
    valid = np.isfinite(Z) & (Z != nodata)
    H, W = image.shape[:2]
    XYZ = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    u, v, d = project_brown(XYZ, rotation, translation, camera, W, H)
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    covered = valid.ravel() & (d > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    out = np.zeros((rows * cols,) + image.shape[2:], dtype=image.dtype)
    out[covered] = image[vi[covered], ui[covered]]
    return out.reshape((rows, cols) + image.shape[2:]), covered.reshape(rows, cols)


def nearest_nadir_mosaic(orthos, covered, nadirs_xy, X, Y):
    """A mosaic of rectified photographs as (mosaic, pick): every cell takes the photograph whose
    nadir (the point straight below its lens, in the frame of X and Y) is nearest among those
    that cover the cell; pick holds that photograph's index, or -1 where none covers the cell."""
    distance = []
    for mask, (nx, ny) in zip(covered, nadirs_xy):
        distance.append(np.where(mask, np.hypot(X - nx, Y - ny), np.inf))
    distance = np.stack(distance)
    pick = np.argmin(distance, axis=0)
    pick = np.where(np.isfinite(distance.min(axis=0)), pick, -1)
    mosaic = np.zeros_like(orthos[0])
    for k, ortho in enumerate(orthos):
        mosaic[pick == k] = ortho[pick == k]
    return mosaic, pick


def cells_of_class(xyz, classification, code, bounds, cell=1.0):
    """A grid over bounds (minx, miny, maxx, maxy) that is True where more than half of the cell's
    points carry the class code, and False where the cell holds no point at all."""
    _, counts = grid_dsm(xyz, bounds, cell)
    of_class = np.asarray(classification) == code
    _, counts_class = grid_dsm(np.asarray(xyz)[of_class], bounds, cell)
    return counts_class > counts / 2


def density(xyz, mask_or_area, bounds=None, cell=1.0):
    """Points per square metre: over an area given in square metres, or on the True cells of a
    grid over bounds (the mean count of those cells divided by the cell's area)."""
    mask = np.asarray(mask_or_area)
    if mask.ndim == 0:
        return len(xyz) / float(mask)
    _, counts = grid_dsm(xyz, bounds, cell)
    return float(counts[mask].mean()) / (cell * cell)


def roof_height(surface, terrain, roof):
    """The height of a roof above a terrain, in metres: the median over the roof's cells of the
    surface model minus the terrain, which may be a grid or one level."""
    difference = np.asarray(surface, dtype=float) - np.asarray(terrain, dtype=float)
    return float(np.nanmedian(difference[roof]))


# --------------------------------------------------------------------------- #
# the window and the roof
# --------------------------------------------------------------------------- #

def window_bounds_local(choices):
    """The pinned window's bounds (minx, miny, maxx, maxy) in the map frame, metres from the
    offset."""
    minx, miny, maxx, maxy = choices["window"]["bounds"]
    ox, oy = choices["offset"]
    return minx - ox, miny - oy, maxx - ox, maxy - oy


def hillshade(grid, cell_m, azdeg=315, altdeg=45):
    """A hillshade of a height grid, 0 dark to 1 bright, lit from the north west and 45 degrees
    up unless told otherwise; cells without a height stay NaN."""
    light = LightSource(azdeg=azdeg, altdeg=altdeg)
    grid = np.asarray(grid, dtype=float)
    missing = np.isnan(grid)
    filled = np.where(missing, np.nanpercentile(grid, 25), grid)
    shade = light.hillshade(filled, vert_exag=1, dx=cell_m, dy=cell_m)
    shade[missing] = np.nan
    return shade


def roof_polygon(choices):
    """The kit's building as a shapely polygon in the products' coordinate system, the
    OpenStreetMap footprint recorded in the choices."""
    return Polygon(choices["roof"]["polygon"])


def roof_mask(polygon, transform, shape, offset=(0.0, 0.0)):
    """A grid of the given shape, True inside the polygon; offset moves a polygon given in the
    products' coordinates into a grid whose transform is in the map frame."""
    moved = translate(polygon, -offset[0], -offset[1])
    return geometry_mask([moved], out_shape=(shape[0], shape[1]), transform=transform, invert=True)


def roof_corner(choices):
    """The x and y, in the products' coordinate system, of the roof's corner inside the window."""
    x, y = choices["roof"]["corner"]
    return float(x), float(y)


def roof_point(dsm, transform, roof, ground_level, towards, offset=(0.0, 0.0), inset_m=1.0):
    """A point on the main roof with its height, (x, y, z) in the transform's coordinates minus
    offset: the roof cell more than 8 m above the ground level that lies nearest the point
    `towards`, moved inset_m towards the middle of the roof so a patch around it sits on the roof
    and not on its edge or on a lower porch."""
    heights = np.nan_to_num(np.asarray(dsm, dtype=float), nan=-np.inf)
    high = np.asarray(roof, dtype=bool) & (heights > ground_level + 8.0)
    X, Y = grid_coordinates(transform, heights.shape, offset)
    xs, ys = X[high], Y[high]
    k = int(np.argmin(np.hypot(xs - towards[0], ys - towards[1])))
    dx, dy = float(xs.mean() - xs[k]), float(ys.mean() - ys[k])
    norm = math.hypot(dx, dy) or 1.0
    x, y = float(xs[k] + inset_m * dx / norm), float(ys[k] + inset_m * dy / norm)
    row = min(max(math.floor((y + offset[1] - transform.f) / transform.e), 0), heights.shape[0] - 1)
    col = min(max(math.floor((x + offset[0] - transform.c) / transform.a), 0), heights.shape[1] - 1)
    return x, y, float(heights[row, col])


# --------------------------------------------------------------------------- #
# plotting, every function draws on the ax it is given
# --------------------------------------------------------------------------- #

def _parts(geometry):
    """The parts of a geometry, a multi geometry, a GeoSeries or a list, one at a time."""
    if isinstance(geometry, BaseGeometry):
        if hasattr(geometry, "geoms"):
            for item in geometry.geoms:
                yield from _parts(item)
        else:
            yield geometry
    elif hasattr(geometry, "geometry"):
        yield from _parts(list(geometry.geometry))
    else:
        for item in geometry:
            yield from _parts(item)


def _finish(ax, title, ticks):
    """The title and, when ticks is False, no tick marks."""
    if title:
        ax.set_title(title, fontsize=11)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def show_image(ax, image, title=None, **kwargs):
    """The image drawn on ax without ticks, grey when it has one channel; returns the handle."""
    kwargs.setdefault("cmap", "gray" if np.ndim(image) == 2 else None)
    kwargs.setdefault("interpolation", "nearest")
    handle = ax.imshow(image, **kwargs)
    _finish(ax, title, False)
    return handle


def show_grid(ax, grid, transform=None, offset=(0.0, 0.0), cmap="viridis", vmin=None, vmax=None,
              label=None, outline=None, points=None, window=None, title=None, ticks=True):
    """A raster drawn on ax in its own coordinates (the transform's, minus offset) with a colour
    bar labelled label when one is given; outline (a shapely geometry or a GeoSeries) in white,
    points (an array of x, y rows) in orange, window (minx, miny, maxx, maxy) as a red rectangle,
    all of them given in the transform's coordinates and moved by the same offset; the view stays
    the grid's. Returns the image handle."""
    extent = None
    if transform is not None:
        minx, miny, maxx, maxy = grid_bounds(transform, np.shape(grid), offset)
        extent = (minx, maxx, miny, maxy)
    handle = ax.imshow(grid, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
    if label:
        ax.figure.colorbar(handle, ax=ax, fraction=0.03, pad=0.02, label=label)
    if outline is not None:
        for part in _parts(outline):
            moved = translate(part, -offset[0], -offset[1])
            x, y = moved.exterior.xy if hasattr(moved, "exterior") else moved.xy
            ax.plot(x, y, color="white", linewidth=1.3)
    if points is not None:
        xy = np.atleast_2d(np.asarray(points, dtype=float))
        ax.scatter(xy[:, 0] - offset[0], xy[:, 1] - offset[1], s=12, color="orange", zorder=3)
    if window is not None:
        minx, miny, maxx, maxy = window
        corner = (minx - offset[0], miny - offset[1])
        box = Rectangle(corner, maxx - minx, maxy - miny, facecolor="none", edgecolor="red",
                        linewidth=1.5)
        ax.add_patch(box)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    _finish(ax, title, ticks)
    return handle


def show_cloud(ax, xyz, rgb=None, values=None, view="plan", size=1, cmap="viridis", vmin=None,
               vmax=None, label=None, title=None):
    """A point cloud scattered on ax, in plan (x east, y north, equal axes) or oblique (x east
    against height plus 0.8 times north), coloured by its rgb, by values with a colour bar
    labelled label, or by height when neither is given. Returns the scatter handle."""
    xyz = np.asarray(xyz, dtype=float)
    y = xyz[:, 1] if view == "plan" else xyz[:, 2] + OBLIQUE_NORTH * xyz[:, 1]
    if rgb is not None:
        colour = np.asarray(rgb) / 255.0
        handle = ax.scatter(xyz[:, 0], y, c=colour, s=size, linewidths=0)
    else:
        colour = xyz[:, 2] if values is None else np.asarray(values)
        handle = ax.scatter(xyz[:, 0], y, c=colour, s=size, cmap=cmap, vmin=vmin, vmax=vmax,
                            linewidths=0)
        if label:
            ax.figure.colorbar(handle, ax=ax, fraction=0.03, pad=0.02, label=label)
    if view == "plan":
        ax.set_aspect("equal")
        ax.set_ylabel("north (m)")
    else:
        ax.set_ylabel(f"height plus {OBLIQUE_NORTH} times north (m)")
    ax.set_xlabel("east (m)")
    _finish(ax, title, True)
    return handle


def show_mesh(ax, vertices, faces, values=None, wire=False, cmap="viridis", vmin=None, vmax=None,
              label=None, title=None):
    """A mesh drawn on ax in plan with equal axes: the triangles filled by their vertices'
    heights (or by values, one per vertex) with a colour bar labelled label, or as a wire frame
    of black edges when wire is True. Returns the handle."""
    vertices = np.asarray(vertices, dtype=float)
    if wire:
        handle = ax.triplot(vertices[:, 0], vertices[:, 1], faces, linewidth=0.4, color="black")
    else:
        colour = vertices[:, 2] if values is None else np.asarray(values)
        handle = ax.tripcolor(vertices[:, 0], vertices[:, 1], faces, colour, shading="gouraud",
                              cmap=cmap, vmin=vmin, vmax=vmax)
        if label:
            ax.figure.colorbar(handle, ax=ax, fraction=0.03, pad=0.02, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    _finish(ax, title, True)
    return handle


def crop_at(ax, image, u, v, half=100, title=None, colour="cyan"):
    """A crop of the image of 2 half pixels on a side around the pixel (u, v), drawn on ax in the
    image's own pixel coordinates with the position ringed; returns the crop."""
    rows, cols = np.shape(image)[0], np.shape(image)[1]
    r0 = int(np.clip(round(v) - half, 0, max(rows - 2 * half, 0)))
    c0 = int(np.clip(round(u) - half, 0, max(cols - 2 * half, 0)))
    crop = image[r0:r0 + 2 * half, c0:c0 + 2 * half]
    extent = (c0, c0 + crop.shape[1], r0 + crop.shape[0], r0)
    cmap = "gray" if np.ndim(crop) == 2 else None
    ax.imshow(crop, cmap=cmap, interpolation="nearest", extent=extent)
    ax.scatter([u], [v], s=110, facecolor="none", edgecolor=colour, linewidth=1.8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _finish(ax, title, False)
    return crop


def descriptor_bars(ax, descriptor, title=None):
    """The 128 values of one descriptor as bars on ax, the sixteen cells of eight direction bins
    separated by faint lines."""
    values = np.asarray(descriptor, dtype=float)
    ax.bar(np.arange(len(values)), values, width=1.0, color="steelblue")
    for cell in range(1, len(values) // 8):
        ax.axvline(cell * 8 - 0.5, color="0.85", linewidth=0.6)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.set_xlabel("sixteen cells of eight direction bins")
    _finish(ax, title, True)
