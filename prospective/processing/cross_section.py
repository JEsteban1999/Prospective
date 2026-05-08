"""Cross-section analysis along a vessel centreline — Feature 2.

Algorithm
---------
For each sample point on the centreline:

1. Build a cutting plane perpendicular to the local tangent vector.
2. Use ``vtkCutter`` to slice the vessel mesh at that plane.
3. Use ``vtkStripper`` to order the cut edges into a closed polyline.
4. Project the polyline onto the cutting plane (2-D) and apply the
   **shoelace formula** to obtain the cross-sectional area.
5. Derive the *equivalent-circle diameter*:  d = 2 · √(A / π)

Outputs
-------
:class:`CrossSectionResult` holds:

* ``arc_positions_mm`` — sample positions along the centreline
* ``areas_mm2``        — cross-sectional area at each position
* ``diameters_mm``     — equivalent diameter (circle with same area)
* ``min/max/mean`` area and diameter
* ``stenosis_ratio``   — min_area / mean_area  (1.0 = no stenosis)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from prospective.processing.centerline import CenterlineResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# Result dataclass                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class CrossSectionResult:
    """Output of :func:`compute_cross_sections`."""

    # Per-sample arrays
    arc_positions_mm: np.ndarray   # (M,) distance along centreline
    areas_mm2:        np.ndarray   # (M,) cross-sectional area
    diameters_mm:     np.ndarray   # (M,) equivalent circle diameter

    # Scalar summary
    min_area_mm2:     float = 0.0
    max_area_mm2:     float = 0.0
    mean_area_mm2:    float = 0.0
    min_diameter_mm:  float = 0.0
    max_diameter_mm:  float = 0.0
    mean_diameter_mm: float = 0.0

    median_area_mm2:     float = 0.0
    median_diameter_mm:  float = 0.0

    # Stenosis estimate: ratio of minimum to median area (1.0 = uniform)
    stenosis_ratio:   float = 1.0


# ──────────────────────────────────────────────────────────────────────────── #
# Main computation                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def compute_cross_sections(
    centerline: CenterlineResult,
    poly_data: vtk.vtkPolyData,
    n_samples: int = 40,
    progress_cb=None,
) -> CrossSectionResult:
    """
    Measure vessel cross-sectional area at *n_samples* evenly-spaced positions
    along *centerline*.

    Parameters
    ----------
    centerline : CenterlineResult
        Output of :func:`~prospective.processing.centerline.extract_centerline`.
    poly_data : vtkPolyData
        The vessel surface mesh (closed, manifold).
    n_samples : int
        Number of cutting planes (default 40).  More → finer profile but slower.
    progress_cb : callable | None
        Optional ``progress_cb(fraction: float)`` for UI progress bars.

    Returns
    -------
    CrossSectionResult
    """
    _progress = progress_cb or (lambda _: None)

    pts = centerline.points        # (N, 3)
    N   = len(pts)
    if N < 2:
        raise ValueError("La línea central debe tener al menos 2 puntos.")

    # ── Compute tangents (forward differences, last point = penultimate) ── #
    tangents        = np.zeros_like(pts)
    tangents[:-1]   = pts[1:] - pts[:-1]
    tangents[-1]    = tangents[-2]
    norms           = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms           = np.where(norms < 1e-9, 1.0, norms)
    tangents       /= norms

    # ── Arc-length positions for each original point ────────────────────── #
    seg_lens    = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total_len   = seg_lens[-1]

    if total_len < 1e-3:
        raise ValueError("La línea central es demasiado corta para el análisis.")

    # ── Choose n_samples positions uniformly along arc ─────────────────── #
    # Skip the very first and last 5 % to avoid noisy boundary cuts
    margin      = 0.05 * total_len
    sample_arcs = np.linspace(margin, total_len - margin, n_samples)

    # Map arc positions back to point indices by interpolation
    sample_indices = np.interp(sample_arcs, seg_lens, np.arange(N)).astype(int)
    sample_indices = np.clip(sample_indices, 0, N - 1)

    # ── Cut mesh at each sample position ────────────────────────────────── #
    areas = np.zeros(n_samples, dtype=np.float64)

    for k, idx in enumerate(sample_indices):
        _progress(k / n_samples)
        pt  = pts[idx]
        nrm = tangents[idx]
        areas[k] = _cut_area(poly_data, pt, nrm)
        if areas[k] == 0.0:
            logger.debug("Zero area at arc pos %.1f mm (pt idx %d)", sample_arcs[k], idx)

    _progress(1.0)

    # ── Filter out zero-area samples (bad cuts at mesh boundary) ────────── #
    valid    = areas > 0.0
    if not valid.any():
        raise ValueError(
            "No se pudo calcular ninguna sección transversal. "
            "Verifique que la malla sea cerrada y los puntos de la línea central "
            "estén dentro del vaso."
        )

    arc_pos  = sample_arcs[valid]
    areas_v  = areas[valid]
    diams    = 2.0 * np.sqrt(areas_v / np.pi)

    # ── IQR outlier filter (removes bifurcation / aneurysm artefacts) ────── #
    if len(areas_v) >= 8:
        q1, q3 = np.percentile(areas_v, [25, 75])
        iqr = q3 - q1
        inliers = (areas_v >= q1 - 3.0 * iqr) & (areas_v <= q3 + 3.0 * iqr)
        if inliers.sum() >= 4:          # keep at least 4 samples
            arc_pos  = arc_pos[inliers]
            areas_v  = areas_v[inliers]
            diams    = diams[inliers]

    min_a    = float(areas_v.min())
    max_a    = float(areas_v.max())
    mean_a   = float(areas_v.mean())
    median_a = float(np.median(areas_v))
    stenosis = min_a / median_a if median_a > 1e-9 else 1.0   # use median as reference

    return CrossSectionResult(
        arc_positions_mm    = arc_pos,
        areas_mm2           = areas_v,
        diameters_mm        = diams,
        min_area_mm2        = min_a,
        max_area_mm2        = max_a,
        mean_area_mm2       = mean_a,
        median_area_mm2     = median_a,
        median_diameter_mm  = 2.0 * float(np.sqrt(median_a / np.pi)),
        min_diameter_mm     = float(diams.min()),
        max_diameter_mm     = float(diams.max()),
        mean_diameter_mm    = float(diams.mean()),
        stenosis_ratio      = stenosis,
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _cut_area(
    poly_data: vtk.vtkPolyData,
    origin: np.ndarray,
    normal: np.ndarray,
) -> float:
    """
    Slice *poly_data* with a plane and return the enclosed 2-D area (mm²).

    Returns 0.0 if the cut produces fewer than 3 points.
    """
    plane = vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(poly_data)
    cutter.Update()

    cut = cutter.GetOutput()
    if cut.GetNumberOfPoints() < 3:
        return 0.0

    # Order the cut edges into a continuous polyline
    stripper = vtk.vtkStripper()
    stripper.SetInputData(cut)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    stripped = stripper.GetOutput()

    if stripped.GetNumberOfPoints() < 3:
        return 0.0

    # Extract the longest polyline (main vessel contour) by following cell ids.
    # vtkStripper produces Line/PolyLine cells whose IDs index into the point array.
    all_pts = vtk_to_numpy(stripped.GetPoints().GetData())   # (K, 3)
    lines   = stripped.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        return 0.0

    best: list[int] = []
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    while lines.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        if n > len(best):
            best = [id_list.GetId(i) for i in range(n)]

    if len(best) < 3:
        return 0.0

    pts_arr = all_pts[best]
    return _shoelace_area(pts_arr, origin, normal)


def _shoelace_area(
    pts: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
) -> float:
    """
    Compute the 2-D area enclosed by *pts* (a planar polygon in 3-D space)
    using the Shoelace (Gauss area) formula.

    The polygon is projected onto the plane defined by *origin* and *normal*.
    """
    if len(pts) < 3:
        return 0.0

    # Build orthonormal basis {u, v} for the plane
    n = normal.astype(float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        return 0.0
    n = n / n_norm

    # Pick a helper vector not collinear with n
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    # Project 3-D points onto the 2-D plane
    rel  = pts.astype(float) - origin.astype(float)
    x    = rel @ u
    y    = rel @ v

    # Sort by angle around centroid (handles cases where Stripper reorders points)
    cx, cy = x.mean(), y.mean()
    angles  = np.arctan2(y - cy, x - cx)
    order   = np.argsort(angles)
    x, y    = x[order], y[order]

    # Shoelace formula
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)
