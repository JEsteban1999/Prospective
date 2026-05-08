"""Parent artery diameter estimation — Feature 5.

Automatically estimates the diameter of the parent artery at the neck
of the aneurysm.  This value is needed to compute the *Size Ratio* (SR =
max_aneurysm_diameter / parent_artery_diameter), which is the most strongly
validated morphometric predictor of rupture risk (Dhar 2008, ISUIA).

Algorithm
---------
1. Find the neck centroid and principal axis from the :class:`MorphometricResult`.
2. Take several cross-sectional cuts of the *vessel* mesh at positions
   just *below* the neck (in the direction opposite the aneurysm dome),
   using the cross-section module already present in the project.
3. For each cut, the intersection may contain two connected components:
   the parent lumen (large circle) and possibly part of the aneurysm.
   The **largest component by area** is taken as the parent artery section.
4. Average the diameters from the valid cuts to produce a stable estimate.

The function works best when the vessel mesh is the full vascular tree and
the aneurysm mesh has been isolated separately (standard PROSPECTIVE workflow).
If only one mesh is provided the function still produces a reasonable estimate
by sampling slightly away from the aneurysm centroid.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from prospective.processing.morphometrics import MorphometricResult

logger = logging.getLogger(__name__)

# How many planes to sample below the neck for the average
_N_SAMPLES     = 8
# How far below the neck (in multiples of the estimated neck radius)
# to start / stop sampling
_OFFSET_START  = 1.0   # begin 1 × neck_radius below neck
_OFFSET_END    = 3.5   # end   3.5 × neck_radius below neck


def estimate_parent_artery_diameter(
    vessel_poly: vtk.vtkPolyData,
    morpho: MorphometricResult,
) -> float:
    """
    Estimate the parent artery outer diameter (mm) near the aneurysm neck.

    Parameters
    ----------
    vessel_poly : vtkPolyData
        The full vascular mesh (not just the aneurysm).
    morpho : MorphometricResult
        Result of morphometric analysis on the aneurysm.

    Returns
    -------
    float
        Estimated parent artery diameter in mm, or 0.0 if estimation fails.
    """
    if vessel_poly is None or vessel_poly.GetNumberOfPoints() == 0:
        logger.warning("parent_artery: empty vessel mesh — returning 0")
        return 0.0

    centroid  = np.asarray(morpho.centroid, dtype=float)
    axis      = np.asarray(morpho.principal_axis, dtype=float)
    norm      = np.linalg.norm(axis)
    if norm < 1e-9:
        return 0.0
    axis /= norm

    # The "below-neck" direction: opposite to dome direction.
    # Dome direction = from centroid toward apex (large positive projection).
    # We already know neck_plane_pos ∈ [0, 1]; if > 0.5 dome is above, else below.
    # Use the sign of (centroid projected to axis) vs neck_plane position.
    pts_arr = vtk_to_numpy(vessel_poly.GetPoints().GetData()).astype(float)
    proj    = (pts_arr - centroid) @ axis
    p_min, p_max = proj.min(), proj.max()
    extent  = p_max - p_min

    neck_abs = p_min + morpho.neck_plane_pos * extent
    # Below-neck is toward the parent artery; direction = −axis (toward p_min)
    neck_r   = max(morpho.neck_diameter_mm / 2.0, 1.0)

    sample_start = neck_abs - _OFFSET_START * neck_r
    sample_end   = neck_abs - _OFFSET_END   * neck_r

    if sample_end > sample_start:
        sample_start, sample_end = sample_end, sample_start   # ensure start < end

    # Clamp to mesh extent
    sample_start = max(sample_start, p_min + 0.5)
    sample_end   = min(sample_end,   p_max - 0.5)

    if sample_start <= sample_end:
        logger.debug(
            "parent_artery: sampling range too small (%.1f–%.1f), "
            "expanding to full mesh extent",
            sample_start, sample_end,
        )
        # Fallback: sample across the whole mesh except near the aneurysm
        sample_start = p_min + extent * 0.05
        sample_end   = neck_abs - neck_r * 0.5

    positions = np.linspace(sample_end, sample_start, _N_SAMPLES)

    diameters: list[float] = []
    for pos in positions:
        origin = centroid + pos * axis
        d = _cut_largest_component_diameter(vessel_poly, origin, axis)
        if d > 0:
            diameters.append(d)
            logger.debug("parent_artery: cut at %.1f mm → Ø=%.2f mm", pos, d)

    if not diameters:
        logger.warning(
            "parent_artery: no valid cross-sections found — returning 0"
        )
        return 0.0

    # Median to suppress outliers
    result = float(np.median(diameters))
    logger.info("parent_artery: estimated Ø = %.2f mm (from %d samples)", result, len(diameters))
    return result


# ──────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _cut_largest_component_diameter(
    poly_data: vtk.vtkPolyData,
    origin: np.ndarray,
    normal: np.ndarray,
) -> float:
    """
    Cut *poly_data* with a plane, find the largest closed contour,
    and return its equivalent-circle diameter (mm).
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

    # Use connectivity to isolate the largest region
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(cut)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    largest = conn.GetOutput()

    if largest.GetNumberOfPoints() < 3:
        return 0.0

    # Strip into polylines and compute area
    stripper = vtk.vtkStripper()
    stripper.SetInputData(largest)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    stripped = stripper.GetOutput()

    if stripped.GetNumberOfPoints() < 3:
        return 0.0

    all_pts = vtk_to_numpy(stripped.GetPoints().GetData())
    lines   = stripped.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        return 0.0

    # Take the longest polyline
    best: list[int] = []
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    while lines.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        if n > len(best):
            best = [id_list.GetId(i) for i in range(n)]

    if len(best) < 3:
        return 0.0

    pts_ordered = all_pts[best]
    area        = _shoelace(pts_ordered, origin, normal)
    if area <= 0:
        return 0.0

    return 2.0 * math.sqrt(area / math.pi)


def _shoelace(pts: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> float:
    """2-D area of a planar polygon via the Shoelace formula."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    rel = pts.astype(float) - origin.astype(float)
    x   = rel @ u
    y   = rel @ v

    # Sort by angle for clean Shoelace
    cx, cy  = x.mean(), y.mean()
    angles  = np.arctan2(y - cy, x - cx)
    order   = np.argsort(angles)
    x, y    = x[order], y[order]

    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
