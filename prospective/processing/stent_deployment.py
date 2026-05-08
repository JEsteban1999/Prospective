"""Virtual stent deployment along a vessel centreline — Feature 4.

Algorithm
---------
1. Extract the sub-path of the centreline between *start_arc_mm* and
   *end_arc_mm* (arc-length positions).
2. Compute **Frenet–Serret frames** (tangent T, normal N, binormal B) at each
   point using double-reflection transport to avoid sudden frame twisting.
3. **Sweep** a circle of radius *stent_radius_mm* along the path using the
   transport frames — each cross-section circle is placed in the (N, B) plane
   at the corresponding centreline point.
4. Optionally generate a **stent wire pattern** (helical braids) on the tube
   surface for visual realism.
5. Return a :class:`DeployedStentResult` with the tube ``vtkPolyData`` and
   summary metrics.

Clinical outputs
----------------
* ``length_mm``       — arc length of the deployed segment
* ``nominal_diameter_mm`` — stent nominal diameter (from spec or explicit arg)
* ``mean_vessel_diameter_mm`` — mean vessel Ø at the deployed segment (from CL radii)
* ``coverage_ratio``  — stent_radius / mean_vessel_radius (< 1 → undersized)
* ``stent_poly_data`` — VTK surface for rendering
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from prospective.processing.centerline import CenterlineResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# Result dataclass                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class DeployedStentResult:
    """Output of :func:`deploy_stent_on_centerline`."""

    # Geometry
    stent_poly_data:          vtk.vtkPolyData     # tube mesh for rendering
    centerline_segment:       np.ndarray          # (M, 3) sub-path points

    # Metrics
    length_mm:                float = 0.0
    nominal_diameter_mm:      float = 0.0         # user-requested stent Ø
    mean_vessel_diameter_mm:  float = 0.0         # actual vessel Ø at segment
    coverage_ratio:           float = 1.0         # stent_r / vessel_r (1.0 = perfect fit)

    def summary(self) -> str:
        return (
            f"Longitud: {self.length_mm:.1f} mm  |  "
            f"Ø stent: {self.nominal_diameter_mm:.2f} mm  |  "
            f"Ø vaso: {self.mean_vessel_diameter_mm:.2f} mm  |  "
            f"Cobertura: {self.coverage_ratio:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Main entry point                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def deploy_stent_on_centerline(
    centerline: CenterlineResult,
    stent_diameter_mm: float,
    start_arc_mm: float | None = None,
    end_arc_mm:   float | None = None,
    circle_resolution: int   = 24,
    braid: bool              = True,
    braid_count: int         = 6,
) -> DeployedStentResult:
    """
    Deploy a virtual stent along a centreline segment.

    Parameters
    ----------
    centerline : CenterlineResult
        Output of :func:`~prospective.processing.centerline.extract_centerline`.
    stent_diameter_mm : float
        Nominal stent diameter in mm (outer diameter).
    start_arc_mm : float | None
        Arc-length start position.  Defaults to 0 (start of centreline).
    end_arc_mm : float | None
        Arc-length end position.  Defaults to full length.
    circle_resolution : int
        Number of sides for each cross-section polygon (≥ 8).
    braid : bool
        If True, add helical braid wires on the tube surface.
    braid_count : int
        Number of helical braid wires (per direction, i.e. 2 × braid_count wires).

    Returns
    -------
    DeployedStentResult
    """
    pts    = centerline.points   # (N, 3)
    radii  = centerline.radii    # (N,)
    N      = len(pts)
    if N < 2:
        raise ValueError("La línea central debe tener al menos 2 puntos.")

    # ── Arc-length table ────────────────────────────────────────────────── #
    seg_lens  = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total_len = float(seg_lens[-1])

    s0 = float(start_arc_mm) if start_arc_mm is not None else 0.0
    s1 = float(end_arc_mm)   if end_arc_mm   is not None else total_len

    s0 = max(0.0, min(s0, total_len))
    s1 = max(s0, min(s1, total_len))

    if s1 - s0 < 0.5:
        raise ValueError("El segmento seleccionado es demasiado corto (< 0.5 mm).")

    # ── Extract sub-segment ─────────────────────────────────────────────── #
    # Find indices within [s0, s1]
    mask      = (seg_lens >= s0 - 1e-6) & (seg_lens <= s1 + 1e-6)
    sub_arcs  = seg_lens[mask]
    sub_pts   = pts[mask]
    sub_radii = radii[mask]

    # Interpolate exact endpoints if needed
    def _interp_at(arc_pos):
        i = int(np.searchsorted(seg_lens, arc_pos, side='left'))
        i = max(1, min(i, N - 1))
        t = (arc_pos - seg_lens[i - 1]) / max(seg_lens[i] - seg_lens[i - 1], 1e-9)
        p = pts[i - 1] + t * (pts[i] - pts[i - 1])
        r = radii[i - 1] + t * (radii[i] - radii[i - 1])
        return p, r, arc_pos

    p_start, r_start, _ = _interp_at(s0)
    p_end,   r_end,   _ = _interp_at(s1)

    sub_pts   = np.vstack([p_start, sub_pts, p_end])
    sub_radii = np.concatenate([[r_start], sub_radii, [r_end]])
    sub_arcs  = np.concatenate([[s0], sub_arcs, [s1]])

    # Deduplicate (in case start/end already in mask)
    _, unique = np.unique(sub_arcs.round(6), return_index=True)
    sub_pts   = sub_pts[unique]
    sub_radii = sub_radii[unique]
    sub_arcs  = sub_arcs[unique]
    M = len(sub_pts)

    if M < 2:
        raise ValueError("No hay suficientes puntos en el segmento seleccionado.")

    # ── Stent and vessel metrics ────────────────────────────────────────── #
    stent_radius_mm = stent_diameter_mm / 2.0
    mean_vessel_r   = float(sub_radii.mean()) if len(sub_radii) else 0.0
    coverage_ratio  = stent_radius_mm / mean_vessel_r if mean_vessel_r > 1e-6 else 1.0
    length_mm       = float(sub_arcs[-1] - sub_arcs[0])

    logger.info(
        "Deploying stent: Ø=%.1f mm, arc %.1f–%.1f mm, coverage=%.2f",
        stent_diameter_mm, s0, s1, coverage_ratio,
    )

    # ── Compute transport frames ────────────────────────────────────────── #
    tangents, normals, binormals = _transport_frames(sub_pts)

    # ── Build tube geometry ─────────────────────────────────────────────── #
    tube_pd = _build_tube(
        sub_pts, tangents, normals, binormals,
        stent_radius_mm, circle_resolution,
    )

    # ── Optionally add braid wires ──────────────────────────────────────── #
    if braid and braid_count > 0:
        braid_pd = _build_braids(
            sub_pts, tangents, normals, binormals,
            stent_radius_mm * 1.02,   # slightly outside tube surface
            braid_count,
        )
        append = vtk.vtkAppendPolyData()
        append.AddInputData(tube_pd)
        append.AddInputData(braid_pd)
        append.Update()
        out_pd = append.GetOutput()
    else:
        out_pd = tube_pd

    return DeployedStentResult(
        stent_poly_data         = out_pd,
        centerline_segment      = sub_pts,
        length_mm               = length_mm,
        nominal_diameter_mm     = stent_diameter_mm,
        mean_vessel_diameter_mm = mean_vessel_r * 2.0,
        coverage_ratio          = coverage_ratio,
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Frenet frame via double-reflection parallel transport                          #
# ──────────────────────────────────────────────────────────────────────────── #

def _transport_frames(
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute tangent / normal / binormal at each of *M* path points using
    double-reflection parallel transport (avoids sudden twisting).

    Returns
    -------
    T, N, B : (M, 3) arrays of unit vectors
    """
    M = len(pts)
    T = np.zeros((M, 3))
    N = np.zeros((M, 3))
    B = np.zeros((M, 3))

    # Tangents via central differences
    T[1:-1] = pts[2:] - pts[:-2]
    T[0]    = pts[1]  - pts[0]
    T[-1]   = pts[-1] - pts[-2]
    norms   = np.linalg.norm(T, axis=1, keepdims=True)
    norms   = np.where(norms < 1e-12, 1.0, norms)
    T      /= norms

    # Seed the first normal perpendicular to T[0]
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(T[0], helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    N[0] = np.cross(T[0], helper)
    N[0] /= np.linalg.norm(N[0])
    B[0]  = np.cross(T[0], N[0])

    # Propagate via double-reflection (Hanson & Ma, 1995)
    for i in range(1, M):
        v1 = pts[i] - pts[i - 1]
        c1 = np.dot(v1, v1)
        if c1 < 1e-12:
            N[i] = N[i - 1]
            B[i] = B[i - 1]
            continue
        r_l = N[i - 1] - (2.0 / c1) * np.dot(v1, N[i - 1]) * v1
        t_l = T[i - 1] - (2.0 / c1) * np.dot(v1, T[i - 1]) * v1
        v2  = T[i] - t_l
        c2  = np.dot(v2, v2)
        if c2 < 1e-12:
            N[i] = r_l
        else:
            N[i] = r_l - (2.0 / c2) * np.dot(v2, r_l) * v2
        n_norm = np.linalg.norm(N[i])
        N[i] /= n_norm if n_norm > 1e-12 else 1.0
        B[i]  = np.cross(T[i], N[i])
        b_norm = np.linalg.norm(B[i])
        B[i] /= b_norm if b_norm > 1e-12 else 1.0

    return T, N, B


# ──────────────────────────────────────────────────────────────────────────── #
# Tube geometry                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def _build_tube(
    pts: np.ndarray,
    T: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    radius: float,
    resolution: int,
) -> vtk.vtkPolyData:
    """Build a swept-tube poly-data following the path."""
    M   = len(pts)
    res = max(8, resolution)
    angles = np.linspace(0, 2 * math.pi, res, endpoint=False)
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)

    # All vertices: M rings × res points
    all_pts = np.zeros((M * res, 3), dtype=np.float64)
    for i in range(M):
        ring = (pts[i]
                + radius * cos_a[:, None] * N[i]
                + radius * sin_a[:, None] * B[i])
        all_pts[i * res: (i + 1) * res] = ring

    # Quad cells connecting adjacent rings
    polys  = vtk.vtkCellArray()
    for i in range(M - 1):
        for j in range(res):
            j1  = (j + 1) % res
            p00 =  i      * res + j
            p01 =  i      * res + j1
            p10 = (i + 1) * res + j
            p11 = (i + 1) * res + j1
            polys.InsertNextCell(4)
            polys.InsertCellPoint(p00)
            polys.InsertCellPoint(p01)
            polys.InsertCellPoint(p11)
            polys.InsertCellPoint(p10)

    # End caps (simple fan triangulation)
    for cap_i, sign in [(0, -1), (M - 1, 1)]:
        centre_id = M * res + (0 if sign < 0 else 1)
        for j in range(res):
            j1 = (j + 1) % res
            polys.InsertNextCell(3)
            if sign < 0:
                polys.InsertCellPoint(cap_i * res + j1)
                polys.InsertCellPoint(cap_i * res + j)
            else:
                polys.InsertCellPoint(cap_i * res + j)
                polys.InsertCellPoint(cap_i * res + j1)
            polys.InsertCellPoint(centre_id)

    cap_pts = np.array([pts[0], pts[-1]], dtype=np.float64)
    all_pts_full = np.vstack([all_pts, cap_pts])

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(all_pts_full, deep=True))

    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetPolys(polys)

    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(pd)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.SplittingOff()
    normals_filter.Update()
    return normals_filter.GetOutput()


# ──────────────────────────────────────────────────────────────────────────── #
# Braid wire pattern                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def _build_braids(
    pts: np.ndarray,
    T: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    radius: float,
    braid_count: int,
    wire_radius: float = 0.12,
) -> vtk.vtkPolyData:
    """
    Generate 2 × braid_count helical wires on the tube surface (Z and S winds).
    Each wire is rendered as a thin tube.
    """
    M        = len(pts)
    arc_lens = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total    = arc_lens[-1]

    append = vtk.vtkAppendPolyData()

    for direction in (+1, -1):         # Z-wind and S-wind
        for k in range(braid_count):
            phase = 2.0 * math.pi * k / braid_count

            wire_pts = vtk.vtkPoints()
            for i in range(M):
                # One full revolution over the stent length
                angle = phase + direction * 2.0 * math.pi * arc_lens[i] / max(total, 1e-6)
                p = (pts[i]
                     + radius * math.cos(angle) * N[i]
                     + radius * math.sin(angle) * B[i])
                wire_pts.InsertNextPoint(*p)

            lines = vtk.vtkCellArray()
            lines.InsertNextCell(M)
            for i in range(M):
                lines.InsertCellPoint(i)

            line_pd = vtk.vtkPolyData()
            line_pd.SetPoints(wire_pts)
            line_pd.SetLines(lines)

            tube = vtk.vtkTubeFilter()
            tube.SetInputData(line_pd)
            tube.SetRadius(wire_radius)
            tube.SetNumberOfSides(6)
            tube.SetCapping(True)
            tube.Update()
            append.AddInputData(tube.GetOutput())

    append.Update()
    return append.GetOutput()
