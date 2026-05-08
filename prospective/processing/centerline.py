"""Vessel centerline extraction — Feature 1 (A-03-03 extension).

Algorithm
---------
1. **Voxelise** the vessel mesh using ``vtkSelectEnclosedPoints`` at a
   configurable resolution (default 0.8 mm/voxel).
2. **Euclidean Distance Transform** (scipy EDT) on the binary mask to obtain
   the local vessel radius at every interior voxel.
3. **Dijkstra shortest path** on the EDT-weighted grid (cost = step_length /
   radius) — minimum-cost path follows the medial axis (tube centre).
4. **Smooth** the raw voxel path with a cardinal B-spline, re-sample at 0.5 mm
   intervals and compute per-point radii.
5. Return a :class:`CenterlineResult` with geometry + clinical metrics.

Clinical outputs
----------------
* ``arc_length_mm``  — total path length along the centreline
* ``chord_length_mm`` — straight-line distance source → target
* ``tortuosity``      — arc / chord  (1.0 = perfectly straight)
* ``tortuosity_index`` — (arc − chord) / chord  (0.0 = straight)
* ``radii``          — array of local vessel radii (≈ parent artery diameter / 2)
* ``mean_radius_mm``, ``min_radius_mm``, ``max_radius_mm``
"""
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import vtk
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────── #
# Result dataclass                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class CenterlineResult:
    """Output of :func:`extract_centerline`."""

    # Geometry
    points: np.ndarray       # (N, 3) world-space mm coordinates
    radii:  np.ndarray       # (N,)   local radius in mm

    # Scalar metrics
    arc_length_mm:    float  = 0.0
    chord_length_mm:  float  = 0.0
    tortuosity:       float  = 1.0   # arc / chord (≥1)
    tortuosity_index: float  = 0.0   # (arc−chord) / chord (≥0)
    mean_radius_mm:   float  = 0.0
    min_radius_mm:    float  = 0.0
    max_radius_mm:    float  = 0.0

    # Raw vtkPolyData for rendering (set by build_centerline_actor)
    poly_data: vtk.vtkPolyData | None = field(default=None, repr=False)

    def summary(self) -> str:
        return (
            f"Longitud: {self.arc_length_mm:.1f} mm  |  "
            f"Tortuosidad: {self.tortuosity:.3f}  |  "
            f"Ø medio: {self.mean_radius_mm*2:.2f} mm"
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Main extractor                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class CenterlineExtractor:
    """
    Extract the medial-axis centreline of a vessel mesh between two endpoints.

    Parameters
    ----------
    voxel_size_mm : float
        Grid resolution for voxelisation and EDT.  0.6–0.8 mm gives a good
        balance between accuracy and speed for cerebral vasculature.
    smooth_sigma : float
        Gaussian σ (in mm) applied to the final point path to remove
        voxel-staircase artefacts.
    resample_spacing_mm : float
        Arc-length re-sampling interval for the output centreline.
    progress_cb : callable | None
        Optional ``progress_cb(fraction: float)`` called during computation.
    """

    def __init__(
        self,
        voxel_size_mm: float = 0.8,
        smooth_sigma: float = 1.5,
        resample_spacing_mm: float = 0.5,
        progress_cb: Callable[[float], None] | None = None,
    ) -> None:
        self.voxel_size_mm     = voxel_size_mm
        self.smooth_sigma      = smooth_sigma
        self.resample_spacing_mm = resample_spacing_mm
        self._progress         = progress_cb or (lambda _: None)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        poly_data: vtk.vtkPolyData,
        source_mm: tuple[float, float, float],
        target_mm: tuple[float, float, float],
    ) -> CenterlineResult:
        """
        Compute the centreline from *source_mm* to *target_mm* through the
        interior of *poly_data*.

        Raises
        ------
        ValueError
            If source or target are outside the mesh, or the path cannot be
            found (disconnected mesh or points too close together).
        """
        logger.info(
            "Centerline: source=%.1f,%.1f,%.1f  target=%.1f,%.1f,%.1f",
            *source_mm, *target_mm,
        )

        # ── Step 1: voxelise ──────────────────────────────────────────── #
        self._progress(0.05)
        mask, origin, vs = self._voxelise(poly_data)
        logger.debug("Voxel grid: %s  inside=%d", mask.shape, mask.sum())

        # ── Step 2: EDT ───────────────────────────────────────────────── #
        self._progress(0.25)
        dt = distance_transform_edt(mask).astype(np.float32) * vs
        # dt values are now in mm

        # ── Step 3: map world coords → voxel indices ───────────────────  #
        src_idx = self._world_to_idx(source_mm, origin, vs, mask.shape)
        tgt_idx = self._world_to_idx(target_mm, origin, vs, mask.shape)

        src_idx = self._snap_to_best_center(src_idx, mask, dt)
        tgt_idx = self._snap_to_best_center(tgt_idx, mask, dt)

        # ── Step 4: Dijkstra on EDT-weighted grid ─────────────────────── #
        self._progress(0.35)
        raw_path = self._dijkstra(dt, mask, src_idx, tgt_idx)
        if len(raw_path) < 2:
            raise ValueError(
                "No se encontró un camino entre los puntos indicados. "
                "Verifique que ambos puntos estén dentro del vaso."
            )
        logger.debug("Raw path length: %d voxels", len(raw_path))

        # ── Step 5: smooth & resample ─────────────────────────────────── #
        self._progress(0.80)
        pts_world = self._path_to_world(raw_path, origin, vs)
        pts_smooth = self._smooth_path(pts_world)
        pts_final, radii = self._resample_with_radii(pts_smooth, dt, origin, vs, mask.shape)

        # ── Step 6: metrics ────────────────────────────────────────────── #
        self._progress(0.95)
        result = self._compute_metrics(pts_final, radii)

        # Build polydata for rendering
        result.poly_data = self._build_polydata(pts_final, radii)

        self._progress(1.0)
        logger.info("Centerline done: %s", result.summary())
        return result

    # ------------------------------------------------------------------ #
    # Step 1 — voxelisation                                                #
    # ------------------------------------------------------------------ #

    def _voxelise(
        self,
        poly_data: vtk.vtkPolyData,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (binary_mask, origin_mm, voxel_size_mm)."""
        vs = self.voxel_size_mm
        bds = poly_data.GetBounds()          # (xmin,xmax,ymin,ymax,zmin,zmax)
        pad = vs * 2

        origin = np.array([bds[0] - pad, bds[2] - pad, bds[4] - pad])
        nx = max(4, int((bds[1] - bds[0] + 2 * pad) / vs) + 1)
        ny = max(4, int((bds[3] - bds[2] + 2 * pad) / vs) + 1)
        nz = max(4, int((bds[5] - bds[4] + 2 * pad) / vs) + 1)

        # Build a point cloud on the regular grid
        xi = origin[0] + np.arange(nx) * vs
        yi = origin[1] + np.arange(ny) * vs
        zi = origin[2] + np.arange(nz) * vs
        gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')   # (nx,ny,nz)
        pts_arr = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)

        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(numpy_to_vtk(pts_arr, deep=True))
        probe_pd = vtk.vtkPolyData()
        probe_pd.SetPoints(vtk_pts)

        enc = vtk.vtkSelectEnclosedPoints()
        enc.SetInputData(probe_pd)
        enc.SetSurfaceData(poly_data)
        enc.SetTolerance(0.001)
        enc.Update()

        inside_arr = vtk_to_numpy(enc.GetOutput().GetPointData().GetArray("SelectedPoints"))
        mask = inside_arr.reshape((nx, ny, nz)).astype(bool)
        return mask, origin, vs

    # ------------------------------------------------------------------ #
    # Step 3 — coordinate helpers                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _world_to_idx(
        pt_mm: tuple,
        origin: np.ndarray,
        vs: float,
        shape: tuple,
    ) -> tuple[int, int, int]:
        i = int(round((pt_mm[0] - origin[0]) / vs))
        j = int(round((pt_mm[1] - origin[1]) / vs))
        k = int(round((pt_mm[2] - origin[2]) / vs))
        i = max(0, min(i, shape[0] - 1))
        j = max(0, min(j, shape[1] - 1))
        k = max(0, min(k, shape[2] - 1))
        return (i, j, k)

    @staticmethod
    def _snap_to_best_center(
        idx: tuple[int, int, int],
        mask: np.ndarray,
        dt: np.ndarray,
    ) -> tuple[int, int, int]:
        """Snap *idx* to the interior voxel closest to the vessel axis.

        Strategy
        --------
        1. Collect all interior voxels within Chebyshev radius 8 (expands to
           20 if none found).
        2. Among those candidates, keep only the ones whose EDT value is within
           10 % of the local maximum — these are all "well-centred" voxels.
        3. Among those near-maximum voxels, pick the one **closest** (L2) to
           the original *idx*.  This prevents the snap from jumping to the far
           end of a uniformly-centered tube when many voxels share the same
           maximum EDT.
        """
        inside = np.argwhere(mask)
        if len(inside) == 0:
            raise ValueError("Mesh interior is empty — check mesh quality.")

        idx_arr = np.array(idx, dtype=np.float64)
        for radius in (8, 20):
            chebyshev = np.max(np.abs(inside - idx_arr), axis=1)
            candidates = inside[chebyshev <= radius]
            if len(candidates) == 0:
                continue

            dt_vals = dt[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
            max_dt  = dt_vals.max()
            # Keep voxels within 10 % of the best EDT (all "axis-level" voxels)
            near_max = candidates[dt_vals >= max_dt * 0.90]
            # Among those, pick the closest to the original user point
            nm_arr = near_max.astype(np.float64)
            sq_dists = np.sum((nm_arr - idx_arr) ** 2, axis=1)
            best = near_max[np.argmin(sq_dists)]
            return tuple(best)

        # Ultimate fallback: nearest interior voxel
        dists = np.sum((inside - idx_arr) ** 2, axis=1)
        nearest = inside[np.argmin(dists)]
        return tuple(nearest)

    # ------------------------------------------------------------------ #
    # Step 4 — Dijkstra on EDT-weighted grid                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dijkstra(
        dt: np.ndarray,
        mask: np.ndarray,
        src: tuple[int, int, int],
        tgt: tuple[int, int, int],
    ) -> list[tuple[int, int, int]]:
        """
        26-connectivity Dijkstra.

        Edge cost = step_distance / (dt[neighbour] + ε)
        This biases the path toward high-radius voxels (vessel axis).
        """
        shape  = dt.shape
        INF    = float("inf")
        EPS    = 0.1           # mm — prevents division by zero

        # cost array
        cost   = np.full(shape, INF, dtype=np.float32)
        cost[src] = 0.0

        # previous-node tracking (flat index)
        stride = (shape[1] * shape[2], shape[2], 1)
        def flat(idx): return idx[0]*stride[0] + idx[1]*stride[1] + idx[2]

        prev   = np.full(shape[0] * shape[1] * shape[2], -1, dtype=np.int32)
        heap   = [(0.0, src)]

        # Pre-compute 26-neighbourhood offsets + step distances
        offsets = [
            (di, dj, dk)
            for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
            if not (di == dj == dk == 0)
        ]
        step_d = {o: math.sqrt(o[0]**2 + o[1]**2 + o[2]**2) for o in offsets}

        while heap:
            c, cur = heapq.heappop(heap)
            if c > cost[cur]:
                continue
            if cur == tgt:
                break

            ci, cj, ck = cur
            for di, dj, dk in offsets:
                ni, nj, nk = ci + di, cj + dj, ck + dk
                if not (0 <= ni < shape[0] and
                        0 <= nj < shape[1] and
                        0 <= nk < shape[2]):
                    continue
                if not mask[ni, nj, nk]:
                    continue
                r = float(dt[ni, nj, nk])
                new_c = c + step_d[(di, dj, dk)] / (r + EPS)
                if new_c < cost[ni, nj, nk]:
                    cost[ni, nj, nk] = new_c
                    prev[flat((ni, nj, nk))] = flat(cur)
                    heapq.heappush(heap, (new_c, (ni, nj, nk)))

        # Back-track
        if cost[tgt] == INF:
            return []

        path: list[tuple[int, int, int]] = []
        cur_flat = flat(tgt)
        while cur_flat >= 0:
            i = cur_flat // stride[0]
            j = (cur_flat % stride[0]) // stride[1]
            k = cur_flat % stride[1]
            path.append((i, j, k))
            cur_flat = int(prev[cur_flat])
        path.reverse()
        return path

    # ------------------------------------------------------------------ #
    # Step 5 — smooth & resample                                           #
    # ------------------------------------------------------------------ #

    def _path_to_world(
        self,
        path: list[tuple[int, int, int]],
        origin: np.ndarray,
        vs: float,
    ) -> np.ndarray:
        arr = np.array(path, dtype=np.float32)   # (N, 3)
        return arr * vs + origin                  # world mm

    def _smooth_path(self, pts: np.ndarray) -> np.ndarray:
        """Per-axis Gaussian smoothing to remove staircase artefacts."""
        sigma_voxels = self.smooth_sigma / self.voxel_size_mm
        smoothed = np.column_stack([
            gaussian_filter1d(pts[:, ax], sigma_voxels, mode="nearest")
            for ax in range(3)
        ])
        # Restore exact endpoint positions — Gaussian boundary padding (mode='nearest')
        # shifts the first/last few points, creating short mis-oriented segments that
        # cause vtkTubeFilter to produce twisted end-caps.
        smoothed[0]  = pts[0]
        smoothed[-1] = pts[-1]
        return smoothed

    def _resample_with_radii(
        self,
        pts: np.ndarray,
        dt: np.ndarray,
        origin: np.ndarray,
        vs: float,
        shape: tuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Arc-length resample to fixed spacing; interpolate EDT radii."""
        spacing = self.resample_spacing_mm
        # Build cumulative arc-length
        diffs  = np.diff(pts, axis=0)
        segs   = np.linalg.norm(diffs, axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(segs)])
        total  = cumlen[-1]
        if total < 1e-3:
            return pts, np.zeros(len(pts))

        n_out  = max(2, int(total / spacing) + 1)
        t_out  = np.linspace(0.0, total, n_out)

        # Linear interpolation for each axis
        resampled = np.column_stack([
            np.interp(t_out, cumlen, pts[:, ax]) for ax in range(3)
        ])

        # Radii from EDT at each resampled point
        radii = np.zeros(n_out, dtype=np.float32)
        for idx, pt in enumerate(resampled):
            vi = int(round((pt[0] - origin[0]) / vs))
            vj = int(round((pt[1] - origin[1]) / vs))
            vk = int(round((pt[2] - origin[2]) / vs))
            vi = max(0, min(vi, shape[0] - 1))
            vj = max(0, min(vj, shape[1] - 1))
            vk = max(0, min(vk, shape[2] - 1))
            radii[idx] = dt[vi, vj, vk]

        # Fix zero radii (resampled point fell outside mask)
        nonzero_mask = radii > 0.0
        if nonzero_mask.any() and not nonzero_mask.all():
            idxs = np.arange(n_out, dtype=float)
            radii = np.interp(idxs, idxs[nonzero_mask], radii[nonzero_mask])
        # Physical floor: half a voxel
        radii = np.maximum(radii, vs * 0.5)

        return resampled, radii

    # ------------------------------------------------------------------ #
    # Step 6 — metrics                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_metrics(
        pts: np.ndarray,
        radii: np.ndarray,
    ) -> CenterlineResult:
        segs       = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc        = float(segs.sum())
        chord_vec  = pts[-1] - pts[0]
        chord      = float(np.linalg.norm(chord_vec))
        tortuosity = arc / chord if chord > 1e-6 else 1.0
        ti         = (arc - chord) / chord if chord > 1e-6 else 0.0

        return CenterlineResult(
            points           = pts,
            radii            = radii,
            arc_length_mm    = arc,
            chord_length_mm  = chord,
            tortuosity       = tortuosity,
            tortuosity_index = ti,
            mean_radius_mm   = float(radii.mean()) if len(radii) else 0.0,
            min_radius_mm    = float(radii.min())  if len(radii) else 0.0,
            max_radius_mm    = float(radii.max())  if len(radii) else 0.0,
        )

    # ------------------------------------------------------------------ #
    # VTK poly data for rendering                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_polydata(
        pts: np.ndarray,
        radii: np.ndarray,
    ) -> vtk.vtkPolyData:
        """Return a vtkPolyData (polyline) with a 'Radius' point scalar."""
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(numpy_to_vtk(pts.astype(np.float64), deep=True))

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(pts))
        for i in range(len(pts)):
            lines.InsertCellPoint(i)

        pd = vtk.vtkPolyData()
        pd.SetPoints(vtk_pts)
        pd.SetLines(lines)

        r_arr = numpy_to_vtk(radii.astype(np.float32), deep=True)
        r_arr.SetName("Radius")
        pd.GetPointData().AddArray(r_arr)
        pd.GetPointData().SetActiveScalars("Radius")
        return pd


# ──────────────────────────────────────────────────────────────────────────── #
# Convenience function                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def extract_centerline(
    poly_data: vtk.vtkPolyData,
    source_mm: tuple[float, float, float],
    target_mm: tuple[float, float, float],
    voxel_size_mm: float = 0.8,
    progress_cb: Callable[[float], None] | None = None,
) -> CenterlineResult:
    """Convenience wrapper around :class:`CenterlineExtractor`."""
    return CenterlineExtractor(
        voxel_size_mm=voxel_size_mm,
        progress_cb=progress_cb,
    ).extract(poly_data, source_mm, target_mm)
