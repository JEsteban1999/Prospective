"""Aneurysm morphometric analysis — F-03 / A-03-10.

Computes the standard clinical metrics used for cerebral aneurysm
clipping pre-operative planning:

  • Volume (mm³) and surface area (mm²)
  • Maximum diameter (mm) — largest bounding-box dimension
  • Neck diameter estimate (mm) — minimum cross-section along the
    principal axis using iterative plane slicing
  • Dome height (mm) — extent from neck plane to apex
  • Dome-to-neck ratio (DNR) — max_diameter / neck_diameter
  • Aspect ratio (AR) — dome_height / neck_diameter
  • Equivalent sphere diameter (mm) — from volume
  • Compactness (Wadell sphericity) — π^(1/3)·(6V)^(2/3)/A, 1 = perfect sphere

Shape-complexity indices (Ankyras-style, Dhar 2008 / Raghavan 2005):
  • Bottleneck Factor (BF) — max_dome_diam / neck_diam
  • Undulation Index (UI) — 1 − V_sac / V_convex_hull
  • Ellipticity Index (EI) — 1 − (18π)^(1/3)·V^(2/3) / A
  • Non-Sphericity Index (NSI) — 1 − Wadell_sphericity
  • Size Ratio (SR) — max_diam / parent_artery_diam  (0 if unknown)

All computations are on a single closed vtkPolyData mesh
(typically the isolated aneurysm from AneurysmDetector).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import vtk
from vtkmodules.util import numpy_support as ns

logger = logging.getLogger(__name__)


@dataclass
class MorphometricResult:
    """All morphometric measurements for one aneurysm candidate."""

    # ── Volumetric ────────────────────────────────────────────────────── #
    volume_mm3: float          # enclosed volume
    surface_area_mm2: float    # total surface area
    eq_sphere_diam_mm: float   # diameter of volume-equivalent sphere

    # ── Bounding-box dimensions ───────────────────────────────────────── #
    bbox_l_mm: float           # longest axis (≈ max diameter)
    bbox_w_mm: float           # intermediate axis
    bbox_h_mm: float           # shortest axis
    max_diameter_mm: float     # = bbox_l_mm

    # ── Neck / dome (via plane slicing) ──────────────────────────────── #
    neck_diameter_mm: float    # minimum cross-section along principal axis
    dome_height_mm: float      # distance from neck plane to apex
    neck_plane_pos: float      # parametric position along principal axis [0–1]

    # ── Clinical ratios ───────────────────────────────────────────────── #
    dome_to_neck_ratio: float  # max_diameter / neck_diameter  (DNR; rupture risk ↑ if >1.6)
    aspect_ratio: float        # dome_height  / neck_diameter  (AR;  risk ↑ if >1.3)
    compactness: float         # Wadell sphericity [0–1]; 1 = perfect sphere

    # ── Shape-complexity indices (Ankyras-style) ──────────────────────── #
    bottleneck_factor:  float = 0.0  # max_dome_diam / neck_diam  (BF > 1.5 → wide neck)
    undulation_index:   float = 0.0  # 1 − V_sac / V_convex  (UI > 0.15 → irregular dome)
    ellipticity_index:  float = 0.0  # 1 − (18π)^(1/3)·V^(2/3)/A  (EI; higher = non-spherical)
    non_sphericity_idx: float = 0.0  # 1 − Wadell_sphericity  (NSI; 0 = perfect sphere)
    size_ratio:         float = 0.0  # max_diam / parent_artery_diam  (0 if unknown)

    # ── PCA axes ─────────────────────────────────────────────────────── #
    principal_axis: tuple[float, float, float] = field(default=(0.0, 0.0, 1.0))
    centroid: tuple[float, float, float]       = field(default=(0.0, 0.0, 0.0))

    # ── Derived labels ────────────────────────────────────────────────── #
    @property
    def rupture_risk_label(self) -> str:
        """Simple heuristic risk label (not a clinical diagnosis).

        Incorporates DNR, AR, UI, and SR (when available).
        Thresholds based on Dhar 2008, Raghavan 2005, Greving 2014.
        """
        sr_alto     = self.size_ratio > 0 and self.size_ratio >= 3.0
        sr_moderado = self.size_ratio > 0 and self.size_ratio >= 2.0
        if (self.aspect_ratio >= 1.6 or self.dome_to_neck_ratio >= 2.0
                or self.undulation_index >= 0.25 or sr_alto):
            return "Alto"
        if (self.aspect_ratio >= 1.3 or self.dome_to_neck_ratio >= 1.6
                or self.ellipticity_index >= 0.35
                or self.undulation_index >= 0.10 or sr_moderado):
            return "Moderado"
        return "Bajo"

    def to_dict(self) -> dict[str, float | str]:
        d: dict[str, float | str] = {
            "Volumen (mm³)":               round(self.volume_mm3, 2),
            "Área superficie (mm²)":       round(self.surface_area_mm2, 2),
            "Diámetro esfera eq. (mm)":    round(self.eq_sphere_diam_mm, 2),
            "Diám. máximo (mm)":           round(self.max_diameter_mm, 2),
            "Diám. cuello (mm)":           round(self.neck_diameter_mm, 2),
            "Altura domo (mm)":            round(self.dome_height_mm, 2),
            "Relación domo/cuello (DNR)":  round(self.dome_to_neck_ratio, 3),
            "Aspect ratio (AR)":           round(self.aspect_ratio, 3),
            "Esfericidad (Wadell)":        round(self.compactness, 4),
            "Bottleneck Factor (BF)":      round(self.bottleneck_factor, 3),
            "Undulation Index (UI)":       round(self.undulation_index, 4),
            "Ellipticity Index (EI)":      round(self.ellipticity_index, 4),
            "Non-Sphericity Index (NSI)":  round(self.non_sphericity_idx, 4),
            "Riesgo ruptura (heurístico)": self.rupture_risk_label,
        }
        if self.size_ratio > 0:
            d["Size Ratio (SR)"] = round(self.size_ratio, 3)
        return d


class MorphometricAnalyzer:
    """
    Compute morphometric indices from a vtkPolyData aneurysm mesh.

    Parameters
    ----------
    n_slices:
        Number of cross-sections sampled along the principal axis
        for neck detection.  Default 40.
    neck_search_fraction:
        Fraction of the dome length (from the base) within which to
        search for the minimum cross-section (neck).  Default 0.40
        (search in the basal 40 % of the aneurysm).
    """

    def __init__(
        self,
        n_slices: int = 40,
        neck_search_fraction: float = 0.40,
    ) -> None:
        self.n_slices = n_slices
        self.neck_search_fraction = neck_search_fraction

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze(self, poly_data: vtk.vtkPolyData) -> MorphometricResult:
        """
        Run all morphometric measurements and return a MorphometricResult.

        The mesh should be the isolated aneurysm polydata from
        AneurysmDetector.  It does not need to be closed — open meshes
        will give approximate volume estimates.
        """
        if poly_data is None or poly_data.GetNumberOfPoints() == 0:
            raise ValueError("Empty mesh — run aneurysm detection first.")

        logger.info(
            "Morphometric analysis started — %d pts / %d tris",
            poly_data.GetNumberOfPoints(), poly_data.GetNumberOfPolys(),
        )

        # ── Volume & surface area ──────────────────────────────────────── #
        volume_mm3, surface_area_mm2 = self._mass_properties(poly_data)

        # ── Bounding box ───────────────────────────────────────────────── #
        bounds = poly_data.GetBounds()   # (xmin,xmax, ymin,ymax, zmin,zmax)
        dims = sorted([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ], reverse=True)
        bbox_l, bbox_w, bbox_h = dims

        # ── Centroid & principal axis (PCA) ───────────────────────────── #
        centroid, principal_axis = self._pca_axis(poly_data)

        # ── Neck detection via plane slicing ──────────────────────────── #
        neck_diam, dome_height, neck_pos = self._find_neck(
            poly_data, centroid, principal_axis
        )

        # ── Derived ───────────────────────────────────────────────────── #
        max_diam    = bbox_l
        eq_diam     = 2.0 * (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0) if volume_mm3 > 0 else 0.0
        dnr         = max_diam / neck_diam  if neck_diam >= 0.1 else 0.0
        ar          = dome_height / neck_diam if neck_diam >= 0.1 else 0.0
        compactness = self._wadell_sphericity(volume_mm3, surface_area_mm2)

        # ── Shape-complexity indices ───────────────────────────────────── #
        bf  = self._bottleneck_factor(poly_data, centroid, principal_axis,
                                      neck_pos, neck_diam)
        ui  = self._undulation_index(poly_data, volume_mm3)
        ei  = self._ellipticity_index(volume_mm3, surface_area_mm2)
        nsi = max(0.0, 1.0 - compactness)

        result = MorphometricResult(
            volume_mm3         = volume_mm3,
            surface_area_mm2   = surface_area_mm2,
            eq_sphere_diam_mm  = eq_diam,
            bbox_l_mm          = bbox_l,
            bbox_w_mm          = bbox_w,
            bbox_h_mm          = bbox_h,
            max_diameter_mm    = max_diam,
            neck_diameter_mm   = neck_diam,
            dome_height_mm     = dome_height,
            neck_plane_pos     = neck_pos,
            dome_to_neck_ratio = dnr,
            aspect_ratio       = ar,
            compactness        = compactness,
            bottleneck_factor  = bf,
            undulation_index   = ui,
            ellipticity_index  = ei,
            non_sphericity_idx = nsi,
            principal_axis     = tuple(principal_axis),
            centroid           = tuple(centroid),
        )

        logger.info(
            "Morphometrics done — V=%.1f mm³  Ø_max=%.1f mm  "
            "Ø_neck=%.1f mm  DNR=%.2f  AR=%.2f  BF=%.2f  UI=%.3f  EI=%.3f",
            volume_mm3, max_diam, neck_diam, dnr, ar, bf, ui, ei,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mass_properties(poly_data: vtk.vtkPolyData) -> tuple[float, float]:
        """Return (volume_mm3, surface_area_mm2) using vtkMassProperties."""
        # vtkMassProperties requires a closed triangulated surface
        # We triangulate first (in case quads exist)
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly_data)
        tri.Update()

        mp = vtk.vtkMassProperties()
        mp.SetInputConnection(tri.GetOutputPort())
        mp.Update()

        vol  = float(mp.GetVolume())
        area = float(mp.GetSurfaceArea())

        # Guard: open meshes can give vol < 0
        return abs(vol), area

    @staticmethod
    def _pca_axis(poly_data: vtk.vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
        """Return (centroid, principal_axis) from point-cloud PCA."""
        pts = ns.vtk_to_numpy(poly_data.GetPoints().GetData()).astype(np.float64)
        centroid = pts.mean(axis=0)
        cov = np.cov((pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Largest eigenvalue → principal axis
        principal = eigvecs[:, np.argmax(eigvals)]
        return centroid, principal

    def _find_neck(
        self,
        poly_data: vtk.vtkPolyData,
        centroid: np.ndarray,
        axis: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Slice the mesh with planes perpendicular to *axis* and find the
        plane with minimum intersection perimeter (= neck estimate).

        Returns (neck_diameter_mm, dome_height_mm, neck_pos_fraction).
        """
        pts = ns.vtk_to_numpy(poly_data.GetPoints().GetData()).astype(np.float64)
        # Project all points onto principal axis
        proj = (pts - centroid) @ axis
        p_min, p_max = proj.min(), proj.max()
        extent = p_max - p_min

        if extent < 1e-3:
            # Degenerate mesh
            bounds = poly_data.GetBounds()
            fallback = min(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            )
            return fallback, extent, 0.0

        # Sample cross-sections in the basal fraction
        search_end = p_min + extent * self.neck_search_fraction
        positions  = np.linspace(p_min, search_end, self.n_slices)

        best_perim  = float("inf")
        best_pos    = p_min
        best_diam   = extent  # fallback

        for pos in positions:
            plane_origin = centroid + pos * axis
            perim, diam  = self._slice_perimeter(poly_data, plane_origin, axis)
            if perim > 0 and perim < best_perim:
                best_perim = perim
                best_pos   = pos
                best_diam  = diam

        dome_height = p_max - best_pos
        neck_pos    = (best_pos - p_min) / extent if extent > 0 else 0.0

        return best_diam, dome_height, float(neck_pos)

    @staticmethod
    def _slice_perimeter(
        poly_data: vtk.vtkPolyData,
        origin: np.ndarray,
        normal: np.ndarray,
    ) -> tuple[float, float]:
        """
        Cut *poly_data* with a plane and return (perimeter_mm, equiv_diam_mm).
        Returns (0, 0) if the plane does not intersect the mesh.
        """
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin.tolist())
        plane.SetNormal(normal.tolist())

        cutter = vtk.vtkCutter()
        cutter.SetInputData(poly_data)
        cutter.SetCutFunction(plane)
        cutter.GenerateValues(1, 0.0, 0.0)
        cutter.Update()
        cut = cutter.GetOutput()

        if cut.GetNumberOfPoints() < 3:
            return 0.0, 0.0

        # Perimeter = sum of all edge lengths in the contour
        pts  = ns.vtk_to_numpy(cut.GetPoints().GetData()).astype(np.float64)
        n    = pts.shape[0]
        diff = pts - np.roll(pts, -1, axis=0)
        perim = float(np.sum(np.linalg.norm(diff, axis=1)))

        # Equivalent circle diameter from perimeter: D = P / π
        diam = perim / math.pi if perim > 0 else 0.0
        return perim, diam

    # ------------------------------------------------------------------ #
    # Shape-complexity indices                                            #
    # ------------------------------------------------------------------ #

    def _bottleneck_factor(
        self,
        poly_data: vtk.vtkPolyData,
        centroid: np.ndarray,
        axis: np.ndarray,
        neck_pos_frac: float,
        neck_diam: float,
    ) -> float:
        """BF = max dome cross-section diameter / neck diameter.

        Searches for the widest slice in the dome region (above neck plane).
        BF > 1.5 suggests wide-neck aneurysm that may need stent assistance.
        """
        if neck_diam < 0.1:
            return 0.0

        pts = ns.vtk_to_numpy(poly_data.GetPoints().GetData()).astype(np.float64)
        proj = (pts - centroid) @ axis
        p_min, p_max = proj.min(), proj.max()
        extent = p_max - p_min
        if extent < 1e-3:
            return 0.0

        # Dome starts just above the neck plane, ends near the apex
        neck_abs   = p_min + neck_pos_frac * extent
        dome_start = neck_abs + extent * 0.05   # 5 % margin above neck
        dome_end   = p_max   - extent * 0.02

        if dome_start >= dome_end:
            return 0.0

        positions   = np.linspace(dome_start, dome_end, 30)
        max_dome_d  = neck_diam  # floor at neck (BF ≥ 1 by definition)

        for pos in positions:
            origin = centroid + pos * axis
            _, d   = self._slice_perimeter(poly_data, origin, axis)
            if d > max_dome_d:
                max_dome_d = d

        return max_dome_d / neck_diam

    @staticmethod
    def _undulation_index(poly_data: vtk.vtkPolyData, volume_mm3: float) -> float:
        """UI = 1 − V_sac / V_convex_hull.

        0 for a perfectly convex shape (sphere/ellipsoid).
        Higher values indicate surface indentations / blebs.
        Requires scipy; returns 0.0 if unavailable.
        """
        try:
            from scipy.spatial import ConvexHull  # type: ignore[import]
            pts = ns.vtk_to_numpy(
                poly_data.GetPoints().GetData()
            ).astype(np.float64)
            if pts.shape[0] < 4:
                return 0.0
            hull = ConvexHull(pts)
            convex_vol = float(hull.volume)
            if convex_vol <= 0:
                return 0.0
            return float(max(0.0, 1.0 - volume_mm3 / convex_vol))
        except Exception:
            return 0.0

    @staticmethod
    def _ellipticity_index(volume: float, area: float) -> float:
        """EI = 1 − (18π)^(1/3) · V^(2/3) / A.

        Measures deviation from a sphere (Dhar et al. 2008).
        For a perfect sphere EI ≈ 0.207; higher = more elongated / irregular.
        """
        if area <= 0 or volume <= 0:
            return 0.0
        return 1.0 - (18.0 * math.pi) ** (1.0 / 3.0) * volume ** (2.0 / 3.0) / area

    @staticmethod
    def _wadell_sphericity(volume: float, area: float) -> float:
        """Wadell compactness: π^(1/3) · (6V)^(2/3) / A."""
        if area <= 0 or volume <= 0:
            return 0.0
        return math.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area
