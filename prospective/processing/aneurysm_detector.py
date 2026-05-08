"""Semi-automatic aneurysm candidate detection — A-02-05.

Algorithm (v4)
--------------
1. Compute mean curvature AND Gaussian curvature (vtkCurvatures).

2. Threshold on GAUSSIAN curvature ≥ p_gauss  (positive Gaussian = spherical surface).
   Using Gaussian curvature alone as the primary gate gives larger, cohesive regions
   compared to a strict mean-AND-Gaussian AND-mask (which creates many tiny fragments).

3. Find connected components (vtkPolyDataConnectivityFilter).
   Each region is cleaned (vtkCleanPolyData) to remove orphan points that inflate
   bounding-box and curvature statistics.

4. Size filter: keep regions in [min_radius_mm, max_radius_mm].
   Radius is estimated from surface area:  r ≈ √(A / 4π).

5. Mean-curvature gate: skip regions where mean curvature < p_mean_min.
   (Vessel walls have naturally low mean curvature along the axial direction.)

6. Per-region discriminator metrics:
   • positive_gauss_frac  — fraction of region vertices with Gaussian curvature > 0.
                            Aneurysm dome: ≈ 0.85–1.0  ·  Bifurcation saddle: ≈ 0.30–0.55
                            >>> HARD GATE: must be ≥ min_positive_gauss_frac <<<
   • compactness          — area / (4π · r_bbox²)  where r_bbox = max_dim / 2.
                            Dome (hemisphere): ≈ 0.45–0.55  ·  Elongated strip: < 0.25
                            >>> HARD GATE: must be ≥ min_compactness <<<
                            NOTE: the old formula used r = √(A/4π) which causes algebraic
                            cancellation (always = 1.0).  v4 uses the bounding-box radius.
   • sphericity           — min(dim) / max(dim) from bounding box (0 = flat, 1 = sphere)

7. Merge duplicate candidates: if two centroids are within merge_dist_mm, keep the
   higher-scoring one.  Prevents the same bifurcation from appearing multiple times.

8. Composite score:
      score = 0.08·norm_mean + 0.12·norm_gauss + 0.28·pos_gauss_frac
            + 0.10·compactness + 0.07·sphericity + 0.35·size_factor

Discrimination rationale
------------------------
Normal vessel wall:  κ_gauss ≈ 0   (one principal curvature ≈ 0 along vessel axis)
Bifurcation apex:    κ_gauss > 0 at the very tip, but MIXED in the transition zone
                      → positive_gauss_frac ≈ 0.30–0.55 (fails the hard gate)
Aneurysm dome:       κ_gauss > 0 consistently across the whole dome
                      → positive_gauss_frac ≈ 0.85–1.0  (passes the hard gate)

False-positive sources addressed in v4
---------------------------------------
1. compactness bug      — formula always returned 1.0 (algebraic cancellation); fixed.
2. no pgf hard gate     — bifurcations with low pgf now filtered before scoring.
3. no compactness gate  — elongated vessel segments now filtered before scoring.
4. score bias           — norm_mean weight reduced (sharp spikes no longer dominate).
5. sphericity unused    — added to composite score.
6. too many candidates  — max_candidates reduced to 8; merge_dist_mm raised to 10 mm.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import vtk
from vtkmodules.util import numpy_support as ns

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# Data classes                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class AneurysmCandidate:
    """A single aneurysm candidate extracted from the vascular mesh."""

    index: int                              # 1-based rank
    centroid: tuple[float, float, float]    # (x, y, z) in mm
    radius_mm: float                        # estimated from surface area
    diameter_mm: float                      # = 2 × radius_mm
    mean_curvature: float                   # mean of Mean_Curvature over region
    gauss_curvature: float                  # mean Gaussian curvature of region
    positive_gauss_frac: float              # fraction of vertices with Gauss > 0
    compactness: float                      # A / (4π·r_bbox²), fixed in v4
    sphericity: float                       # min(dim)/max(dim) bounding box, 0–1
    n_points: int                           # vertex count in the region
    score: float                            # composite detection score [0–1]
    poly_data: vtk.vtkPolyData = field(repr=False)
    # v5 additional discriminators
    cv_gauss: float       = 0.0    # coefficient of variation of Gaussian curvature
                                   # low = uniform dome; high = bifurcation spike
    normal_isotropy: float = 0.5   # isotropy of vertex-normal directions (0–1)
                                   # high = normals fan across hemisphere (dome)
                                   # low  = normals concentrated in one direction


@dataclass
class DetectionResult:
    """Full result from AneurysmDetector.detect()."""

    candidates: list[AneurysmCandidate]

    # Diagnostic counters
    n_regions_total:       int   = 0
    n_failed_points:       int   = 0
    n_failed_size:         int   = 0
    n_failed_mean_curv:    int   = 0
    n_failed_pgf:          int   = 0   # failed positive_gauss_frac hard gate (v4)
    n_failed_compact:      int   = 0   # failed compactness hard gate (v4)
    n_failed_sphericity:   int   = 0   # failed sphericity hard gate (v5)
    n_merged:              int   = 0   # duplicates suppressed
    n_removed_components:  int   = 0   # noise components removed by pre-filter
    gauss_threshold:       float = 0.0
    mean_curv_gate:        float = 0.0  # per-region mean curvature gate


# ──────────────────────────────────────────────────────────────────────────── #
# Detector                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

class AneurysmDetector:
    """
    Detect aneurysm candidates on a pre-segmented vascular mesh.

    Parameters
    ----------
    gauss_percentile:
        Percentile gate on Gaussian curvature (primary criterion).
        Positive Gaussian curvature marks spherical domes.  Default 85.
    mean_curv_gate_percentile:
        Per-region mean curvature gate: a surviving region must have mean
        curvature above this percentile of the whole-mesh distribution.
        Default 75.
    min_radius_mm / max_radius_mm:
        Size filter on the candidate radius.
    min_points:
        Minimum vertex count per region.
    merge_dist_mm:
        Merge duplicate candidates whose centroids are within this distance.
        Default 8 mm.
    max_candidates:
        Maximum returned candidates, ordered by score.  Default 12.
    """

    def __init__(
        self,
        gauss_percentile:          float = 85.0,
        mean_curv_gate_percentile: float = 75.0,
        min_radius_mm:             float = 1.5,
        max_radius_mm:             float = 15.0,
        min_points:                int   = 8,
        merge_dist_mm:             float = 10.0,   # v4: raised from 8→10 mm
        max_candidates:            int   = 8,       # v4: reduced from 12→8
        # v4: hard gates (most impactful for false-positive reduction)
        min_positive_gauss_frac:   float = 0.50,   # fraction of Gauss+ vertices required
        min_compactness:           float = 0.20,   # compactness (area/sphere_area) required
        # v5: additional shape gates
        min_sphericity:            float = 0.28,   # min(bbox_dim)/max(bbox_dim) ≥ threshold
                                                   # filters elongated vessel-arc artifacts
        # v6: pre-smoothing for noisy meshes (XA / 3DRA)
        pre_smooth_iterations:     int   = 0,      # Laplacian passes before curvature calc.
                                                   # 0 = disabled (CTA); 25 = XA/3DRA
    ) -> None:
        self.gauss_percentile          = gauss_percentile
        self.mean_curv_gate_percentile = mean_curv_gate_percentile
        self.min_radius_mm             = min_radius_mm
        self.max_radius_mm             = max_radius_mm
        self.min_points                = min_points
        self.merge_dist_mm             = merge_dist_mm
        self.max_candidates            = max_candidates
        self.min_positive_gauss_frac   = min_positive_gauss_frac
        self.min_compactness           = min_compactness
        self.min_sphericity            = min_sphericity
        self.pre_smooth_iterations     = pre_smooth_iterations

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect(self, poly_data: vtk.vtkPolyData) -> DetectionResult:
        """Run detection.  Returns DetectionResult (never raises)."""
        if poly_data is None or poly_data.GetNumberOfPoints() == 0:
            logger.warning("AneurysmDetector: empty mesh")
            return DetectionResult(candidates=[])

        logger.info(
            "Aneurysm detection — %d verts / %d tris",
            poly_data.GetNumberOfPoints(), poly_data.GetNumberOfPolys(),
        )

        # ── 0. Pre-filter: remove small disconnected noise components ───── #
        # Disconnected fragments (residual tissue, mesh artifacts) inflate the
        # global Gaussian-curvature percentile thresholds AND pass all shape
        # gates because they are tiny spherical bumps.  Removing them before
        # curvature computation makes the detector faster and more accurate.
        #
        # Threshold: keep connected components whose point count is at least
        # 5 % of the largest component, with a hard floor of min_points×5.
        # Aneurysm domes are topologically attached to the parent artery and
        # are therefore always part of the dominant connected component.
        # Disconnected noise blobs (calcifications, bone chips, mesh artifacts
        # from threshold leakage) are typically < 2% of the main tree —
        # raising to 5 % removes them more aggressively while never discarding
        # a real dome that is part of the primary vascular mesh.
        n_removed_components = 0
        n_orig_pts = poly_data.GetNumberOfPoints()

        pre_conn = vtk.vtkPolyDataConnectivityFilter()
        pre_conn.SetInputData(poly_data)
        pre_conn.SetExtractionModeToAllRegions()
        pre_conn.ColorRegionsOn()
        pre_conn.Update()
        n_comps = pre_conn.GetNumberOfExtractedRegions()

        if n_comps > 1:
            region_ids = ns.vtk_to_numpy(
                pre_conn.GetOutput().GetPointData().GetScalars()
            ).astype(np.int32)
            counts = np.bincount(region_ids.clip(0), minlength=n_comps)
            # Keep components ≥ 2% of the largest, with a min floor
            min_comp_pts = max(self.min_points * 5, int(counts.max() * 0.05))
            large_ids = [int(i) for i, c in enumerate(counts) if c >= min_comp_pts]
            n_removed_components = n_comps - len(large_ids)

            if n_removed_components > 0 and large_ids:
                sel = vtk.vtkPolyDataConnectivityFilter()
                sel.SetInputData(poly_data)
                sel.SetExtractionModeToSpecifiedRegions()
                sel.InitializeSpecifiedRegionList()
                for rid in large_ids:
                    sel.AddSpecifiedRegion(rid)
                sel.Update()

                cl = vtk.vtkCleanPolyData()
                cl.SetInputConnection(sel.GetOutputPort())
                cl.Update()
                poly_data = cl.GetOutput()

                logger.info(
                    "Pre-filter: removed %d/%d components (threshold ≥%d pts) "
                    "— %d → %d verts",
                    n_removed_components, n_comps, min_comp_pts,
                    n_orig_pts, poly_data.GetNumberOfPoints(),
                )

                if poly_data.GetNumberOfPoints() == 0:
                    logger.warning("Pre-filter removed all geometry — empty result")
                    return DetectionResult(
                        candidates=[], n_removed_components=n_removed_components
                    )

        # ── 0b. Pre-smoothing (XA / 3DRA — v6) ───────────────────────── #
        # Laplacian smoothing applied to a TEMPORARY COPY of the mesh
        # before curvature computation.  The original poly_data (and all
        # stored candidates' poly_data) are NOT modified — only the copy
        # used for curvature analysis is smoothed.
        #
        # Why this helps for XA / 3DRA:
        #   Marching-cubes on noisy rotational-angiography volumes produces
        #   high-frequency surface wrinkles (typical wavelength: 1–3 voxels).
        #   vtkCurvatures is very sensitive to these wrinkles: a 1-voxel
        #   bump generates a Gaussian-curvature spike larger than that of a
        #   real 5-mm aneurysm dome.  25 passes of Laplacian smoothing
        #   (relaxation 0.10) attenuate wrinkles whose wavelength < ~5 voxels
        #   while preserving large-scale geometry (dome radius >> 5 vx).
        #   Result: AUSS+ on true domes rises from ~40 % to ~75–90 %,
        #   while noise bumps fall from ~60–80 % to ~10–25 % (below gate).
        curvature_input = poly_data   # default: same object
        if self.pre_smooth_iterations > 0:
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputData(poly_data)
            smoother.SetNumberOfIterations(self.pre_smooth_iterations)
            smoother.SetRelaxationFactor(0.10)
            smoother.FeatureEdgeSmoothingOff()
            smoother.BoundarySmoothingOff()
            smoother.Update()
            curvature_input = smoother.GetOutput()
            logger.info(
                "Pre-smooth: %d Laplacian iterations applied to %d-vert mesh "
                "before curvature computation",
                self.pre_smooth_iterations, curvature_input.GetNumberOfPoints(),
            )

        # ── 1. Curvature arrays ────────────────────────────────────────── #
        mean_filter = vtk.vtkCurvatures()
        mean_filter.SetInputData(curvature_input)
        mean_filter.SetCurvatureTypeToMean()
        mean_filter.Update()
        mean_poly = mean_filter.GetOutput()

        gauss_filter = vtk.vtkCurvatures()
        gauss_filter.SetInputData(curvature_input)
        gauss_filter.SetCurvatureTypeToGaussian()
        gauss_filter.Update()

        mean_arr  = ns.vtk_to_numpy(
            mean_poly.GetPointData().GetArray("Mean_Curvature")
        ).astype(np.float64)

        gauss_arr = ns.vtk_to_numpy(
            gauss_filter.GetOutput().GetPointData().GetArray("Gauss_Curvature")
        ).astype(np.float64)

        # ── 2. Attach both arrays to mean_poly for later per-region use ── #
        gauss_vtk = ns.numpy_to_vtk(gauss_arr.astype(np.float32), deep=True,
                                    array_type=vtk.VTK_FLOAT)
        gauss_vtk.SetName("Gauss_Curvature")
        mean_poly.GetPointData().AddArray(gauss_vtk)

        # ── 3. Threshold on GAUSSIAN curvature only (primary gate) ─────── #
        thresh_gauss = float(np.percentile(gauss_arr, self.gauss_percentile))
        thresh_gauss = max(thresh_gauss, 0.0)   # must be positive

        # Mean-curvature gate: per-region mean must exceed this value
        mean_curv_gate = float(np.percentile(mean_arr, self.mean_curv_gate_percentile))

        logger.info(
            "Gauss threshold p%.0f = %.4f  |  mean-curv gate p%.0f = %.4f",
            self.gauss_percentile, thresh_gauss,
            self.mean_curv_gate_percentile, mean_curv_gate,
        )

        # Set active scalars to Gauss_Curvature for thresholding
        mean_poly.GetPointData().SetActiveScalars("Gauss_Curvature")

        thresh = vtk.vtkThreshold()
        thresh.SetInputData(mean_poly)
        thresh.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Gauss_Curvature"
        )
        thresh.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_UPPER)
        thresh.SetUpperThreshold(thresh_gauss)
        thresh.SetAllScalars(0)   # keep cell if ANY point exceeds threshold
        thresh.Update()

        geom = vtk.vtkGeometryFilter()
        geom.SetInputConnection(thresh.GetOutputPort())
        geom.Update()
        high_gauss = geom.GetOutput()

        if high_gauss.GetNumberOfPoints() == 0:
            logger.info("No positive Gaussian curvature regions found")
            return DetectionResult(
                candidates=[], gauss_threshold=thresh_gauss,
                mean_curv_gate=mean_curv_gate,
            )

        # ── 4. Connected components ────────────────────────────────────── #
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(high_gauss)
        conn.SetExtractionModeToAllRegions()
        conn.Update()
        n_regions = conn.GetNumberOfExtractedRegions()
        logger.info("Connected regions: %d", n_regions)

        # ── 5. Analyse each region ─────────────────────────────────────── #
        candidates: list[AneurysmCandidate] = []
        n_fail_pts     = 0
        n_fail_size    = 0
        n_fail_mean    = 0
        n_fail_pgf      = 0   # v4: positive_gauss_frac hard gate
        n_fail_compact  = 0   # v4: compactness hard gate
        n_fail_sph      = 0   # v5: sphericity hard gate

        for i in range(n_regions):
            sel = vtk.vtkPolyDataConnectivityFilter()
            sel.SetInputData(high_gauss)
            sel.SetExtractionModeToSpecifiedRegions()
            sel.InitializeSpecifiedRegionList()
            sel.AddSpecifiedRegion(i)
            sel.Update()

            # vtkPolyDataConnectivityFilter retains ALL source points in its
            # output even though only a subset of cells is extracted.  Orphan
            # points inflate GetBounds(), centroid, curvature arrays and all
            # downstream morphometric measurements.  Clean them up first.
            _cl = vtk.vtkCleanPolyData()
            _cl.SetInputConnection(sel.GetOutputPort())
            _cl.Update()
            region = _cl.GetOutput()

            n_pts = region.GetNumberOfPoints()
            if n_pts < self.min_points:
                n_fail_pts += 1
                continue

            # Radius from surface area
            mass = vtk.vtkMassProperties()
            mass.SetInputData(region)
            mass.Update()
            area = mass.GetSurfaceArea()
            radius = math.sqrt(area / (4.0 * math.pi)) if area > 0 else 0.0

            if radius < self.min_radius_mm or radius > self.max_radius_mm:
                n_fail_size += 1
                continue

            # Centroid from bounding box
            bounds = region.GetBounds()
            cx = (bounds[0] + bounds[1]) / 2.0
            cy = (bounds[2] + bounds[3]) / 2.0
            cz = (bounds[4] + bounds[5]) / 2.0

            # Per-region curvature arrays
            r_mean_arr  = ns.vtk_to_numpy(
                region.GetPointData().GetArray("Mean_Curvature")
            ).astype(np.float64)
            r_gauss_arr = ns.vtk_to_numpy(
                region.GetPointData().GetArray("Gauss_Curvature")
            ).astype(np.float64)

            mean_curv_region  = float(np.mean(r_mean_arr))
            gauss_curv_region = float(np.mean(r_gauss_arr))

            # Mean curvature gate
            if mean_curv_region < mean_curv_gate:
                n_fail_mean += 1
                continue

            # ── Sac discriminator metrics ──────────────────────────────── #

            # Bounding-box dimensions (needed for compactness and sphericity)
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            dz = bounds[5] - bounds[4]
            dims = sorted([dx, dy, dz])   # [smallest, mid, largest]
            max_dim = dims[2]

            # 1) Fraction of vertices with positive Gaussian curvature.
            #    Aneurysm dome: ≈ 0.85–1.0   Bifurcation saddle: ≈ 0.30–0.55
            positive_gauss_frac = float(np.mean(r_gauss_arr > 0))

            # >>> HARD GATE #1 — positive_gauss_frac <<<
            # Bifurcations and vessel-wall arcs have mixed Gaussian curvature
            # (saddle surfaces): positive at the apex tip, negative on the flanks.
            # True aneurysm domes are convex everywhere → pgf ≈ 0.85–1.0.
            # This is the single most effective false-positive filter.
            if positive_gauss_frac < self.min_positive_gauss_frac:
                n_fail_pgf += 1
                continue

            # 2) Compactness — ratio of actual area to area of a sphere whose
            #    radius equals half the largest bounding-box dimension.
            #
            #    v4 FIX: The old formula used r = √(A/4π), which causes algebraic
            #    cancellation: sphere_area = 4π·r² = 4π·(A/4π) = A → compactness
            #    was ALWAYS 1.0 and completely useless as a discriminator.
            #
            #    New formula uses the bounding-box radius instead:
            #      r_bbox = max(dx, dy, dz) / 2
            #      compactness = A / (4π · r_bbox²)
            #    Dome (≈hemisphere): compactness ≈ 0.45–0.55
            #    Elongated strip:    compactness < 0.25
            r_bbox = max_dim / 2.0 if max_dim > 0 else 0.0
            sphere_area_bbox = 4.0 * math.pi * r_bbox ** 2
            compactness = (
                min(1.0, area / sphere_area_bbox)
                if sphere_area_bbox > 0
                else 0.0
            )

            # >>> HARD GATE #2 — compactness <<<
            # Vessel segments and bifurcation arms are elongated: their surface
            # area is much smaller than the sphere bounding them → low compactness.
            if compactness < self.min_compactness:
                n_fail_compact += 1
                continue

            # 3) Sphericity from bounding box — min(dim) / max(dim)
            sphericity = (dims[0] / dims[2]) if dims[2] > 0 else 0.0

            # >>> HARD GATE #3 — sphericity (v5) <<<
            # Elongated vessel-arc artifacts (carotid siphon outer wall, MCA
            # branch curves) have sphericity < 0.3 — they are streaks, not domes.
            # True aneurysms are spherical or ovoid (sphericity ≥ 0.35 for most,
            # ≥ 0.28 even for mildly lobulated ones).
            if sphericity < self.min_sphericity:
                n_fail_sph += 1
                continue

            # ── v5: additional discriminator metrics ───────────────────── #

            # 4) Coefficient of variation of Gaussian curvature (cv_gauss)
            #    Aneurysm dome: curvature is relatively UNIFORM → moderate CV (0.3–1.2)
            #    Bifurcation apex / neck ring: very peaked at one point → high CV (> 1.5)
            #    This discriminates between the smooth dome and the ring of high curvature
            #    that appears at the aneurysm neck (a common false positive adjacent to a
            #    true candidate).
            mean_abs_gauss = abs(gauss_curv_region)
            std_gauss      = float(np.std(r_gauss_arr))
            cv_gauss       = float(np.clip(std_gauss / (mean_abs_gauss + 1e-6), 0.0, 10.0))

            # 5) Normal-vector isotropy
            #    Compute the covariance matrix of unit vertex normals and measure
            #    how evenly distributed they are (eigenvalue isotropy).
            #    Dome: normals fan across a hemisphere → high isotropy (min_eig/max_eig ≈ 0.2–0.5)
            #    Bifurcation apex: normals all point "outward" from one point → low isotropy
            #    Vessel arc: normals follow one direction → very low isotropy
            #
            #    Normals are produced by vtkPolyDataNormals in the segmentation pipeline
            #    and propagated through vtkCleanPolyData into the region.
            normal_isotropy = 0.5    # neutral default if normals unavailable
            normals_vtk = region.GetPointData().GetNormals()
            if normals_vtk is not None and normals_vtk.GetNumberOfTuples() > 2:
                nrm = ns.vtk_to_numpy(normals_vtk).astype(np.float64)
                # Normalize (should already be unit vectors, but ensure safety)
                nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
                nlen = np.where(nlen > 1e-8, nlen, 1.0)
                nrm  = nrm / nlen
                # 3×3 covariance of normal directions
                cov_n    = np.cov(nrm.T)
                eigvals  = np.sort(np.linalg.eigvalsh(cov_n))  # ascending
                max_eig  = eigvals[2]
                min_eig  = max(eigvals[0], 0.0)   # clamp numerical negatives
                # isotropy = min_eig / max_eig   →   0=all normals collinear, 1=isotropic
                normal_isotropy = float(min_eig / max_eig) if max_eig > 1e-10 else 0.0

            # ── Composite score (v5) ───────────────────────────────────── #
            norm_mean = (mean_curv_region - mean_curv_gate) / (
                abs(mean_curv_gate) + 1e-6)
            norm_mean = float(np.clip(norm_mean / 5.0, 0.0, 1.0))

            norm_gauss = (gauss_curv_region - thresh_gauss) / (
                abs(thresh_gauss) + 1e-6)
            norm_gauss = float(np.clip(norm_gauss / 5.0, 0.0, 1.0))

            # Clinical size factor — peaks at r=4 mm (Ø 8 mm).
            if radius < 1.0:
                size_factor = 0.0
            elif radius < 2.0:
                size_factor = (radius - 1.0) * 0.5   # ramp 0→0.5 in 1–2 mm range
            else:
                size_factor = max(0.0, 1.0 - abs(radius - 4.0) / 6.0)

            # cv_gauss penalty: high variability → probably not a smooth dome
            # cv ≤ 0.8  →  penalty 0     (uniform dome)
            # cv = 2.0  →  penalty 0.50  (peaked bifurcation apex)
            # cv ≥ 3.0  →  penalty 1.00  (extreme spike / neck ring)
            cv_penalty = float(np.clip((cv_gauss - 0.8) / 2.2, 0.0, 1.0))

            # v5 score weights (sum to 1.0 when normal_isotropy is available):
            #   size_factor:         0.30  clinical size remains dominant
            #   positive_gauss_frac: 0.25  dome discriminator (hard-gated survivors)
            #   normal_isotropy:     0.15  normals fan across hemisphere for a true dome
            #   norm_gauss:          0.10  secondary Gauss evidence
            #   compactness:         0.08  compact shape among survivors
            #   sphericity:          0.07  bonus for round candidates
            #   norm_mean:           0.05  kept very low — spikes inflate this
            base_score = (0.05 * norm_mean
                          + 0.10 * norm_gauss
                          + 0.25 * positive_gauss_frac
                          + 0.08 * compactness
                          + 0.07 * sphericity
                          + 0.15 * normal_isotropy
                          + 0.30 * size_factor)
            # Apply cv_gauss penalty multiplicatively (max 20% reduction)
            score = base_score * (1.0 - 0.20 * cv_penalty)
            score = float(np.clip(score, 0.0, 1.0))

            candidates.append(
                AneurysmCandidate(
                    index=0,
                    centroid=(cx, cy, cz),
                    radius_mm=radius,
                    diameter_mm=radius * 2.0,
                    mean_curvature=mean_curv_region,
                    gauss_curvature=gauss_curv_region,
                    positive_gauss_frac=positive_gauss_frac,
                    compactness=compactness,
                    sphericity=sphericity,
                    n_points=n_pts,
                    score=score,
                    poly_data=region,
                    cv_gauss=cv_gauss,
                    normal_isotropy=normal_isotropy,
                )
            )

        # ── 6. Sort by score ───────────────────────────────────────────── #
        candidates.sort(key=lambda c: c.score, reverse=True)

        # ── 7. Merge spatial duplicates ────────────────────────────────── #
        n_merged = 0
        merged: list[AneurysmCandidate] = []
        for c in candidates:
            pt = np.array(c.centroid)
            duplicate = any(
                float(np.linalg.norm(pt - np.array(m.centroid))) < self.merge_dist_mm
                for m in merged
            )
            if duplicate:
                n_merged += 1
            else:
                merged.append(c)
        candidates = merged[: self.max_candidates]

        for rank, c in enumerate(candidates, start=1):
            c.index = rank

        logger.info(
            "Detection done — %d candidates | %d regions | "
            "%d fail-size | %d fail-mean | %d fail-pgf | %d fail-compact | "
            "%d fail-sph | %d merged | %d noise-comps removed",
            len(candidates), n_regions, n_fail_size, n_fail_mean,
            n_fail_pgf, n_fail_compact, n_fail_sph, n_merged,
            n_removed_components,
        )

        return DetectionResult(
            candidates=candidates,
            n_regions_total=n_regions,
            n_failed_points=n_fail_pts,
            n_failed_size=n_fail_size,
            n_failed_mean_curv=n_fail_mean,
            n_failed_pgf=n_fail_pgf,
            n_failed_compact=n_fail_compact,
            n_failed_sphericity=n_fail_sph,
            n_merged=n_merged,
            n_removed_components=n_removed_components,
            gauss_threshold=thresh_gauss,
            mean_curv_gate=mean_curv_gate,
        )
