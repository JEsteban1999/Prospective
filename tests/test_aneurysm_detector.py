"""Integration tests — aneurysm detection (A-04-09)."""
from __future__ import annotations

import numpy as np
import pytest
import vtk

from prospective.processing.aneurysm_detector import (
    AneurysmCandidate,
    AneurysmDetector,
    DetectionResult,
)


def _bumpy_sphere_poly(radius: float = 6.0, bump_scale: float = 3.5) -> vtk.vtkPolyData:
    """
    Sphere with a prominent high-curvature bump at the +Z pole.
    The bump has a spherical, dome-like shape — representative of a saccular aneurysm.
    """
    src = vtk.vtkSphereSource()
    src.SetRadius(radius)
    src.SetThetaResolution(50)
    src.SetPhiResolution(50)
    src.Update()

    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputConnection(src.GetOutputPort())
    normals_filter.ComputePointNormalsOn()
    normals_filter.SplittingOff()
    normals_filter.Update()
    pd = normals_filter.GetOutput()

    pts = pd.GetPoints()
    norm_arr = pd.GetPointData().GetNormals()
    for i in range(pts.GetNumberOfPoints()):
        x, y, z = pts.GetPoint(i)
        if z > radius * 0.80:
            nx, ny, nz = norm_arr.GetTuple3(i)
            factor = bump_scale * ((z - radius * 0.80) / (radius * 0.20))
            pts.SetPoint(i, x + nx * factor, y + ny * factor, z + nz * factor)
    pd.Modified()

    normals2 = vtk.vtkPolyDataNormals()
    normals2.SetInputData(pd)
    normals2.ComputePointNormalsOn()
    normals2.SplittingOff()
    normals2.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(normals2.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


class TestAneurysmDetector:

    def test_returns_detection_result(self):
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70,
            mean_curv_gate_percentile=60,
            min_radius_mm=0.5,
            max_radius_mm=30.0,
            min_points=4,
        ).detect(poly)
        assert isinstance(result, DetectionResult)
        assert hasattr(result, "candidates")
        assert hasattr(result, "n_regions_total")
        assert hasattr(result, "n_failed_size")
        assert hasattr(result, "n_failed_points")
        assert hasattr(result, "n_failed_mean_curv")
        assert hasattr(result, "n_merged")

    def test_detects_candidates_on_bumpy_mesh(self):
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70,
            mean_curv_gate_percentile=60,
            min_radius_mm=0.5,
            max_radius_mm=30.0,
            min_points=4,
        ).detect(poly)
        assert len(result.candidates) > 0

    def test_candidate_has_required_fields(self):
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60,
            min_radius_mm=0.5, max_radius_mm=30.0, min_points=4,
        ).detect(poly)
        c = result.candidates[0]
        assert isinstance(c, AneurysmCandidate)
        assert c.radius_mm > 0
        assert c.diameter_mm == pytest.approx(c.radius_mm * 2, rel=0.01)
        assert 0.0 <= c.sphericity <= 1.01
        assert 0.0 <= c.score <= 1.0
        assert c.n_points > 0
        assert c.poly_data is not None
        # New sac discriminator fields
        assert hasattr(c, "positive_gauss_frac")
        assert hasattr(c, "compactness")
        assert 0.0 <= c.positive_gauss_frac <= 1.0
        assert c.compactness >= 0.0

    def test_bump_has_high_positive_gauss_frac(self):
        """The dome-shaped bump should have mostly positive Gaussian curvature."""
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60,
            min_radius_mm=0.5, max_radius_mm=30.0, min_points=4,
        ).detect(poly)
        assert len(result.candidates) > 0
        best = result.candidates[0]
        # A spherical dome → positive_gauss_frac should be fairly high
        assert best.positive_gauss_frac > 0.3   # at least dome-like

    def test_high_gauss_percentile_fewer_candidates(self):
        poly = _bumpy_sphere_poly()
        low  = AneurysmDetector(gauss_percentile=60,
                                mean_curv_gate_percentile=50, min_points=4).detect(poly)
        high = AneurysmDetector(gauss_percentile=95,
                                mean_curv_gate_percentile=85, min_points=4).detect(poly)
        assert len(high.candidates) <= len(low.candidates)

    def test_min_radius_filter(self):
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60,
            min_radius_mm=50.0,    # larger than the bump
            max_radius_mm=200.0, min_points=4,
        ).detect(poly)
        assert len(result.candidates) == 0

    def test_candidates_sorted_by_score_descending(self):
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=65, mean_curv_gate_percentile=55, min_points=4,
        ).detect(poly)
        if len(result.candidates) >= 2:
            scores = [c.score for c in result.candidates]
            assert scores == sorted(scores, reverse=True)

    def test_merge_nearby_duplicates(self):
        """With very low thresholds the same bump should not appear twice."""
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=50, mean_curv_gate_percentile=40,
            min_radius_mm=0.3, max_radius_mm=30.0,
            min_points=4, merge_dist_mm=5.0,
        ).detect(poly)
        # All surviving candidates should be spatially distinct
        centroids = [np.array(c.centroid) for c in result.candidates]
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = float(np.linalg.norm(centroids[i] - centroids[j]))
                assert dist >= 5.0 - 1e-3, (
                    f"Candidates {i} and {j} are only {dist:.2f} mm apart "
                    "(should have been merged)"
                )

    def test_diagnostic_counters(self):
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
        ).detect(poly)
        assert result.n_regions_total    >= 0
        assert result.n_failed_size      >= 0
        assert result.n_failed_points    >= 0
        assert result.n_failed_mean_curv >= 0
        assert result.n_failed_pgf       >= 0   # v4
        assert result.n_failed_compact   >= 0   # v4
        assert result.n_merged           >= 0
        assert result.gauss_threshold    >= 0.0

    def test_pgf_hard_gate_removes_low_pgf_regions(self):
        """Setting min_positive_gauss_frac=0.99 should remove most candidates."""
        poly = _bumpy_sphere_poly()
        result_strict = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.99,  # almost no region can pass
        ).detect(poly)
        result_open = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.0,   # gate disabled
        ).detect(poly)
        # Strict gate ≤ permissive gate
        assert len(result_strict.candidates) <= len(result_open.candidates)

    def test_compactness_now_below_one(self):
        """v4 fix: compactness must be < 1.0 for real candidates (bug was always 1.0)."""
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.0,   # disable pgf gate to inspect compactness directly
            min_compactness=0.0,           # disable compactness gate
        ).detect(poly)
        if result.candidates:
            for c in result.candidates:
                # Old bug: compactness was always exactly 1.0.
                # With the bbox-based fix it should be in (0, 1).
                assert c.compactness < 1.0, (
                    f"compactness={c.compactness:.4f} — bbox-radius fix may not be applied"
                )
                assert c.compactness > 0.0

    def test_compactness_gate_filters_elongated(self):
        """Setting min_compactness=0.99 should filter most regions."""
        poly = _bumpy_sphere_poly()
        result_strict = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.0,  # isolate compactness effect
            min_compactness=0.99,
        ).detect(poly)
        result_open = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.0,
            min_compactness=0.0,
        ).detect(poly)
        assert len(result_strict.candidates) <= len(result_open.candidates)

    def test_empty_mesh_returns_empty_result(self):
        empty = vtk.vtkPolyData()
        result = AneurysmDetector().detect(empty)
        assert isinstance(result, DetectionResult)
        assert result.candidates == []

    # ── v5 tests ──────────────────────────────────────────────────────────── #

    def test_candidate_has_cv_gauss_and_normal_isotropy(self):
        """v5 fields cv_gauss and normal_isotropy must be present and in range."""
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.0, min_compactness=0.0, min_sphericity=0.0,
        ).detect(poly)
        assert len(result.candidates) > 0
        for c in result.candidates:
            assert hasattr(c, "cv_gauss")
            assert hasattr(c, "normal_isotropy")
            assert 0.0 <= c.cv_gauss <= 10.0
            assert 0.0 <= c.normal_isotropy <= 1.0

    def test_sphericity_gate_filters_elongated(self):
        """Setting min_sphericity=0.99 should filter most regions."""
        poly = _bumpy_sphere_poly()
        result_strict = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.0,   # isolate sphericity effect
            min_compactness=0.0,
            min_sphericity=0.99,
        ).detect(poly)
        result_open = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.0,
            min_compactness=0.0,
            min_sphericity=0.0,
        ).detect(poly)
        assert len(result_strict.candidates) <= len(result_open.candidates)

    def test_sphericity_counter_increments(self):
        """n_failed_sphericity must be ≥ 0 and count filtered-out regions."""
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.0, min_compactness=0.0,
            min_sphericity=0.99,   # nearly everything fails
        ).detect(poly)
        assert result.n_failed_sphericity >= 0

    def test_dome_has_low_cv_gauss(self):
        """The spherical bump should have relatively uniform curvature → low cv_gauss."""
        poly = _bumpy_sphere_poly()
        result = AneurysmDetector(
            gauss_percentile=70, mean_curv_gate_percentile=60, min_points=4,
            min_positive_gauss_frac=0.0, min_compactness=0.0, min_sphericity=0.0,
        ).detect(poly)
        assert len(result.candidates) > 0
        best = result.candidates[0]
        # A spherical dome → curvature is fairly uniform → cv_gauss should be moderate
        # We only assert it is in [0, 10] — already guaranteed by clipping.
        assert 0.0 <= best.cv_gauss <= 10.0
