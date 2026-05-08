"""Tests for Feature 4 — Virtual stent deployment along centreline.

Covers:
  * _transport_frames  — orthonormality, tangent alignment
  * deploy_stent_on_centerline  — geometry, metrics, edge cases
  * DeployedStentResult dataclass
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import vtk

from prospective.processing.centerline import CenterlineResult
from prospective.processing.stent_deployment import (
    DeployedStentResult,
    _transport_frames,
    deploy_stent_on_centerline,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ══════════════════════════════════════════════════════════════════════════════

def _straight_cl(length: float = 30.0, n: int = 60, radius: float = 2.0) -> CenterlineResult:
    z   = np.linspace(0, length, n)
    pts = np.column_stack([np.zeros(n), np.zeros(n), z])
    rad = np.full(n, radius, dtype=np.float32)
    return CenterlineResult(
        points=pts, radii=rad,
        arc_length_mm=length, chord_length_mm=length,
        tortuosity=1.0, tortuosity_index=0.0,
        mean_radius_mm=radius, min_radius_mm=radius, max_radius_mm=radius,
    )


def _curved_cl(arc_angle_deg: float = 90.0, radius: float = 20.0,
               tube_r: float = 2.0, n: int = 60) -> CenterlineResult:
    """Quarter-circle arc in the XZ plane, centred at (radius, 0, 0)."""
    angles = np.linspace(0, math.radians(arc_angle_deg), n)
    x = radius - radius * np.cos(angles)
    z = radius * np.sin(angles)
    pts = np.column_stack([x, np.zeros(n), z])
    arc = float(radius * math.radians(arc_angle_deg))
    rad = np.full(n, tube_r, dtype=np.float32)
    return CenterlineResult(
        points=pts, radii=rad,
        arc_length_mm=arc, chord_length_mm=float(np.linalg.norm(pts[-1] - pts[0])),
        tortuosity=arc / float(np.linalg.norm(pts[-1] - pts[0])),
        tortuosity_index=0.0,
        mean_radius_mm=tube_r, min_radius_mm=tube_r, max_radius_mm=tube_r,
    )


@pytest.fixture(scope="module")
def straight_cl():
    return _straight_cl()


@pytest.fixture(scope="module")
def curved_cl():
    return _curved_cl()


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests: _transport_frames
# ══════════════════════════════════════════════════════════════════════════════

class TestTransportFrames:
    def test_returns_three_arrays(self):
        cl = _straight_cl(n=10)
        T, N, B = _transport_frames(cl.points)
        assert T.shape == N.shape == B.shape == (10, 3)

    def test_tangents_unit_length(self):
        cl = _straight_cl(n=20)
        T, _, _ = _transport_frames(cl.points)
        norms = np.linalg.norm(T, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_normals_unit_length(self):
        cl = _straight_cl(n=20)
        _, N, _ = _transport_frames(cl.points)
        norms = np.linalg.norm(N, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_binormals_unit_length(self):
        cl = _straight_cl(n=20)
        _, _, B = _transport_frames(cl.points)
        norms = np.linalg.norm(B, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_frames_orthogonal(self):
        """T · N ≈ 0 and T · B ≈ 0 at each point."""
        cl = _curved_cl(n=30)
        T, N, B = _transport_frames(cl.points)
        np.testing.assert_allclose(np.einsum('ij,ij->i', T, N), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.einsum('ij,ij->i', T, B), 0.0, atol=1e-5)

    def test_straight_tangent_is_z(self):
        """For a straight Z-axis path, T should be (0,0,1) everywhere."""
        cl = _straight_cl(n=10)
        T, _, _ = _transport_frames(cl.points)
        np.testing.assert_allclose(np.abs(T[:, 2]), 1.0, atol=1e-6)

    def test_single_segment(self):
        """Two-point path should not raise."""
        pts = np.array([[0, 0, 0], [0, 0, 10]], dtype=float)
        T, N, B = _transport_frames(pts)
        assert T.shape == (2, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests: DeployedStentResult
# ══════════════════════════════════════════════════════════════════════════════

class TestDeployedStentResult:
    def test_summary_string(self, straight_cl):
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=False)
        s = r.summary()
        assert "mm" in s
        assert "Longitud" in s

    def test_fields_present(self, straight_cl):
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=False)
        assert r.length_mm         > 0
        assert r.nominal_diameter_mm == pytest.approx(4.0)
        assert r.mean_vessel_diameter_mm > 0
        assert r.coverage_ratio    > 0
        assert r.stent_poly_data   is not None


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests: deploy_stent_on_centerline
# ══════════════════════════════════════════════════════════════════════════════

class TestDeployStentOnCenterline:
    def test_returns_result(self, straight_cl):
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=False)
        assert isinstance(r, DeployedStentResult)

    def test_poly_data_has_points(self, straight_cl):
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=False)
        assert r.stent_poly_data.GetNumberOfPoints() > 0

    def test_poly_data_has_cells(self, straight_cl):
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=False)
        assert r.stent_poly_data.GetNumberOfCells() > 0

    def test_length_approximates_arc(self, straight_cl):
        """Full deployment should have length ≈ arc_length_mm."""
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=False)
        assert abs(r.length_mm - straight_cl.arc_length_mm) / straight_cl.arc_length_mm < 0.01

    def test_partial_segment(self, straight_cl):
        r = deploy_stent_on_centerline(
            straight_cl, stent_diameter_mm=4.0,
            start_arc_mm=5.0, end_arc_mm=20.0, braid=False,
        )
        assert abs(r.length_mm - 15.0) < 0.5

    def test_coverage_ratio_formula(self, straight_cl):
        """coverage = stent_r / vessel_r = 4/(2×2) = 1.0."""
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=False)
        # vessel radius = 2.0 mm → vessel diameter = 4.0 mm → ratio = 1.0
        assert r.coverage_ratio == pytest.approx(1.0, abs=0.05)

    def test_undersized_stent_coverage_lt_1(self, straight_cl):
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=2.0, braid=False)
        assert r.coverage_ratio < 1.0

    def test_oversized_stent_coverage_gt_1(self, straight_cl):
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=6.0, braid=False)
        assert r.coverage_ratio > 1.0

    def test_nominal_diameter_stored(self, straight_cl):
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=3.5, braid=False)
        assert r.nominal_diameter_mm == pytest.approx(3.5)

    def test_braid_adds_more_geometry(self, straight_cl):
        r_no  = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=False)
        r_yes = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=True)
        assert r_yes.stent_poly_data.GetNumberOfPoints() > r_no.stent_poly_data.GetNumberOfPoints()

    def test_curved_cl_deploys(self, curved_cl):
        """Deployment on a curved path should not raise."""
        r = deploy_stent_on_centerline(curved_cl, stent_diameter_mm=4.0, braid=False)
        assert r.stent_poly_data.GetNumberOfPoints() > 0

    def test_too_short_segment_raises(self, straight_cl):
        with pytest.raises(ValueError):
            deploy_stent_on_centerline(
                straight_cl, stent_diameter_mm=4.0,
                start_arc_mm=10.0, end_arc_mm=10.3, braid=False,
            )

    def test_single_point_cl_raises(self):
        cl = CenterlineResult(
            points=np.array([[0, 0, 0]]),
            radii=np.array([2.0]),
        )
        with pytest.raises(ValueError):
            deploy_stent_on_centerline(cl, stent_diameter_mm=4.0)

    def test_segment_clipped_to_arc_bounds(self, straight_cl):
        """start/end beyond arc bounds should be clamped, not raise."""
        r = deploy_stent_on_centerline(
            straight_cl, stent_diameter_mm=4.0,
            start_arc_mm=-5.0, end_arc_mm=999.0, braid=False,
        )
        assert r.length_mm > 0

    def test_centerline_segment_stored(self, straight_cl):
        r = deploy_stent_on_centerline(straight_cl, stent_diameter_mm=4.0, braid=False)
        assert r.centerline_segment.ndim == 2
        assert r.centerline_segment.shape[1] == 3
