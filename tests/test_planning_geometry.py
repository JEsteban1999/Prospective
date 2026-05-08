"""Unit tests — planning geometry helpers (A-03-07 / trajectory placement).

Tests cover the pure-geometry static methods extracted from PlanningWindow:
  - _get_point_and_tangent   (arc-length interpolation)
  - _transform_z_to_tangent  (rotation transform)

Both are @staticmethod so they can be exercised without a Qt/VTK display.
The arc-length table for the 2-point (straight line) case is also tested
via a lightweight helper that mirrors the internal logic, avoiding any
QApplication dependency.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import vtk

from prospective.ui.windows.planning_window import PlanningWindow


# ══════════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ══════════════════════════════════════════════════════════════════════════════


def _straight_table(
    p0: tuple, p1: tuple
) -> tuple[np.ndarray, np.ndarray, float]:
    """Arc-length table for a 2-point (straight) trajectory — mirrors _build_arc_length_table."""
    pts   = np.array([p0, p1], dtype=float)
    total = float(np.linalg.norm(pts[1] - pts[0]))
    return pts, np.array([0.0, total]), total


def _spline_table_vtk(
    points: list[tuple],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Arc-length table for ≥ 3 control points using VTK spline (mirrors PlanningWindow)."""
    spline_pts = vtk.vtkPoints()
    for p in points:
        spline_pts.InsertNextPoint(*p)

    param_spline = vtk.vtkParametricSpline()
    param_spline.SetPoints(spline_pts)
    param_spline.ClosedOff()

    spline_src = vtk.vtkParametricFunctionSource()
    spline_src.SetParametricFunction(param_spline)
    spline_src.SetUResolution(500)
    spline_src.Update()

    poly     = spline_src.GetOutput()
    n_pts    = poly.GetNumberOfPoints()
    pts_arr  = np.array([poly.GetPoint(i) for i in range(n_pts)])
    diffs    = np.diff(pts_arr, axis=0)
    seg_len  = np.linalg.norm(diffs, axis=1)
    arc_lens = np.concatenate([[0.0], np.cumsum(seg_len)])
    return pts_arr, arc_lens, float(arc_lens[-1])


# ══════════════════════════════════════════════════════════════════════════════
# _get_point_and_tangent
# ══════════════════════════════════════════════════════════════════════════════


class TestGetPointAndTangent:

    # ── straight-line table (2 pts) ────────────────────────────────────── #

    def test_midpoint_of_straight_line(self):
        pts, arcs, total = _straight_table((0, 0, 0), (10, 0, 0))
        pt, tan = PlanningWindow._get_point_and_tangent(5.0, pts, arcs)
        assert pytest.approx(pt[0], abs=1e-6) == 5.0
        assert pytest.approx(pt[1], abs=1e-6) == 0.0
        assert pytest.approx(pt[2], abs=1e-6) == 0.0

    def test_start_of_straight_line(self):
        pts, arcs, total = _straight_table((0, 0, 0), (8, 0, 0))
        pt, tan = PlanningWindow._get_point_and_tangent(0.0, pts, arcs)
        # Arc clamped to first segment — result inside [start, end]
        assert 0.0 <= pt[0] <= 8.0

    def test_end_of_straight_line(self):
        pts, arcs, total = _straight_table((0, 0, 0), (12, 0, 0))
        pt, tan = PlanningWindow._get_point_and_tangent(12.0, pts, arcs)
        assert pytest.approx(pt[0], abs=1e-4) == 12.0

    def test_tangent_is_unit_vector(self):
        pts, arcs, _ = _straight_table((0, 0, 0), (10, 0, 0))
        _, tan = PlanningWindow._get_point_and_tangent(5.0, pts, arcs)
        assert pytest.approx(float(np.linalg.norm(tan)), abs=1e-9) == 1.0

    def test_tangent_direction_matches_segment(self):
        """Along X-axis trajectory tangent should be (1, 0, 0)."""
        pts, arcs, _ = _straight_table((0, 0, 0), (10, 0, 0))
        _, tan = PlanningWindow._get_point_and_tangent(5.0, pts, arcs)
        assert pytest.approx(tan[0], abs=1e-6) == 1.0
        assert pytest.approx(tan[1], abs=1e-6) == 0.0
        assert pytest.approx(tan[2], abs=1e-6) == 0.0

    def test_clamping_below_zero(self):
        """Arc-s < 0 must be clamped to the start — no exception raised."""
        pts, arcs, _ = _straight_table((5, 5, 5), (15, 5, 5))
        pt, tan = PlanningWindow._get_point_and_tangent(-99.0, pts, arcs)
        assert np.all(np.isfinite(pt))
        assert np.all(np.isfinite(tan))

    def test_clamping_above_total(self):
        """Arc-s > total must be clamped — no exception raised."""
        pts, arcs, total = _straight_table((0, 0, 0), (6, 0, 0))
        pt, tan = PlanningWindow._get_point_and_tangent(999.0, pts, arcs)
        assert np.all(np.isfinite(pt))
        assert np.all(np.isfinite(tan))

    def test_diagonal_trajectory(self):
        """Point at half arc-length of a 3-D diagonal segment."""
        p0, p1 = (0.0, 0.0, 0.0), (6.0, 6.0, 6.0)
        pts, arcs, total = _straight_table(p0, p1)
        pt, _ = PlanningWindow._get_point_and_tangent(total / 2, pts, arcs)
        expected = np.array([3.0, 3.0, 3.0])
        assert np.allclose(pt, expected, atol=1e-5)

    # ── spline table (3 pts) ───────────────────────────────────────────── #

    def test_spline_start_near_first_control_point(self):
        pts_arr, arcs, total = _spline_table_vtk(
            [(0, 0, 0), (5, 5, 0), (10, 0, 0)]
        )
        pt, _ = PlanningWindow._get_point_and_tangent(0.0, pts_arr, arcs)
        assert np.allclose(pt, [0, 0, 0], atol=0.5)

    def test_spline_total_length_reasonable(self):
        """Spline length through a gentle curve > straight-line distance."""
        pts_arr, arcs, total = _spline_table_vtk(
            [(0, 0, 0), (5, 5, 0), (10, 0, 0)]
        )
        straight = math.sqrt(100)  # 10 mm
        assert total > straight - 1e-3, "Spline should be at least as long as chord"

    @pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_spline_points_are_finite(self, frac: float):
        pts_arr, arcs, total = _spline_table_vtk(
            [(0, 0, 0), (5, 3, 0), (10, 0, 0)]
        )
        pt, tan = PlanningWindow._get_point_and_tangent(
            frac * total, pts_arr, arcs
        )
        assert np.all(np.isfinite(pt))
        assert np.all(np.isfinite(tan))
        assert pytest.approx(float(np.linalg.norm(tan)), abs=1e-9) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# _transform_z_to_tangent
# ══════════════════════════════════════════════════════════════════════════════


class TestTransformZToTangent:

    def _get_matrix(self, transform: vtk.vtkTransform) -> np.ndarray:
        m = vtk.vtkMatrix4x4()
        transform.GetMatrix(m)
        return np.array([[m.GetElement(r, c) for c in range(4)] for r in range(4)])

    def test_returns_vtk_transform(self):
        pos = np.array([0.0, 0.0, 0.0])
        tan = np.array([0.0, 0.0, 1.0])
        t   = PlanningWindow._transform_z_to_tangent(pos, tan)
        assert isinstance(t, vtk.vtkTransform)

    def test_translation_column_matches_position(self):
        pos = np.array([3.0, 7.0, -2.0])
        tan = np.array([0.0, 0.0, 1.0])
        mat = self._get_matrix(PlanningWindow._transform_z_to_tangent(pos, tan))
        # Column 3 (homogeneous translation)
        assert pytest.approx(mat[0, 3], abs=1e-5) == pos[0]
        assert pytest.approx(mat[1, 3], abs=1e-5) == pos[1]
        assert pytest.approx(mat[2, 3], abs=1e-5) == pos[2]

    def test_z_aligned_tangent_produces_identity_rotation(self):
        """When tangent == Z the rotation block should be identity."""
        pos = np.array([0.0, 0.0, 0.0])
        tan = np.array([0.0, 0.0, 1.0])
        mat = self._get_matrix(PlanningWindow._transform_z_to_tangent(pos, tan))
        rot = mat[:3, :3]
        assert np.allclose(rot, np.eye(3), atol=1e-6)

    def test_minus_z_tangent_rotates_180(self):
        """When tangent == -Z the rotation block should flip Z."""
        pos = np.array([0.0, 0.0, 0.0])
        tan = np.array([0.0, 0.0, -1.0])
        mat = self._get_matrix(PlanningWindow._transform_z_to_tangent(pos, tan))
        # The rotated Z column should point in -Z direction
        rotated_z = mat[:3, 2]
        assert pytest.approx(float(rotated_z[2]), abs=1e-5) == -1.0

    def test_x_tangent_rotates_z_to_x(self):
        """When tangent == X the local Z axis should map to X."""
        pos = np.array([0.0, 0.0, 0.0])
        tan = np.array([1.0, 0.0, 0.0])
        mat = self._get_matrix(PlanningWindow._transform_z_to_tangent(pos, tan))
        rotated_z = mat[:3, 2]   # third column = where local Z goes
        assert np.allclose(rotated_z, [1.0, 0.0, 0.0], atol=1e-5)

    def test_y_tangent_rotates_z_to_y(self):
        pos = np.array([0.0, 0.0, 0.0])
        tan = np.array([0.0, 1.0, 0.0])
        mat = self._get_matrix(PlanningWindow._transform_z_to_tangent(pos, tan))
        rotated_z = mat[:3, 2]
        assert np.allclose(rotated_z, [0.0, 1.0, 0.0], atol=1e-5)

    def test_unnormalized_tangent_still_works(self):
        """Input tangent need not be unit-length."""
        pos = np.array([0.0, 0.0, 0.0])
        tan = np.array([5.0, 0.0, 0.0])  # length 5
        mat = self._get_matrix(PlanningWindow._transform_z_to_tangent(pos, tan))
        # Rotation block should still be orthogonal
        rot = mat[:3, :3]
        should_be_identity = rot @ rot.T
        assert np.allclose(should_be_identity, np.eye(3), atol=1e-5)

    def test_rotation_block_is_orthogonal(self):
        """For any tangent the rotation block must be an orthogonal matrix."""
        pos = np.array([1.0, 2.0, 3.0])
        for tan in [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([1.0, 1.0, 1.0]) / math.sqrt(3),
            np.array([-1.0, 1.0, 0.0]) / math.sqrt(2),
        ]:
            mat = self._get_matrix(PlanningWindow._transform_z_to_tangent(pos, tan))
            rot = mat[:3, :3]
            assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5), (
                f"Non-orthogonal rotation for tangent {tan}"
            )

    def test_arbitrary_position_and_tangent(self):
        pos = np.array([10.0, -5.0, 3.0])
        tan = np.array([0.6, 0.8, 0.0])
        mat = self._get_matrix(PlanningWindow._transform_z_to_tangent(pos, tan))
        # Translation preserved
        assert pytest.approx(mat[0, 3], abs=1e-5) == 10.0
        assert pytest.approx(mat[1, 3], abs=1e-5) == -5.0
        assert pytest.approx(mat[2, 3], abs=1e-5) == 3.0
        # Rotation block orthogonal
        rot = mat[:3, :3]
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# Arc-length table — straight-line (2-point) case
# ══════════════════════════════════════════════════════════════════════════════


class TestArcLengthTableStraight:
    """Tests for the 2-point (degenerate) arc-length logic used by _build_arc_length_table."""

    def test_total_equals_euclidean_distance(self):
        p0, p1 = (0, 0, 0), (3, 4, 0)
        _, _, total = _straight_table(p0, p1)
        assert pytest.approx(total, abs=1e-9) == 5.0

    def test_arc_lengths_starts_at_zero(self):
        _, arcs, _ = _straight_table((0, 0, 0), (10, 0, 0))
        assert arcs[0] == 0.0

    def test_arc_lengths_ends_at_total(self):
        _, arcs, total = _straight_table((0, 0, 0), (7, 0, 0))
        assert pytest.approx(arcs[-1], abs=1e-9) == total

    def test_arc_lengths_monotone(self):
        _, arcs, _ = _straight_table((0, 0, 0), (10, 5, 3))
        diffs = np.diff(arcs)
        assert np.all(diffs >= 0)

    def test_unit_diagonal(self):
        p0 = (0, 0, 0)
        p1 = (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))
        _, _, total = _straight_table(p0, p1)
        assert pytest.approx(total, abs=1e-9) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Arc-length table — spline (3+ points) via VTK
# ══════════════════════════════════════════════════════════════════════════════


class TestArcLengthTableSpline:

    def test_three_collinear_points_total_is_sum(self):
        """Collinear control points → spline behaves like a straight line."""
        pts_arr, arcs, total = _spline_table_vtk(
            [(0, 0, 0), (5, 0, 0), (10, 0, 0)]
        )
        assert pytest.approx(total, rel=0.01) == 10.0

    def test_arc_starts_at_zero(self):
        _, arcs, _ = _spline_table_vtk([(0, 0, 0), (5, 5, 0), (10, 0, 0)])
        assert arcs[0] == 0.0

    def test_arc_monotone(self):
        _, arcs, _ = _spline_table_vtk([(0, 0, 0), (5, 5, 0), (10, 0, 0)])
        diffs = np.diff(arcs)
        assert np.all(diffs >= 0)

    def test_four_points_reasonable_total(self):
        """Total arc length must be ≥ the chord between start and end."""
        control_pts = [(0, 0, 0), (5, 8, 0), (12, 3, 0), (20, 0, 0)]
        _, _, total = _spline_table_vtk(control_pts)
        chord = math.sqrt((20 - 0) ** 2)
        assert total >= chord - 1e-3
