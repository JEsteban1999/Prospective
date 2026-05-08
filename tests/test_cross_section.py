"""Tests for prospective.processing.cross_section — Feature 2.

Fixtures
--------
straight_tube  — 30 mm long cylinder, r = 3 mm  (uniform cross-section)
tapered_tube   — 30 mm long cone, r 3 mm → 1.5 mm  (variable cross-section)
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from prospective.processing.centerline import CenterlineResult
from prospective.processing.cross_section import (
    CrossSectionResult,
    _shoelace_area,
    _cut_area,
    compute_cross_sections,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_cylinder(length: float, radius: float, resolution: int = 60) -> vtk.vtkPolyData:
    """Closed cylinder mesh along Z axis from 0 to *length*."""
    src = vtk.vtkCylinderSource()
    src.SetRadius(radius)
    src.SetHeight(length)
    src.SetResolution(resolution)
    src.SetCapping(True)
    src.Update()
    # vtkCylinderSource creates a Y-axis cylinder centred at origin.
    # PostMultiply: operations applied in the order they are called.
    # 1. Rotate 90° around X  → cylinder becomes Z-axis, z in [-L/2, +L/2]
    # 2. Translate +L/2 in Z  → z in [0, L]
    tf = vtk.vtkTransformPolyDataFilter()
    rot = vtk.vtkTransform()
    rot.PostMultiply()
    rot.RotateX(90)
    rot.Translate(0, 0, length / 2)
    tf.SetTransform(rot)
    tf.SetInputConnection(src.GetOutputPort())
    tf.Update()
    return tf.GetOutput()


def _make_cone(length: float, r_start: float, r_end: float, resolution: int = 60) -> vtk.vtkPolyData:
    """Approximate tapered tube by stacking cylinder rings."""
    append = vtk.vtkAppendPolyData()
    n_segs = 20
    for i in range(n_segs):
        t0 = i / n_segs
        t1 = (i + 1) / n_segs
        z0 = t0 * length
        z1 = t1 * length
        r0 = r_start + t0 * (r_end - r_start)
        r1 = r_start + t1 * (r_end - r_start)
        # single frustum segment
        seg = vtk.vtkCylinderSource()
        seg.SetRadius((r0 + r1) / 2)
        seg.SetHeight(z1 - z0)
        seg.SetResolution(resolution)
        seg.SetCapping(False)
        seg.Update()
        tf = vtk.vtkTransformPolyDataFilter()
        rot = vtk.vtkTransform()
        rot.RotateX(90)
        rot.Translate(0, 0, (z0 + z1) / 2)
        tf.SetTransform(rot)
        tf.SetInputConnection(seg.GetOutputPort())
        tf.Update()
        append.AddInputData(tf.GetOutput())

    # End caps
    for z, r in [(0, r_start), (length, r_end)]:
        disk = vtk.vtkDiskSource()
        disk.SetInnerRadius(0)
        disk.SetOuterRadius(r)
        disk.SetRadialResolution(1)
        disk.SetCircumferentialResolution(resolution)
        disk.Update()
        tf = vtk.vtkTransformPolyDataFilter()
        t = vtk.vtkTransform()
        t.Translate(0, 0, z)
        tf.SetTransform(t)
        tf.SetInputConnection(disk.GetOutputPort())
        tf.Update()
        append.AddInputData(tf.GetOutput())

    append.Update()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(append.GetOutputPort())
    cleaner.Update()
    return cleaner.GetOutput()


def _straight_centerline(length: float, n: int = 60) -> CenterlineResult:
    """CenterlineResult for a straight path along Z, radius ≈ 3 mm."""
    z = np.linspace(0, length, n)
    pts = np.column_stack([np.zeros(n), np.zeros(n), z])
    radii = np.full(n, 3.0, dtype=np.float32)
    arc = float(length)
    chord = float(length)
    return CenterlineResult(
        points=pts,
        radii=radii,
        arc_length_mm=arc,
        chord_length_mm=chord,
        tortuosity=1.0,
        tortuosity_index=0.0,
        mean_radius_mm=3.0,
        min_radius_mm=3.0,
        max_radius_mm=3.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def straight_tube():
    return _make_cylinder(length=30.0, radius=3.0)


@pytest.fixture(scope="module")
def tapered_tube():
    return _make_cone(length=30.0, r_start=3.0, r_end=1.5)


@pytest.fixture(scope="module")
def straight_cl():
    return _straight_centerline(length=30.0)


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests: _shoelace_area
# ══════════════════════════════════════════════════════════════════════════════

class TestShoelaceArea:
    def test_square_xy_plane(self):
        """Unit square in XY plane → area = 1.0."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        origin = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        area = _shoelace_area(pts, origin, normal)
        assert abs(area - 1.0) < 0.01

    def test_circle_approximation(self):
        """Regular 360-gon ≈ π r² for r = 5."""
        r   = 5.0
        n   = 360
        t   = np.linspace(0, 2 * math.pi, n, endpoint=False)
        pts = np.column_stack([r * np.cos(t), r * np.sin(t), np.zeros(n)])
        origin = np.zeros(3)
        normal = np.array([0.0, 0.0, 1.0])
        area   = _shoelace_area(pts, origin, normal)
        expected = math.pi * r ** 2
        assert abs(area - expected) / expected < 0.01

    def test_too_few_points_returns_zero(self):
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        area = _shoelace_area(pts, np.zeros(3), np.array([0.0, 0.0, 1.0]))
        assert area == 0.0

    def test_degenerate_normal_returns_zero(self):
        pts = np.ones((4, 3))
        area = _shoelace_area(pts, np.zeros(3), np.zeros(3))
        assert area == 0.0

    def test_tilted_plane(self):
        """Circle in a tilted plane: area should be the same as in XY."""
        r   = 4.0
        n   = 360
        t   = np.linspace(0, 2 * math.pi, n, endpoint=False)
        # Build circle in XZ plane (normal = Y)
        pts = np.column_stack([r * np.cos(t), np.zeros(n), r * np.sin(t)])
        origin = np.zeros(3)
        normal = np.array([0.0, 1.0, 0.0])
        area   = _shoelace_area(pts, origin, normal)
        expected = math.pi * r ** 2
        assert abs(area - expected) / expected < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests: _cut_area
# ══════════════════════════════════════════════════════════════════════════════

class TestCutArea:
    def test_cylinder_midplane(self, straight_tube):
        """Mid-plane cut of a 3 mm radius cylinder ≈ π × 3² = 28.27 mm²."""
        origin = np.array([0.0, 0.0, 15.0])
        normal = np.array([0.0, 0.0, 1.0])
        area   = _cut_area(straight_tube, origin, normal)
        expected = math.pi * 3.0 ** 2
        # Allow 10 % tolerance — mesh is discrete
        assert abs(area - expected) / expected < 0.10, f"area={area:.2f}, expected≈{expected:.2f}"

    def test_zero_points_returns_zero(self):
        """A plane that misses the mesh returns 0."""
        cylinder = _make_cylinder(10.0, 2.0)
        origin = np.array([0.0, 0.0, 100.0])   # far away
        normal = np.array([0.0, 0.0, 1.0])
        area   = _cut_area(cylinder, origin, normal)
        assert area == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests: compute_cross_sections
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeCrossSections:
    def test_returns_cross_section_result(self, straight_tube, straight_cl):
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=10)
        assert isinstance(r, CrossSectionResult)

    def test_array_lengths_match(self, straight_tube, straight_cl):
        n = 15
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=n)
        # Valid samples may be fewer than n_samples (boundary samples filtered)
        assert len(r.arc_positions_mm) == len(r.areas_mm2) == len(r.diameters_mm)
        assert len(r.arc_positions_mm) > 0

    def test_uniform_cylinder_consistent_diameter(self, straight_tube, straight_cl):
        """Uniform cylinder: max diameter ≈ min diameter (within 15 %)."""
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=20)
        assert r.max_diameter_mm > 0
        variation = (r.max_diameter_mm - r.min_diameter_mm) / r.mean_diameter_mm
        assert variation < 0.15, f"Diameter variation {variation:.2%} too high"

    def test_cylinder_diameter_near_expected(self, straight_tube, straight_cl):
        """Mean diameter should be close to 2×3 = 6 mm (within 15 %)."""
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=20)
        assert abs(r.mean_diameter_mm - 6.0) / 6.0 < 0.15

    def test_stenosis_ratio_uniform(self, straight_tube, straight_cl):
        """Uniform cylinder: stenosis_ratio should be near 1.0."""
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=20)
        assert r.stenosis_ratio > 0.8

    def test_scalar_summary_consistency(self, straight_tube, straight_cl):
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=15)
        tol = 1e-9
        assert r.min_diameter_mm <= r.mean_diameter_mm + tol
        assert r.mean_diameter_mm <= r.max_diameter_mm + tol
        assert r.min_area_mm2    <= r.mean_area_mm2    + tol
        assert r.mean_area_mm2   <= r.max_area_mm2     + tol

    def test_arc_positions_monotonic(self, straight_tube, straight_cl):
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=20)
        diffs = np.diff(r.arc_positions_mm)
        assert (diffs > 0).all(), "Arc positions must be strictly increasing"

    def test_arc_positions_within_range(self, straight_tube, straight_cl):
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=20)
        assert r.arc_positions_mm[0]  >= 0.0
        assert r.arc_positions_mm[-1] <= 30.0

    def test_areas_positive(self, straight_tube, straight_cl):
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=10)
        assert (r.areas_mm2 > 0).all()

    def test_diameters_equal_circle_formula(self, straight_tube, straight_cl):
        """diameters_mm must satisfy d = 2·sqrt(A/π) for every sample."""
        r = compute_cross_sections(straight_cl, straight_tube, n_samples=10)
        expected = 2.0 * np.sqrt(r.areas_mm2 / math.pi)
        np.testing.assert_allclose(r.diameters_mm, expected, rtol=1e-6)

    def test_progress_callback_called(self, straight_tube, straight_cl):
        calls = []
        compute_cross_sections(straight_cl, straight_tube, n_samples=5,
                               progress_cb=lambda f: calls.append(f))
        assert len(calls) > 0
        assert calls[-1] == pytest.approx(1.0)

    def test_too_short_centerline_raises(self, straight_tube):
        """A degenerate 1-point centerline must raise ValueError."""
        cl = CenterlineResult(
            points=np.array([[0, 0, 0]]),
            radii=np.array([3.0]),
        )
        with pytest.raises(ValueError):
            compute_cross_sections(cl, straight_tube)
