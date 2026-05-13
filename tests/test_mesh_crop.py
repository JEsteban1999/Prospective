"""Unit tests for prospective.processing.mesh_crop — clip_box / clip_sphere.

All tests use analytic VTK primitives so the expected vertex counts and
bounds are predictable without loading real patient data.
"""
from __future__ import annotations

import vtk
import pytest

from prospective.processing.mesh_crop import clip_box, clip_sphere


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _sphere(center=(0.0, 0.0, 0.0), radius: float = 10.0,
            theta: int = 32, phi: int = 32) -> vtk.vtkPolyData:
    src = vtk.vtkSphereSource()
    src.SetCenter(*center)
    src.SetRadius(radius)
    src.SetThetaResolution(theta)
    src.SetPhiResolution(phi)
    src.Update()
    return src.GetOutput()


def _empty_mesh() -> vtk.vtkPolyData:
    return vtk.vtkPolyData()


def _bounds(poly: vtk.vtkPolyData) -> tuple[float, ...]:
    """Return (xmin, xmax, ymin, ymax, zmin, zmax)."""
    b = [0.0] * 6
    poly.GetBounds(b)
    return tuple(b)


# ──────────────────────────────────────────────────────────────────────────── #
# clip_box                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

class TestClipBox:

    def test_empty_mesh_passthrough(self):
        """Empty input is returned immediately without processing."""
        empty = _empty_mesh()
        result = clip_box(empty, -1, 1, -1, 1, -1, 1)
        assert result.GetNumberOfPoints() == 0

    def test_identity_box_keeps_all_points(self):
        """Box larger than mesh — all points should be retained."""
        poly = _sphere(radius=10.0)
        n_in = poly.GetNumberOfPoints()
        result = clip_box(poly, -20, 20, -20, 20, -20, 20)
        # Clean may merge coincident points; we allow ±2 for VTK internals
        assert result.GetNumberOfPoints() >= n_in - 2

    def test_half_box_reduces_points(self):
        """Box that crops exactly the positive-x half should halve the mesh."""
        poly = _sphere(center=(0, 0, 0), radius=10.0)
        n_in = poly.GetNumberOfPoints()
        # Keep only x >= 0 half
        result = clip_box(poly, 0, 20, -20, 20, -20, 20)
        n_out = result.GetNumberOfPoints()
        assert 0 < n_out < n_in

    def test_excluded_box_returns_empty(self):
        """Box far away from mesh → result should have no geometry."""
        poly   = _sphere(center=(0, 0, 0), radius=5.0)
        result = clip_box(poly, 100, 200, 100, 200, 100, 200)
        assert result.GetNumberOfPoints() == 0

    def test_result_bounds_within_box(self):
        """Every remaining vertex must lie within the clip box."""
        poly = _sphere(radius=10.0)
        xmin, xmax = -5.0, 5.0
        ymin, ymax = -20.0, 20.0
        zmin, zmax = -20.0, 20.0
        result = clip_box(poly, xmin, xmax, ymin, ymax, zmin, zmax)
        if result.GetNumberOfPoints() == 0:
            return   # nothing to check
        bx0, bx1, by0, by1, bz0, bz1 = _bounds(result)
        tol = 1e-6
        assert bx0 >= xmin - tol
        assert bx1 <= xmax + tol
        assert by0 >= ymin - tol
        assert by1 <= ymax + tol
        assert bz0 >= zmin - tol
        assert bz1 <= zmax + tol

    def test_returns_vtkpolydata(self):
        result = clip_box(_sphere(), -5, 5, -5, 5, -5, 5)
        assert isinstance(result, vtk.vtkPolyData)

    def test_does_not_modify_input(self):
        """clip_box must be non-destructive."""
        poly = _sphere(radius=10.0)
        n_before = poly.GetNumberOfPoints()
        clip_box(poly, 0, 5, 0, 5, 0, 5)
        assert poly.GetNumberOfPoints() == n_before


# ──────────────────────────────────────────────────────────────────────────── #
# clip_sphere                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

class TestClipSphere:

    def test_empty_mesh_passthrough(self):
        """Empty input is returned immediately without processing."""
        empty = _empty_mesh()
        result = clip_sphere(empty, center=(0, 0, 0), radius=5.0)
        assert result.GetNumberOfPoints() == 0

    def test_zero_radius_returns_empty(self):
        """radius=0 should return an empty mesh, not raise."""
        poly = _sphere(radius=10.0)
        result = clip_sphere(poly, center=(0, 0, 0), radius=0.0)
        assert result.GetNumberOfPoints() == 0

    def test_negative_radius_returns_empty(self):
        """Negative radius is invalid — return empty mesh."""
        poly = _sphere(radius=10.0)
        result = clip_sphere(poly, center=(0, 0, 0), radius=-3.0)
        assert result.GetNumberOfPoints() == 0

    def test_large_sphere_keeps_all_points(self):
        """Clip sphere larger than the mesh keeps all vertices."""
        poly = _sphere(radius=5.0)
        n_in = poly.GetNumberOfPoints()
        result = clip_sphere(poly, center=(0, 0, 0), radius=50.0)
        # vtkCleanPolyData may merge a few coincident vertices
        assert result.GetNumberOfPoints() >= n_in - 2

    def test_partial_clip_sphere_reduces_points(self):
        """Clip sphere that covers only one hemisphere — keeps fewer than all points.

        Mesh: sphere at origin r=10.  Clip sphere: center=(10,0,0) r=12.
        Vertices near (10,0,0) are inside the clip sphere (kept);
        vertices near (-10,0,0) are outside (removed).
        """
        poly = _sphere(radius=10.0, theta=32, phi=32)
        n_in = poly.GetNumberOfPoints()
        result = clip_sphere(poly, center=(10.0, 0.0, 0.0), radius=12.0)
        n_out = result.GetNumberOfPoints()
        assert 0 < n_out < n_in

    def test_off_center_clip(self):
        """Clip sphere offset from mesh — should keep only overlapping region."""
        poly = _sphere(center=(0, 0, 0), radius=10.0)
        # Sphere centered at (15, 0, 0) with radius 8 overlaps x ∈ [7, 10]
        result = clip_sphere(poly, center=(15, 0, 0), radius=8.0)
        n_out = result.GetNumberOfPoints()
        assert 0 < n_out < poly.GetNumberOfPoints()

    def test_disjoint_clip_sphere_returns_empty(self):
        """Clip sphere entirely outside the mesh → empty result."""
        poly = _sphere(center=(0, 0, 0), radius=5.0)
        result = clip_sphere(poly, center=(100, 0, 0), radius=2.0)
        assert result.GetNumberOfPoints() == 0

    def test_returns_vtkpolydata(self):
        result = clip_sphere(_sphere(), center=(0, 0, 0), radius=5.0)
        assert isinstance(result, vtk.vtkPolyData)

    def test_does_not_modify_input(self):
        """clip_sphere must be non-destructive."""
        poly = _sphere(radius=10.0)
        n_before = poly.GetNumberOfPoints()
        clip_sphere(poly, center=(0, 0, 0), radius=4.0)
        assert poly.GetNumberOfPoints() == n_before
