"""Tests for vessel centreline extraction — Feature 1.

Uses synthetic tube geometries so no real DICOM data is needed.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers — build synthetic tube meshes                                         #
# ──────────────────────────────────────────────────────────────────────────── #

def _make_straight_tube(
    length_mm: float = 30.0,
    radius_mm: float = 3.0,
    n_sides: int = 32,
) -> vtk.vtkPolyData:
    """Return a closed cylindrical tube along the Z axis."""
    line = vtk.vtkLineSource()
    line.SetPoint1(0.0, 0.0, 0.0)
    line.SetPoint2(0.0, 0.0, length_mm)
    line.SetResolution(50)
    line.Update()

    tube = vtk.vtkTubeFilter()
    tube.SetInputConnection(line.GetOutputPort())
    tube.SetRadius(radius_mm)
    tube.SetNumberOfSides(n_sides)
    tube.SetCapping(True)
    tube.Update()
    return tube.GetOutput()


def _make_curved_tube(
    arc_deg: float = 90.0,
    arc_radius_mm: float = 20.0,
    tube_radius_mm: float = 2.5,
    n_sides: int = 32,
) -> vtk.vtkPolyData:
    """Return a curved tube following a circular arc in the XZ plane."""
    n_pts = 40
    angles = np.linspace(0.0, math.radians(arc_deg), n_pts)
    pts_arr = np.column_stack([
        arc_radius_mm * np.sin(angles),
        np.zeros(n_pts),
        arc_radius_mm * (1.0 - np.cos(angles)),
    ]).astype(np.float32)

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(pts_arr, deep=True))
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(n_pts)
    for i in range(n_pts):
        lines.InsertCellPoint(i)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetLines(lines)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly)
    tube.SetRadius(tube_radius_mm)
    tube.SetNumberOfSides(n_sides)
    tube.SetCapping(True)
    tube.Update()
    return tube.GetOutput()


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture(scope="module")
def straight_tube():
    return _make_straight_tube(length_mm=30.0, radius_mm=3.0)


@pytest.fixture(scope="module")
def curved_tube():
    return _make_curved_tube(arc_deg=90.0, arc_radius_mm=20.0, tube_radius_mm=2.5)


@pytest.fixture(scope="module")
def straight_result(straight_tube):
    from prospective.processing.centerline import CenterlineExtractor
    extractor = CenterlineExtractor(voxel_size_mm=1.0)
    return extractor.extract(
        straight_tube,
        source_mm=(0.0, 0.0, 1.0),
        target_mm=(0.0, 0.0, 29.0),
    )


@pytest.fixture(scope="module")
def curved_result(curved_tube):
    from prospective.processing.centerline import CenterlineExtractor
    extractor = CenterlineExtractor(voxel_size_mm=1.0)
    # Source and target follow the arc
    arc_r = 20.0
    src = (0.0, 0.0, 1.0)
    tgt = (arc_r, 0.0, arc_r - 1.0)
    return extractor.extract(curved_tube, source_mm=src, target_mm=tgt)


# ──────────────────────────────────────────────────────────────────────────── #
# CenterlineResult dataclass                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

class TestCenterlineResult:
    def test_import(self):
        from prospective.processing.centerline import CenterlineResult
        assert CenterlineResult is not None

    def test_summary(self, straight_result):
        s = straight_result.summary()
        assert "mm" in s
        assert "Tortuosidad" in s

    def test_points_shape(self, straight_result):
        pts = straight_result.points
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert len(pts) >= 2

    def test_radii_length_matches_points(self, straight_result):
        assert len(straight_result.radii) == len(straight_result.points)

    def test_radii_positive(self, straight_result):
        assert (straight_result.radii >= 0).all()

    def test_poly_data_present(self, straight_result):
        assert straight_result.poly_data is not None
        assert straight_result.poly_data.GetNumberOfPoints() > 0


# ──────────────────────────────────────────────────────────────────────────── #
# Straight tube — geometry metrics                                               #
# ──────────────────────────────────────────────────────────────────────────── #

class TestStraightTube:
    def test_arc_length_approx_28mm(self, straight_result):
        """Source at z=1, target at z=29 → path ≈ 28 mm."""
        assert 24.0 <= straight_result.arc_length_mm <= 32.0

    def test_tortuosity_near_one(self, straight_result):
        """Straight tube → tortuosity ≈ 1.0 (tolerance ±0.15)."""
        assert 1.0 <= straight_result.tortuosity <= 1.15

    def test_tortuosity_index_near_zero(self, straight_result):
        assert straight_result.tortuosity_index < 0.15

    def test_chord_length_approx_arc(self, straight_result):
        """Chord and arc should be close for a straight tube."""
        ratio = straight_result.arc_length_mm / straight_result.chord_length_mm
        assert 1.0 <= ratio <= 1.15

    def test_mean_radius_approx_3mm(self, straight_result):
        """Tube radius = 3 mm → EDT radius ≈ 3 mm (within 40%)."""
        assert 1.5 <= straight_result.mean_radius_mm <= 4.5

    def test_min_radius_positive(self, straight_result):
        assert straight_result.min_radius_mm > 0.0

    def test_max_radius_geq_mean(self, straight_result):
        assert straight_result.max_radius_mm >= straight_result.mean_radius_mm


# ──────────────────────────────────────────────────────────────────────────── #
# Curved tube — tortuosity                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

class TestCurvedTube:
    def test_tortuosity_above_one(self, curved_result):
        """Curved tube must have tortuosity > 1.0."""
        assert curved_result.tortuosity > 1.0

    def test_arc_longer_than_chord(self, curved_result):
        assert curved_result.arc_length_mm > curved_result.chord_length_mm

    def test_tortuosity_index_positive(self, curved_result):
        assert curved_result.tortuosity_index > 0.0

    def test_result_has_poly_data(self, curved_result):
        assert curved_result.poly_data is not None


# ──────────────────────────────────────────────────────────────────────────── #
# CenterlineExtractor API                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

class TestExtractorAPI:
    def test_convenience_function(self, straight_tube):
        from prospective.processing.centerline import extract_centerline
        r = extract_centerline(
            straight_tube,
            source_mm=(0.0, 0.0, 1.0),
            target_mm=(0.0, 0.0, 29.0),
            voxel_size_mm=1.0,
        )
        assert r.arc_length_mm > 0

    def test_progress_callback_called(self, straight_tube):
        from prospective.processing.centerline import CenterlineExtractor
        calls = []
        def cb(f): calls.append(f)
        CenterlineExtractor(voxel_size_mm=1.0, progress_cb=cb).extract(
            straight_tube, (0, 0, 1), (0, 0, 29)
        )
        assert len(calls) > 0
        assert calls[-1] == 1.0

    def test_invalid_points_raise(self):
        """Points far outside mesh should still find a nearest interior voxel."""
        from prospective.processing.centerline import extract_centerline
        tube = _make_straight_tube(30.0, 3.0)
        # Points slightly outside but snap to interior
        r = extract_centerline(tube, (0.0, 0.0, 0.5), (0.0, 0.0, 29.5), voxel_size_mm=1.0)
        assert r.arc_length_mm > 0

    def test_coarser_voxel_still_works(self, straight_tube):
        from prospective.processing.centerline import extract_centerline
        r = extract_centerline(straight_tube, (0, 0, 1), (0, 0, 29), voxel_size_mm=1.5)
        assert r.arc_length_mm > 0
        assert r.tortuosity >= 1.0 - 1e-9   # FP tolerance for perfectly straight path


# ──────────────────────────────────────────────────────────────────────────── #
# Actor factories                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

class TestCenterlineActors:
    def test_build_centerline_actor(self, straight_result):
        from prospective.rendering.centerline_actor import build_centerline_actor
        actor = build_centerline_actor(straight_result)
        assert actor is not None
        assert actor.GetMapper() is not None

    def test_build_endpoint_actors(self, straight_result):
        from prospective.rendering.centerline_actor import build_endpoint_actors
        src, tgt = build_endpoint_actors(straight_result)
        assert src is not None
        assert tgt is not None

    def test_build_scalar_bar(self, straight_result):
        from prospective.rendering.centerline_actor import build_scalar_bar
        bar = build_scalar_bar(straight_result)
        assert bar is not None
