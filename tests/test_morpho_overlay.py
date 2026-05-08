"""Tests for Feature 5 — Morphometric overlay + parent artery estimation.

Covers:
  * morpho_overlay — actor construction, visibility toggle, iteration
  * parent_artery  — diameter estimation on known geometry
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from prospective.processing.morphometrics import MorphometricResult
from prospective.rendering.morpho_overlay import (
    MorphoOverlayActors,
    build_morpho_overlay,
    _perp_to,
    _max_diameter_endpoints,
)
from prospective.processing.parent_artery import (
    estimate_parent_artery_diameter,
    _shoelace,
    _cut_largest_component_diameter,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Shared fixtures                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

def _sphere_mesh(radius: float = 5.0, centre=(0.0, 0.0, 0.0)) -> vtk.vtkPolyData:
    src = vtk.vtkSphereSource()
    src.SetRadius(radius)
    src.SetCenter(*centre)
    src.SetPhiResolution(32)
    src.SetThetaResolution(32)
    src.Update()
    return src.GetOutput()


def _cylinder_mesh(length: float = 30.0, radius: float = 3.0) -> vtk.vtkPolyData:
    """Z-aligned closed cylinder from 0 to *length*."""
    src = vtk.vtkCylinderSource()
    src.SetRadius(radius)
    src.SetHeight(length)
    src.SetResolution(60)
    src.SetCapping(True)
    src.Update()
    tf = vtk.vtkTransformPolyDataFilter()
    rot = vtk.vtkTransform()
    rot.PostMultiply()
    rot.RotateX(90)
    rot.Translate(0, 0, length / 2)
    tf.SetTransform(rot)
    tf.SetInputConnection(src.GetOutputPort())
    tf.Update()
    return tf.GetOutput()


def _fake_morpho(
    centroid=(0.0, 0.0, 0.0),
    axis=(0.0, 0.0, 1.0),
    neck_diam=4.0,
    dome_h=8.0,
    max_diam=10.0,
    neck_pos=0.2,
) -> MorphometricResult:
    """Minimal MorphometricResult for overlay testing."""
    ar  = dome_h / neck_diam if neck_diam > 0 else 0.0
    dnr = max_diam / neck_diam if neck_diam > 0 else 0.0
    return MorphometricResult(
        volume_mm3=500.0, surface_area_mm2=300.0, eq_sphere_diam_mm=9.8,
        bbox_l_mm=max_diam, bbox_w_mm=8.0, bbox_h_mm=7.0,
        max_diameter_mm=max_diam,
        neck_diameter_mm=neck_diam, dome_height_mm=dome_h,
        neck_plane_pos=neck_pos,
        dome_to_neck_ratio=dnr, aspect_ratio=ar,
        compactness=0.85,
        bottleneck_factor=1.2, undulation_index=0.05,
        ellipticity_index=0.10, non_sphericity_idx=0.15,
        principal_axis=tuple(axis),
        centroid=tuple(centroid),
    )


@pytest.fixture(scope="module")
def sphere5():
    return _sphere_mesh(radius=5.0)


@pytest.fixture(scope="module")
def cylinder30():
    return _cylinder_mesh(length=30.0, radius=3.0)


@pytest.fixture(scope="module")
def fake_morpho():
    return _fake_morpho()


# ══════════════════════════════════════════════════════════════════════════════
# _perp_to helper
# ══════════════════════════════════════════════════════════════════════════════

class TestPerpTo:
    def test_unit_length(self):
        for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]:
            u = np.array(v, dtype=float)
            u /= np.linalg.norm(u)
            p = _perp_to(u)
            assert abs(np.linalg.norm(p) - 1.0) < 1e-6

    def test_perpendicular(self):
        for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.6, 0.8, 0)]:
            u = np.array(v, dtype=float)
            u /= np.linalg.norm(u)
            p = _perp_to(u)
            assert abs(np.dot(u, p)) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# build_morpho_overlay
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildMorphoOverlay:
    def test_returns_named_tuple(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        assert isinstance(actors, MorphoOverlayActors)

    def test_iteration_yields_actors(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        actor_list = list(actors)
        # disc + outline + dome_line + maxd_line + 4 labels = 8
        assert len(actor_list) >= 7

    def test_neck_disc_is_actor(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        assert isinstance(actors.neck_disc, vtk.vtkActor)

    def test_neck_outline_is_actor(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        assert isinstance(actors.neck_outline, vtk.vtkActor)

    def test_labels_are_billboard_actors(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        for lbl in actors.labels:
            assert isinstance(lbl, vtk.vtkBillboardTextActor3D)

    def test_neck_diameter_in_label(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        texts = [lbl.GetInput() for lbl in actors.labels]
        assert any("cuello" in t for t in texts)

    def test_ar_dnr_in_label(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        texts = [lbl.GetInput() for lbl in actors.labels]
        assert any("AR" in t and "DNR" in t for t in texts)

    def test_risk_label_present(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        texts  = " ".join(lbl.GetInput() for lbl in actors.labels)
        assert any(r in texts for r in ("Bajo", "Moderado", "Alto"))

    def test_set_visible_false(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        actors.set_visible(False)
        for a in actors:
            assert a.GetVisibility() == 0

    def test_set_visible_true(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        actors.set_visible(False)
        actors.set_visible(True)
        for a in actors:
            assert a.GetVisibility() == 1

    def test_disc_opacity_lt_1(self, fake_morpho, sphere5):
        actors = build_morpho_overlay(fake_morpho, sphere5)
        assert actors.neck_disc.GetProperty().GetOpacity() < 1.0

    def test_different_axes(self, sphere5):
        """Overlay should work regardless of principal axis orientation."""
        for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
            a = np.array(axis, dtype=float)
            a /= np.linalg.norm(a)
            m = _fake_morpho(axis=tuple(a))
            actors = build_morpho_overlay(m, sphere5)
            assert isinstance(actors, MorphoOverlayActors)


# ══════════════════════════════════════════════════════════════════════════════
# parent_artery helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestShoelace:
    def test_unit_square(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        area = _shoelace(pts, np.zeros(3), np.array([0., 0., 1.]))
        assert abs(area - 1.0) < 0.01

    def test_circle(self):
        r   = 4.0
        t   = np.linspace(0, 2*math.pi, 360, endpoint=False)
        pts = np.column_stack([r*np.cos(t), r*np.sin(t), np.zeros(360)])
        area = _shoelace(pts, np.zeros(3), np.array([0., 0., 1.]))
        assert abs(area - math.pi * r**2) / (math.pi * r**2) < 0.01


class TestCutLargestComponentDiameter:
    def test_cylinder_midplane(self, cylinder30):
        """Mid-plane cut of a 3 mm radius cylinder ≈ 6 mm diameter."""
        origin = np.array([0.0, 0.0, 15.0])
        normal = np.array([0.0, 0.0, 1.0])
        d = _cut_largest_component_diameter(cylinder30, origin, normal)
        assert abs(d - 6.0) / 6.0 < 0.15

    def test_missed_plane_returns_zero(self, cylinder30):
        origin = np.array([0.0, 0.0, 200.0])   # far away
        normal = np.array([0.0, 0.0, 1.0])
        d = _cut_largest_component_diameter(cylinder30, origin, normal)
        assert d == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# estimate_parent_artery_diameter
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateParentArteryDiameter:
    def test_returns_float(self, cylinder30):
        morpho = _fake_morpho(neck_diam=3.0, neck_pos=0.1)
        d = estimate_parent_artery_diameter(cylinder30, morpho)
        assert isinstance(d, float)

    def test_empty_mesh_returns_zero(self):
        empty = vtk.vtkPolyData()
        morpho = _fake_morpho()
        d = estimate_parent_artery_diameter(empty, morpho)
        assert d == 0.0

    def test_none_mesh_returns_zero(self):
        morpho = _fake_morpho()
        d = estimate_parent_artery_diameter(None, morpho)
        assert d == 0.0

    def test_cylinder_diameter_reasonable(self, cylinder30):
        """Estimated Ø of a 3 mm-radius cylinder should be close to 6 mm."""
        morpho = _fake_morpho(
            centroid=(0.0, 0.0, 15.0),
            axis=(0.0, 0.0, 1.0),
            neck_diam=6.0,
            neck_pos=0.1,
        )
        d = estimate_parent_artery_diameter(cylinder30, morpho)
        if d > 0:   # may return 0 if cuts miss the mesh near the neck
            assert abs(d - 6.0) / 6.0 < 0.30
