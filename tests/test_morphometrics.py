"""Integration tests — morphometric analysis (A-04-09)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import vtk

from prospective.processing.morphometrics import MorphometricAnalyzer, MorphometricResult


def _sphere_poly_data(radius_mm: float = 5.0, resolution: int = 32) -> vtk.vtkPolyData:
    """Generate a VTK sphere as ground truth for morphometric checks."""
    src = vtk.vtkSphereSource()
    src.SetRadius(radius_mm)
    src.SetThetaResolution(resolution)
    src.SetPhiResolution(resolution)
    src.Update()

    # Close the surface with triangle filter + normals
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(src.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


class TestMorphometricResult:

    def test_rupture_risk_high(self):
        r = MorphometricResult(
            volume_mm3=0, surface_area_mm2=0, eq_sphere_diam_mm=0,
            bbox_l_mm=0, bbox_w_mm=0, bbox_h_mm=0, max_diameter_mm=0,
            neck_diameter_mm=1.0,
            dome_height_mm=2.0,
            neck_plane_pos=0.3,
            dome_to_neck_ratio=2.1,   # > 2.0 → Alto
            aspect_ratio=1.0,
            compactness=0.8,
        )
        assert r.rupture_risk_label == "Alto"

    def test_rupture_risk_moderate_by_dnr(self):
        r = MorphometricResult(
            volume_mm3=0, surface_area_mm2=0, eq_sphere_diam_mm=0,
            bbox_l_mm=0, bbox_w_mm=0, bbox_h_mm=0, max_diameter_mm=0,
            neck_diameter_mm=1.0, dome_height_mm=1.0, neck_plane_pos=0.3,
            dome_to_neck_ratio=1.7,   # ≥1.6 → Moderado
            aspect_ratio=1.0,
            compactness=0.8,
        )
        assert r.rupture_risk_label == "Moderado"

    def test_rupture_risk_moderate_by_ar(self):
        r = MorphometricResult(
            volume_mm3=0, surface_area_mm2=0, eq_sphere_diam_mm=0,
            bbox_l_mm=0, bbox_w_mm=0, bbox_h_mm=0, max_diameter_mm=0,
            neck_diameter_mm=1.0, dome_height_mm=1.4, neck_plane_pos=0.3,
            dome_to_neck_ratio=1.2,
            aspect_ratio=1.4,         # ≥1.3 → Moderado
            compactness=0.8,
        )
        assert r.rupture_risk_label == "Moderado"

    def test_rupture_risk_low(self):
        r = MorphometricResult(
            volume_mm3=0, surface_area_mm2=0, eq_sphere_diam_mm=0,
            bbox_l_mm=0, bbox_w_mm=0, bbox_h_mm=0, max_diameter_mm=0,
            neck_diameter_mm=1.0, dome_height_mm=1.0, neck_plane_pos=0.3,
            dome_to_neck_ratio=1.2,
            aspect_ratio=1.1,
            compactness=0.8,
        )
        assert r.rupture_risk_label == "Bajo"


class TestMorphometricAnalyzer:

    def test_sphere_volume_approx(self):
        """Volume of a 5 mm radius sphere ≈ 523.6 mm³."""
        poly = _sphere_poly_data(radius_mm=5.0)
        result = MorphometricAnalyzer().analyze(poly)
        assert result is not None
        expected = (4 / 3) * math.pi * 5.0 ** 3   # ≈ 523.6
        assert result.volume_mm3 == pytest.approx(expected, rel=0.10)

    def test_sphere_surface_area_approx(self):
        """Surface area of a 5 mm sphere ≈ 314.2 mm²."""
        poly = _sphere_poly_data(radius_mm=5.0)
        result = MorphometricAnalyzer().analyze(poly)
        expected = 4 * math.pi * 5.0 ** 2   # ≈ 314.2
        assert result.surface_area_mm2 == pytest.approx(expected, rel=0.10)

    def test_sphere_compactness_near_one(self):
        """Wadell sphericity of a perfect sphere == 1.0."""
        poly = _sphere_poly_data(radius_mm=5.0, resolution=64)
        result = MorphometricAnalyzer().analyze(poly)
        assert result.compactness == pytest.approx(1.0, abs=0.05)

    def test_sphere_max_diameter_approx(self):
        """Max diameter of a 5 mm sphere ≈ 10 mm."""
        poly = _sphere_poly_data(radius_mm=5.0)
        result = MorphometricAnalyzer().analyze(poly)
        assert result.max_diameter_mm == pytest.approx(10.0, rel=0.10)

    def test_eq_sphere_diameter_consistent_with_volume(self):
        poly = _sphere_poly_data(radius_mm=4.0)
        result = MorphometricAnalyzer().analyze(poly)
        # eq_sphere_diam = (6*V/pi)^(1/3)
        expected = (6 * result.volume_mm3 / math.pi) ** (1 / 3)
        assert result.eq_sphere_diam_mm == pytest.approx(expected, rel=0.01)

    def test_principal_axis_is_unit_vector(self):
        poly = _sphere_poly_data(radius_mm=5.0)
        result = MorphometricAnalyzer().analyze(poly)
        ax = result.principal_axis
        length = math.sqrt(sum(v ** 2 for v in ax))
        assert length == pytest.approx(1.0, abs=1e-6)

    def test_neck_diameter_less_than_max(self):
        poly = _sphere_poly_data(radius_mm=5.0)
        result = MorphometricAnalyzer().analyze(poly)
        assert result.neck_diameter_mm <= result.max_diameter_mm + 0.1
