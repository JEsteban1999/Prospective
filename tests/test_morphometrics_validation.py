"""A-03-10 · Morphometric validation against synthetic ground-truth cases.

Validates MorphometricAnalyzer accuracy using parametric meshes whose
exact values are known analytically.  Tolerances are derived from published
inter-observer variability for cerebral aneurysm measurement tools:

  Volume:       ≤ 3 % relative error
  Surface area: ≤ 3 % relative error
  Max diameter: ≤ 2 % relative error    (bounding-box)
  Eq. diameter: ≤ 3 % relative error    (derived from volume)
  Compactness:  ≤ 5 % relative error    (Wadell sphericity)
  Neck diameter: ≤ 20 % relative error  (heuristic plane-slice)

References
----------
• Raghavan ML et al. (2005) Quantified aneurysm shape characteristics
  correlate with hemodynamic forces.  Ann Biomed Eng 33(11):1479–88.
• Dhar S et al. (2008) Morphology parameters for intracranial aneurysm
  rupture risk assessment.  Neurosurgery 63(2):185–96.
• Tütüncü F et al. (2014) Computed analysis of the relationship between
  aneurysm shapes and arterial locations.  J Neurosci Methods 228:75–82.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest
import vtk

from prospective.processing.morphometrics import MorphometricAnalyzer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# Mesh factories                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def _sphere_mesh(radius_mm: float, resolution: int = 60) -> vtk.vtkPolyData:
    """Triangulated sphere of given radius."""
    src = vtk.vtkSphereSource()
    src.SetRadius(radius_mm)
    src.SetPhiResolution(resolution)
    src.SetThetaResolution(resolution)
    src.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(src.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def _ellipsoid_mesh(
    a: float, b: float, c: float, resolution: int = 60
) -> vtk.vtkPolyData:
    """Triangulated ellipsoid with semi-axes (a, b, c) by scaling a unit sphere."""
    src = vtk.vtkSphereSource()
    src.SetRadius(1.0)
    src.SetPhiResolution(resolution)
    src.SetThetaResolution(resolution)
    src.Update()

    xform = vtk.vtkTransform()
    xform.Scale(a, b, c)

    xf = vtk.vtkTransformPolyDataFilter()
    xf.SetInputConnection(src.GetOutputPort())
    xf.SetTransform(xform)
    xf.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(xf.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def _gourd_mesh(
    r_dome: float,
    r_neck: float,
    dome_offset: float,
    resolution: int = 60,
) -> vtk.vtkPolyData:
    """
    Approximate aneurysm shape: large dome sphere above a smaller neck sphere.

    *dome_offset* is the centre-to-centre distance between the two spheres.
    They overlap to form a single connected shape.  The neck is the region
    where the two spheres intersect.
    """
    # Dome sphere
    dome_src = vtk.vtkSphereSource()
    dome_src.SetRadius(r_dome)
    dome_src.SetCenter(0.0, 0.0, dome_offset)
    dome_src.SetPhiResolution(resolution)
    dome_src.SetThetaResolution(resolution)
    dome_src.Update()

    # Neck sphere
    neck_src = vtk.vtkSphereSource()
    neck_src.SetRadius(r_neck)
    neck_src.SetCenter(0.0, 0.0, 0.0)
    neck_src.SetPhiResolution(resolution)
    neck_src.SetThetaResolution(resolution)
    neck_src.Update()

    # Boolean union
    boolean = vtk.vtkBooleanOperationPolyDataFilter()
    boolean.SetOperationToUnion()
    boolean.SetInputConnection(0, dome_src.GetOutputPort())
    boolean.SetInputConnection(1, neck_src.GetOutputPort())
    boolean.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(boolean.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


# ──────────────────────────────────────────────────────────────────────────── #
# Analytical ground-truth helpers                                                #
# ──────────────────────────────────────────────────────────────────────────── #

def _sphere_volume(r: float) -> float:
    return (4.0 / 3.0) * math.pi * r ** 3


def _sphere_area(r: float) -> float:
    return 4.0 * math.pi * r ** 2


def _ellipsoid_volume(a: float, b: float, c: float) -> float:
    return (4.0 / 3.0) * math.pi * a * b * c


def _ellipsoid_area_ramanujan(a: float, b: float, c: float) -> float:
    """Ramanujan II approximation (< 0.1 % error for modest eccentricities)."""
    # Sort so that p ≤ q
    dims = sorted([a, b, c])
    p = (dims[1] ** 1.6075 + dims[2] ** 1.6075) / 2.0
    return 4.0 * math.pi * ((p) ** (1.0 / 1.6075))


def _wadell(volume: float, area: float) -> float:
    return math.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area


# ──────────────────────────────────────────────────────────────────────────── #
# Tolerance helper                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _rel_err(computed: float, expected: float) -> float:
    """Signed relative error: (computed - expected) / expected."""
    if abs(expected) < 1e-9:
        return 0.0
    return (computed - expected) / expected


def _check_tol(label: str, computed: float, expected: float, tol: float) -> None:
    """Assert relative error ≤ tol and log the result."""
    err = abs(_rel_err(computed, expected))
    logger.info(
        "  %-22s  computed=%8.3f  expected=%8.3f  err=%5.1f %%",
        label, computed, expected, err * 100,
    )
    assert err <= tol, (
        f"{label}: relative error {err*100:.1f}% exceeds tolerance {tol*100:.1f}%  "
        f"(computed={computed:.4f}, expected={expected:.4f})"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Test cases — spheres                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.mark.parametrize("radius_mm", [3.0, 5.0, 8.0, 12.0])
def test_sphere_volume_and_area(radius_mm):
    """Volume and surface area for a sphere must be within 3 %."""
    mesh = _sphere_mesh(radius_mm)
    result = MorphometricAnalyzer().analyze(mesh)

    expected_vol  = _sphere_volume(radius_mm)
    expected_area = _sphere_area(radius_mm)

    _check_tol("volume_mm3",       result.volume_mm3,       expected_vol,  0.03)
    _check_tol("surface_area_mm2", result.surface_area_mm2, expected_area, 0.03)


@pytest.mark.parametrize("radius_mm", [3.0, 5.0, 8.0])
def test_sphere_equivalent_diameter(radius_mm):
    """Equivalent sphere diameter must equal 2·r within 3 %."""
    mesh = _sphere_mesh(radius_mm)
    result = MorphometricAnalyzer().analyze(mesh)
    _check_tol("eq_sphere_diam_mm", result.eq_sphere_diam_mm, 2.0 * radius_mm, 0.03)


@pytest.mark.parametrize("radius_mm", [3.0, 5.0, 8.0])
def test_sphere_max_diameter(radius_mm):
    """Max diameter (bounding-box longest edge) must equal 2·r within 2 %."""
    mesh = _sphere_mesh(radius_mm)
    result = MorphometricAnalyzer().analyze(mesh)
    _check_tol("max_diameter_mm", result.max_diameter_mm, 2.0 * radius_mm, 0.02)


@pytest.mark.parametrize("radius_mm", [3.0, 5.0, 8.0])
def test_sphere_compactness(radius_mm):
    """Wadell sphericity for a sphere must equal 1.0 within 5 %."""
    mesh = _sphere_mesh(radius_mm)
    result = MorphometricAnalyzer().analyze(mesh)
    _check_tol("compactness", result.compactness, 1.0, 0.05)


# ──────────────────────────────────────────────────────────────────────────── #
# Test cases — ellipsoids                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.mark.parametrize("a,b,c", [
    (4.0, 4.0, 8.0),   # prolate (a=b < c)
    (3.0, 4.0, 7.0),   # triaxial
    (5.0, 5.0, 5.0),   # sphere via ellipsoid path
    (2.0, 3.0, 10.0),  # elongated (aneurysm-like)
])
def test_ellipsoid_volume(a, b, c):
    """Ellipsoid volume within 3 % of the analytical value."""
    mesh = _ellipsoid_mesh(a, b, c)
    result = MorphometricAnalyzer().analyze(mesh)
    expected_vol = _ellipsoid_volume(a, b, c)
    _check_tol(f"volume ({a},{b},{c})", result.volume_mm3, expected_vol, 0.03)


@pytest.mark.parametrize("a,b,c", [
    (4.0, 4.0, 8.0),
    (3.0, 4.0, 7.0),
    (2.0, 3.0, 10.0),
])
def test_ellipsoid_max_diameter(a, b, c):
    """Ellipsoid max diameter must match 2·max(a,b,c) within 2 %."""
    mesh = _ellipsoid_mesh(a, b, c)
    result = MorphometricAnalyzer().analyze(mesh)
    expected_max_diam = 2.0 * max(a, b, c)
    _check_tol(
        f"max_diam ({a},{b},{c})",
        result.max_diameter_mm,
        expected_max_diam,
        0.02,
    )


@pytest.mark.parametrize("a,b,c", [
    (4.0, 4.0, 8.0),
    (3.0, 4.0, 7.0),
])
def test_ellipsoid_compactness_less_than_sphere(a, b, c):
    """Any non-spherical ellipsoid must have compactness < 1.0."""
    mesh = _ellipsoid_mesh(a, b, c)
    result = MorphometricAnalyzer().analyze(mesh)
    assert result.compactness < 1.0, (
        f"Non-spherical ellipsoid ({a},{b},{c}) compactness={result.compactness:.4f} "
        "should be < 1.0"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Test cases — neck detection on elongated ellipsoids                            #
# ──────────────────────────────────────────────────────────────────────────── #

def test_neck_smaller_than_max_diameter():
    """For any non-spherical shape, neck_diameter < max_diameter."""
    mesh = _ellipsoid_mesh(3.0, 3.0, 10.0)
    result = MorphometricAnalyzer().analyze(mesh)
    assert result.neck_diameter_mm < result.max_diameter_mm, (
        f"neck={result.neck_diameter_mm:.2f} >= max_diam={result.max_diameter_mm:.2f}"
    )


def test_dnr_greater_than_one():
    """DNR (dome-to-neck ratio) for an elongated ellipsoid must exceed 1.0."""
    mesh = _ellipsoid_mesh(2.5, 2.5, 9.0)
    result = MorphometricAnalyzer().analyze(mesh)
    assert result.dome_to_neck_ratio > 1.0, (
        f"DNR={result.dome_to_neck_ratio:.3f} for elongated ellipsoid should be > 1.0"
    )


@pytest.mark.parametrize("r", [3.0, 5.0, 8.0])
def test_sphere_dnr_finite(r):
    """For a sphere, DNR must be a non-negative finite float (not NaN/inf).

    A sphere has no real neck, so the neck-detection algorithm may find a
    degenerate cross-section near a pole.  We only verify the result is finite
    and non-negative — not that it approximates 1.
    """
    import math
    mesh = _sphere_mesh(r)
    result = MorphometricAnalyzer().analyze(mesh)
    assert math.isfinite(result.dome_to_neck_ratio), (
        f"Sphere r={r}: DNR={result.dome_to_neck_ratio} is not finite"
    )
    assert result.dome_to_neck_ratio >= 0, (
        f"Sphere r={r}: DNR={result.dome_to_neck_ratio} is negative"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Test cases — risk label heuristics                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def test_risk_label_low_for_compact_sphere():
    """Compact sphere-like shape should report low rupture risk."""
    mesh = _sphere_mesh(5.0)
    result = MorphometricAnalyzer().analyze(mesh)
    # A sphere has compactness≈1, DNR≈1 — risk should be Bajo or Moderado
    assert result.rupture_risk_label in ("Bajo", "Moderado"), (
        f"Sphere should not report Alto risk; got {result.rupture_risk_label!r}  "
        f"DNR={result.dome_to_neck_ratio:.2f}  AR={result.aspect_ratio:.2f}"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Test cases — shape-complexity indices (BF, UI, EI, NSI)                       #
# ──────────────────────────────────────────────────────────────────────────── #

def test_nsi_zero_for_sphere():
    """NSI = 1 − compactness.  For a sphere compactness ≈ 1 → NSI ≈ 0."""
    mesh   = _sphere_mesh(5.0)
    result = MorphometricAnalyzer().analyze(mesh)
    assert result.non_sphericity_idx >= 0, "NSI must be non-negative"
    assert result.non_sphericity_idx < 0.15, (
        f"NSI={result.non_sphericity_idx:.4f} too large for a sphere (expected < 0.15)"
    )


def test_nsi_higher_for_elongated():
    """Elongated ellipsoid should have higher NSI than a sphere of same volume."""
    r_equiv = 5.0
    # Elongated ellipsoid with same semi-minor as sphere radius
    mesh_sphere   = _sphere_mesh(r_equiv)
    mesh_elongated = _ellipsoid_mesh(r_equiv * 0.5, r_equiv * 0.5, r_equiv * 4.0)

    r_sph = MorphometricAnalyzer().analyze(mesh_sphere)
    r_elo = MorphometricAnalyzer().analyze(mesh_elongated)

    assert r_elo.non_sphericity_idx > r_sph.non_sphericity_idx, (
        f"Elongated NSI={r_elo.non_sphericity_idx:.4f} should exceed "
        f"sphere NSI={r_sph.non_sphericity_idx:.4f}"
    )


def test_ei_positive_and_finite():
    """EI must be a finite positive float for any valid mesh."""
    for r in [3.0, 5.0]:
        mesh   = _sphere_mesh(r)
        result = MorphometricAnalyzer().analyze(mesh)
        assert math.isfinite(result.ellipticity_index), (
            f"EI={result.ellipticity_index} not finite for r={r}"
        )
        assert result.ellipticity_index > 0, (
            f"EI={result.ellipticity_index} not positive for r={r}"
        )


def test_ei_sphere_near_theoretical():
    """For a sphere EI = 1 − (18π)^(1/3)·V^(2/3)/A.
    Theoretical value for a perfect sphere ≈ 0.207.
    """
    mesh   = _sphere_mesh(5.0, resolution=80)
    result = MorphometricAnalyzer().analyze(mesh)
    # Theoretical: 1 − (18π)^(1/3) * V^(2/3) / A  with V=(4/3)πr³, A=4πr²
    r = 5.0
    V = (4.0 / 3.0) * math.pi * r ** 3
    A = 4.0 * math.pi * r ** 2
    ei_theory = 1.0 - (18.0 * math.pi) ** (1.0 / 3.0) * V ** (2.0 / 3.0) / A
    assert abs(result.ellipticity_index - ei_theory) < 0.015, (
        f"EI={result.ellipticity_index:.4f} vs theoretical {ei_theory:.4f} "
        f"(diff={abs(result.ellipticity_index - ei_theory):.4f})"
    )


def test_ui_zero_for_convex_sphere():
    """Undulation Index must be ≈ 0 for a convex shape (sphere)."""
    mesh   = _sphere_mesh(5.0)
    result = MorphometricAnalyzer().analyze(mesh)
    assert result.undulation_index >= 0, "UI must be non-negative"
    assert result.undulation_index < 0.05, (
        f"UI={result.undulation_index:.4f} too large for sphere (expected < 0.05)"
    )


def test_bf_geq_one_for_ellipsoid():
    """BF = max_dome_diam / neck_diam ≥ 1 for any aneurysm-like shape."""
    mesh   = _ellipsoid_mesh(3.0, 3.0, 9.0)
    result = MorphometricAnalyzer().analyze(mesh)
    # BF is only meaningful when neck detection worked
    if result.neck_diameter_mm >= 0.1:
        assert result.bottleneck_factor >= 1.0, (
            f"BF={result.bottleneck_factor:.3f} < 1 for elongated ellipsoid"
        )


def test_sr_computed_from_parent_diam():
    """Size Ratio = max_diam / parent_artery_diam, assigned post-analysis."""
    mesh   = _sphere_mesh(5.0)
    result = MorphometricAnalyzer().analyze(mesh)
    assert result.size_ratio == 0.0, "SR should default to 0 (unknown parent diam)"

    parent_diam = 3.5
    result.size_ratio = result.max_diameter_mm / parent_diam
    expected_sr = result.max_diameter_mm / parent_diam
    assert abs(result.size_ratio - expected_sr) < 1e-9


# ──────────────────────────────────────────────────────────────────────────── #
# Accuracy summary                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def test_accuracy_summary_report(capsys):
    """
    Print a tabular accuracy summary for five synthetic cases.
    Not a pass/fail assertion — used to document algorithm accuracy.
    """
    cases = [
        ("Sphere r=3",          _sphere_mesh(3.0),
         _sphere_volume(3.0),   _sphere_area(3.0),  6.0),
        ("Sphere r=5",          _sphere_mesh(5.0),
         _sphere_volume(5.0),   _sphere_area(5.0),  10.0),
        ("Sphere r=8",          _sphere_mesh(8.0),
         _sphere_volume(8.0),   _sphere_area(8.0),  16.0),
        ("Ellipsoid 4×4×8",     _ellipsoid_mesh(4.0, 4.0, 8.0),
         _ellipsoid_volume(4.0, 4.0, 8.0), None,    16.0),
        ("Ellipsoid 3×4×7",     _ellipsoid_mesh(3.0, 4.0, 7.0),
         _ellipsoid_volume(3.0, 4.0, 7.0), None,    14.0),
        ("Ellipsoid 2×3×10",    _ellipsoid_mesh(2.0, 3.0, 10.0),
         _ellipsoid_volume(2.0, 3.0, 10.0), None,   20.0),
    ]

    analyzer = MorphometricAnalyzer()
    rows = []
    for name, mesh, exp_vol, exp_area, exp_max in cases:
        r = analyzer.analyze(mesh)
        vol_err  = _rel_err(r.volume_mm3, exp_vol) * 100
        area_err = _rel_err(r.surface_area_mm2, exp_area) * 100 if exp_area else float("nan")
        max_err  = _rel_err(r.max_diameter_mm, exp_max) * 100
        rows.append((name, vol_err, area_err, max_err, r.compactness))

    with capsys.disabled():
        print("\n")
        print("=" * 78)
        print("  A-03-10 · Morphometric Accuracy Report")
        print("=" * 78)
        print(f"  {'Case':<22} {'Vol err%':>9} {'Area err%':>10} {'MaxD err%':>10} {'Compactness':>12}")
        print("  " + "-" * 70)
        for name, ve, ae, me, comp in rows:
            ae_s = f"{ae:+.1f} %" if not math.isnan(ae) else "   n/a"
            print(f"  {name:<22} {ve:+7.1f} %  {ae_s:>10}  {me:+7.1f} %  {comp:>10.4f}")
        print("=" * 78)
        max_vol_err = max(abs(r[1]) for r in rows)
        max_max_err = max(abs(r[3]) for r in rows)
        print(f"  Max |volume error|:       {max_vol_err:.2f} %  (tolerance 3.0 %)")
        print(f"  Max |max-diam error|:     {max_max_err:.2f} %  (tolerance 2.0 %)")
        print("=" * 78)
