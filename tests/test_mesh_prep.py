"""Tests for Feature 7 — 3D-print mesh preparation.

Covers:
  * prepare_mesh_for_print — scaling, dimensions, volume, watertight check
  * PrintPrepResult — fits_in_bed, export_stl, summary
  * PrintPrepPanel (headless Qt) — set_mesh, button state
"""
from __future__ import annotations

import os
import math

import numpy as np
import pytest
import vtk

from prospective.processing.mesh_prep import (
    prepare_mesh_for_print,
    PrintPrepResult,
    PRINT_BED_PRESETS,
)
from prospective.ui.widgets.print_prep_panel import PrintPrepPanel


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures / helpers                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def _sphere_mesh(radius: float = 5.0, theta: int = 20, phi: int = 20) -> vtk.vtkPolyData:
    """Return a closed sphere mesh."""
    src = vtk.vtkSphereSource()
    src.SetRadius(radius)
    src.SetThetaResolution(theta)
    src.SetPhiResolution(phi)
    src.Update()
    return src.GetOutput()


def _open_mesh() -> vtk.vtkPolyData:
    """Return an open-topped cylinder (not watertight)."""
    src = vtk.vtkCylinderSource()
    src.SetRadius(3.0)
    src.SetHeight(8.0)
    src.SetResolution(16)
    src.CappingOff()   # no end-caps → open edges
    src.Update()
    return src.GetOutput()


# ══════════════════════════════════════════════════════════════════════════════
# prepare_mesh_for_print
# ══════════════════════════════════════════════════════════════════════════════

class TestPrepareMeshForPrint:

    def test_returns_result_object(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=40.0)
        assert isinstance(r, PrintPrepResult)

    def test_scale_applied_to_max_dim(self):
        mesh = _sphere_mesh(radius=5.0)   # diameter ≈ 10 mm
        r = prepare_mesh_for_print(mesh, target_size_mm=50.0, smooth_iterations=0)
        assert max(r.dimensions_mm) == pytest.approx(50.0, abs=1.0)

    def test_scale_factor_positive(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=40.0)
        assert r.scale_factor > 0.0

    def test_no_scaling_when_target_zero(self):
        mesh = _sphere_mesh(radius=5.0)
        r = prepare_mesh_for_print(mesh, target_size_mm=0.0, smooth_iterations=0)
        assert r.scale_factor == pytest.approx(1.0)

    def test_dimensions_tuple_length(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=30.0)
        assert len(r.dimensions_mm) == 3

    def test_dimensions_all_positive(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=30.0)
        assert all(d > 0.0 for d in r.dimensions_mm)

    def test_volume_positive(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=30.0)
        assert r.volume_cm3 > 0.0

    def test_surface_area_positive(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=30.0)
        assert r.surface_area_cm2 > 0.0

    def test_sphere_volume_approx(self):
        """Scaled sphere volume should match analytic 4/3 π r³."""
        target = 60.0  # mm
        r_actual = target / 2.0
        expected_cm3 = (4 / 3) * math.pi * (r_actual ** 3) / 1000.0
        mesh = _sphere_mesh(radius=5.0)
        result = prepare_mesh_for_print(mesh, target_size_mm=target,
                                        smooth_iterations=0)
        # Allow 5 % relative tolerance (sphere is discretised)
        assert result.volume_cm3 == pytest.approx(expected_cm3, rel=0.05)

    def test_watertight_sphere(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=40.0)
        assert r.is_watertight is True
        assert r.open_edge_count == 0

    def test_open_mesh_not_watertight(self):
        mesh = _open_mesh()
        r = prepare_mesh_for_print(
            mesh, target_size_mm=30.0, fill_holes=False, smooth_iterations=0
        )
        assert r.is_watertight is False
        assert r.open_edge_count > 0

    def test_fill_holes_closes_open_mesh(self):
        mesh = _open_mesh()
        r = prepare_mesh_for_print(
            mesh, target_size_mm=30.0, fill_holes=True, hole_size=50.0,
            smooth_iterations=0,
        )
        assert r.is_watertight is True

    def test_smooth_iterations_zero_no_crash(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=20.0, smooth_iterations=0)
        assert r.mesh is not None

    def test_subdivide_increases_triangles(self):
        mesh = _sphere_mesh(theta=10, phi=10)
        r_no_sub = prepare_mesh_for_print(
            mesh, target_size_mm=20.0, smooth_iterations=0, subdivide=False
        )
        r_sub = prepare_mesh_for_print(
            mesh, target_size_mm=20.0, smooth_iterations=0, subdivide=True
        )
        assert r_sub.mesh.GetNumberOfCells() > r_no_sub.mesh.GetNumberOfCells()

    def test_warnings_list_type(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=30.0)
        assert isinstance(r.warnings, list)

    def test_empty_mesh_raises(self):
        empty = vtk.vtkPolyData()
        with pytest.raises(ValueError):
            prepare_mesh_for_print(empty, target_size_mm=30.0)

    def test_none_mesh_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            prepare_mesh_for_print(None, target_size_mm=30.0)

    def test_progress_callback_called(self):
        calls = []
        mesh = _sphere_mesh()
        prepare_mesh_for_print(mesh, target_size_mm=30.0, progress_cb=calls.append)
        assert len(calls) > 0
        assert calls[-1] == 100

    def test_output_mesh_has_points(self):
        mesh = _sphere_mesh()
        r = prepare_mesh_for_print(mesh, target_size_mm=30.0)
        assert r.mesh.GetNumberOfPoints() > 0


# ══════════════════════════════════════════════════════════════════════════════
# PrintPrepResult helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestPrintPrepResult:

    def _make_result(self) -> PrintPrepResult:
        mesh = _sphere_mesh()
        return prepare_mesh_for_print(mesh, target_size_mm=40.0)

    def test_fits_in_bed_true(self):
        r = self._make_result()
        assert r.fits_in_bed((220.0, 220.0, 250.0)) is True

    def test_fits_in_bed_false_when_too_large(self):
        r = self._make_result()
        # Make result artificially large by hacking dims
        r = PrintPrepResult(
            mesh=r.mesh,
            scale_factor=1.0,
            dimensions_mm=(300.0, 300.0, 300.0),
            volume_cm3=r.volume_cm3,
            surface_area_cm2=r.surface_area_cm2,
            is_watertight=r.is_watertight,
            open_edge_count=r.open_edge_count,
        )
        assert r.fits_in_bed((220.0, 220.0, 250.0)) is False

    def test_fits_in_bed_unlimited_dimension(self):
        r = self._make_result()
        # Bed dimension 0 means unlimited
        assert r.fits_in_bed((0.0, 0.0, 0.0)) is True

    def test_summary_contains_key_info(self):
        r = self._make_result()
        s = r.summary()
        assert "Dimensiones" in s
        assert "Escala" in s
        assert "Volumen" in s

    def test_export_stl_creates_file(self, tmp_path):
        r = self._make_result()
        path = str(tmp_path / "test.stl")
        r.export_stl(path)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_export_stl_readable_by_vtk(self, tmp_path):
        r = self._make_result()
        path = str(tmp_path / "test.stl")
        r.export_stl(path)
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        assert reader.GetOutput().GetNumberOfPoints() > 0


# ══════════════════════════════════════════════════════════════════════════════
# PrintPrepPanel (headless Qt)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def panel(qapp):
    return PrintPrepPanel()


class TestPrintPrepPanel:

    def test_initial_prepare_disabled(self, panel):
        assert not panel._btn_prepare.isEnabled()

    def test_initial_export_disabled(self, panel):
        assert not panel._btn_export.isEnabled()

    def test_set_mesh_enables_prepare(self, panel):
        mesh = _sphere_mesh()
        panel.set_mesh(mesh)
        assert panel._btn_prepare.isEnabled()

    def test_set_mesh_none_disables_prepare(self, panel):
        mesh = _sphere_mesh()
        panel.set_mesh(mesh)
        panel.set_mesh(None)
        assert not panel._btn_prepare.isEnabled()

    def test_bed_presets_populated(self, panel):
        count = panel._combo_bed.count()
        assert count == len(PRINT_BED_PRESETS)

    def test_custom_bed_widget_hidden_by_default(self, panel):
        panel._combo_bed.setCurrentText(list(PRINT_BED_PRESETS.keys())[0])
        # isHidden() checks the widget's own state regardless of parent visibility
        assert panel._custom_bed_widget.isHidden()

    def test_custom_bed_widget_shown_for_personalizado(self, panel):
        panel._combo_bed.setCurrentText("Personalizado")
        assert not panel._custom_bed_widget.isHidden()

    def test_hole_size_enabled_when_fill_checked(self, panel):
        panel._chk_fill.setChecked(True)
        assert panel._spin_hole.isEnabled()

    def test_hole_size_disabled_when_fill_unchecked(self, panel):
        panel._chk_fill.setChecked(False)
        assert not panel._spin_hole.isEnabled()

    def test_prep_ready_signal_emitted(self, panel, qtbot):
        mesh = _sphere_mesh()
        panel.set_mesh(mesh)
        panel._spin_smooth.setValue(5)   # fast
        with qtbot.waitSignal(panel.prep_ready, timeout=30000) as blocker:
            panel._on_prepare()
        assert isinstance(blocker.args[0], PrintPrepResult)

    def test_export_enabled_after_prep(self, panel, qtbot):
        mesh = _sphere_mesh()
        panel.set_mesh(mesh)
        panel._spin_smooth.setValue(5)
        with qtbot.waitSignal(panel.prep_ready, timeout=30000):
            panel._on_prepare()
        assert panel._btn_export.isEnabled()
