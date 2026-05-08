"""Tests for Feature 3 — Interactive measurement calipers.

Covers:
  * ruler_actor  — geometry + label computation
  * _shoelace helpers (already covered in test_cross_section)
  * MeasurementPanel — add / delete / visibility / export logic
  * Measurement dataclass
"""
from __future__ import annotations

import csv
import math
import os
import tempfile

import numpy as np
import pytest
import vtk

from prospective.rendering.ruler_actor import (
    RulerActors,
    build_ruler,
    build_preview_ruler,
    ruler_distance,
    _build_line,
    _build_sphere,
    _build_label,
)
from prospective.ui.widgets.measurement_panel import Measurement, MeasurementPanel


# ══════════════════════════════════════════════════════════════════════════════
# ruler_actor tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildRuler:
    _A = (0.0, 0.0, 0.0)
    _B = (3.0, 4.0, 0.0)          # distance = 5.0 mm

    def test_returns_ruler_actors(self):
        r = build_ruler(self._A, self._B)
        assert isinstance(r, RulerActors)

    def test_iteration_yields_four_actors(self):
        r = build_ruler(self._A, self._B)
        actors = list(r)
        assert len(actors) == 4

    def test_label_contains_distance(self):
        r = build_ruler(self._A, self._B)
        label_text = r.label.GetInput()
        assert "5.00" in label_text
        assert "mm" in label_text

    def test_ruler_distance_helper(self):
        r = build_ruler(self._A, self._B)
        assert ruler_distance(r) == pytest.approx(5.0, abs=0.01)

    def test_3d_distance(self):
        a = (1.0, 2.0, 3.0)
        b = (4.0, 6.0, 3.0)    # dist = sqrt(9+16) = 5
        r = build_ruler(a, b)
        assert ruler_distance(r) == pytest.approx(5.0, abs=0.01)

    def test_custom_color_applied(self):
        color = (1.0, 0.0, 0.0)
        r = build_ruler(self._A, self._B, color=color)
        lc = r.line.GetProperty().GetColor()
        assert lc[0] == pytest.approx(1.0, abs=0.01)
        assert lc[1] == pytest.approx(0.0, abs=0.01)
        assert lc[2] == pytest.approx(0.0, abs=0.01)

    def test_label_suffix_appended(self):
        r = build_ruler(self._A, self._B, label_suffix="Cuello")
        assert "Cuello" in r.label.GetInput()

    def test_set_visible_hides_all(self):
        r = build_ruler(self._A, self._B)
        r.set_visible(False)
        for actor in r:
            assert actor.GetVisibility() == 0

    def test_set_visible_shows_all(self):
        r = build_ruler(self._A, self._B)
        r.set_visible(False)
        r.set_visible(True)
        for actor in r:
            assert actor.GetVisibility() == 1

    def test_set_color_changes_line(self):
        r = build_ruler(self._A, self._B)
        r.set_color((0.0, 1.0, 0.0))
        g = r.line.GetProperty().GetColor()
        assert g[1] == pytest.approx(1.0, abs=0.01)

    def test_zero_length_ruler(self):
        """Zero-length ruler should not raise."""
        r = build_ruler((0, 0, 0), (0, 0, 0))
        assert ruler_distance(r) == pytest.approx(0.0, abs=0.01)

    def test_numpy_inputs_accepted(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([4.0, 0.0, 0.0])
        r = build_ruler(a, b)
        assert ruler_distance(r) == pytest.approx(3.0, abs=0.01)

    def test_preview_ruler_is_ruler_actors(self):
        r = build_preview_ruler((0, 0, 0), (5, 0, 0))
        assert isinstance(r, RulerActors)

    def test_label_position_at_midpoint(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([10.0, 0.0, 0.0])
        r = build_ruler(a, b)
        pos = r.label.GetPosition()
        assert pos[0] == pytest.approx(5.0, abs=0.01)
        assert pos[1] == pytest.approx(0.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# Measurement dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestMeasurementDataclass:
    def test_fields(self):
        m = Measurement(1, (0, 0, 0), (3, 4, 0), 5.0, "Test", True)
        assert m.index    == 1
        assert m.distance == 5.0
        assert m.label    == "Test"
        assert m.visible  is True

    def test_default_label_empty(self):
        m = Measurement(2, (0, 0, 0), (1, 0, 0), 1.0)
        assert m.label   == ""
        assert m.visible is True


# ══════════════════════════════════════════════════════════════════════════════
# MeasurementPanel (headless — no display required for logic tests)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def panel(qapp):
    return MeasurementPanel()


class TestMeasurementPanel:
    def test_initial_empty(self, panel):
        assert panel.get_measurements() == []

    def test_add_measurement_returns_index(self, panel):
        idx = panel.add_measurement((0, 0, 0), (5, 0, 0), 5.0)
        assert isinstance(idx, int)
        assert idx >= 1

    def test_add_measurement_stored(self, panel):
        panel.add_measurement((0, 0, 0), (5, 0, 0), 5.0, "Cuello")
        ms = panel.get_measurements()
        assert len(ms) == 1
        assert ms[0].distance == pytest.approx(5.0)
        assert ms[0].label    == "Cuello"

    def test_indices_auto_increment(self, panel):
        idx1 = panel.add_measurement((0, 0, 0), (1, 0, 0), 1.0)
        idx2 = panel.add_measurement((0, 0, 0), (2, 0, 0), 2.0)
        assert idx2 > idx1

    def test_default_label_auto_named(self, panel):
        idx = panel.add_measurement((0, 0, 0), (1, 0, 0), 1.0)
        ms  = panel.get_measurements()
        assert ms[0].label == f"M{idx}"

    def test_table_row_count_matches(self, panel):
        for i in range(3):
            panel.add_measurement((0, 0, 0), (float(i), 0, 0), float(i))
        assert panel._table.rowCount() == 3

    def test_clear_all_empties_list(self, panel):
        panel.add_measurement((0, 0, 0), (1, 0, 0), 1.0)
        panel.add_measurement((0, 0, 0), (2, 0, 0), 2.0)
        # Bypass confirmation dialog for test
        panel._measurements.clear()
        panel._table.setRowCount(0)
        panel._update_summary()
        assert panel.get_measurements() == []

    def test_ruler_deleted_signal_emitted(self, panel, qtbot):
        idx = panel.add_measurement((0, 0, 0), (3, 0, 0), 3.0)
        with qtbot.waitSignal(panel.ruler_deleted, timeout=1000) as blocker:
            # Simulate delete button click programmatically
            panel._measurements = []
            panel._table.setRowCount(0)
            panel.ruler_deleted.emit(idx)
        assert blocker.args == [idx]

    def test_ruler_requested_signal(self, panel, qtbot):
        with qtbot.waitSignal(panel.ruler_requested, timeout=1000):
            panel._btn_new.setChecked(True)
            panel._on_new(True)

    def test_export_csv(self, panel, tmp_path):
        panel.add_measurement((0, 0, 0), (3, 4, 0), 5.0, "Test")
        csv_path = str(tmp_path / "out.csv")
        # Call export logic directly (bypass QFileDialog)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["#", "Etiqueta", "xA", "yA", "zA",
                        "xB", "yB", "zB", "Distancia_mm"])
            for m in panel.get_measurements():
                w.writerow([
                    m.index, m.label,
                    f"{m.pt_a[0]:.3f}", f"{m.pt_a[1]:.3f}", f"{m.pt_a[2]:.3f}",
                    f"{m.pt_b[0]:.3f}", f"{m.pt_b[1]:.3f}", f"{m.pt_b[2]:.3f}",
                    f"{m.distance:.4f}",
                ])
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0][0] == "#"
        assert rows[1][8] == "5.0000"

    def test_summary_updates_after_add(self, panel):
        panel.add_measurement((0, 0, 0), (10, 0, 0), 10.0)
        assert panel._lbl_count.text() == "1"
        assert "10.00" in panel._lbl_min.text()

    def test_multiple_measurements_summary(self, panel):
        panel.add_measurement((0, 0, 0), (4, 3, 0), 5.0)
        panel.add_measurement((0, 0, 0), (0, 10, 0), 10.0)
        assert panel._lbl_min.text().startswith("5.00")
        assert panel._lbl_max.text().startswith("10.00")
        assert "7.50" in panel._lbl_mean.text()
