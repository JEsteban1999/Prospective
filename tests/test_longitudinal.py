"""Tests for Feature 6 — Longitudinal morphometric comparison.

Covers:
  * MorphometricSnapshot — construction, from_morpho_result
  * LongitudinalSeries — add/remove, sorting, deltas, trends, export/import
  * LongitudinalPanel (headless logic)
"""
from __future__ import annotations

import csv
import math
import os
from datetime import date

import numpy as np
import pytest

from prospective.processing.longitudinal import (
    MorphometricSnapshot,
    LongitudinalSeries,
    TRACKED_METRICS,
    METRIC_LABELS,
)
from prospective.processing.morphometrics import MorphometricResult
from prospective.ui.widgets.longitudinal_panel import LongitudinalPanel


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def _fake_morpho_result(
    volume: float = 500.0,
    max_diam: float = 10.0,
    neck: float = 4.0,
    dome_h: float = 7.0,
    ar: float | None = None,
    dnr: float | None = None,
) -> MorphometricResult:
    ar  = ar  if ar  is not None else dome_h / neck if neck else 0.0
    dnr = dnr if dnr is not None else max_diam / neck if neck else 0.0
    return MorphometricResult(
        volume_mm3=volume, surface_area_mm2=300.0, eq_sphere_diam_mm=9.8,
        bbox_l_mm=max_diam, bbox_w_mm=8.0, bbox_h_mm=7.0,
        max_diameter_mm=max_diam,
        neck_diameter_mm=neck, dome_height_mm=dome_h,
        neck_plane_pos=0.2,
        dome_to_neck_ratio=dnr, aspect_ratio=ar,
        compactness=0.85,
        bottleneck_factor=1.2, undulation_index=0.05,
        ellipticity_index=0.10, non_sphericity_idx=0.15,
    )


def _snap(
    d: date,
    label: str = "",
    volume: float = 500.0,
    max_diam: float = 10.0,
) -> MorphometricSnapshot:
    r = _fake_morpho_result(volume=volume, max_diam=max_diam)
    return MorphometricSnapshot.from_morpho_result(r, session_date=d, label=label)


# ══════════════════════════════════════════════════════════════════════════════
# MorphometricSnapshot
# ══════════════════════════════════════════════════════════════════════════════

class TestMorphometricSnapshot:
    def test_from_morpho_result(self):
        r = _fake_morpho_result()
        s = MorphometricSnapshot.from_morpho_result(r, date(2024, 1, 1), "Baseline")
        assert s.session_date == date(2024, 1, 1)
        assert s.label        == "Baseline"
        assert s.volume_mm3   == pytest.approx(500.0)
        assert s.max_diameter_mm == pytest.approx(10.0)

    def test_default_date_is_today(self):
        r = _fake_morpho_result()
        s = MorphometricSnapshot.from_morpho_result(r)
        assert s.session_date == date.today()

    def test_get_metric(self):
        r = _fake_morpho_result(volume=123.4)
        s = MorphometricSnapshot.from_morpho_result(r)
        assert s.get_metric("volume_mm3") == pytest.approx(123.4)

    def test_get_unknown_metric_returns_zero(self):
        r = _fake_morpho_result()
        s = MorphometricSnapshot.from_morpho_result(r)
        assert s.get_metric("nonexistent_field") == 0.0

    def test_to_dict_has_iso_date(self):
        r = _fake_morpho_result()
        s = MorphometricSnapshot.from_morpho_result(r, date(2025, 6, 15))
        d = s.to_dict()
        assert d["session_date"] == "2025-06-15"

    def test_rupture_risk_label_set(self):
        r = _fake_morpho_result(ar=2.0)   # high AR → Alto
        s = MorphometricSnapshot.from_morpho_result(r)
        assert s.rupture_risk in ("Bajo", "Moderado", "Alto")


# ══════════════════════════════════════════════════════════════════════════════
# LongitudinalSeries
# ══════════════════════════════════════════════════════════════════════════════

class TestLongitudinalSeries:
    def test_empty_series(self):
        s = LongitudinalSeries()
        assert len(s) == 0

    def test_add_snapshot(self):
        s    = LongitudinalSeries()
        snap = _snap(date(2024, 1, 1))
        s.add_snapshot(snap)
        assert len(s) == 1

    def test_snapshots_sorted_by_date(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 6, 1), "B"))
        s.add_snapshot(_snap(date(2024, 1, 1), "A"))
        assert s.snapshots[0].label == "A"
        assert s.snapshots[1].label == "B"

    def test_remove_snapshot(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1)))
        s.add_snapshot(_snap(date(2024, 6, 1)))
        s.remove_snapshot(0)
        assert len(s) == 1

    def test_remove_out_of_range_no_error(self):
        s = LongitudinalSeries()
        s.remove_snapshot(99)   # should not raise

    def test_clear(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1)))
        s.clear()
        assert len(s) == 0

    def test_dates(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1)))
        s.add_snapshot(_snap(date(2024, 6, 1)))
        assert s.dates() == [date(2024, 1, 1), date(2024, 6, 1)]

    def test_days_from_first(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1)))
        s.add_snapshot(_snap(date(2024, 4, 1)))   # 91 days
        days = s.days_from_first()
        assert days[0] == 0.0
        assert days[1] == pytest.approx(91.0)

    def test_values(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1), volume=400.0))
        s.add_snapshot(_snap(date(2024, 6, 1), volume=500.0))
        vals = s.values("volume_mm3")
        assert vals == pytest.approx([400.0, 500.0])

    def test_deltas(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1), volume=400.0))
        s.add_snapshot(_snap(date(2024, 6, 1), volume=500.0))
        s.add_snapshot(_snap(date(2024, 12, 1), volume=600.0))
        d = s.deltas("volume_mm3")
        assert len(d) == 2
        assert d[0] == pytest.approx(100.0)
        assert d[1] == pytest.approx(100.0)

    def test_deltas_single_snapshot_empty(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1)))
        assert s.deltas("volume_mm3") == []

    def test_percent_change(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1), volume=400.0))
        s.add_snapshot(_snap(date(2024, 6, 1), volume=600.0))
        pcts = s.percent_change("volume_mm3")
        assert pcts[0] == pytest.approx(50.0)   # +50 %

    def test_percent_change_zero_base_returns_zeros(self):
        s = LongitudinalSeries()
        r = _fake_morpho_result(volume=0.0)
        s.add_snapshot(MorphometricSnapshot.from_morpho_result(r, date(2024, 1, 1)))
        s.add_snapshot(MorphometricSnapshot.from_morpho_result(r, date(2024, 6, 1)))
        pcts = s.percent_change("volume_mm3")
        assert all(p == 0.0 for p in pcts)

    def test_trend_slope_positive(self):
        s = LongitudinalSeries()
        for i, v in enumerate([400.0, 500.0, 600.0, 700.0]):
            s.add_snapshot(_snap(date(2024, i + 1, 1), volume=v))
        slope, r2 = s.trend("volume_mm3")
        assert slope > 0
        assert r2 > 0.9

    def test_trend_single_snapshot(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1)))
        slope, r2 = s.trend("volume_mm3")
        assert slope == 0.0
        assert r2 == 0.0

    def test_summary_table_length(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1)))
        rows = s.summary_table()
        assert len(rows) == len(TRACKED_METRICS)

    def test_to_dataframe_dict_keys(self):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1)))
        df = s.to_dataframe_dict()
        assert "Fecha" in df
        assert "Volumen (mm³)" in df

    # ── Export / Import ──────────────────────────────────────────────── #

    def test_export_csv_creates_file(self, tmp_path):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1), "T1"))
        s.add_snapshot(_snap(date(2024, 6, 1), "T2"))
        path = str(tmp_path / "out.csv")
        s.export_csv(path)
        assert os.path.isfile(path)

    def test_export_csv_has_header(self, tmp_path):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1)))
        path = str(tmp_path / "out.csv")
        s.export_csv(path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0][0] == "Fecha"

    def test_from_csv_round_trip(self, tmp_path):
        s = LongitudinalSeries(patient_id="P001")
        s.add_snapshot(_snap(date(2024, 1, 1), "T1", volume=400.0))
        s.add_snapshot(_snap(date(2024, 6, 1), "T2", volume=520.0))
        path = str(tmp_path / "roundtrip.csv")
        s.export_csv(path)

        s2 = LongitudinalSeries.from_csv(path, patient_id="P001")
        assert len(s2) == 2
        assert s2.snapshots[0].label == "T1"
        assert s2.snapshots[1].label == "T2"

    def test_from_csv_volume_preserved(self, tmp_path):
        s = LongitudinalSeries()
        s.add_snapshot(_snap(date(2024, 1, 1), volume=412.5))
        path = str(tmp_path / "vol.csv")
        s.export_csv(path)
        s2 = LongitudinalSeries.from_csv(path)
        assert s2.snapshots[0].volume_mm3 == pytest.approx(412.5, abs=0.01)

    def test_from_csv_bad_file_returns_empty(self, tmp_path):
        path = str(tmp_path / "bad.csv")
        with open(path, "w") as f:
            f.write("garbage,data\n1,2\n")
        s = LongitudinalSeries.from_csv(path)
        # May have entries but shouldn't raise
        assert isinstance(s, LongitudinalSeries)


# ══════════════════════════════════════════════════════════════════════════════
# LongitudinalPanel (headless Qt)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def panel(qapp):
    return LongitudinalPanel()


class TestLongitudinalPanel:
    def test_initial_empty(self, panel):
        assert len(panel.get_series()) == 0

    def test_set_current_result_enables_button(self, panel):
        r = _fake_morpho_result()
        panel.set_current_result(r)
        assert panel._btn_add_current.isEnabled()

    def test_add_snapshot_via_method(self, panel):
        r = _fake_morpho_result()
        panel.add_snapshot(r, date(2024, 1, 1), "T1")
        assert len(panel.get_series()) == 1

    def test_table_row_count_matches(self, panel):
        r = _fake_morpho_result()
        panel.add_snapshot(r, date(2024, 1, 1), "T1")
        panel.add_snapshot(r, date(2024, 6, 1), "T2")
        assert panel._session_table.rowCount() == 2

    def test_snapshot_count_signal(self, panel, qtbot):
        r = _fake_morpho_result()
        with qtbot.waitSignal(panel.snapshot_count_changed, timeout=1000) as blocker:
            panel.add_snapshot(r, date(2024, 1, 1))
        assert blocker.args == [1]

    def test_set_series_replaces(self, panel):
        r = _fake_morpho_result()
        panel.add_snapshot(r, date(2024, 1, 1))
        new_s = LongitudinalSeries()
        new_s.add_snapshot(_snap(date(2025, 1, 1), "New"))
        panel.set_series(new_s)
        assert len(panel.get_series()) == 1
        assert panel.get_series().snapshots[0].label == "New"
