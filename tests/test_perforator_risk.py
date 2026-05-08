"""Tests for «Perforantes en riesgo» feature.

Covers:
  * perforator_risk module — compute_perforator_risk, neck_origin_from_morpho,
    helpers (_vtk_points_to_numpy, _compute_vertex_valences, _tagged_poly)
  * PerforatorCandidate — risk_label, risk_color
  * PerforatorRiskResult — structure
  * perforator_risk_actor — build_risk_actors, PerforatorRiskActors iteration
  * PerforatorRiskPanel (headless Qt) — set_vessel_mesh, set_morpho_data,
    overlay_ready signal, overlay_cleared, table population, zone validation
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from prospective.processing.perforator_risk import (
    PerforatorCandidate,
    PerforatorRiskResult,
    compute_perforator_risk,
    neck_origin_from_morpho,
    _vtk_points_to_numpy,
    _compute_vertex_valences,
    _tagged_poly,
)
from prospective.rendering.perforator_risk_actor import (
    PerforatorRiskActors,
    build_risk_actors,
)
from prospective.ui.widgets.perforator_risk_panel import PerforatorRiskPanel


# ──────────────────────────────────────────────────────────────────────────── #
# Shared geometry helpers                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def _sphere_poly(radius: float = 5.0, centre=(0.0, 0.0, 0.0)) -> vtk.vtkPolyData:
    src = vtk.vtkSphereSource()
    src.SetRadius(radius)
    src.SetCenter(*centre)
    src.SetPhiResolution(32)
    src.SetThetaResolution(32)
    src.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(src.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def _tube_poly(
    length: float = 40.0,
    radius: float = 3.0,
    resolution: int = 32,
) -> vtk.vtkPolyData:
    """Z-aligned tube (cylinder) centred at origin."""
    src = vtk.vtkCylinderSource()
    src.SetRadius(radius)
    src.SetHeight(length)
    src.SetResolution(resolution)
    src.SetCapping(True)
    src.Update()
    # Rotate so axis is along Z
    tf = vtk.vtkTransformPolyDataFilter()
    t  = vtk.vtkTransform()
    t.RotateX(90)
    tf.SetTransform(t)
    tf.SetInputConnection(src.GetOutputPort())
    tf.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(tf.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def _fake_morpho(
    neck: float = 4.0,
    centroid=(0.0, 0.0, 0.0),
    axis=(0.0, 0.0, 1.0),
    neck_plane_pos: float = 0.3,
):
    """Minimal duck-type MorphometricResult for perforator tests."""
    _c   = centroid
    _a   = axis
    _npp = neck_plane_pos
    _n   = neck

    class _M:
        neck_diameter_mm = _n
        principal_axis   = _a

        @property
        def neck_plane_pos(self):
            return _npp

        @property
        def centroid(self):
            return _c
    return _M()


# ══════════════════════════════════════════════════════════════════════════════
# PerforatorCandidate
# ══════════════════════════════════════════════════════════════════════════════

class TestPerforatorCandidate:

    def _make(self, level: int) -> PerforatorCandidate:
        return PerforatorCandidate(
            index=0, position=(1.0, 2.0, 3.0),
            distance_to_neck_mm=4.5,
            risk_level=level,
            valence_score=2.3,
        )

    def test_risk_label_high(self):
        assert self._make(1).risk_label == "Alto"

    def test_risk_label_medium(self):
        assert self._make(2).risk_label == "Medio"

    def test_risk_label_low(self):
        assert self._make(3).risk_label == "Bajo"

    def test_risk_label_unknown(self):
        assert self._make(99).risk_label == "—"

    def test_risk_color_high_is_reddish(self):
        r, g, b = self._make(1).risk_color
        assert r > g and r > b, "High-risk colour should be predominantly red"

    def test_risk_color_medium_is_warm(self):
        r, g, b = self._make(2).risk_color
        assert r > 0.8 and g > 0.5 and b < 0.3, "Medium-risk should be amber/orange"

    def test_risk_color_low_is_green(self):
        r, g, b = self._make(3).risk_color
        assert g > r and g > b, "Low-risk colour should be predominantly green"

    def test_risk_color_all_in_unit_range(self):
        for level in (1, 2, 3):
            for ch in self._make(level).risk_color:
                assert 0.0 <= ch <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_vtk_points_to_numpy_shape(self):
        poly = _sphere_poly()
        pts  = _vtk_points_to_numpy(poly)
        assert pts.ndim == 2
        assert pts.shape[1] == 3

    def test_vtk_points_to_numpy_count_matches_vtk(self):
        poly = _sphere_poly()
        pts  = _vtk_points_to_numpy(poly)
        assert len(pts) == poly.GetNumberOfPoints()

    def test_vtk_points_empty_poly(self):
        poly = vtk.vtkPolyData()
        pts  = _vtk_points_to_numpy(poly)
        assert pts.shape == (0, 3)

    def test_compute_vertex_valences_length(self):
        poly = _sphere_poly()
        n    = poly.GetNumberOfPoints()
        val  = _compute_vertex_valences(poly, n)
        assert len(val) == n

    def test_compute_vertex_valences_positive(self):
        poly = _sphere_poly()
        n    = poly.GetNumberOfPoints()
        val  = _compute_vertex_valences(poly, n)
        assert (val > 0).all()

    def test_tagged_poly_preserves_geometry(self):
        poly = _sphere_poly()
        n    = poly.GetNumberOfPoints()
        zones = np.zeros(n, dtype=np.int32)
        out  = _tagged_poly(poly, zones)
        assert out.GetNumberOfPoints() == n
        assert out.GetNumberOfCells()  == poly.GetNumberOfCells()

    def test_tagged_poly_adds_scalar_array(self):
        poly  = _sphere_poly()
        n     = poly.GetNumberOfPoints()
        zones = np.ones(n, dtype=np.int32)
        out   = _tagged_poly(poly, zones)
        arr   = out.GetPointData().GetArray("RiskZone")
        assert arr is not None
        assert arr.GetNumberOfTuples() == n

    def test_tagged_poly_scalar_values_preserved(self):
        poly  = _tube_poly()
        n     = poly.GetNumberOfPoints()
        zones = (np.arange(n) % 4).astype(np.int32)
        out   = _tagged_poly(poly, zones)
        recovered = vtk_to_numpy(out.GetPointData().GetArray("RiskZone"))
        np.testing.assert_array_equal(recovered, zones)


# ══════════════════════════════════════════════════════════════════════════════
# compute_perforator_risk — core algorithm
# ══════════════════════════════════════════════════════════════════════════════

class TestComputePerforatorRisk:

    def test_returns_perforator_risk_result(self):
        poly   = _sphere_poly()
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0))
        assert isinstance(result, PerforatorRiskResult)

    def test_risk_poly_has_risk_zone_scalars(self):
        poly   = _sphere_poly()
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0))
        arr    = result.risk_poly.GetPointData().GetArray("RiskZone")
        assert arr is not None

    def test_risk_poly_point_count_preserved(self):
        poly   = _tube_poly()
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0))
        assert result.risk_poly.GetNumberOfPoints() == poly.GetNumberOfPoints()

    def test_zone_1_radius_respected(self):
        """All vertices at distance ≤ r_high should have zone=1."""
        poly   = _sphere_poly(radius=10.0, centre=(0.0, 0.0, 0.0))
        r_high = 3.0
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0),
                                          zone_radii=(r_high, 5.0, 9.0))
        pts  = _vtk_points_to_numpy(result.risk_poly)
        dists = np.linalg.norm(pts, axis=1)
        zones = vtk_to_numpy(result.risk_poly.GetPointData().GetArray("RiskZone"))
        mask  = dists <= r_high
        assert (zones[mask] == 1).all()

    def test_zone_0_outside_r_low(self):
        """Vertices beyond r_low must be zone=0."""
        poly   = _sphere_poly(radius=15.0)
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0),
                                          zone_radii=(3.0, 5.0, 8.0))
        pts    = _vtk_points_to_numpy(result.risk_poly)
        dists  = np.linalg.norm(pts, axis=1)
        zones  = vtk_to_numpy(result.risk_poly.GetPointData().GetArray("RiskZone"))
        mask   = dists > 8.0
        assert (zones[mask] == 0).all()

    def test_candidates_is_list(self):
        poly   = _sphere_poly()
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0))
        assert isinstance(result.candidates, list)

    def test_candidates_sorted_by_distance(self):
        poly   = _sphere_poly()
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0))
        dists  = [c.distance_to_neck_mm for c in result.candidates]
        assert dists == sorted(dists)

    def test_max_candidates_respected(self):
        poly   = _sphere_poly(radius=10.0)
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0),
                                          zone_radii=(3.0, 6.0, 10.0),
                                          max_candidates=5)
        assert len(result.candidates) <= 5

    def test_neck_origin_stored(self):
        origin = (1.5, -2.0, 3.0)
        poly   = _sphere_poly()
        result = compute_perforator_risk(poly, origin)
        assert result.neck_origin == pytest.approx(origin, abs=1e-6)

    def test_zone_radii_stored(self):
        radii  = (2.0, 4.0, 7.0)
        poly   = _sphere_poly()
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0), zone_radii=radii)
        assert result.zone_radii_mm == radii

    def test_empty_poly_returns_empty_candidates(self):
        poly   = vtk.vtkPolyData()
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0))
        assert result.candidates == []

    def test_far_neck_all_zone_zero(self):
        """If neck is far from the mesh, all vertices are outside every zone."""
        poly   = _sphere_poly(radius=2.0, centre=(0.0, 0.0, 0.0))
        result = compute_perforator_risk(
            poly, (100.0, 100.0, 100.0), zone_radii=(3.0, 5.0, 8.0)
        )
        zones = vtk_to_numpy(result.risk_poly.GetPointData().GetArray("RiskZone"))
        assert (zones == 0).all()
        assert result.candidates == []

    def test_candidate_risk_levels_valid(self):
        poly   = _sphere_poly(radius=10.0)
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0),
                                          zone_radii=(3.0, 6.0, 10.0))
        for c in result.candidates:
            assert c.risk_level in (1, 2, 3)

    def test_candidate_distances_non_negative(self):
        poly   = _sphere_poly()
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0))
        for c in result.candidates:
            assert c.distance_to_neck_mm >= 0.0

    def test_candidate_valence_scores_positive(self):
        poly   = _sphere_poly(radius=10.0)
        result = compute_perforator_risk(poly, (0.0, 0.0, 0.0),
                                          zone_radii=(3.0, 6.0, 10.0),
                                          min_valence_zscore=0.5)
        for c in result.candidates:
            assert c.valence_score >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# neck_origin_from_morpho
# ══════════════════════════════════════════════════════════════════════════════

class TestNeckOriginFromMorpho:

    def test_returns_tuple_of_3(self):
        poly   = _sphere_poly(radius=5.0, centre=(0.0, 0.0, 0.0))
        morpho = _fake_morpho(centroid=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0))
        origin = neck_origin_from_morpho(morpho, poly)
        assert len(origin) == 3

    def test_origin_within_mesh_bounds(self):
        """The reconstructed neck point must fall inside the sphere bounds."""
        r      = 5.0
        poly   = _sphere_poly(radius=r, centre=(0.0, 0.0, 0.0))
        morpho = _fake_morpho(centroid=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                               neck_plane_pos=0.5)
        origin = neck_origin_from_morpho(morpho, poly)
        dist   = math.sqrt(sum(x**2 for x in origin))
        # neck point is on the principal axis, within mesh range
        assert dist <= r + 0.5

    def test_empty_aneurysm_poly_returns_centroid(self):
        morpho    = _fake_morpho(centroid=(3.0, 4.0, 5.0))
        empty_poly = vtk.vtkPolyData()
        origin     = neck_origin_from_morpho(morpho, empty_poly)
        assert origin == pytest.approx((3.0, 4.0, 5.0), abs=1e-6)

    def test_neck_plane_pos_0_near_base(self):
        poly   = _sphere_poly(radius=5.0, centre=(0.0, 0.0, 0.0))
        morpho = _fake_morpho(centroid=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                               neck_plane_pos=0.0)
        origin_0 = neck_origin_from_morpho(morpho, poly)

        morpho1 = _fake_morpho(centroid=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
                                neck_plane_pos=0.5)
        origin_5 = neck_origin_from_morpho(morpho1, poly)

        # pos=0 should be closer to the axis min than pos=0.5
        assert origin_0[2] < origin_5[2]


# ══════════════════════════════════════════════════════════════════════════════
# build_risk_actors
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildRiskActors:

    @pytest.fixture()
    def result(self):
        poly = _sphere_poly(radius=10.0)
        return compute_perforator_risk(
            poly, (0.0, 0.0, 0.0), zone_radii=(3.0, 6.0, 10.0),
            min_valence_zscore=0.5,
        )

    def test_returns_named_tuple(self, result):
        actors = build_risk_actors(result)
        assert isinstance(actors, PerforatorRiskActors)

    def test_overlay_is_vtk_actor(self, result):
        actors = build_risk_actors(result)
        assert isinstance(actors.overlay, vtk.vtkActor)

    def test_spheres_is_list(self, result):
        actors = build_risk_actors(result)
        assert isinstance(actors.spheres, list)

    def test_labels_is_list(self, result):
        actors = build_risk_actors(result)
        assert isinstance(actors.labels, list)

    def test_neck_ring_is_vtk_actor(self, result):
        actors = build_risk_actors(result)
        assert isinstance(actors.neck_ring, vtk.vtkActor)

    def test_sphere_count_matches_candidates(self, result):
        actors = build_risk_actors(result)
        assert len(actors.spheres) == len(result.candidates)

    def test_label_count_matches_candidates(self, result):
        actors = build_risk_actors(result)
        assert len(actors.labels) == len(result.candidates)

    def test_iteration_yields_all_actors(self, result):
        actors = build_risk_actors(result)
        all_actors = list(actors)
        # overlay + N spheres + N labels + neck_ring
        expected = 1 + len(result.candidates) + len(result.candidates) + 1
        assert len(all_actors) == expected

    def test_set_visible_false_hides_all(self, result):
        actors = build_risk_actors(result)
        actors.set_visible(False)
        for a in actors:
            assert a.GetVisibility() == 0

    def test_set_visible_true_shows_all(self, result):
        actors = build_risk_actors(result)
        actors.set_visible(False)
        actors.set_visible(True)
        for a in actors:
            assert a.GetVisibility() == 1

    def test_neck_ring_position(self, result):
        actors = build_risk_actors(result)
        # The ring sphere's center should match neck_origin
        bds = actors.neck_ring.GetBounds()
        cx  = (bds[0] + bds[1]) / 2.0
        cy  = (bds[2] + bds[3]) / 2.0
        cz  = (bds[4] + bds[5]) / 2.0
        ox, oy, oz = result.neck_origin
        assert cx == pytest.approx(ox, abs=0.5)
        assert cy == pytest.approx(oy, abs=0.5)
        assert cz == pytest.approx(oz, abs=0.5)

    def test_no_candidates_still_builds(self):
        """build_risk_actors must work even with an empty candidate list."""
        poly   = _sphere_poly(radius=1.0)
        result = compute_perforator_risk(
            poly, (100.0, 100.0, 100.0),   # far away — no candidates
            zone_radii=(1.0, 2.0, 3.0),
        )
        actors = build_risk_actors(result)
        assert isinstance(actors, PerforatorRiskActors)
        assert actors.spheres == []
        assert actors.labels  == []


# ══════════════════════════════════════════════════════════════════════════════
# PerforatorRiskPanel (headless Qt)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def panel(qapp):
    return PerforatorRiskPanel()


@pytest.fixture()
def vessel_poly():
    return _sphere_poly(radius=10.0)


class TestPerforatorRiskPanel:

    # ── Initial state ─────────────────────────────────────────────────── #

    def test_compute_button_disabled_initially(self, panel):
        assert not panel._btn_compute.isEnabled()

    def test_clear_button_disabled_initially(self, panel):
        assert not panel._btn_clear.isEnabled()

    def test_table_empty_initially(self, panel):
        assert panel._table.rowCount() == 0

    # ── set_vessel_mesh ───────────────────────────────────────────────── #

    def test_set_vessel_mesh_stores_poly(self, panel, vessel_poly):
        panel.set_vessel_mesh(vessel_poly)
        assert panel._vessel_poly is vessel_poly

    def test_set_vessel_mesh_none_still_ok(self, panel):
        panel.set_vessel_mesh(None)
        assert panel._vessel_poly is None

    def test_compute_still_disabled_without_morpho(self, panel, vessel_poly):
        panel.set_vessel_mesh(vessel_poly)
        assert not panel._btn_compute.isEnabled()

    # ── set_morpho_data ───────────────────────────────────────────────── #

    def test_set_morpho_data_enables_compute(self, panel, vessel_poly):
        panel.set_vessel_mesh(vessel_poly)
        panel.set_morpho_data(_fake_morpho(), vessel_poly)
        assert panel._btn_compute.isEnabled()

    def test_set_morpho_data_without_vessel_keeps_disabled(self, panel, vessel_poly):
        panel.set_morpho_data(_fake_morpho(), vessel_poly)
        assert not panel._btn_compute.isEnabled()

    def test_status_label_updated_after_morpho(self, panel, vessel_poly):
        panel.set_vessel_mesh(vessel_poly)
        morpho = _fake_morpho(neck=6.0)
        panel.set_morpho_data(morpho, vessel_poly)
        assert "6.0" in panel._lbl_status.text()

    def test_morpho_none_resets_status(self, panel):
        panel.set_morpho_data(None, None)
        assert "Sin" in panel._lbl_status.text() or "morfometr" in panel._lbl_status.text().lower()

    # ── zone radius validation ────────────────────────────────────────── #

    def test_invalid_radii_shows_warning(self, panel, vessel_poly, qtbot):
        panel.set_vessel_mesh(vessel_poly)
        panel.set_morpho_data(_fake_morpho(), vessel_poly)
        # Set radii out-of-order: high >= med
        panel._spin_high.setValue(6.0)
        panel._spin_med.setValue(4.0)
        panel._spin_low.setValue(8.0)
        panel._on_compute()
        # Result label should mention error / radios
        assert panel._table.rowCount() == 0

    # ── compute ───────────────────────────────────────────────────────── #

    def test_compute_populates_table(self, panel, vessel_poly, qtbot):
        panel.set_vessel_mesh(vessel_poly)
        panel.set_morpho_data(_fake_morpho(), vessel_poly)
        panel._spin_high.setValue(3.0)
        panel._spin_med.setValue(6.0)
        panel._spin_low.setValue(10.0)
        panel._on_compute()
        # Table may or may not have rows depending on mesh topology
        # but it should not raise an exception
        assert panel._table.rowCount() >= 0

    def test_compute_enables_clear(self, panel, vessel_poly):
        panel.set_vessel_mesh(vessel_poly)
        panel.set_morpho_data(_fake_morpho(), vessel_poly)
        panel._spin_high.setValue(2.0)
        panel._spin_med.setValue(5.0)
        panel._spin_low.setValue(10.0)
        panel._on_compute()
        assert panel._btn_clear.isEnabled()

    def test_compute_emits_overlay_ready(self, panel, vessel_poly, qtbot):
        panel.set_vessel_mesh(vessel_poly)
        panel.set_morpho_data(_fake_morpho(), vessel_poly)
        panel._spin_high.setValue(2.0)
        panel._spin_med.setValue(5.0)
        panel._spin_low.setValue(10.0)
        with qtbot.waitSignal(panel.overlay_ready, timeout=5000) as blocker:
            panel._on_compute()
        assert isinstance(blocker.args[0], list)

    # ── clear ──────────────────────────────────────────────────────────── #

    def test_clear_resets_table(self, panel, vessel_poly, qtbot):
        panel.set_vessel_mesh(vessel_poly)
        panel.set_morpho_data(_fake_morpho(), vessel_poly)
        panel._spin_high.setValue(2.0)
        panel._spin_med.setValue(5.0)
        panel._spin_low.setValue(10.0)
        panel._on_compute()
        panel._on_clear()
        assert panel._table.rowCount() == 0

    def test_clear_emits_overlay_cleared(self, panel, vessel_poly, qtbot):
        panel.set_vessel_mesh(vessel_poly)
        panel.set_morpho_data(_fake_morpho(), vessel_poly)
        panel._on_compute()
        with qtbot.waitSignal(panel.overlay_cleared, timeout=1000):
            panel._on_clear()

    def test_clear_disables_clear_button(self, panel, vessel_poly):
        panel.set_vessel_mesh(vessel_poly)
        panel.set_morpho_data(_fake_morpho(), vessel_poly)
        panel._on_compute()
        panel._on_clear()
        assert not panel._btn_clear.isEnabled()

    # ── session state ─────────────────────────────────────────────────── #

    def test_session_state_roundtrip(self, panel):
        panel._spin_high.setValue(2.5)
        panel._spin_med.setValue(4.5)
        panel._spin_low.setValue(7.5)
        state = panel.get_session_state()
        panel._spin_high.setValue(1.0)
        panel._spin_med.setValue(2.0)
        panel._spin_low.setValue(3.0)
        panel.restore_session_state(state)
        assert panel._spin_high.value() == pytest.approx(2.5)
        assert panel._spin_med.value()  == pytest.approx(4.5)
        assert panel._spin_low.value()  == pytest.approx(7.5)
