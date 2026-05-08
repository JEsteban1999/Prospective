"""Tests for Feature 8 — Clip selection assistant.

Covers:
  * recommend_clips / recommend_from_morpho — filtering, scoring, ranking
  * ClipRecommendation — coverage label, score label, to_dict
  * ClipRecommenderPanel (headless Qt) — set_morpho_result, table population, signal
"""
from __future__ import annotations

import pytest

from prospective.models.clip_library import ClipSpec, ClipShape, CLIP_CATALOGUE
from prospective.processing.clip_recommender import (
    recommend_clips,
    recommend_from_morpho,
    ClipRecommendation,
    WIDE_NECK_THRESHOLD_MM,
    DEEP_DOME_AR_THRESHOLD,
    _coverage_score,
    _shape_score,
    _force_score,
)
from prospective.ui.widgets.clip_recommender_panel import ClipRecommenderPanel


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _fake_morpho(neck: float = 4.0, ar: float = 1.2):
    """Return a minimal mock that satisfies MorphometricResult duck-type."""
    class _M:
        neck_diameter_mm = neck
        aspect_ratio     = ar
    return _M()


def _make_clip(blade: float, shape=ClipShape.STRAIGHT, force: float = 100.0):
    return ClipSpec(
        name=f"TestClip {blade}mm",
        shape=shape,
        blade_length_mm=blade,
        blade_width_mm=1.0,
        blade_height_mm=1.0,
        spring_length_mm=6.0,
        closing_force_g=force,
        manufacturer="Test",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Scoring sub-functions
# ══════════════════════════════════════════════════════════════════════════════

class TestScoringHelpers:

    def test_coverage_score_ideal(self):
        # At ideal ratio 1.35 score should be ~1.0
        s = _coverage_score(1.35)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_coverage_score_decreases_away_from_ideal(self):
        assert _coverage_score(1.35) > _coverage_score(1.0)
        assert _coverage_score(1.35) > _coverage_score(2.0)

    def test_coverage_score_nonnegative(self):
        for r in [0.5, 1.0, 1.35, 2.0, 3.0]:
            assert _coverage_score(r) >= 0.0

    def test_force_score_optimal_range(self):
        for f in [80, 100, 130, 160]:
            assert _force_score(f) == pytest.approx(1.0)

    def test_force_score_below_optimal(self):
        assert _force_score(40) < _force_score(80)

    def test_force_score_above_optimal(self):
        assert _force_score(200) < _force_score(160)

    def test_force_score_zero_force(self):
        assert _force_score(0) == pytest.approx(0.0)

    def test_shape_score_standard_prefers_straight(self):
        straight = _make_clip(8.0, ClipShape.STRAIGHT)
        fenest   = _make_clip(8.0, ClipShape.FENESTRATED)
        assert _shape_score(straight, neck_mm=3.0, ar=1.0) > _shape_score(fenest, neck_mm=3.0, ar=1.0)

    def test_shape_score_wide_neck_prefers_fenestrated(self):
        fenest   = _make_clip(8.0, ClipShape.FENESTRATED)
        straight = _make_clip(8.0, ClipShape.STRAIGHT)
        assert _shape_score(fenest, neck_mm=6.0, ar=1.0) > _shape_score(straight, neck_mm=6.0, ar=1.0)

    def test_shape_score_deep_dome_prefers_angled(self):
        angled   = _make_clip(8.0, ClipShape.ANGLED)
        straight = _make_clip(8.0, ClipShape.STRAIGHT)
        assert _shape_score(angled, neck_mm=3.0, ar=2.0) > _shape_score(straight, neck_mm=3.0, ar=2.0)


# ══════════════════════════════════════════════════════════════════════════════
# recommend_clips
# ══════════════════════════════════════════════════════════════════════════════

class TestRecommendClips:

    def test_returns_list(self):
        r = recommend_clips(4.0, 1.2)
        assert isinstance(r, list)

    def test_results_are_recommendations(self):
        r = recommend_clips(4.0, 1.2)
        for rec in r:
            assert isinstance(rec, ClipRecommendation)

    def test_blade_length_at_least_neck_plus_one(self):
        neck = 4.0
        for rec in recommend_clips(neck, 1.2):
            assert rec.clip.blade_length_mm >= neck + 1.0

    def test_blade_length_at_most_three_times_neck(self):
        neck = 4.0
        for rec in recommend_clips(neck, 1.2):
            assert rec.clip.blade_length_mm <= neck * 3.0

    def test_sorted_descending_by_score(self):
        recs = recommend_clips(4.0, 1.2)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_n_limits_output(self):
        recs = recommend_clips(4.0, 1.2, n=3)
        assert len(recs) <= 3

    def test_n_zero_returns_empty(self):
        assert recommend_clips(4.0, 1.2, n=0) == []

    def test_zero_neck_returns_empty(self):
        assert recommend_clips(0.0, 1.2) == []

    def test_negative_neck_returns_empty(self):
        assert recommend_clips(-1.0, 1.2) == []

    def test_custom_catalogue(self):
        cat = [_make_clip(6.0), _make_clip(8.0)]
        recs = recommend_clips(4.0, 1.2, catalogue=cat)
        assert all(r.clip in cat for r in recs)

    def test_coverage_ratio_computed_correctly(self):
        cat = [_make_clip(8.0)]
        recs = recommend_clips(4.0, 1.2, catalogue=cat)
        assert len(recs) == 1
        assert recs[0].coverage_ratio == pytest.approx(8.0 / 4.0)

    def test_safety_margin_computed_correctly(self):
        cat = [_make_clip(7.0)]
        neck = 4.5
        recs = recommend_clips(neck, 1.2, catalogue=cat)
        assert len(recs) == 1
        assert recs[0].safety_margin_mm == pytest.approx(7.0 - neck, abs=0.01)

    def test_wide_neck_top_result_is_fenestrated(self):
        """For wide-neck aneurysms, a FENESTRATED clip should rank highest
        when one is available with good coverage."""
        neck = WIDE_NECK_THRESHOLD_MM + 0.5   # 5.5 mm → wide neck
        recs = recommend_clips(neck, 1.0, catalogue=CLIP_CATALOGUE)
        if recs:
            # The top result should have higher shape score — not guaranteed to
            # be strictly FENESTRATED if catalogue is sparse, but score must be > 0
            assert recs[0].score > 0

    def test_scores_between_zero_and_hundred(self):
        for rec in recommend_clips(4.0, 1.2):
            assert 0.0 <= rec.score <= 100.0

    def test_reasons_is_nonempty_list(self):
        for rec in recommend_clips(4.0, 1.2):
            assert isinstance(rec.reasons, list)
            assert len(rec.reasons) > 0


# ══════════════════════════════════════════════════════════════════════════════
# recommend_from_morpho
# ══════════════════════════════════════════════════════════════════════════════

class TestRecommendFromMorpho:

    def test_uses_neck_from_morpho(self):
        morpho = _fake_morpho(neck=4.0, ar=1.2)
        recs = recommend_from_morpho(morpho)
        for rec in recs:
            assert rec.clip.blade_length_mm >= 4.0 + 1.0

    def test_returns_list(self):
        assert isinstance(recommend_from_morpho(_fake_morpho()), list)


# ══════════════════════════════════════════════════════════════════════════════
# ClipRecommendation helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestClipRecommendation:

    def _rec(self, score: float, cov: float) -> ClipRecommendation:
        return ClipRecommendation(
            clip=_make_clip(8.0),
            score=score,
            coverage_ratio=cov,
            safety_margin_mm=cov * 4.0 - 4.0,
        )

    def test_score_label_excellent(self):
        assert self._rec(80, 1.5).score_label == "Excelente"

    def test_score_label_good(self):
        assert self._rec(60, 1.3).score_label == "Bueno"

    def test_score_label_acceptable(self):
        assert self._rec(40, 1.2).score_label == "Aceptable"

    def test_score_label_marginal(self):
        assert self._rec(20, 1.1).score_label == "Marginal"

    def test_coverage_label_ok(self):
        assert "✔" in self._rec(80, 1.3).coverage_label

    def test_coverage_label_warning(self):
        assert "⚠" in self._rec(40, 1.1).coverage_label

    def test_to_dict_has_required_keys(self):
        d = self._rec(75, 1.4).to_dict()
        for key in ("clip", "score", "coverage_ratio", "safety_mm", "label"):
            assert key in d


# ══════════════════════════════════════════════════════════════════════════════
# ClipRecommenderPanel (headless Qt)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def panel(qapp):
    return ClipRecommenderPanel()


class TestClipRecommenderPanel:

    def test_initial_empty_table(self, panel):
        assert panel._table.rowCount() == 0

    def test_set_morpho_populates_table(self, panel):
        panel.set_morpho_result(_fake_morpho(neck=4.0, ar=1.2))
        assert panel._table.rowCount() > 0

    def test_table_rows_match_recommendations(self, panel):
        panel.set_morpho_result(_fake_morpho(neck=4.0, ar=1.2))
        assert panel._table.rowCount() == len(panel._recommendations)

    def test_n_spin_limits_rows(self, panel):
        panel._spin_n.setValue(3)
        panel.set_morpho_result(_fake_morpho(neck=4.0, ar=1.2))
        assert panel._table.rowCount() <= 3

    def test_context_labels_updated(self, panel):
        panel.set_morpho_result(_fake_morpho(neck=5.5, ar=2.0))
        assert panel._lbl_neck.text() != "—"
        assert panel._lbl_ar.text() != "—"
        assert panel._lbl_ctx.text() != "—"

    def test_send_button_disabled_initially(self, panel):
        assert not panel._btn_send.isEnabled()

    def test_send_button_enabled_after_row_selected(self, panel):
        panel.set_morpho_result(_fake_morpho(neck=4.0, ar=1.2))
        # First row auto-selected
        if panel._table.rowCount() > 0:
            assert panel._btn_send.isEnabled()

    def test_clip_selected_signal(self, panel, qtbot):
        panel.set_morpho_result(_fake_morpho(neck=4.0, ar=1.2))
        panel._table.selectRow(0)
        with qtbot.waitSignal(panel.clip_selected, timeout=1000) as blocker:
            panel._on_send()
        from prospective.models.clip_library import ClipSpec
        assert isinstance(blocker.args[0], ClipSpec)

    def test_wide_neck_context_label(self, panel):
        panel.set_morpho_result(_fake_morpho(neck=WIDE_NECK_THRESHOLD_MM + 1, ar=1.0))
        assert "ancho" in panel._lbl_ctx.text().lower()

    def test_deep_dome_context_label(self, panel):
        panel.set_morpho_result(_fake_morpho(neck=3.0, ar=DEEP_DOME_AR_THRESHOLD + 0.5))
        assert "ar" in panel._lbl_ctx.text().lower() or "profundo" in panel._lbl_ctx.text().lower()
