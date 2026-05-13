"""Unit tests for compute_auto_thresholds() and its CT/MR sub-functions.

Exercises the four modality paths of the universal threshold dispatcher:
  - XA / RF / DX  → delegates to compute_xa_thresholds (strategy reported)
  - CT             → _ct_auto_thresholds (ct_stats or ct_wc_ww)
  - MR             → _mr_auto_thresholds (mr_percentile or wc_ww)
  - Unknown        → percentile fallback or wc_ww

Each test describes *what behaviour it locks in* and *why* the assertion is
correct, so the intent is self-documenting even without reading the source.
"""
from __future__ import annotations

import numpy as np
import pytest

from prospective.ui.widgets.segmentation_panel import compute_auto_thresholds


# ──────────────────────────────────────────────────────────────────────────── #
# Volume factories                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def _cta_volume(n: int = 100_000) -> np.ndarray:
    """Simulate a CTA volume: background air + soft tissue + contrast vessels.

    > 2 % of voxels above 150 HU → triggers ct_stats CTA branch.
    """
    rng = np.random.default_rng(3)
    vol = rng.normal(-200, 300, n).astype("float32")   # mixed soft tissue
    n_vessels = n // 10                                  # 10 % bright
    idx = rng.choice(n, n_vessels, replace=False)
    vol[idx] = rng.uniform(200, 800, n_vessels)          # contrast vessels
    return vol.reshape(10, 100, 100)


def _noncontrast_ct_volume(n: int = 100_000) -> np.ndarray:
    """Non-contrast CT: almost no voxels above 150 HU.

    < 2 % above 150 HU → triggers ct_stats non-contrast branch.
    """
    rng = np.random.default_rng(5)
    vol = rng.normal(30, 30, n).astype("float32")       # soft tissue centred ~30 HU
    return vol.reshape(10, 100, 100)


def _mr_volume(n: int = 100_000) -> np.ndarray:
    """Simulate an MR volume: near-zero background + bright tissue tail."""
    rng = np.random.default_rng(11)
    vol = np.concatenate([
        rng.uniform(0, 50, int(n * 0.15)),      # background / air
        rng.normal(400, 120, int(n * 0.85)),    # tissue + vessels
    ]).astype("float32")
    rng.shuffle(vol)
    return vol.reshape(10, 100, 100)


def _dsa_volume(n: int = 100_000) -> np.ndarray:
    """DSA subtraction: p99 < 0, sparse bright vessels at top."""
    rng = np.random.default_rng(42)
    vol = np.full(n, -1024.0, dtype="float32")
    idx = rng.choice(n, n // 300, replace=False)
    vol[idx] = rng.uniform(2_000, 15_000, n // 300)
    return vol.reshape(10, 100, 100)


def _xa_wide_ww_volume(n: int = 100_000) -> np.ndarray:
    """Standard 3DRA: mixed background + bright vessel tail, WW > 2000."""
    rng = np.random.default_rng(7)
    vol = rng.normal(-400, 300, n).astype("float32")
    idx = rng.choice(n, n // 20, replace=False)
    vol[idx] = rng.uniform(800, 5_000, n // 20)
    return vol.reshape(10, 100, 100)


# ──────────────────────────────────────────────────────────────────────────── #
# Return type contract                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

class TestReturnTypeContract:
    """compute_auto_thresholds always returns (float, float, str) with lower < upper."""

    @pytest.mark.parametrize("modality", ["CT", "MR", "XA", "RF", "DX", "CR", "DR", "PT", ""])
    def test_returns_float_float_str(self, modality):
        lower, upper, strategy = compute_auto_thresholds(None, modality, 200.0, 1000.0)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert isinstance(strategy, str)

    @pytest.mark.parametrize("modality", ["CT", "MR", "XA", "RF", ""])
    def test_lower_less_than_upper_no_volume(self, modality):
        lower, upper, _ = compute_auto_thresholds(None, modality, 200.0, 1000.0)
        assert lower < upper

    @pytest.mark.parametrize("modality", ["CT", "MR", "XA"])
    def test_lower_less_than_upper_with_volume(self, modality):
        vol = _cta_volume()
        lower, upper, _ = compute_auto_thresholds(vol, modality, 200.0, 2000.0)
        assert lower < upper

    def test_none_volume_does_not_raise(self):
        """All modalities must handle volume=None gracefully."""
        for mod in ("CT", "MR", "XA", "RF", "DX", "CR", "DR", "PT", "", None):
            compute_auto_thresholds(None, mod or "", 100.0, 500.0)

    def test_strategy_is_known_value(self):
        known = {"dsa", "xa_band_pass", "xa_wc_ww", "ct_stats", "ct_wc_ww",
                 "mr_percentile", "wc_ww"}
        for mod in ("CT", "MR", "XA"):
            _, _, s = compute_auto_thresholds(_cta_volume(), mod, 200.0, 1200.0)
            assert s in known, f"Unknown strategy '{s}' for modality {mod}"


# ──────────────────────────────────────────────────────────────────────────── #
# CT path                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

class TestCTPath:

    def test_modality_ct_dispatches_to_ct(self):
        """CT modality must never return xa_* or mr_* strategy."""
        _, _, s = compute_auto_thresholds(_cta_volume(), "CT", 300.0, 1500.0)
        assert s.startswith("ct_")

    def test_cta_uses_ct_stats_strategy(self):
        """CTA volume (≥2 % bright voxels) → strategy='ct_stats'."""
        _, _, s = compute_auto_thresholds(_cta_volume(), "CT", 300.0, 1500.0)
        assert s == "ct_stats"

    def test_cta_lower_bounded_at_150(self):
        """CTA lower = clip(WC - WW×0.10, 150, 500) — never below 150 HU.

        150 HU is the parenchyma/vessel boundary: brain tissue sits at 30–60 HU,
        contrast-enhanced vessels at ≥150 HU.  The old 80 HU floor captured
        parenchyma alongside vessels and produced noisy meshes.
        """
        lower, _, _ = compute_auto_thresholds(_cta_volume(), "CT", 0.0, 100.0)
        assert lower >= 150.0

    def test_cta_lower_bounded_at_500(self):
        """Extreme WC should not push lower above 500 HU."""
        lower, _, _ = compute_auto_thresholds(_cta_volume(), "CT", 10_000.0, 100.0)
        assert lower <= 500.0

    def test_cta_upper_is_1500(self):
        """CT upper threshold is fixed at 1500 HU for all CT branches."""
        _, upper, _ = compute_auto_thresholds(_cta_volume(), "CT", 300.0, 1500.0)
        assert upper == pytest.approx(1500.0, rel=1e-3)

    def test_noncontrast_ct_uses_ct_stats_strategy(self):
        """Non-contrast CT also returns ct_stats (different internal branch)."""
        _, _, s = compute_auto_thresholds(_noncontrast_ct_volume(), "CT", 35.0, 80.0)
        assert s == "ct_stats"

    def test_noncontrast_ct_lower_clamped_to_tissue_range(self):
        """Non-contrast lower = clip(p80_tissue, 20, 150) — within physiological range."""
        lower, _, _ = compute_auto_thresholds(_noncontrast_ct_volume(), "CT", 35.0, 80.0)
        assert 20.0 <= lower <= 150.0

    def test_ct_no_volume_uses_ct_wc_ww_strategy(self):
        """Volume=None → ct_wc_ww strategy."""
        _, _, s = compute_auto_thresholds(None, "CT", 300.0, 600.0)
        assert s == "ct_wc_ww"

    def test_ct_wc_ww_lower_formula(self):
        """Fallback: lower = max(WC - WW×0.10, 150)."""
        wc, ww = 400.0, 600.0
        lower, _, _ = compute_auto_thresholds(None, "CT", wc, ww)
        expected = max(wc - ww * 0.10, 150.0)
        assert lower == pytest.approx(expected, rel=1e-3)

    def test_ct_wc_ww_upper_is_1500(self):
        _, upper, _ = compute_auto_thresholds(None, "CT", 400.0, 600.0)
        assert upper == pytest.approx(1500.0, rel=1e-3)

    def test_ct_case_insensitive(self):
        """Modality string matching is case-insensitive."""
        l1, u1, s1 = compute_auto_thresholds(None, "CT", 300.0, 1200.0)
        l2, u2, s2 = compute_auto_thresholds(None, "ct", 300.0, 1200.0)
        assert l1 == pytest.approx(l2) and u1 == pytest.approx(u2) and s1 == s2


# ──────────────────────────────────────────────────────────────────────────── #
# MR path                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

class TestMRPath:

    def test_modality_mr_dispatches_to_mr(self):
        """MR modality must not return ct_* or xa_* strategy."""
        _, _, s = compute_auto_thresholds(_mr_volume(), "MR", 500.0, 800.0)
        assert not s.startswith("ct_") and not s.startswith("xa_") and "dsa" not in s

    def test_mr_with_volume_uses_mr_percentile(self):
        """MR + valid volume → strategy='mr_percentile'."""
        _, _, s = compute_auto_thresholds(_mr_volume(), "MR", 500.0, 800.0)
        assert s == "mr_percentile"

    def test_mr_lower_above_background(self):
        """MR lower must be above the 10th-percentile cutoff (above background)."""
        vol = _mr_volume()
        flat = vol.ravel()
        vmin = float(flat.min())
        cutoff = vmin + (float(flat.max()) - vmin) * 0.10
        lower, _, _ = compute_auto_thresholds(vol, "MR", 0.0, 1000.0)
        assert lower > cutoff

    def test_mr_upper_above_lower(self):
        lower, upper, _ = compute_auto_thresholds(_mr_volume(), "MR", 0.0, 1000.0)
        assert upper > lower

    def test_mr_no_volume_uses_wc_ww_fallback(self):
        """Volume=None → strategy='wc_ww' (MR fallback)."""
        _, _, s = compute_auto_thresholds(None, "MR", 500.0, 800.0)
        assert s == "wc_ww"

    def test_mr_wc_ww_lower_formula(self):
        """MR fallback lower = max(-200, WC - WW×0.35)."""
        wc, ww = 500.0, 800.0
        lower, _, _ = compute_auto_thresholds(None, "MR", wc, ww)
        expected = max(-200.0, wc - ww * 0.35)
        assert lower == pytest.approx(expected, rel=1e-3)

    def test_mr_wc_ww_upper_formula(self):
        """MR fallback upper = WC + WW×0.45."""
        wc, ww = 500.0, 800.0
        _, upper, _ = compute_auto_thresholds(None, "MR", wc, ww)
        assert upper == pytest.approx(wc + ww * 0.45, rel=1e-3)

    def test_mr_case_insensitive(self):
        l1, u1, s1 = compute_auto_thresholds(None, "MR", 400.0, 900.0)
        l2, u2, s2 = compute_auto_thresholds(None, "mr", 400.0, 900.0)
        assert l1 == pytest.approx(l2) and u1 == pytest.approx(u2) and s1 == s2


# ──────────────────────────────────────────────────────────────────────────── #
# XA / RF / DX / CR / DR paths (delegation to compute_xa_thresholds)           #
# ──────────────────────────────────────────────────────────────────────────── #

class TestXAPath:

    @pytest.mark.parametrize("modality", ["XA", "RF", "DX", "CR", "DR"])
    def test_xa_family_dispatches_to_xa(self, modality):
        """All XA-family modalities must return an xa_* or 'dsa' strategy."""
        _, _, s = compute_auto_thresholds(_xa_wide_ww_volume(), modality, 0.0, 7000.0)
        assert s in ("dsa", "xa_band_pass", "xa_wc_ww")

    def test_dsa_returns_dsa_strategy(self):
        """DSA volume → strategy='dsa'."""
        _, _, s = compute_auto_thresholds(_dsa_volume(), "XA", 0.0, 1000.0)
        assert s == "dsa"

    def test_wide_ww_returns_xa_band_pass(self):
        """Wide WW (>2000) + volume → strategy='xa_band_pass'."""
        _, _, s = compute_auto_thresholds(_xa_wide_ww_volume(), "XA", 0.0, 7000.0)
        assert s == "xa_band_pass"

    def test_narrow_ww_no_volume_returns_xa_wc_ww(self):
        """Narrow WW (≤2000) + no volume → strategy='xa_wc_ww'."""
        _, _, s = compute_auto_thresholds(None, "XA", 500.0, 1000.0)
        assert s == "xa_wc_ww"

    def test_narrow_ww_with_volume_returns_xa_wc_ww(self):
        """Narrow WW (≤2000) even with a volume → WC/WW path, not band-pass."""
        _, _, s = compute_auto_thresholds(_xa_wide_ww_volume(), "XA", 0.0, 1000.0)
        assert s == "xa_wc_ww"

    def test_dsa_lower_positive(self):
        """DSA lower threshold must be positive (vessel voxels are bright)."""
        lower, _, _ = compute_auto_thresholds(_dsa_volume(), "XA", 0.0, 1000.0)
        assert lower > 0

    def test_xa_ignores_modality_case(self):
        l1, u1, s1 = compute_auto_thresholds(None, "XA", 0.0, 1000.0)
        l2, u2, s2 = compute_auto_thresholds(None, "xa", 0.0, 1000.0)
        assert l1 == pytest.approx(l2) and s1 == s2


# ──────────────────────────────────────────────────────────────────────────── #
# Unknown / fallback path                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

class TestFallbackPath:

    def test_unknown_modality_with_volume_uses_wc_ww_strategy(self):
        """Unknown modality + volume → p90/p99 path, strategy='wc_ww'."""
        _, _, s = compute_auto_thresholds(_cta_volume(), "PT", 200.0, 1000.0)
        assert s == "wc_ww"

    def test_unknown_modality_no_volume_uses_wc_ww_strategy(self):
        _, _, s = compute_auto_thresholds(None, "PT", 200.0, 1000.0)
        assert s == "wc_ww"

    def test_unknown_modality_no_volume_lower_from_wc_ww(self):
        """Fallback without volume: lower = max(-200, WC - WW×0.35)."""
        wc, ww = 200.0, 1000.0
        lower, _, _ = compute_auto_thresholds(None, "PT", wc, ww)
        expected = max(-200.0, wc - ww * 0.35)
        assert lower == pytest.approx(expected, rel=1e-3)

    def test_empty_modality_string_uses_fallback(self):
        lower, upper, s = compute_auto_thresholds(None, "", 100.0, 600.0)
        assert s == "wc_ww"
        assert lower < upper

    def test_unknown_modality_with_volume_lower_is_p90(self):
        """Fallback with volume: lower = p90."""
        vol = _cta_volume()
        flat = vol.ravel()
        expected_lower = float(np.percentile(flat, 90))
        lower, _, _ = compute_auto_thresholds(vol, "PT", 200.0, 1000.0)
        assert lower == pytest.approx(expected_lower, rel=1e-3)

    def test_unknown_modality_with_volume_upper_is_p99(self):
        vol = _cta_volume()
        flat = vol.ravel()
        expected_upper = float(np.percentile(flat, 99))
        _, upper, _ = compute_auto_thresholds(vol, "PT", 200.0, 1000.0)
        assert upper == pytest.approx(expected_upper, rel=1e-3)
