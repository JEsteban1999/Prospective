"""Unit tests for compute_xa_thresholds() — XA/3DRA threshold auto-detection.

Tests cover the three dataset archetypes present in the real DICOM library:

  Case 9  → DSA subtraction (3DANGIO SUB): p99 < 0, max >> 0
  Case 39 → Standard 3DRA, wide WW (> 2 000): p90–p99 band-pass
  Case 2  → Standard 3DRA, narrow WW (≤ 2 000): WC/WW derivation
  Case 3  → 16-bit 3DRA_PROP, no rescale: wide WW → p90–p99 band-pass
"""
from __future__ import annotations

import numpy as np
import pytest

from prospective.ui.widgets.segmentation_panel import compute_xa_thresholds


# ──────────────────────────────────────────────────────────────────────────── #
# Synthetic volume factories                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

def _dsa_volume(n: int = 100_000) -> np.ndarray:
    """Simulate a DSA subtraction volume.

    99 % of voxels: background at ~-1024 HU (subtracted tissue).
    <1 %: bright vessel voxels at 2 000–15 000 HU.
    Signature: p99 < 0, max > 500.
    """
    rng = np.random.default_rng(42)
    vol = np.full(n, -1024.0, dtype="float32")
    # Sprinkle a handful of very bright vessel voxels (< 0.5 %)
    n_vessels = n // 300
    idx = rng.choice(n, n_vessels, replace=False)
    vol[idx] = rng.uniform(2_000, 15_000, n_vessels)
    return vol.reshape(10, 100, 100)


def _std_3dra_wide_ww(n: int = 100_000) -> np.ndarray:
    """Simulate a standard cone-beam 3DRA without subtraction.

    Distribution: background ≈ -500 HU, vessels/bone in the bright tail.
    p90 ~ 800 HU, p99 ~ 3 000 HU.  WW is typically > 2 000.
    """
    rng = np.random.default_rng(7)
    vol = rng.normal(-400, 300, n).astype("float32")
    # Add bright vessel tail (top 5 %)
    n_bright = n // 20
    idx = rng.choice(n, n_bright, replace=False)
    vol[idx] = rng.uniform(800, 5_000, n_bright)
    return vol.reshape(10, 100, 100)


# ──────────────────────────────────────────────────────────────────────────── #
# DSA subtraction (Case 9 archetype)                                           #
# ──────────────────────────────────────────────────────────────────────────── #

class TestDsaSubtraction:

    def setup_method(self):
        self.vol = _dsa_volume()
        flat = self.vol.ravel()
        self.p999 = float(np.percentile(flat, 99.9))
        self.vmax  = float(flat.max())

    def test_is_dsa_flag_is_true(self):
        _, _, is_dsa = compute_xa_thresholds(self.vol, 0.0, 1000.0)
        assert is_dsa is True

    def test_lower_is_at_least_p999(self):
        lower, _, _ = compute_xa_thresholds(self.vol, 0.0, 1000.0)
        assert lower >= self.p999 - 1   # allow float rounding

    def test_lower_is_positive(self):
        """DSA lower threshold must be > 0 — vessel voxels are bright."""
        lower, _, _ = compute_xa_thresholds(self.vol, 0.0, 1000.0)
        assert lower > 0

    def test_upper_is_below_max(self):
        """Upper threshold must trim top saturation artefacts."""
        _, upper, _ = compute_xa_thresholds(self.vol, 0.0, 1000.0)
        assert upper < self.vmax

    def test_upper_close_to_95pct_max(self):
        _, upper, _ = compute_xa_thresholds(self.vol, 0.0, 1000.0)
        assert upper == pytest.approx(self.vmax * 0.95, rel=1e-3)

    def test_lower_upper_ordered(self):
        lower, upper, _ = compute_xa_thresholds(self.vol, 0.0, 1000.0)
        assert lower < upper

    def test_wc_ww_not_used(self):
        """WC/WW values should be ignored when DSA is detected."""
        lower1, upper1, _ = compute_xa_thresholds(self.vol, 0.0, 1000.0)
        lower2, upper2, _ = compute_xa_thresholds(self.vol, 300.0, 5000.0)
        assert lower1 == pytest.approx(lower2, rel=1e-3)
        assert upper1 == pytest.approx(upper2, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────────── #
# Standard 3DRA — wide WW (Case 39 / Case 3 archetype)                        #
# ──────────────────────────────────────────────────────────────────────────── #

class TestStandard3DRAWideWW:

    def setup_method(self):
        self.vol = _std_3dra_wide_ww()
        flat = self.vol.ravel()
        self.p90 = float(np.percentile(flat, 90))
        self.p99 = float(np.percentile(flat, 99))

    def test_is_dsa_flag_is_false(self):
        _, _, is_dsa = compute_xa_thresholds(self.vol, 0.0, 7000.0)
        assert is_dsa is False

    def test_lower_equals_p90(self):
        lower, _, _ = compute_xa_thresholds(self.vol, 0.0, 7000.0)
        assert lower == pytest.approx(self.p90, rel=1e-3)

    def test_upper_equals_p99(self):
        _, upper, _ = compute_xa_thresholds(self.vol, 0.0, 7000.0)
        assert upper == pytest.approx(self.p99, rel=1e-3)

    def test_lower_upper_ordered(self):
        lower, upper, _ = compute_xa_thresholds(self.vol, 0.0, 7000.0)
        assert lower < upper

    def test_not_triggered_when_ww_narrow(self):
        """With WW ≤ 2 000 the WC/WW path is used, not p90–p99."""
        lower_wide, _, _ = compute_xa_thresholds(self.vol, 0.0, 7000.0)
        lower_narr, _, _ = compute_xa_thresholds(self.vol, 0.0, 1000.0)
        # Narrow WW should derive a different lower threshold
        assert lower_wide != pytest.approx(lower_narr, rel=0.05)


# ──────────────────────────────────────────────────────────────────────────── #
# Narrow WW — calibrated header (Case 2 archetype)                             #
# ──────────────────────────────────────────────────────────────────────────── #

class TestNarrowWW:

    def test_lower_from_wc_ww(self):
        lower, _, is_dsa = compute_xa_thresholds(None, 500.0, 1000.0)
        assert is_dsa is False
        assert lower == pytest.approx(max(-200.0, 500 - 1000 * 0.35), rel=1e-3)

    def test_upper_from_wc_ww(self):
        _, upper, _ = compute_xa_thresholds(None, 500.0, 1000.0)
        assert upper == pytest.approx(500 + 1000 * 0.45, rel=1e-3)

    def test_no_volume_uses_wc_ww(self):
        """When volume is None the function must not raise."""
        lower, upper, is_dsa = compute_xa_thresholds(None, 200.0, 800.0)
        assert is_dsa is False
        assert lower < upper

    def test_lower_clamped_to_minus_200(self):
        """lower = max(-200, WC - 0.35·WW) — never below -200."""
        lower, _, _ = compute_xa_thresholds(None, 0.0, 100.0)
        assert lower >= -200.0

    def test_volume_with_narrow_ww_uses_wc_ww(self):
        """Standard 3DRA volume + WW ≤ 2 000 → WC/WW path."""
        vol = _std_3dra_wide_ww()
        wc, ww = 400.0, 1200.0
        lower, upper, is_dsa = compute_xa_thresholds(vol, wc, ww)
        assert is_dsa is False
        assert lower == pytest.approx(max(-200.0, wc - ww * 0.35), rel=1e-3)
        assert upper == pytest.approx(wc + ww * 0.45, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────────── #
# Raw 16-bit 3DRA (Case 3 archetype — 3DRA_PROP, no RescaleSlope)             #
# ──────────────────────────────────────────────────────────────────────────── #

def _raw16_3dra_volume(n: int = 100_000) -> np.ndarray:
    """Simulate a 3DRA_PROP volume with raw 16-bit counts (no HU rescale).

    Background: dense cluster at 19 000–21 000 counts (99 % of voxels).
    Vessels: sparse bright voxels at 22 500–40 000 counts (top ~1 %).
    Signature: p90 > 5 000 AND max > 50 000.
    """
    rng = np.random.default_rng(99)
    vol = rng.normal(20_000, 600, n).astype("float32")   # background
    n_vessels = n // 100                                   # 1 % vessel voxels
    idx = rng.choice(n, n_vessels, replace=False)
    vol[idx] = rng.uniform(22_500, 40_000, n_vessels)
    # Ensure max is close to 16-bit ceiling
    vol[0] = 60_000.0
    return vol.reshape(10, 100, 100)


class TestRaw16Bit3DRA:
    """Tests for the raw-16-bit 3DRA branch (Case 3 / 3DRA_PROP archetype)."""

    def setup_method(self):
        self.vol = _raw16_3dra_volume()
        flat = self.vol.ravel()
        self.p99  = float(np.percentile(flat, 99))
        self.p999 = float(np.percentile(flat, 99.9))

    def test_is_dsa_flag_is_false(self):
        _, _, is_dsa = compute_xa_thresholds(self.vol, -343.0, 7577.0)
        assert is_dsa is False

    def test_lower_equals_p99(self):
        """Raw-16-bit branch: lower = p99 (top 1% = vessel-intensity voxels)."""
        lower, _, _ = compute_xa_thresholds(self.vol, -343.0, 7577.0)
        assert lower == pytest.approx(self.p99, rel=1e-3)

    def test_upper_equals_p999(self):
        """Upper threshold = p99.9 to exclude saturation artefacts."""
        _, upper, _ = compute_xa_thresholds(self.vol, -343.0, 7577.0)
        assert upper == pytest.approx(self.p999, rel=1e-3)

    def test_lower_upper_ordered(self):
        lower, upper, _ = compute_xa_thresholds(self.vol, -343.0, 7577.0)
        assert lower < upper

    def test_lower_captures_vessel_fraction(self):
        """lower = p99 → exactly 1 % of voxels are above it."""
        lower, _, _ = compute_xa_thresholds(self.vol, -343.0, 7577.0)
        frac = float(np.mean(self.vol >= lower))
        # Allow 0.5–2 % (some spread from the rng)
        assert 0.005 <= frac <= 0.02

    def test_not_triggered_for_normal_hu_volume(self):
        """A normal HU volume (p90 << 5000) must NOT use the raw-16-bit branch."""
        normal_vol = _std_3dra_wide_ww()   # p90 is typically ~800 HU
        lower_raw, _, _ = compute_xa_thresholds(self.vol, 0.0, 7000.0)
        lower_hu,  _, _ = compute_xa_thresholds(normal_vol, 0.0, 7000.0)
        # The two volumes should give very different lower thresholds
        assert lower_raw > 5_000    # raw-16-bit branch
        assert lower_hu  < 5_000    # wide-WW / p90 branch


# ──────────────────────────────────────────────────────────────────────────── #
# Edge cases                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

class TestEdgeCases:

    def test_all_zeros_volume(self):
        """All-zero volume should not raise and return lower < upper."""
        vol = np.zeros((10, 10, 10), dtype="float32")
        lower, upper, _ = compute_xa_thresholds(vol, 0.0, 500.0)
        assert lower < upper

    def test_constant_positive_volume(self):
        """All-bright volume: p99 > 0 → not DSA, even with large max."""
        vol = np.full((10, 10, 10), 5000.0, dtype="float32")
        _, _, is_dsa = compute_xa_thresholds(vol, 0.0, 7000.0)
        assert is_dsa is False

    def test_returns_tuple_of_three(self):
        result = compute_xa_thresholds(None, 100.0, 400.0)
        assert len(result) == 3
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
        assert isinstance(result[2], bool)
