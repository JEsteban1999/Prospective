"""Integration tests — real 3DRA DICOM pipeline end-to-end (XA preset).

Covers two distinct datasets to validate cross-vendor compatibility:

  Case 9  (ANKYRAS)
      Siemens AXIOM-Artis · 3DRA subtracted (3DANGIO SUB)
      198 classic CT Image Storage files · 512×512×198 · 0.362 mm isotropic

  Case 3  (unnamed)
      Philips Interventional Workspot · 3DRA proportional (3DRA_PROP)
      1 Enhanced XA multi-frame file (SOP 1.2.840.10008.5.1.4.1.1.13.1.1)
      384×384×384 · 0.3223 mm isotropic

Both test classes are skipped automatically when the expected resource
directories are absent, so the CI suite remains green on machines that
don't have the source DICOM data.

Run the full set locally with::

    pytest tests/test_pipeline_xa.py -v

Or together with the rest of the slow tests::

    pytest -m "slow or real_data" -v
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from prospective.dicom.loader import DICOMLoader
from prospective.dicom.series import DicomSeries
from prospective.processing.aneurysm_detector import AneurysmDetector, DetectionResult
from prospective.processing.segmentation import SegmentationPipeline

# ── Resource paths ────────────────────────────────────────────────────────────
_BASE = (
    pathlib.Path(__file__).parent.parent
    / "resources" / "DICOM" / "DICOM"
)
_CASE9_DIR = (
    _BASE / "ANKYRAS" / "case 9" / "ANGIOGRAFIA CEREBRAL" / "XA ACI DERECHA VAS"
)
_CASE3_DIR = _BASE / "Case 3" / "Case 3" / "Unknown Study" / "XA"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _xa_band_pass(volume: np.ndarray) -> tuple[float, float]:
    """Derive XA segmentation thresholds from the actual volume distribution.

    Adapts to two distinct 3DRA flavours:

    SUB (subtracted) — e.g. Siemens AXIOM-Artis Case 9:
        Background after subtraction is nearly all −1024 (median ≈ −1024).
        Contrast-filled vessels represent only ~0.5–1% of voxels.
        Threshold must reach p99.5 to cross zero into the vessel region.

    PROP (proportional) — e.g. Philips Interventional Workspot Case 3:
        Background is spread around ~−400 (median ≈ −400).
        Vessels are the bright positive minority.
        p90 → p99 band-pass captures vessels cleanly.

    Heuristic: if the median is below −800 (very dark, heavily subtracted),
    we use high percentiles (p99.5 / no upper); otherwise the standard
    p90 / p99 band-pass.
    """
    flat = volume.ravel().astype(np.float32)
    median_val = float(np.median(flat))

    if median_val < -800:
        # Dark-background subtracted 3DRA (e.g. Siemens AXIOM-Artis):
        # background ≈ −1024, vessels are the top ~0.5% of bright voxels.
        # p99.5 places the lower threshold in the vessel-only region (~192 IS).
        # No upper bound: subtracted images contain no bone/skull above vessels.
        lower = float(np.percentile(flat, 99.5))
        upper = 0.0   # 0 = disabled in SegmentationPipeline
    else:
        # Proportional 3DRA (e.g. Philips Interventional Workspot):
        # bone and tissue are present; vessels are the top 1%.
        # p99 / p99.5 band captures only the brightest vessel cores and
        # keeps the mesh small enough for the detector's region loop.
        lower = float(np.percentile(flat, 99.0))
        upper = float(np.percentile(flat, 99.5))

    return lower, upper


def _run_segmentation(series: DicomSeries) -> "SegmentationPipeline":
    """Return a SegmentationResult from a loaded XA series.

    Parameters chosen for test robustness on real 3DRA data:
    - keep_top_n=5 retains only the 5 largest connected components,
      pruning micro-fragments before Marching Cubes so the detector's
      per-region loop receives a manageable mesh on 512³ / 384³ volumes.
    - target_reduction=0.90 keeps polygon count practical.
    - threshold_max_hu=0 (the default) is intentional for SUB 3DRA where
      _xa_band_pass() returns upper=0 to disable the band-pass upper bound.
    """
    lower, upper = _xa_band_pass(series.volume)
    return SegmentationPipeline(
        threshold_hu=lower,
        threshold_max_hu=upper,   # 0 = disabled for SUB; positive for PROP
        smooth_iterations=15,
        smooth_pass_band=0.08,
        target_reduction=0.90,
        gaussian_sigma=0.0,       # 3DRA volumes are already smooth from FBP
        keep_top_n=5,             # keep the 5 largest vessel components
    ).run(series.volume, series.spacing)


def _run_detection(seg_result) -> DetectionResult:
    """Return a DetectionResult from a segmented 3DRA mesh.

    gauss_percentile=82 is intentionally conservative for real data to keep
    the number of high-curvature regions manageable on large vessel meshes.
    """
    return AneurysmDetector(
        gauss_percentile=82,
        mean_curv_gate_percentile=70,
        min_radius_mm=1.0,
        max_radius_mm=25.0,
        min_points=8,
        min_positive_gauss_frac=0.3,
        min_compactness=0.1,
        min_sphericity=0.2,
        pre_smooth_iterations=10,
        merge_dist_mm=3.0,
    ).detect(seg_result.poly_data)


# ── Case 9: Siemens AXIOM-Artis, 3DRA SUB, 198 classic files ─────────────────

@pytest.mark.real_data
@pytest.mark.slow
class TestCase9SiemensSUB:
    """Pipeline smoke-test for Case 9 — Siemens 3DRA subtracted, 198-file series."""

    @pytest.fixture(scope="class")
    def series(self):
        if not _CASE9_DIR.exists():
            pytest.skip(f"Case 9 DICOM directory not found: {_CASE9_DIR}")
        return DICOMLoader().load_directory(str(_CASE9_DIR))

    @pytest.fixture(scope="class")
    def seg(self, series):
        return _run_segmentation(series)

    @pytest.fixture(scope="class")
    def detection(self, seg):
        return _run_detection(seg)

    # ── Loading ──────────────────────────────────────────────────────── #

    def test_loads_as_dicom_series(self, series):
        assert isinstance(series, DicomSeries)
        assert series.is_loaded

    def test_volume_shape(self, series):
        z, y, x = series.volume.shape
        assert x == 512
        assert y == 512
        assert z == 198

    def test_volume_dtype_float32(self, series):
        assert series.volume.dtype == np.float32

    def test_spacing_approx_isotropic(self, series):
        sz, sy, sx = series.spacing
        # Siemens 3DRA: 0.362 mm voxels
        assert sx == pytest.approx(0.362, abs=0.01)
        assert sy == pytest.approx(0.362, abs=0.01)

    def test_metadata_modality_xa(self, series):
        assert series.metadata.modality.upper() == "XA"

    def test_metadata_pixel_spacing_correct(self, series):
        """Classic multi-file: PixelSpacing at top level — must not be 1.0 mm."""
        ps_row, ps_col = series.metadata.pixel_spacing
        assert ps_row == pytest.approx(0.362, abs=0.01), (
            f"metadata.pixel_spacing row={ps_row:.4f} mm (expected ≈0.362)"
        )
        assert ps_col == pytest.approx(0.362, abs=0.01)

    # ── Segmentation ─────────────────────────────────────────────────── #

    def test_segmentation_produces_mesh(self, seg):
        assert seg.poly_data.GetNumberOfPoints() > 0
        assert seg.poly_data.GetNumberOfCells() > 0

    def test_segmentation_threshold_above_background(self, series):
        """For subtracted 3DRA, the adaptive lower threshold must be positive.

        Siemens SUB 3DRA: background ≈ −1024, vessels are the top ~0.5% of
        bright voxels.  The adaptive heuristic (median < −800 → use p99.5)
        should produce a positive lower threshold that targets only vessels.
        """
        lower, upper = _xa_band_pass(series.volume)
        # With median ≈ −1024 the heuristic switches to p99.5 ≈ +192
        assert lower > 0, (
            f"lower threshold {lower:.0f} IS is not positive — "
            "adaptive heuristic should select p99.5 for dark-background SUB data"
        )

    # ── Detection ────────────────────────────────────────────────────── #

    def test_detection_returns_result(self, detection):
        assert isinstance(detection, DetectionResult)

    def test_detection_counters_non_negative(self, detection):
        assert detection.n_regions_total >= 0
        assert detection.n_failed_size >= 0
        assert detection.n_failed_points >= 0
        assert detection.n_failed_mean_curv >= 0
        assert detection.n_merged >= 0

    def test_detection_analyzed_regions(self, detection):
        """The detector must have analysed a non-trivial number of mesh regions.

        The DICOM series for Case 9 is a partial export (198 of a full scan,
        with slice gaps up to 23.9 mm), so the aneurysm may not be within the
        exported region.  We therefore only assert that the detector ran the
        full pipeline (n_regions_total > 0) rather than requiring a candidate.
        """
        assert detection.n_regions_total > 0, (
            "n_regions_total=0 — the curvature threshold may have rejected the "
            "entire mesh before the region loop even ran"
        )

    def test_candidates_sorted_by_score(self, detection):
        if len(detection.candidates) >= 2:
            scores = [c.score for c in detection.candidates]
            assert scores == sorted(scores, reverse=True)

    def test_candidates_within_size_bounds(self, detection):
        for c in detection.candidates:
            assert 1.0 <= c.radius_mm <= 25.0
            assert c.n_points >= 8
            assert 0.0 <= c.score <= 1.0


# ── Case 3: Philips Interventional Workspot, 3DRA PROP, single Enhanced XA ───

@pytest.mark.real_data
@pytest.mark.slow
class TestCase3PhilipsPROP:
    """Pipeline smoke-test for Case 3 — Philips 3DRA proportional, Enhanced XA."""

    @pytest.fixture(scope="class")
    def series(self):
        if not _CASE3_DIR.exists():
            pytest.skip(f"Case 3 DICOM directory not found: {_CASE3_DIR}")
        return DICOMLoader().load_directory(str(_CASE3_DIR))

    @pytest.fixture(scope="class")
    def seg(self, series):
        return _run_segmentation(series)

    @pytest.fixture(scope="class")
    def detection(self, seg):
        return _run_detection(seg)

    # ── Loading ──────────────────────────────────────────────────────── #

    def test_loads_as_dicom_series(self, series):
        assert isinstance(series, DicomSeries)
        assert series.is_loaded

    def test_volume_is_perfect_cube(self, series):
        z, y, x = series.volume.shape
        assert x == 384
        assert y == 384
        assert z == 384

    def test_volume_dtype_float32(self, series):
        assert series.volume.dtype == np.float32

    def test_spacing_isotropic(self, series):
        sz, sy, sx = series.spacing
        # Philips 3DRA: 0.3223 mm isotropic, perfectly uniform
        for s in (sx, sy, sz):
            assert s == pytest.approx(0.3223, abs=0.001), (
                f"spacing component {s:.4f} mm (expected ≈0.3223)"
            )

    def test_metadata_modality_xa(self, series):
        assert series.metadata.modality.upper() == "XA"

    def test_metadata_pixel_spacing_from_functional_groups(self, series):
        """Enhanced XA: PixelSpacing is in SharedFunctionalGroupsSequence.

        Regression test for the metadata extraction bug — without the fix,
        _extract_metadata() returned (1.0, 1.0) for Enhanced XA files because
        it only looked at top-level DICOM tags where the tag is absent.
        """
        ps_row, ps_col = series.metadata.pixel_spacing
        assert ps_row == pytest.approx(0.3223, abs=0.001), (
            f"metadata.pixel_spacing row={ps_row:.4f} mm — "
            "Enhanced XA SharedFunctionalGroupsSequence fallback may not be applied"
        )
        assert ps_col == pytest.approx(0.3223, abs=0.001)

    def test_metadata_slice_thickness_from_functional_groups(self, series):
        """SliceThickness must also be resolved from SharedFunctionalGroupsSequence."""
        st = series.metadata.slice_thickness
        assert st == pytest.approx(0.3223, abs=0.001), (
            f"metadata.slice_thickness={st:.4f} mm — "
            "Enhanced XA SharedFunctionalGroupsSequence fallback may not be applied"
        )

    def test_volume_values_rescaled(self, series):
        """SimpleITK must apply RescaleSlope/Intercept automatically.

        Raw uint16 pixel range is [0, 65535].  After slope=0.7365 and
        intercept=−15093, the minimum becomes ≈ −15093.  If values are
        still in raw uint16 range, rescaling was not applied.
        """
        assert series.volume.min() < 0, (
            "Volume minimum is not negative — RescaleIntercept may not have been applied "
            f"(min={series.volume.min():.0f})"
        )
        # Background should be well below −1000 IS
        assert series.volume.min() < -1000

    # ── Segmentation ─────────────────────────────────────────────────── #

    def test_segmentation_produces_mesh(self, seg):
        assert seg.poly_data.GetNumberOfPoints() > 0
        assert seg.poly_data.GetNumberOfCells() > 0

    def test_segmentation_threshold_above_background(self, series):
        """Adaptive threshold must target the bright vessel region (> 0 IS).

        Philips PROP 3DRA: background median ≈ −400 IS; the top-1% vessel
        cores (p99 ≈ 1471 IS) are well above zero.
        """
        lower, upper = _xa_band_pass(series.volume)
        assert lower > 0, (
            f"lower threshold {lower:.0f} IS unexpectedly ≤ 0 "
            "(vessel cores should be positive in proportional 3DRA)"
        )
        assert upper > lower

    # ── Detection ────────────────────────────────────────────────────── #

    def test_detection_returns_result(self, detection):
        assert isinstance(detection, DetectionResult)

    def test_detection_counters_non_negative(self, detection):
        assert detection.n_regions_total >= 0
        assert detection.n_failed_size >= 0
        assert detection.n_failed_points >= 0
        assert detection.n_merged >= 0

    def test_detection_result_has_candidate_fields(self, detection):
        """If any candidates survive, they must have all required v5 fields."""
        for c in detection.candidates:
            assert c.radius_mm > 0
            assert 0.0 <= c.sphericity <= 1.01
            assert 0.0 <= c.score <= 1.0
            assert hasattr(c, "positive_gauss_frac")
            assert hasattr(c, "compactness")
            assert hasattr(c, "cv_gauss")
            assert hasattr(c, "normal_isotropy")
            assert 0.0 <= c.normal_isotropy <= 1.0


# ── Cross-vendor comparison ───────────────────────────────────────────────────

@pytest.mark.real_data
class TestCrossVendorLoader:
    """Verify that both datasets load with consistent types regardless of format."""

    def test_case9_and_case3_both_float32(self):
        """Both datasets must yield float32 volumes regardless of on-disk format."""
        for path, label in ((_CASE9_DIR, "Case 9"), (_CASE3_DIR, "Case 3")):
            if not path.exists():
                pytest.skip(f"{label} DICOM directory not found")
        s9 = DICOMLoader().load_directory(str(_CASE9_DIR))
        s3 = DICOMLoader().load_directory(str(_CASE3_DIR))
        assert s9.volume.dtype == np.float32, "Case 9 volume not float32"
        assert s3.volume.dtype == np.float32, "Case 3 volume not float32"

    def test_both_have_sub_mm_voxels(self):
        """Both 3DRA datasets are high-resolution (< 0.5 mm voxels)."""
        for path, label in ((_CASE9_DIR, "Case 9"), (_CASE3_DIR, "Case 3")):
            if not path.exists():
                pytest.skip(f"{label} DICOM directory not found")
        s9 = DICOMLoader().load_directory(str(_CASE9_DIR))
        s3 = DICOMLoader().load_directory(str(_CASE3_DIR))
        for s, label in ((s9, "Case 9"), (s3, "Case 3")):
            _, sy, sx = s.spacing
            assert sx < 0.5, f"{label}: sx={sx:.3f} mm is not sub-mm"
            assert sy < 0.5, f"{label}: sy={sy:.3f} mm is not sub-mm"

    def test_both_metadata_pixel_spacing_not_default(self):
        """After the Enhanced XA metadata fix, neither series may report 1.0 mm spacing."""
        for path, label in ((_CASE9_DIR, "Case 9"), (_CASE3_DIR, "Case 3")):
            if not path.exists():
                pytest.skip(f"{label} DICOM directory not found")
        s9 = DICOMLoader().load_directory(str(_CASE9_DIR))
        s3 = DICOMLoader().load_directory(str(_CASE3_DIR))
        for s, label in ((s9, "Case 9"), (s3, "Case 3")):
            ps = s.metadata.pixel_spacing
            assert ps != (1.0, 1.0), (
                f"{label}: metadata.pixel_spacing is still the default (1.0, 1.0) — "
                "pixel spacing not resolved correctly"
            )
