"""Integration tests — segmentation pipeline (A-04-09, S-3)."""
from __future__ import annotations

import numpy as np
import pytest

from prospective.processing.segmentation import (
    GrowResult,
    GrowSegmentationPipeline,
    SegmentationPipeline,
    SegmentationResult,
)


class TestSegmentationPipeline:

    def test_returns_result(self, synthetic_series):
        pipeline = SegmentationPipeline(
            threshold_hu=30.0,
            gaussian_sigma=0.5,
            smooth_iterations=5,
            target_reduction=0.5,
        )
        result = pipeline.run(synthetic_series.volume, synthetic_series.spacing)
        assert result is not None
        assert isinstance(result, SegmentationResult)

    def test_poly_data_has_points(self, synthetic_series):
        pipeline = SegmentationPipeline(threshold_hu=30.0, gaussian_sigma=0.5,
                                        smooth_iterations=5)
        result = pipeline.run(synthetic_series.volume, synthetic_series.spacing)
        assert result.poly_data is not None
        assert result.poly_data.GetNumberOfPoints() > 0

    def test_vertex_count_reasonable(self, synthetic_series):
        pipeline = SegmentationPipeline(threshold_hu=30.0, smooth_iterations=5)
        result = pipeline.run(synthetic_series.volume, synthetic_series.spacing)
        assert result.n_vertices > 10
        assert result.n_triangles > 10

    def test_high_threshold_yields_no_mesh(self, synthetic_series):
        """HU > 2000 captures nothing — pipeline raises ValueError or returns empty."""
        pipeline = SegmentationPipeline(threshold_hu=2000.0, smooth_iterations=0)
        try:
            result = pipeline.run(synthetic_series.volume, synthetic_series.spacing)
            if result is not None:
                assert result.n_vertices == 0
        except ValueError:
            pass   # expected: no iso-surface found

    def test_threshold_stored_in_result(self, synthetic_series):
        pipeline = SegmentationPipeline(threshold_hu=42.0)
        result = pipeline.run(synthetic_series.volume, synthetic_series.spacing)
        if result is not None:
            assert result.threshold_hu == pytest.approx(42.0)

    def test_lower_threshold_more_geometry(self, synthetic_series):
        """Lowering threshold should include more voxels → more vertices."""
        high = SegmentationPipeline(threshold_hu=60.0, smooth_iterations=3,
                                    target_reduction=0.5)
        low  = SegmentationPipeline(threshold_hu=10.0, smooth_iterations=3,
                                    target_reduction=0.5)
        r_high = high.run(synthetic_series.volume, synthetic_series.spacing)
        r_low  = low .run(synthetic_series.volume, synthetic_series.spacing)
        if r_high is not None and r_low is not None:
            assert r_low.n_vertices >= r_high.n_vertices


# ──────────────────────────────────────────────────────────────────────────── #
# S-3 — Grow-from-seeds                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

@pytest.fixture()
def bright_sphere_volume():
    """Small volume with a bright sphere (HU≈200) embedded in background (HU=-500)."""
    vol = np.full((32, 32, 32), -500.0, dtype=np.float32)
    cz, cy, cx = 16, 16, 16
    for z in range(32):
        for y in range(32):
            for x in range(32):
                if (z-cz)**2 + (y-cy)**2 + (x-cx)**2 <= 6**2:
                    vol[z, y, x] = 200.0
    spacing = (1.0, 1.0, 1.0)
    return vol, spacing


class TestGrowSegmentationPipeline:

    def test_returns_grow_result(self, bright_sphere_volume):
        vol, spacing = bright_sphere_volume
        pipeline = GrowSegmentationPipeline(
            lower_hu=100.0, upper_hu=300.0,
            smooth_iterations=3, target_reduction=0.4,
        )
        result = pipeline.run(vol, spacing, seeds=[(16, 16, 16)])
        assert isinstance(result, GrowResult)

    def test_mesh_has_geometry(self, bright_sphere_volume):
        vol, spacing = bright_sphere_volume
        pipeline = GrowSegmentationPipeline(lower_hu=100.0, upper_hu=300.0,
                                            smooth_iterations=3, target_reduction=0.4)
        result = pipeline.run(vol, spacing, seeds=[(16, 16, 16)])
        assert result.n_vertices > 10
        assert result.n_triangles > 10

    def test_voxel_count_positive(self, bright_sphere_volume):
        vol, spacing = bright_sphere_volume
        pipeline = GrowSegmentationPipeline(lower_hu=100.0, upper_hu=300.0)
        result = pipeline.run(vol, spacing, seeds=[(16, 16, 16)])
        # Sphere of radius 6 voxels ≈ 904 voxels
        assert result.n_voxels > 100

    def test_hu_range_stored(self, bright_sphere_volume):
        vol, spacing = bright_sphere_volume
        pipeline = GrowSegmentationPipeline(lower_hu=80.0, upper_hu=500.0)
        result = pipeline.run(vol, spacing, seeds=[(16, 16, 16)])
        assert result.lower_hu == pytest.approx(80.0)
        assert result.upper_hu == pytest.approx(500.0)

    def test_seeds_stored(self, bright_sphere_volume):
        vol, spacing = bright_sphere_volume
        seeds = [(16, 16, 16), (15, 16, 16)]
        pipeline = GrowSegmentationPipeline(lower_hu=100.0, upper_hu=300.0)
        result = pipeline.run(vol, spacing, seeds=seeds)
        assert len(result.seeds) == 2

    def test_no_seeds_raises(self, bright_sphere_volume):
        vol, spacing = bright_sphere_volume
        pipeline = GrowSegmentationPipeline(lower_hu=100.0, upper_hu=300.0)
        with pytest.raises(ValueError, match="seed"):
            pipeline.run(vol, spacing, seeds=[])

    def test_wrong_hu_range_yields_no_voxels(self, bright_sphere_volume):
        """HU range that matches nothing → ValueError (no voxels)."""
        vol, spacing = bright_sphere_volume
        pipeline = GrowSegmentationPipeline(lower_hu=1000.0, upper_hu=2000.0,
                                            smooth_iterations=0)
        with pytest.raises(ValueError):
            pipeline.run(vol, spacing, seeds=[(16, 16, 16)])
