"""Tests for DicomSeries data model."""
from __future__ import annotations

import numpy as np
import pytest

from prospective.dicom.series import DicomMetadata, DicomSeries


class TestDicomMetadata:
    def test_defaults(self):
        m = DicomMetadata()
        assert m.patient_name == ""
        assert m.window_center == 40.0
        assert m.window_width == 400.0
        assert m.pixel_spacing == (1.0, 1.0)

    def test_custom_values(self):
        m = DicomMetadata(patient_name="Smith^John", modality="CT", num_slices=64)
        assert m.patient_name == "Smith^John"
        assert m.modality == "CT"
        assert m.num_slices == 64


class TestDicomSeries:
    def test_shape(self, synthetic_series):
        assert synthetic_series.shape == (50, 128, 128)

    def test_is_loaded(self, synthetic_series):
        assert synthetic_series.is_loaded is True

    def test_not_loaded(self):
        s = DicomSeries()
        assert s.is_loaded is False
        assert s.shape == (0, 0, 0)

    # ── Slice accessors ──────────────────────────────────────────────── #

    def test_axial_slice_shape(self, synthetic_series):
        slc = synthetic_series.get_axial_slice(25)
        assert slc.shape == (128, 128)

    def test_coronal_slice_shape(self, synthetic_series):
        slc = synthetic_series.get_coronal_slice(64)
        assert slc.shape == (50, 128)

    def test_sagital_slice_shape(self, synthetic_series):
        slc = synthetic_series.get_sagital_slice(64)
        assert slc.shape == (50, 128)

    def test_generic_get_slice_axial(self, synthetic_series):
        assert synthetic_series.get_slice("axial", 10).shape == (128, 128)

    def test_generic_get_slice_invalid_plane(self, synthetic_series):
        with pytest.raises(ValueError, match="Unknown plane"):
            synthetic_series.get_slice("oblique", 0)

    # ── Index helpers ─────────────────────────────────────────────────── #

    def test_max_index(self, synthetic_series):
        assert synthetic_series.max_index("axial") == 49
        assert synthetic_series.max_index("coronal") == 127
        assert synthetic_series.max_index("sagital") == 127

    def test_center_index(self, synthetic_series):
        assert synthetic_series.center_index("axial") == 24

    # ── Volume content ───────────────────────────────────────────────── #

    def test_volume_dtype(self, synthetic_series):
        assert synthetic_series.volume.dtype == np.float32

    def test_aneurysm_voxels_present(self, synthetic_series):
        # The fixture places a ~50 HU sphere at the centre
        centre_slice = synthetic_series.get_axial_slice(25)
        assert centre_slice.max() > 40.0

    def test_skull_voxels_present(self, synthetic_series):
        assert synthetic_series.volume.max() > 600.0
