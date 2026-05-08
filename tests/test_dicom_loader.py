"""Tests for DICOMLoader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from prospective.dicom.loader import DICOMLoader
from prospective.dicom.series import DicomSeries


def _write_synthetic_dicom_series(directory: Path, num_slices: int = 5) -> None:
    """Write a minimal synthetic DICOM series to *directory* using SimpleITK."""
    import pydicom
    from pydicom.uid import generate_uid

    volume = np.random.randint(-100, 1000, (num_slices, 64, 64), dtype=np.int16)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((0.5, 0.5, 1.0))

    series_uid   = generate_uid()
    study_uid    = generate_uid()
    frame_of_ref = generate_uid()

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    for i in range(num_slices):
        slc = image[:, :, i]
        slc.SetMetaData("0008|0060", "CT")
        slc.SetMetaData("0008|0016", "1.2.840.10008.5.1.4.1.1.2")  # CT SOP Class
        slc.SetMetaData("0010|0020", "SYNTH001")
        slc.SetMetaData("0020|000D", study_uid)        # Study Instance UID
        slc.SetMetaData("0020|000E", series_uid)       # Series Instance UID (same for all)
        slc.SetMetaData("0020|0013", str(i + 1))       # Instance Number
        slc.SetMetaData("0020|0052", frame_of_ref)     # Frame of Reference UID
        slc.SetMetaData("0020|0032", f"0\\0\\{i * 1.0}")  # Image Position Patient
        slc.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")  # Image Orientation Patient
        out = str(directory / f"slice_{i:03d}.dcm")
        writer.SetFileName(out)
        writer.Execute(slc)


class TestDICOMLoader:
    def test_missing_directory(self):
        loader = DICOMLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory("/nonexistent/path/xyz")

    def test_empty_directory(self, tmp_path):
        loader = DICOMLoader()
        with pytest.raises(ValueError):
            loader.load_directory(str(tmp_path))

    def test_load_synthetic_series(self, tmp_path):
        _write_synthetic_dicom_series(tmp_path, num_slices=5)
        loader = DICOMLoader()
        series = loader.load_directory(str(tmp_path))

        assert isinstance(series, DicomSeries)
        assert series.is_loaded
        assert series.volume.dtype == np.float32
        assert series.volume.shape[0] == 5
        assert series.volume.shape[1] == 64
        assert series.volume.shape[2] == 64

    def test_load_files_empty(self):
        loader = DICOMLoader()
        with pytest.raises(ValueError, match="empty"):
            loader.load_files([])

    def test_spacing_parsed(self, tmp_path):
        _write_synthetic_dicom_series(tmp_path, num_slices=5)
        series = DICOMLoader().load_directory(str(tmp_path))
        sz, sy, sx = series.spacing
        assert sx > 0
        assert sy > 0
        assert sz > 0


class TestDicomPreprocessor:
    def test_clip_hu(self, synthetic_series):
        from prospective.dicom.preprocessor import DicomPreprocessor

        # Inject extreme HU values
        synthetic_series.volume[0, 0, 0] = 5000.0
        synthetic_series.volume[0, 0, 1] = -2000.0

        proc = DicomPreprocessor(clip_hu=True)
        result = proc.process(synthetic_series)

        assert result.volume.max() <= 3000.0
        assert result.volume.min() >= -1000.0

    def test_shape_preserved_without_resample(self, synthetic_series):
        from prospective.dicom.preprocessor import DicomPreprocessor

        original_shape = synthetic_series.shape
        proc = DicomPreprocessor(clip_hu=True, resample_isotropic=False)
        result = proc.process(synthetic_series)
        assert result.shape == original_shape

    def test_metadata_copied(self, synthetic_series):
        from prospective.dicom.preprocessor import DicomPreprocessor

        proc = DicomPreprocessor(clip_hu=True)
        result = proc.process(synthetic_series)
        assert result.metadata.patient_id == synthetic_series.metadata.patient_id
