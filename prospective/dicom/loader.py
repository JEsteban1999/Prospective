"""DICOM loading using pydicom + SimpleITK."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pydicom
import SimpleITK as sitk

from prospective.dicom.series import DicomMetadata, DicomSeries

logger = logging.getLogger(__name__)


class DICOMLoader:
    """
    Loads a DICOM series from a directory or an explicit file list.

    Uses SimpleITK's ImageSeriesReader for robust multi-file sorting and
    pydicom for metadata extraction.
    """

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_directory(self, directory: str) -> DicomSeries:
        """
        Load all DICOM files in *directory* as a single 3-D series.

        Handles both multi-file series and single multi-frame DICOM files.

        Raises
        ------
        FileNotFoundError
            If *directory* does not exist.
        ValueError
            If no DICOM files are found inside *directory*.
        """
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))

        if series_ids:
            # Standard multi-file series
            dicom_names = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
            if dicom_names:
                return self._read_series(list(dicom_names))

        # Fallback: single multi-frame .dcm file (Enhanced CT/MR, secondary capture, etc.)
        dcm_files = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in (".dcm", "") and p.is_file()
        )
        if not dcm_files:
            raise ValueError(f"No DICOM files found in: {directory}")

        logger.info("Falling back to single-file load: %s", dcm_files[0])
        return self._read_single_file(str(dcm_files[0]))

    def load_files(self, file_paths: List[str]) -> DicomSeries:
        """Load a series from an explicit ordered list of .dcm files."""
        if not file_paths:
            raise ValueError("file_paths must not be empty")
        return self._read_series(file_paths)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _read_series(self, file_paths: List[str]) -> DicomSeries:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_paths)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image: sitk.Image = reader.Execute()
        return self._image_to_series(image, file_paths[0])

    def _read_single_file(self, file_path: str) -> DicomSeries:
        """Load a single (possibly multi-frame) DICOM file."""
        image: sitk.Image = sitk.ReadImage(file_path)
        return self._image_to_series(image, file_path)

    def _image_to_series(self, image: sitk.Image, ref_path: str) -> DicomSeries:
        """Convert a SimpleITK image of any dimensionality to a DicomSeries."""
        # GetArrayFromImage: for a 3D image → (z, y, x); for 2D → (y, x)
        array = sitk.GetArrayFromImage(image).astype(np.float32)

        # Squeeze out any leading size-1 dimensions (e.g. single-phase 4D volumes)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]

        # If still >3D take the first volume along axis 0
        if array.ndim > 3:
            logger.warning(
                "Volume has %d dimensions (shape=%s); taking first 3D sub-volume.",
                array.ndim, array.shape,
            )
            array = array[0]

        # Ensure 3D — add a z-axis if the image is 2D
        if array.ndim == 2:
            array = array[np.newaxis, ...]  # (1, y, x)

        # Take only the first 3 spacing components regardless of image dimensionality
        raw_spacing = image.GetSpacing()
        sx = float(raw_spacing[0]) if len(raw_spacing) > 0 else 1.0
        sy = float(raw_spacing[1]) if len(raw_spacing) > 1 else 1.0
        sz = float(raw_spacing[2]) if len(raw_spacing) > 2 else 1.0

        raw_origin = image.GetOrigin()
        origin = (
            float(raw_origin[0]) if len(raw_origin) > 0 else 0.0,
            float(raw_origin[1]) if len(raw_origin) > 1 else 0.0,
            float(raw_origin[2]) if len(raw_origin) > 2 else 0.0,
        )

        metadata = self._extract_metadata(ref_path, array.shape)
        metadata.pixel_spacing = (sy, sx)
        metadata.slice_thickness = sz

        series = DicomSeries(
            metadata=metadata,
            volume=array,
            spacing=(sz, sy, sx),   # (z, y, x) to match volume axes
            origin=origin,
        )

        logger.info(
            "Loaded DICOM: shape=%s  spacing=(%.3f, %.3f, %.3f) mm",
            array.shape, sz, sy, sx,
        )
        return series

    def _extract_metadata(self, dcm_path: str, shape: tuple) -> DicomMetadata:
        """Read DICOM tags from the first file of the series.

        Handles both classic DICOM (CT/MR multi-file) and Enhanced multi-frame
        formats (Enhanced XA, Enhanced CT/MR — SOP 1.2.840.10008.5.1.4.1.1.13.*).
        In Enhanced formats, PixelSpacing and SliceThickness live inside
        SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0] rather than
        at the top level.
        """
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        except Exception as exc:
            logger.warning("Could not read DICOM metadata from %s: %s", dcm_path, exc)
            return DicomMetadata(num_slices=shape[0])

        def get_str(tag: str, default: str = "") -> str:
            try:
                val = getattr(ds, tag, None)
                return str(val).strip() if val is not None else default
            except Exception:
                return default

        def get_float(tag: str, default: float = 0.0) -> float:
            try:
                val = getattr(ds, tag, None)
                if val is None:
                    return default
                # DSfloat / MultiValue
                if hasattr(val, "__iter__") and not isinstance(val, str):
                    return float(val[0])
                return float(val)
            except Exception:
                return default

        # ── Pixel spacing / slice thickness ───────────────────────────────── #
        # Classic DICOM (CT, MR single-frame): PixelSpacing and SliceThickness
        # are top-level tags.
        # Enhanced XA / Enhanced CT / Enhanced MR (multi-frame): they are nested
        # inside SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].
        # We try the top level first; fall back to the functional group only
        # when the top-level tag is absent (None).
        pixel_spacing: list[float] = [1.0, 1.0]
        slice_thickness: float = 1.0

        raw_ps_top = getattr(ds, "PixelSpacing", None)
        raw_st_top = getattr(ds, "SliceThickness", None)

        if raw_ps_top is not None:
            try:
                pixel_spacing = [float(raw_ps_top[0]), float(raw_ps_top[1])]
            except Exception:
                logger.debug("Could not parse PixelSpacing %r — using default", raw_ps_top, exc_info=True)

        if raw_st_top is not None:
            try:
                slice_thickness = float(raw_st_top)
            except Exception:
                logger.debug("Could not parse SliceThickness %r — using default", raw_st_top, exc_info=True)

        # Fallback for Enhanced XA and other multi-frame Enhanced IODs
        if raw_ps_top is None or raw_st_top is None:
            try:
                pms = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
                if raw_ps_top is None:
                    raw_ps2 = getattr(pms, "PixelSpacing", None)
                    if raw_ps2 is not None:
                        pixel_spacing = [float(raw_ps2[0]), float(raw_ps2[1])]
                        logger.debug(
                            "PixelSpacing resolved from SharedFunctionalGroupsSequence: %s",
                            pixel_spacing,
                        )
                if raw_st_top is None:
                    raw_st2 = getattr(pms, "SliceThickness", None)
                    if raw_st2 is not None:
                        slice_thickness = float(raw_st2)
                        logger.debug(
                            "SliceThickness resolved from SharedFunctionalGroupsSequence: %.4f",
                            slice_thickness,
                        )
            except Exception:
                logger.debug("SharedFunctionalGroupsSequence absent — using default spacing", exc_info=True)

        return DicomMetadata(
            patient_name=get_str("PatientName"),
            patient_id=get_str("PatientID"),
            study_date=get_str("StudyDate"),
            study_description=get_str("StudyDescription"),
            series_description=get_str("SeriesDescription"),
            modality=get_str("Modality", "CT"),
            rows=int(get_float("Rows", shape[1])),
            columns=int(get_float("Columns", shape[2])),
            num_slices=shape[0],
            pixel_spacing=(pixel_spacing[0], pixel_spacing[1]),
            slice_thickness=slice_thickness,
            window_center=get_float("WindowCenter", 40.0),
            window_width=get_float("WindowWidth", 400.0),
        )
