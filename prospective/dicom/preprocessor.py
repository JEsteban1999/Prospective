"""DICOM volume preprocessing utilities."""
from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk

from prospective.dicom.series import DicomSeries

logger = logging.getLogger(__name__)

# HU clipping range suitable for intracranial CTA
HU_MIN = -1000.0
HU_MAX = 3000.0


class DicomPreprocessor:
    """
    Preprocessing pipeline for CTA volumes.

    Steps (all optional, controlled by constructor flags):
    1. HU clipping           — remove extreme outliers
    2. Isotropic resampling  — resample to cubic voxels
    3. Gaussian smoothing    — mild noise reduction
    """

    def __init__(
        self,
        clip_hu: bool = True,
        resample_isotropic: bool = False,
        target_spacing_mm: float = 0.5,
        smooth: bool = False,
        smooth_sigma: float = 0.5,
    ) -> None:
        self.clip_hu = clip_hu
        self.resample_isotropic = resample_isotropic
        self.target_spacing_mm = target_spacing_mm
        self.smooth = smooth
        self.smooth_sigma = smooth_sigma

    def process(self, series: DicomSeries) -> DicomSeries:
        """Return a new DicomSeries with the preprocessing applied."""
        if not series.is_loaded:
            raise ValueError("Series has no loaded volume")

        image = self._to_sitk(series)

        if self.clip_hu:
            image = self._clip_hu(image)

        if self.resample_isotropic:
            image = self._resample(image, self.target_spacing_mm)

        if self.smooth:
            image = self._gaussian_smooth(image, self.smooth_sigma)

        return self._to_series(image, series.metadata)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_sitk(series: DicomSeries) -> sitk.Image:
        """Convert DicomSeries volume (z,y,x) float32 → SimpleITK image."""
        image = sitk.GetImageFromArray(series.volume)  # expects (z,y,x)
        sz, sy, sx = series.spacing
        image.SetSpacing((sx, sy, sz))   # SimpleITK uses (x,y,z)
        image.SetOrigin(series.origin)
        return image

    @staticmethod
    def _clip_hu(image: sitk.Image) -> sitk.Image:
        return sitk.Clamp(image, sitk.sitkFloat32, HU_MIN, HU_MAX)

    @staticmethod
    def _resample(image: sitk.Image, target_mm: float) -> sitk.Image:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_spacing = (target_mm, target_mm, target_mm)
        new_size = [
            int(round(osz * ospc / target_mm))
            for osz, ospc in zip(original_size, original_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1000.0)
        resampler.SetInterpolator(sitk.sitkLinear)

        logger.info(
            "Resampling %s → %s at %.2f mm isotropic",
            original_size,
            new_size,
            target_mm,
        )
        return resampler.Execute(image)

    @staticmethod
    def _gaussian_smooth(image: sitk.Image, sigma: float) -> sitk.Image:
        return sitk.SmoothingRecursiveGaussian(image, sigma)

    # ------------------------------------------------------------------ #
    # Standalone utility — bone subtraction                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def subtract_bone(
        volume: np.ndarray,
        bone_threshold_hu: float = 300.0,
    ) -> np.ndarray:
        """Return a copy of *volume* with bone voxels replaced by -1000 HU.

        All voxels above *bone_threshold_hu* are classified as bone and set to
        air (-1000 HU), effectively removing them from the volume so that the
        CTA transfer function can isolate contrast-enhanced vessels without
        cortical-bone bleed-through.

        Parameters
        ----------
        volume:
            Float32 array of shape (z, y, x) in Hounsfield units.
        bone_threshold_hu:
            HU value above which a voxel is classified as bone (default 300 HU).
            Typical CTA: cancellous bone starts ~200 HU, cortical bone ~400 HU.
            300 HU is a conservative threshold that removes most calvaria while
            preserving calcified plaques at the vessel wall (<300 HU typically).
        """
        result = volume.copy()
        result[result > bone_threshold_hu] = -1000.0
        return result

    @staticmethod
    def _to_series(image: sitk.Image, metadata) -> DicomSeries:
        from prospective.dicom.series import DicomSeries

        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        sx, sy, sz = image.GetSpacing()

        new_meta = metadata.__class__(**{
            k: v for k, v in metadata.__dict__.items()
        })
        new_meta.num_slices = volume.shape[0]
        new_meta.rows = volume.shape[1]
        new_meta.columns = volume.shape[2]
        new_meta.slice_thickness = float(sz)
        new_meta.pixel_spacing = (float(sy), float(sx))

        return DicomSeries(
            metadata=new_meta,
            volume=volume,
            spacing=(float(sz), float(sy), float(sx)),
            origin=image.GetOrigin(),
        )
