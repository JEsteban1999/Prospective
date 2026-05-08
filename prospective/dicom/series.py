"""DICOM series data model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DicomMetadata:
    """Patient and study metadata extracted from DICOM headers."""

    patient_name: str = ""
    patient_id: str = ""
    study_date: str = ""
    study_description: str = ""
    series_description: str = ""
    modality: str = ""
    rows: int = 0
    columns: int = 0
    num_slices: int = 0
    pixel_spacing: tuple[float, float] = (1.0, 1.0)   # (row_spacing, col_spacing) mm
    slice_thickness: float = 1.0                        # mm
    window_center: float = 40.0                         # HU
    window_width: float = 400.0                         # HU


@dataclass
class DicomSeries:
    """
    A loaded DICOM series represented as a 3-D NumPy volume.

    Volume axes: (z, y, x) — z = slice index (superior→inferior for axial CTA).
    Spacing:     (z_spacing, y_spacing, x_spacing) in mm.
    """

    metadata: DicomMetadata = field(default_factory=DicomMetadata)
    volume: Optional[np.ndarray] = None  # float32, HU values
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.volume.shape if self.volume is not None else (0, 0, 0)

    @property
    def is_loaded(self) -> bool:
        return self.volume is not None

    # ------------------------------------------------------------------ #
    # Slice accessors                                                      #
    # ------------------------------------------------------------------ #

    def get_axial_slice(self, z: int) -> np.ndarray:
        """Return the axial slice at index *z* (shape: y × x)."""
        return self.volume[z, :, :]

    def get_coronal_slice(self, y: int) -> np.ndarray:
        """Return the coronal slice at index *y* (shape: z × x)."""
        return self.volume[:, y, :]

    def get_sagital_slice(self, x: int) -> np.ndarray:
        """Return the sagital slice at index *x* (shape: z × y)."""
        return self.volume[:, :, x]

    def get_slice(self, plane: str, index: int) -> np.ndarray:
        """Generic accessor: plane in {'axial', 'coronal', 'sagital'}."""
        if plane == "axial":
            return self.get_axial_slice(index)
        elif plane == "coronal":
            return self.get_coronal_slice(index)
        elif plane == "sagital":
            return self.get_sagital_slice(index)
        raise ValueError(f"Unknown plane: {plane!r}")

    def max_index(self, plane: str) -> int:
        """Return the last valid slice index for the given plane."""
        z, y, x = self.shape
        return {"axial": z - 1, "coronal": y - 1, "sagital": x - 1}[plane]

    def center_index(self, plane: str) -> int:
        return self.max_index(plane) // 2
