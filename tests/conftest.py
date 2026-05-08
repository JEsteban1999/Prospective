"""Shared pytest fixtures and configuration."""
from __future__ import annotations

import numpy as np
import pytest

from prospective.dicom.series import DicomMetadata, DicomSeries


@pytest.fixture
def synthetic_series() -> DicomSeries:
    """
    A 50×128×128 synthetic CTA-like volume with a spherical 'aneurysm'
    and a background skull ring — useful for testing without real DICOM data.
    """
    volume = np.full((50, 128, 128), fill_value=-100.0, dtype=np.float32)  # soft tissue

    # Simulated skull ring (bone HU ~700)
    z_c, y_c, x_c = 25, 64, 64
    z, y, x = np.ogrid[:50, :128, :128]
    r_outer = ((z - z_c) ** 2 + (y - y_c) ** 2 + (x - x_c) ** 2) ** 0.5
    skull = (r_outer > 52) & (r_outer < 60)
    volume[skull] = 700.0

    # Simulated aneurysm sac (blood HU ~50)
    aneurysm = r_outer < 8
    volume[aneurysm] = 50.0

    metadata = DicomMetadata(
        patient_name="Ficticio^Paciente",
        patient_id="TEST-001",
        study_date="20260101",
        study_description="CTA Cerebral",
        series_description="CTA HEAD W CONTRAST",
        modality="CT",
        rows=128,
        columns=128,
        num_slices=50,
        pixel_spacing=(0.5, 0.5),
        slice_thickness=0.6,
        window_center=170.0,
        window_width=600.0,
    )

    return DicomSeries(
        metadata=metadata,
        volume=volume,
        spacing=(0.6, 0.5, 0.5),
        origin=(0.0, 0.0, 0.0),
    )


@pytest.fixture(scope="session")
def vtk_sphere_poly():
    """5 mm radius VTK sphere — reused across morphometric / detector tests."""
    import vtk
    src = vtk.vtkSphereSource()
    src.SetRadius(5.0)
    src.SetThetaResolution(32)
    src.SetPhiResolution(32)
    src.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(src.GetOutputPort())
    tri.Update()
    return tri.GetOutput()
