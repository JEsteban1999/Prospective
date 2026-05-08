"""Oblique MPR viewer — 3D Slicer-inspired.

Opens a standalone dialog with a single oblique plane.
The user controls the cut-plane orientation (rotX, rotY) and
scroll position via sliders or mouse-wheel, with real-time update
via ``vtkImageReslice``.

Usage
-----
    dlg = ObliqueMPRDialog(volume_array, spacing, parent=window)
    dlg.show()
"""
from __future__ import annotations

import math

import numpy as np
import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


class ObliqueMPRDialog(QDialog):
    """Oblique MPR slice viewer.

    Parameters
    ----------
    volume : np.ndarray, shape (Z, Y, X)
        3-D greyscale volume (int16 or float32).
    spacing : tuple[float, float, float]
        Voxel spacing in mm — (dz, dy, dx) order matching the array.
    wc, ww : float
        Initial window centre / width for HU display.
    """

    def __init__(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        wc: float = 40.0,
        ww: float = 400.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PROSPECTIVE — MPR Oblicuo")
        self.setMinimumSize(640, 560)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._spacing = spacing
        self._wc      = wc
        self._ww      = ww

        self._vtk_image = self._numpy_to_vtk_image(volume, spacing)
        self._build_ui()
        self._init_vtk()
        self._update_slice()

    # ------------------------------------------------------------------ #
    # Build                                                                #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        vl = QVBoxLayout(self)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(6)

        hdr = QLabel(
            "<b style='color:#A8B8C6'>MPR Oblicuo</b>"
            "<small style='color:#9B9B9B'>  —  define el plano de corte con los "
            "deslizadores de rotación</small>"
        )
        vl.addWidget(hdr)

        self._iren = QVTKRenderWindowInteractor()
        self._iren.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(self._iren, stretch=1)

        # ── Controls ──────────────────────────────────────────────────── #
        ctrl = QWidget()
        cl   = QHBoxLayout(ctrl)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(12)

        def _make_slider(label: str, lo: int, hi: int, val: int):
            w  = QWidget()
            wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(2)
            lbl = QLabel(f"<small>{label}</small>")
            lbl.setAlignment(Qt.AlignCenter)
            s = QSlider(Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            s.setTickInterval((hi - lo) // 10)
            s.setTickPosition(QSlider.TicksBelow)
            wl.addWidget(lbl)
            wl.addWidget(s)
            return w, s

        rot_x_grp, self._sl_rotx = _make_slider("Rotación X  (°)", -90, 90, 0)
        rot_y_grp, self._sl_roty = _make_slider("Rotación Y  (°)", -90, 90, 0)

        # Slice position scroll bar
        extent = self._vtk_image.GetExtent()   # (x0,x1,y0,y1,z0,z1)
        max_z  = extent[5]
        pos_grp, self._sl_pos = _make_slider("Posición  (voxel)", 0, max_z, max_z // 2)

        cl.addWidget(rot_x_grp, stretch=1)
        cl.addWidget(rot_y_grp, stretch=1)
        cl.addWidget(pos_grp,   stretch=1)
        vl.addWidget(ctrl)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet(
            "color:#9B9B9B; font-size:10px;" if _is_dark()
            else "color:#6B6B6B; font-size:10px;"
        )
        vl.addWidget(self._lbl_status)

        # Connect sliders
        for sl in (self._sl_rotx, self._sl_roty, self._sl_pos):
            sl.valueChanged.connect(self._update_slice)

    def _init_vtk(self) -> None:
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.04, 0.06, 0.10)
        self._iren.GetRenderWindow().AddRenderer(self._renderer)

        # Interactor style — disable 3D rotation (2D pan/zoom only)
        style = vtk.vtkInteractorStyleImage()
        self._iren.SetInteractorStyle(style)

        # vtkImageReslice
        self._reslice = vtk.vtkImageReslice()
        self._reslice.SetInputData(self._vtk_image)
        self._reslice.SetOutputDimensionality(2)
        self._reslice.SetInterpolationModeToLinear()

        # Lookup table for HU → grey
        self._lut = vtk.vtkWindowLevelLookupTable()
        self._lut.SetWindow(self._ww)
        self._lut.SetLevel(self._wc)
        self._lut.Build()

        # Image colour map
        self._colour = vtk.vtkImageMapToColors()
        self._colour.SetLookupTable(self._lut)
        self._colour.SetInputConnection(self._reslice.GetOutputPort())

        # Actor
        self._slice_actor = vtk.vtkImageActor()
        self._slice_actor.GetMapper().SetInputConnection(self._colour.GetOutputPort())
        self._renderer.AddActor(self._slice_actor)

        self._iren.Initialize()

    # ------------------------------------------------------------------ #
    # Reslice update                                                       #
    # ------------------------------------------------------------------ #

    def _update_slice(self) -> None:
        """Recompute the oblique slice from current slider values."""
        rot_x_deg = self._sl_rotx.value()
        rot_y_deg = self._sl_roty.value()
        pos_vox   = self._sl_pos.value()

        bounds = self._vtk_image.GetBounds()
        cx = (bounds[0] + bounds[1]) / 2.0
        cy = (bounds[2] + bounds[3]) / 2.0
        # Physical Z position of the slice
        dz = self._spacing[0]
        pz = bounds[4] + pos_vox * dz

        # Build reslice axes matrix from euler angles
        # Start with identity, rotate around X then Y
        rx = math.radians(rot_x_deg)
        ry = math.radians(rot_y_deg)

        # Rotation matrices
        Rx = np.array([
            [1,          0,           0],
            [0,  math.cos(rx), -math.sin(rx)],
            [0,  math.sin(rx),  math.cos(rx)],
        ])
        Ry = np.array([
            [ math.cos(ry), 0, math.sin(ry)],
            [0,             1, 0           ],
            [-math.sin(ry), 0, math.cos(ry)],
        ])
        R = Ry @ Rx   # combined rotation (Y after X)

        # Reslice axes: rows are the output x, y, z axes in VTK image space
        axes = vtk.vtkMatrix4x4()
        axes.Identity()
        for col in range(3):
            for row in range(3):
                axes.SetElement(row, col, R[row, col])
        # Origin: centre of the volume in XY, user-chosen Z
        axes.SetElement(0, 3, cx)
        axes.SetElement(1, 3, cy)
        axes.SetElement(2, 3, pz)

        self._reslice.SetResliceAxes(axes)
        self._reslice.Update()
        self._colour.Update()
        self._slice_actor.GetMapper().Update()

        self._renderer.ResetCameraClippingRange()
        if rot_x_deg == 0 and rot_y_deg == 0:
            # Only reset camera on the first call or when angle returns to 0
            self._renderer.ResetCamera()

        self._lbl_status.setText(
            f"Rot X: {rot_x_deg:+d}°  ·  Rot Y: {rot_y_deg:+d}°  ·  "
            f"Posición Z: {pz:.1f} mm  (voxel {pos_vox})"
        )
        self._iren.GetRenderWindow().Render()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_to_vtk_image(
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> vtk.vtkImageData:
        """Convert (Z, Y, X) numpy array to vtkImageData."""
        import vtk.util.numpy_support as vnp

        img = vtk.vtkImageData()
        nz, ny, nx = volume.shape
        img.SetDimensions(nx, ny, nz)
        img.SetSpacing(spacing[2], spacing[1], spacing[0])   # dx, dy, dz
        img.SetOrigin(0.0, 0.0, 0.0)

        flat = volume.ravel(order="C").astype(np.float32)
        arr  = vnp.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        arr.SetName("HU")
        img.GetPointData().SetScalars(arr)
        return img
