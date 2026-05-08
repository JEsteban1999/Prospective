"""Multi-Planar Reconstruction viewer — 2×2 grid layout.

Layout:
  ┌──────────┬──────────┐
  │  AXIAL   │ SAGITAL  │
  ├──────────┼──────────┤
  │ CORONAL  │   3-D    │
  │          │  (VTK)   │
  └──────────┴──────────┘
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QHBoxLayout, QPushButton, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from prospective.dicom.series import DicomSeries
from prospective.ui.viewers.slice_widget import SliceWidget
from prospective.ui.viewers.vtk_volume_widget import VTKVolumeWidget


def _is_dark() -> bool:
    """Return True when the application is using a dark-background theme."""
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


class MPRViewer(QWidget):
    """
    Four-pane viewer: three orthogonal 2-D slice planes + interactive 3-D volume.
    """

    #: Relay of hu_hovered from any slice pane — carries the raw HU value.
    hu_hovered = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: DicomSeries | None = None
        self._oblique_dlg = None   # ObliqueMPRDialog instance (lazy)
        self._last_hu: float = 0.0
        self._sizes_equalized = False   # guard: equalize only once on first show
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_series(self, series: DicomSeries) -> None:
        """Load a DicomSeries into all four panes."""
        self._series = series

        # 2-D planes — pass spacing so each plane renders with correct physical aspect ratio
        for widget in self._slice_widgets:
            widget.set_volume(series.volume, series.spacing)
            widget.set_window(series.metadata.window_center, series.metadata.window_width)

        # 3-D volume — pass modality so the widget can validate bone-subtraction eligibility
        self._vtk3d.set_volume(series.volume, series.spacing, series.metadata.modality)

        # Enable oblique MPR button now that data is loaded
        self._btn_oblique.setEnabled(True)

    def set_window(self, center: float, width: float) -> None:
        """Apply window / level to all 2-D planes."""
        for widget in self._slice_widgets:
            widget.set_window(center, width)

    def set_3d_preset(self, name: str) -> None:
        self._vtk3d.set_preset(name)

    def set_3d_blend_mode(self, name: str) -> None:
        self._vtk3d.set_blend_mode(name)

    def set_mesh(self, poly_data) -> None:
        self._vtk3d.set_mesh(poly_data)

    def set_mesh_visible(self, visible: bool) -> None:
        self._vtk3d.set_mesh_visible(visible)

    def set_volume_visible(self, visible: bool) -> None:
        self._vtk3d.set_volume_visible(visible)

    def set_aneurysm(self, poly_data) -> None:
        self._vtk3d.set_aneurysm(poly_data)

    def set_aneurysm_visible(self, visible: bool) -> None:
        self._vtk3d.set_aneurysm_visible(visible)

    def set_trajectory(self, entry, target) -> None:
        self._vtk3d.set_trajectory(entry, target)

    def add_clip(self, index: int, spec, transform, custom_poly_data=None) -> None:
        self._vtk3d.add_clip(index, spec, transform, custom_poly_data)

    def remove_clip(self, index: int) -> None:
        self._vtk3d.remove_clip(index)

    def set_clip_visible(self, index: int, visible: bool) -> None:
        self._vtk3d.set_clip_visible(index, visible)

    def update_clip_transform(self, index: int, transform) -> None:
        self._vtk3d.update_clip_transform(index, transform)

    # ------------------------------------------------------------------ #
    # Build UI                                                             #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)

        # ── Toolbar row ───────────────────────────────────────────────── #
        toolbar = QWidget()
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(4, 2, 4, 2)
        tl.setSpacing(8)

        from prospective.ui.icons import I as _I
        self._btn_oblique = QPushButton(f"{_I.OBLIQUE} Abrir MPR Oblicuo")
        self._btn_oblique.setEnabled(False)
        self._btn_oblique.setToolTip(
            "Abre un visor de corte oblicuo — permite rotar el plano de corte "
            "en cualquier dirección con controles deslizantes (3D Slicer-inspired)."
        )
        self._btn_oblique.clicked.connect(self._open_oblique_viewer)
        tl.addWidget(self._btn_oblique)
        tl.addStretch()
        outer.addWidget(toolbar)

        # ── 2×2 viewer grid — nested QSplitters guarantee equal quadrants ── #
        self._axial   = SliceWidget("axial")
        self._sagital = SliceWidget("sagital")
        self._coronal = SliceWidget("coronal")
        self._vtk3d   = VTKVolumeWidget()

        # Override size policies so Qt ignores intrinsic sizeHints (VTK's is large)
        # and allocates space purely from stretch factors.
        _exp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        for viewer in (self._axial, self._sagital, self._coronal, self._vtk3d):
            viewer.setSizePolicy(_exp)
            viewer.setMinimumSize(10, 10)

        # Left column: AXIAL (top) / CORONAL (bottom)
        self._left_splitter = QSplitter(Qt.Vertical)
        self._left_splitter.setChildrenCollapsible(False)
        self._left_splitter.addWidget(self._axial)
        self._left_splitter.addWidget(self._coronal)
        self._left_splitter.setStretchFactor(0, 1)
        self._left_splitter.setStretchFactor(1, 1)
        self._left_splitter.setHandleWidth(3)
        self._left_splitter.setSizePolicy(_exp)

        # Right column: SAGITAL (top) / 3-D VTK (bottom)
        self._right_splitter = QSplitter(Qt.Vertical)
        self._right_splitter.setChildrenCollapsible(False)
        self._right_splitter.addWidget(self._sagital)
        self._right_splitter.addWidget(self._vtk3d)
        self._right_splitter.setStretchFactor(0, 1)
        self._right_splitter.setStretchFactor(1, 1)
        self._right_splitter.setHandleWidth(3)
        self._right_splitter.setSizePolicy(_exp)

        # Horizontal splitter: left column | right column
        self._h_splitter = QSplitter(Qt.Horizontal)
        self._h_splitter.setChildrenCollapsible(False)
        self._h_splitter.addWidget(self._left_splitter)
        self._h_splitter.addWidget(self._right_splitter)
        self._h_splitter.setStretchFactor(0, 1)
        self._h_splitter.setStretchFactor(1, 1)
        self._h_splitter.setHandleWidth(3)

        outer.addWidget(self._h_splitter, stretch=1)

        self._slice_widgets = [self._axial, self._sagital, self._coronal]

        for w in self._slice_widgets:
            w.position_changed.connect(self._on_position_changed)
            w.hu_hovered.connect(self._on_hu_hovered)

        self._apply_styles()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        # Schedule equalization after the first event loop iteration so that the
        # parent layout has had time to assign a non-zero geometry to this widget.
        if not self._sizes_equalized:
            self._sizes_equalized = True
            QTimer.singleShot(0, self._equalize_splitters)

    def _equalize_splitters(self) -> None:
        """Force all four quadrants to equal size after geometry is available."""
        w = self._h_splitter.width()
        h = self._h_splitter.height()
        if w >= 4:
            hw = w // 2
            self._h_splitter.setSizes([hw, hw])
        if h >= 4:
            hh = h // 2
            self._left_splitter.setSizes([hh, hh])
            self._right_splitter.setSizes([hh, hh])

    # ------------------------------------------------------------------ #
    # Theme                                                                #
    # ------------------------------------------------------------------ #

    def _apply_styles(self) -> None:
        """Apply theme-aware stylesheet to toolbar widgets."""
        dark = _is_dark()
        if dark:
            btn_bg     = "#1F1F1F"
            btn_border = "#363636"
            btn_clr    = "#9B9B9B"
            btn_en_clr = "#EBEBEB"
            hover_bg   = "#1C303F"
            hover_bdr  = "#A8B8C6"
            hover_clr  = "#A8B8C6"
        else:
            btn_bg     = "#F7F7F7"
            btn_border = "#E5E5E5"
            btn_clr    = "#6B6B6B"
            btn_en_clr = "#0D0D0D"
            hover_bg   = "#DDE5EC"
            hover_bdr  = "#8B9BAA"
            hover_clr  = "#8B9BAA"

        self._btn_oblique.setStyleSheet(
            f"QPushButton{{background:{btn_bg};border:1px solid {btn_border};"
            f"border-radius:8px;color:{btn_clr};padding:3px 10px;}}"
            f"QPushButton:hover{{background:{hover_bg};border-color:{hover_bdr};"
            f"color:{hover_clr};}}"
            f"QPushButton:enabled{{color:{btn_en_clr};}}"
        )

    def apply_theme(self) -> None:
        """Called by the main window when the user toggles the colour theme."""
        self._apply_styles()
        self._vtk3d.apply_theme()
        # Cascade to each slice pane (header, nav bar, buttons, footer).
        for sw in (self._axial, self._coronal, self._sagital):
            sw.apply_theme()

    # ------------------------------------------------------------------ #
    # Position tracking                                                    #
    # ------------------------------------------------------------------ #

    def get_last_hu(self) -> float:
        """Return the most recent HU value seen by any of the three slice panes."""
        return self._last_hu

    def _on_hu_hovered(self, hu: float) -> None:
        self._last_hu = hu
        self.hu_hovered.emit(hu)

    def get_current_voxel(self) -> tuple[int, int, int]:
        """
        Return the current (z, y, x) voxel from the three slice indices.

        axial._current_index   = z
        coronal._current_index = y
        sagital._current_index = x
        """
        return (
            self._axial._current_index,
            self._coronal._current_index,
            self._sagital._current_index,
        )

    def _on_position_changed(self, plane: str, index: int) -> None:
        # Crosshair sync across planes — A-02 follow-up
        pass

    def _open_oblique_viewer(self) -> None:
        """Open (or raise) the oblique MPR dialog for the loaded series."""
        if self._series is None:
            return
        from prospective.ui.viewers.oblique_viewer import ObliqueMPRDialog
        if self._oblique_dlg is None or not self._oblique_dlg.isVisible():
            wc = self._series.metadata.window_center
            ww = self._series.metadata.window_width
            self._oblique_dlg = ObliqueMPRDialog(
                self._series.volume, self._series.spacing, wc, ww, parent=self
            )
        self._oblique_dlg.show()
        self._oblique_dlg.raise_()
