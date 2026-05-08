"""TriplanarPanel — three orthogonal MPR planes shown side-by-side.

Layout (left → right):
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  AXIAL   │  │ CORONAL  │  │ SAGITAL  │
    │  (Z slider)│  (Y slider)│  (X slider)│
    └──────────┘  └──────────┘  └──────────┘

Each plane reuses :class:`SliceWidget` which already includes a horizontal
slider and a spinbox for slice navigation.

Crosshair sync
--------------
When the user scrolls / drags a slider in any plane the crosshairs in the
other two planes update automatically to reflect the new intersection point
in physical (mm) coordinates:

  plane      rows         cols
  ─────────  ───────────  ───────────
  axial      Y position   X position
  coronal    Z position   X position
  sagital    Z position   Y position

Public API
----------
set_volume(volume, spacing)
    Load a 3-D numpy array and its voxel spacing (sz, sy, sx) into all three planes.

set_window(center, width)
    Apply window / level to all three planes.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from prospective.ui.viewers.slice_widget import SliceWidget


class TriplanarPanel(QWidget):
    """
    Side-by-side display of the three orthogonal MPR planes.

    Slice navigation (slider + spinbox) is handled internally by each
    :class:`SliceWidget`.  This widget additionally keeps the crosshairs of
    all three planes in sync whenever any slice position changes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
        # Current voxel indices along each axis (updated by position_changed signals)
        self._z: int = 0   # axial index
        self._y: int = 0   # coronal index
        self._x: int = 0   # sagital index
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_volume(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Load *volume* (shape Z×Y×X, HU values) into all three planes."""
        self._spacing = spacing
        z, y, x = volume.shape
        self._z = z // 2
        self._y = y // 2
        self._x = x // 2
        for w in (self._axial, self._coronal, self._sagital):
            w.set_volume(volume, spacing)
        # After set_volume the slice widgets centre their sliders at mid-index;
        # sync the crosshairs to match those initial positions.
        self._sync_crosshairs()

    def set_window(self, center: float, width: float) -> None:
        """Apply window / level to all three planes."""
        for w in (self._axial, self._coronal, self._sagital):
            w.set_window(center, width)

    # ------------------------------------------------------------------ #
    # Build UI                                                             #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── Header ────────────────────────────────────────────────────── #
        hdr = QLabel("Vista triplanar  —  Axial (Z) · Coronal (Y) · Sagital (X)")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet(
            "color:#8B9BAA; font-size:11px; font-weight:bold;"
            "letter-spacing:1px; padding:4px 0;"
        )
        outer.addWidget(hdr)

        # ── Three slice panels ─────────────────────────────────────────── #
        row = QWidget()
        hl  = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)

        self._axial   = SliceWidget("axial")
        self._coronal = SliceWidget("coronal")
        self._sagital = SliceWidget("sagital")

        for w in (self._axial, self._coronal, self._sagital):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            hl.addWidget(w)

        outer.addWidget(row, stretch=1)

        # ── Crosshair sync ─────────────────────────────────────────────── #
        self._axial.position_changed.connect(self._on_axial_changed)
        self._coronal.position_changed.connect(self._on_coronal_changed)
        self._sagital.position_changed.connect(self._on_sagital_changed)

    # ------------------------------------------------------------------ #
    # Crosshair sync                                                       #
    # ------------------------------------------------------------------ #

    def _on_axial_changed(self, _plane: str, idx: int) -> None:
        """Axial slider moved → new Z position; update coronal & sagital crosshairs."""
        self._z = idx
        self._sync_crosshairs(skip="axial")

    def _on_coronal_changed(self, _plane: str, idx: int) -> None:
        """Coronal slider moved → new Y position; update axial & sagital crosshairs."""
        self._y = idx
        self._sync_crosshairs(skip="coronal")

    def _on_sagital_changed(self, _plane: str, idx: int) -> None:
        """Sagital slider moved → new X position; update axial & coronal crosshairs."""
        self._x = idx
        self._sync_crosshairs(skip="sagital")

    def _sync_crosshairs(self, skip: str = "") -> None:
        """
        Update all crosshairs (except the plane named *skip*) to reflect
        the current Z/Y/X voxel indices.

        Physical coordinate mapping (mm):
          axial   → row = Y_mm, col = X_mm
          coronal → row = Z_mm, col = X_mm
          sagital → row = Z_mm, col = Y_mm
        """
        sz, sy, sx = self._spacing
        z_mm = self._z * sz
        y_mm = self._y * sy
        x_mm = self._x * sx

        if skip != "axial":
            self._axial.set_crosshair(y_mm, x_mm)
        if skip != "coronal":
            self._coronal.set_crosshair(z_mm, x_mm)
        if skip != "sagital":
            self._sagital.set_crosshair(z_mm, y_mm)
