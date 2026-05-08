"""Single-plane orthogonal slice display widget.

Uses PyQtGraph's ImageItem for GPU-accelerated rendering.

Controls
--------
  Mouse wheel            → scroll slices
  Right-click drag       → window / level
  Middle-click / scroll  → pan / zoom (PyQtGraph built-in)

Measurement tools (A-03-03 / A-03-05)
--------------------------------------
  Cal.  — distance caliper: click two points → distance in mm
  Ang.  — angle tool:       click three points (A, vertex, B) → angle in °
  ✕     — clear all measurements on this plane
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Force row-major axis order so array[row, col] maps directly to image[y, x]
pg.setConfigOptions(imageAxisOrder="row-major")

_PLANE_LABELS = {"axial": "AXIAL", "coronal": "CORONAL", "sagital": "SAGITAL"}

# Per-plane header colours — (dark_bg, dark_fg, light_bg, light_fg).
# The fg colours are medical-convention (blue/green/orange) and are designed
# to be legible on both the dark and light header backgrounds.
_HEADER_COLORS = {
    "axial":   ("#0d2137", "#4fc3f7", "#e3f4fd", "#1565C0"),
    "coronal": ("#0d2a1a", "#69f0ae", "#e0f6ea", "#1B6B30"),
    "sagital": ("#2a1a0d", "#ffcc80", "#fdf3e3", "#C05000"),
}


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


def _slice_theme(dark: bool) -> dict:
    """Return a dict of colour tokens for the navigation/button/footer areas."""
    if dark:
        return dict(
            btn_bg   = "#1F1F1F", btn_bdr  = "#363636",
            btn_clr  = "#9B9B9B", btn_en   = "#EBEBEB",
            btn_hov  = "#1C303F",
            nav_bg   = "#090e15",
            spin_bg  = "#1F1F1F", spin_clr = "#9B9B9B", spin_bdr = "#363636",
            ftr_bg   = "#090e15", ftr_clr  = "#5a6a7a",
            meas_clr_dist = "#e6e64a", meas_clr_ang = "#64dcff",
        )
    return dict(
        btn_bg   = "#F7F7F7", btn_bdr  = "#D8E0E8",
        btn_clr  = "#6B7A8A", btn_en   = "#0D1520",
        btn_hov  = "#DDE5EC",
        nav_bg   = "#E8EDF2",
        spin_bg  = "#FFFFFF", spin_clr = "#3A4A5A", spin_bdr = "#C8D4DC",
        ftr_bg   = "#E0E8F0", ftr_clr  = "#7A8A9A",
        meas_clr_dist = "#B07A00", meas_clr_ang = "#1565C0",
    )


# Measurement colour palette (canvas — always dark because the VTK canvas is always dark)
_COL_CALIPER = (255, 230, 50)    # yellow
_COL_ANGLE   = (100, 220, 255)   # cyan
_COL_POINT   = (255, 120, 50)    # orange — in-progress first point


class SliceWidget(QWidget):
    """Displays one orthogonal plane of a 3-D volume with window/level control."""

    #: Emitted whenever the displayed slice index changes (plane, index).
    position_changed = pyqtSignal(str, int)
    #: Emitted when cursor moves over the image — carries the raw HU value.
    hu_hovered = pyqtSignal(float)

    def __init__(self, plane: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if plane not in _PLANE_LABELS:
            raise ValueError(f"plane must be one of {list(_PLANE_LABELS)}, got {plane!r}")

        self.plane = plane
        self._volume:        np.ndarray | None         = None
        self._spacing:       tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._current_index: int                        = 0
        self._max_index:     int                        = 0
        self._wc:            float                      = 40.0
        self._ww:            float                      = 400.0

        # Right-drag state for interactive window/level
        self._drag_origin: tuple[int, int] | None = None
        self._drag_wc0:    float                  = 0.0
        self._drag_ww0:    float                  = 0.0

        # Measurement state
        # _meas_mode: "off" | "distance" | "angle"
        self._meas_mode: str                       = "off"
        self._meas_pts:  list[tuple[float, float]] = []   # in-progress click points (mm)
        self._meas_items: list                     = []   # placed pg items (line/text/scatter)
        self._pending_items: list                  = []   # items for the current in-progress meas

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def set_volume(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self._volume  = volume
        self._spacing = spacing
        self._max_index     = self._dim() - 1
        self._current_index = self._max_index // 2

        for w in (self._slider, self._spinbox):
            w.blockSignals(True)
        self._slider.setMaximum(self._max_index)
        self._slider.setValue(self._current_index)
        self._spinbox.setMaximum(self._max_index + 1)
        self._spinbox.setValue(self._current_index + 1)
        for w in (self._slider, self._spinbox):
            w.blockSignals(False)

        self._refresh()
        QTimer.singleShot(0, self._fit_view)

    def set_window(self, center: float, width: float) -> None:
        self._wc = center
        self._ww = max(1.0, width)
        self._refresh()

    def set_index(self, index: int) -> None:
        if self._volume is None:
            return
        self._current_index = int(np.clip(index, 0, self._max_index))
        for w in (self._slider, self._spinbox):
            w.blockSignals(True)
        self._slider.setValue(self._current_index)
        self._spinbox.setValue(self._current_index + 1)
        for w in (self._slider, self._spinbox):
            w.blockSignals(False)
        self._refresh()

    def set_crosshair(self, row: float, col: float) -> None:
        self._h_line.setValue(row)
        self._v_line.setValue(col)

    # ------------------------------------------------------------------ #
    # Build UI                                                             #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header row (label + measurement buttons) ───────────────────── #
        self._hdr_row = QWidget()
        hl = QHBoxLayout(self._hdr_row)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(4)

        self._header = QLabel(_PLANE_LABELS[self.plane])
        self._header.setStyleSheet("font-weight:bold; font-size:11px; letter-spacing:1px;")
        hl.addWidget(self._header, stretch=1)

        # Measurement mode buttons — styled by apply_theme()
        self._btn_cal = QPushButton("Cal.")
        self._btn_cal.setCheckable(True)
        self._btn_cal.setFixedHeight(22)
        self._btn_cal.setToolTip("Calibre de distancia: clic en dos puntos → distancia en mm")
        self._btn_cal.toggled.connect(lambda c: self._set_meas_mode("distance" if c else "off"))
        hl.addWidget(self._btn_cal)

        self._btn_ang = QPushButton("Ang.")
        self._btn_ang.setCheckable(True)
        self._btn_ang.setFixedHeight(22)
        self._btn_ang.setToolTip("Medición de ángulo: clic en tres puntos (A, vértice, B) → ángulo en °")
        self._btn_ang.toggled.connect(lambda c: self._set_meas_mode("angle" if c else "off"))
        hl.addWidget(self._btn_ang)

        self._btn_clr = QPushButton("✕")
        self._btn_clr.setFixedHeight(22)
        self._btn_clr.setToolTip("Borrar todas las mediciones de este plano")
        self._btn_clr.clicked.connect(self._clear_measurements)
        hl.addWidget(self._btn_clr)

        layout.addWidget(self._hdr_row)

        # ── PyQtGraph canvas ──────────────────────────────────────────── #
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#111111")
        self._glw.installEventFilter(self)

        self._vb = self._glw.addViewBox(row=0, col=0)
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)

        self._img = pg.ImageItem()
        self._vb.addItem(self._img)

        # Crosshair — use the dark fg colour from _HEADER_COLORS (canvas is always dark)
        _dark_bg, _xhair_fg, _light_bg, _light_fg = _HEADER_COLORS[self.plane]
        self._h_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen(_xhair_fg, width=1, style=Qt.DashLine)
        )
        self._v_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(_xhair_fg, width=1, style=Qt.DashLine)
        )
        self._vb.addItem(self._h_line)
        self._vb.addItem(self._v_line)

        # Connect scene click for measurement tools
        self._vb.scene().sigMouseClicked.connect(self._on_scene_click)
        # NOTE: sigMouseMoved (HU cursor tracking) intentionally NOT connected —
        # firing on every pixel of mouse movement caused visual glitches in all
        # four slice views.  HU readout is removed; footer shows only slice info.

        layout.addWidget(self._glw, stretch=1)

        # ── Slice navigation bar ───────────────────────────────────────── #
        self._nav_bar = QWidget()
        nav_layout = QHBoxLayout(self._nav_bar)
        nav_layout.setContentsMargins(4, 1, 4, 1)
        nav_layout.setSpacing(4)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.setTickPosition(QSlider.NoTicks)
        self._slider.valueChanged.connect(self._on_slider_changed)
        nav_layout.addWidget(self._slider, stretch=1)

        self._spinbox = QSpinBox()
        self._spinbox.setMinimum(1)
        self._spinbox.setMaximum(1)
        self._spinbox.setValue(1)
        self._spinbox.setFixedWidth(52)
        self._spinbox.valueChanged.connect(self._on_spinbox_changed)
        nav_layout.addWidget(self._spinbox)

        layout.addWidget(self._nav_bar)

        # ── Footer ────────────────────────────────────────────────────── #
        self._footer = QLabel("—")
        self._footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._footer)

        # Apply initial theme-aware styles.
        self.apply_theme()

    # ------------------------------------------------------------------ #
    # Measurement mode                                                      #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Theme                                                                #
    # ------------------------------------------------------------------ #

    def apply_theme(self) -> None:
        """Re-apply all inline stylesheets to match the current UI theme."""
        dark = _is_dark()
        tc   = _slice_theme(dark)
        hc   = _HEADER_COLORS[self.plane]
        bg   = hc[0] if dark else hc[2]
        fg   = hc[1] if dark else hc[3]

        # Header row
        self._hdr_row.setStyleSheet(f"background:{bg};")
        self._header.setStyleSheet(
            f"color:{fg}; font-weight:bold; font-size:11px; letter-spacing:1px;"
        )

        # Measurement buttons
        _btn_qss = (
            f"QPushButton{{background:{tc['btn_bg']};border:1px solid {tc['btn_bdr']};"
            f"border-radius:5px;color:{tc['btn_clr']};font-size:10px;"
            f"padding:2px 6px;min-height:0;}}"
            f"QPushButton:checked{{background:#1a2a40;border-color:#4fc3f7;"
            f"color:#4fc3f7;font-weight:bold;}}"
            f"QPushButton:hover{{background:{tc['btn_hov']};color:{tc['btn_en']};}}"
        )
        self._btn_cal.setStyleSheet(_btn_qss)
        self._btn_ang.setStyleSheet(_btn_qss)
        self._btn_clr.setStyleSheet(
            f"QPushButton{{background:{tc['btn_bg']};border:1px solid {tc['btn_bdr']};"
            f"border-radius:5px;color:#f85149;font-size:10px;padding:2px 6px;min-height:0;}}"
            f"QPushButton:hover{{background:{'#2d1117' if dark else '#fde8e8'};}}"
        )

        # Navigation bar + spinbox
        self._nav_bar.setStyleSheet(f"background:{tc['nav_bg']};")
        self._spinbox.setStyleSheet(
            f"background:{tc['spin_bg']};color:{tc['spin_clr']};"
            f"border:1px solid {tc['spin_bdr']};border-radius:5px;padding:1px;"
        )

        # Footer (idle state; active measurement text re-set by _set_meas_mode)
        if self._meas_mode == "distance":
            self._footer.setStyleSheet(
                f"background:{tc['ftr_bg']};color:{tc['meas_clr_dist']};"
                "font-size:10px;padding:2px;"
            )
        elif self._meas_mode == "angle":
            self._footer.setStyleSheet(
                f"background:{tc['ftr_bg']};color:{tc['meas_clr_ang']};"
                "font-size:10px;padding:2px;"
            )
        else:
            self._footer.setStyleSheet(
                f"background:{tc['ftr_bg']};color:{tc['ftr_clr']};"
                "font-size:10px;padding:2px;"
            )

    def _set_meas_mode(self, mode: str) -> None:
        """Switch measurement mode; cancel any in-progress measurement."""
        self._cancel_pending()
        self._meas_mode = mode

        # Keep buttons mutually exclusive
        self._btn_cal.blockSignals(True)
        self._btn_ang.blockSignals(True)
        self._btn_cal.setChecked(mode == "distance")
        self._btn_ang.setChecked(mode == "angle")
        self._btn_cal.blockSignals(False)
        self._btn_ang.blockSignals(False)

        tc = _slice_theme(_is_dark())
        if mode == "distance":
            self._footer.setText("Calibre: haga clic en el primer punto")
            self._footer.setStyleSheet(
                f"background:{tc['ftr_bg']};color:{tc['meas_clr_dist']};"
                "font-size:10px;padding:2px;"
            )
        elif mode == "angle":
            self._footer.setText("Ángulo: haga clic en el punto A")
            self._footer.setStyleSheet(
                f"background:{tc['ftr_bg']};color:{tc['meas_clr_ang']};"
                "font-size:10px;padding:2px;"
            )
        else:
            self._footer.setStyleSheet(
                f"background:{tc['ftr_bg']};color:{tc['ftr_clr']};"
                "font-size:10px;padding:2px;"
            )
            self._refresh_footer()

    def _on_scene_click(self, event) -> None:
        """Handle scene mouse-click events for measurement tools."""
        if self._meas_mode == "off" or self._volume is None:
            return
        if event.button() != Qt.LeftButton:
            # Right-click → cancel current measurement
            self._cancel_pending()
            return

        # Map scene coordinates → ViewBox mm coordinates
        vb_pos = self._vb.mapSceneToView(event.scenePos())
        x_mm, y_mm = vb_pos.x(), vb_pos.y()

        # Bounds check
        w_mm, h_mm = self._physical_size()
        if not (0 <= x_mm <= w_mm and 0 <= y_mm <= h_mm):
            return

        self._meas_pts.append((x_mm, y_mm))

        if self._meas_mode == "distance":
            self._handle_distance_click()
        elif self._meas_mode == "angle":
            self._handle_angle_click()

        event.accept()

    # ── Distance caliper ──────────────────────────────────────────────── #

    def _handle_distance_click(self) -> None:
        n = len(self._meas_pts)

        if n == 1:
            # First point: draw a small dot
            dot = self._make_dot(*self._meas_pts[0], color=_COL_POINT)
            self._pending_items.append(dot)
            self._footer.setText("Calibre: haga clic en el segundo punto  (clic derecho = cancelar)")

        elif n == 2:
            # Second point: finish measurement
            p1, p2 = self._meas_pts
            dist_mm = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

            # Clear pending dot
            self._cancel_pending(clear_pts=False)

            # Permanent items
            line = self._make_line(p1, p2, color=_COL_CALIPER, width=1.5)
            d1   = self._make_dot(*p1, color=_COL_CALIPER)
            d2   = self._make_dot(*p2, color=_COL_CALIPER)
            lbl  = self._make_label(
                f"{dist_mm:.2f} mm",
                ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2),
                color=_COL_CALIPER,
            )
            for item in (line, d1, d2, lbl):
                self._meas_items.append(item)

            self._meas_pts.clear()
            self._footer.setText(
                f"Calibre: {dist_mm:.2f} mm  —  haga clic para otra medición"
            )

    # ── Angle measurement ─────────────────────────────────────────────── #

    def _handle_angle_click(self) -> None:
        n = len(self._meas_pts)

        if n == 1:
            dot = self._make_dot(*self._meas_pts[0], color=_COL_POINT)
            self._pending_items.append(dot)
            self._footer.setText("Ángulo: haga clic en el vértice  (clic derecho = cancelar)")

        elif n == 2:
            # Draw preliminary line A→vertex
            self._cancel_pending(clear_pts=False)
            line_av = self._make_line(self._meas_pts[0], self._meas_pts[1],
                                      color=_COL_POINT, width=1.2, dashed=True)
            d1 = self._make_dot(*self._meas_pts[0], color=_COL_POINT)
            d2 = self._make_dot(*self._meas_pts[1], color=_COL_POINT)
            for item in (line_av, d1, d2):
                self._pending_items.append(item)
            self._footer.setText("Ángulo: haga clic en el punto B  (clic derecho = cancelar)")

        elif n == 3:
            pa, pv, pb = self._meas_pts
            angle_deg  = self._compute_angle(pa, pv, pb)

            self._cancel_pending(clear_pts=False)

            # Permanent items: two line segments + dots + label
            line_av = self._make_line(pa, pv, color=_COL_ANGLE, width=1.5)
            line_vb = self._make_line(pv, pb, color=_COL_ANGLE, width=1.5)
            for pt in (pa, pv, pb):
                self._meas_items.append(self._make_dot(*pt, color=_COL_ANGLE))
            self._meas_items.extend([line_av, line_vb])

            # Label near vertex, offset slightly
            lbl = self._make_label(
                f"{angle_deg:.1f}°",
                pv,
                color=_COL_ANGLE,
                offset=(5, -5),
            )
            self._meas_items.append(lbl)

            self._meas_pts.clear()
            self._footer.setText(
                f"Ángulo: {angle_deg:.1f}°  —  haga clic para otra medición"
            )

    # ── Helpers ───────────────────────────────────────────────────────── #

    def _cancel_pending(self, clear_pts: bool = True) -> None:
        """Remove in-progress measurement items."""
        for item in self._pending_items:
            self._vb.removeItem(item)
        self._pending_items.clear()
        if clear_pts:
            self._meas_pts.clear()

    def _clear_measurements(self) -> None:
        """Remove all placed measurements + cancel any in-progress one."""
        self._cancel_pending()
        for item in self._meas_items:
            self._vb.removeItem(item)
        self._meas_items.clear()
        self._set_meas_mode("off")

    def _make_line(
        self,
        p1: tuple,
        p2: tuple,
        color: tuple,
        width: float = 1.5,
        dashed: bool = False,
    ) -> pg.PlotDataItem:
        style = Qt.DashLine if dashed else Qt.SolidLine
        pen   = pg.mkPen(color, width=width, style=style)
        item  = pg.PlotDataItem([p1[0], p2[0]], [p1[1], p2[1]], pen=pen)
        self._vb.addItem(item)
        return item

    def _make_dot(self, x: float, y: float, color: tuple) -> pg.ScatterPlotItem:
        item = pg.ScatterPlotItem(
            [x], [y], size=7, pen=None,
            brush=pg.mkBrush(*color, 220),
        )
        self._vb.addItem(item)
        return item

    def _make_label(
        self,
        text: str,
        pos: tuple,
        color: tuple,
        offset: tuple = (4, -4),
    ) -> pg.TextItem:
        item = pg.TextItem(
            text,
            color=color,
            anchor=(0.0, 1.0),
        )
        item.setPos(pos[0] + offset[0] * 0.5, pos[1] + offset[1] * 0.5)
        # Slightly larger font for readability
        font = item.textItem.font()
        font.setPointSize(8)
        item.textItem.setFont(font)
        self._vb.addItem(item)
        return item

    @staticmethod
    def _compute_angle(pa: tuple, pv: tuple, pb: tuple) -> float:
        """Angle at vertex *pv* between rays pv→pa and pv→pb (degrees)."""
        ax, ay = pa[0] - pv[0], pa[1] - pv[1]
        bx, by = pb[0] - pv[0], pb[1] - pv[1]
        dot  = ax*bx + ay*by
        mag_a = math.sqrt(ax*ax + ay*ay)
        mag_b = math.sqrt(bx*bx + by*by)
        if mag_a < 1e-9 or mag_b < 1e-9:
            return 0.0
        cos_a = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        return math.degrees(math.acos(cos_a))

    # ------------------------------------------------------------------ #
    # Event handling                                                        #
    # ------------------------------------------------------------------ #

    # _on_mouse_moved intentionally removed — sigMouseMoved is no longer connected.
    # Per-pixel HU readout on mouse hover caused severe visual glitches in all
    # slice views (pyqtgraph fired on every mouse-move, triggering full repaints).
    # The hu_hovered signal is kept defined for API compatibility but never emitted.

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self._glw and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            step  = 1 if delta > 0 else -1
            self.set_index(self._current_index + step)
            self.position_changed.emit(self.plane, self._current_index)
            return True
        return False

    def leaveEvent(self, event) -> None:
        """Restore default footer when cursor leaves the widget."""
        self._refresh_footer()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._drag_origin = (event.x(), event.y())
            self._drag_wc0    = self._wc
            self._drag_ww0    = self._ww
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin and event.buttons() & Qt.RightButton:
            dx     = event.x() - self._drag_origin[0]
            dy     = event.y() - self._drag_origin[1]
            new_ww = max(1.0, self._drag_ww0 + dx * 4.0)
            new_wc = self._drag_wc0 - dy * 2.0
            self.set_window(new_wc, new_ww)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._drag_origin = None
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._volume is not None:
            self._fit_view()

    def _on_slider_changed(self, value: int) -> None:
        self.set_index(value)
        self.position_changed.emit(self.plane, self._current_index)

    def _on_spinbox_changed(self, value: int) -> None:
        self.set_index(value - 1)
        self.position_changed.emit(self.plane, self._current_index)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _fit_view(self) -> None:
        if self._volume is None:
            return
        w_mm, h_mm = self._physical_size()
        self._vb.setRange(rect=QRectF(0.0, 0.0, w_mm, h_mm), padding=0.02)

    def _dim(self) -> int:
        z, y, x = self._volume.shape
        return {"axial": z, "coronal": y, "sagital": x}[self.plane]

    def _get_slice(self) -> np.ndarray:
        i = self._current_index
        if self.plane == "axial":
            return self._volume[i, :, :]
        elif self.plane == "coronal":
            return self._volume[:, i, :]
        else:
            return self._volume[:, :, i]

    def _physical_size(self) -> tuple[float, float]:
        sz, sy, sx = self._spacing
        z, y, x    = self._volume.shape
        if self.plane == "axial":
            return x * sx, y * sy
        elif self.plane == "coronal":
            return x * sx, z * sz
        else:
            return y * sy, z * sz

    def _apply_window(self, slc: np.ndarray) -> np.ndarray:
        lo = self._wc - self._ww / 2.0
        hi = self._wc + self._ww / 2.0
        return ((np.clip(slc, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)

    def _refresh(self) -> None:
        if self._volume is None:
            return
        windowed = self._apply_window(self._get_slice())
        self._img.setImage(windowed, autoLevels=False, levels=(0, 255))
        w_mm, h_mm = self._physical_size()
        self._img.setRect(QRectF(0.0, 0.0, w_mm, h_mm))
        if self._meas_mode == "off":
            self._refresh_footer()

    def _refresh_footer(self) -> None:
        self._footer.setText(
            f"Slice {self._current_index + 1} / {self._max_index + 1}  |  "
            f"WC {self._wc:.0f}  WW {self._ww:.0f}"
        )
        tc = _slice_theme(_is_dark())
        self._footer.setStyleSheet(
            f"background:{tc['ftr_bg']};color:{tc['ftr_clr']};font-size:10px;padding:2px;"
        )
