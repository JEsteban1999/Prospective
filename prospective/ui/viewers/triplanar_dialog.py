"""TriplanarDialog — ventana flotante no-modal con los tres planos MPR.

Layout (3 columnas redimensionables con QSplitter):

    ┌──────────────┬──────────────┬──────────────┐
    │    AXIAL     │   SAGITAL    │   CORONAL    │
    │   (imagen)   │   (imagen)   │   (imagen)   │
    │  ──────────  │  ──────────  │  ──────────  │
    │  slider Z    │  slider X    │  slider Y    │
    │  Z = 25.1 mm │  X = 38.0 mm│  Y = 14.2 mm │
    └──────────────┴──────────────┴──────────────┘

Diseño intencional
------------------
* Sin botones de medición ni spinbox — solo imagen, slider y posición en mm.
* Crosshair sync: mover cualquier slider actualiza los crosshairs de los
  otros dos planos en coordenadas físicas (mm).
* El scroll de rueda en cualquier imagen también mueve el slice y sincroniza.
* La ventana se oculta (hide) al cerrar para preservar la posición de los
  sliders cuando se vuelva a abrir.

API pública
-----------
set_volume(volume, spacing)
    Carga el volumen 3-D en los tres planos y centra los sliders.

set_window(center, width)
    Aplica ventana/nivel a los tres planos.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

pg.setConfigOptions(imageAxisOrder="row-major")


# ── Theme helper ─────────────────────────────────────────────────────────── #

def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


# ── Palette por plano (dark_bg, dark_fg, light_bg, light_fg) ─────────────── #
_PLANE_CFG: dict[str, dict] = {
    "axial":   {
        "label":    "AXIAL",
        "dark_bg":  "#0d2137", "dark_fg":  "#4fc3f7",
        "light_bg": "#e3f4fd", "light_fg": "#1565C0",
    },
    "sagital": {
        "label":    "SAGITAL",
        "dark_bg":  "#2a1a0d", "dark_fg":  "#ffcc80",
        "light_bg": "#fdf3e3", "light_fg": "#C05000",
    },
    "coronal": {
        "label":    "CORONAL",
        "dark_bg":  "#0d2a1a", "dark_fg":  "#69f0ae",
        "light_bg": "#e0f6ea", "light_fg": "#1B6B30",
    },
}
_AXIS_LABEL = {"axial": "Z", "sagital": "X", "coronal": "Y"}


def _plane_colors(plane: str) -> tuple[str, str, str]:
    """Return (bg, fg, canvas_bg) for *plane* based on the current theme."""
    cfg  = _PLANE_CFG[plane]
    dark = _is_dark()
    bg   = cfg["dark_bg"]  if dark else cfg["light_bg"]
    fg   = cfg["dark_fg"]  if dark else cfg["light_fg"]
    cvs  = "#090e15"       if dark else "#FFFFFF"
    return bg, fg, cvs


# ─────────────────────────────────────────────────────────────────────── #
# _PlaneView                                                               #
# ─────────────────────────────────────────────────────────────────────── #

class _PlaneView(QWidget):
    """
    Vista mínima de un único plano ortogonal.

    Contenido (de arriba a abajo):
      1. Header coloreado con el nombre del plano.
      2. Canvas pyqtgraph con la imagen + crosshairs.
      3. Footer: slider + etiqueta de posición en mm.
    """

    #: Emitida cuando el slice cambia, ya sea por slider o por scroll.
    position_changed = pyqtSignal(str, int)   # (plane, index)

    def __init__(self, plane: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        cfg = _PLANE_CFG[plane]
        self.plane        = plane
        self._cfg         = cfg
        self._volume:        np.ndarray | None          = None
        self._spacing:       tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._current_index: int                        = 0
        self._max_index:     int                        = 0
        self._wc:            float                      = 40.0
        self._ww:            float                      = 400.0
        self._build_ui()

    # ── Public API ──────────────────────────────────────────────── #

    def set_volume(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        self._volume        = volume
        self._spacing       = spacing
        self._max_index     = self._dim() - 1
        self._current_index = self._max_index // 2

        self._slider.blockSignals(True)
        self._slider.setMaximum(self._max_index)
        self._slider.setValue(self._current_index)
        self._slider.blockSignals(False)

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
        self._slider.blockSignals(True)
        self._slider.setValue(self._current_index)
        self._slider.blockSignals(False)
        self._refresh()

    def set_crosshair(self, row: float, col: float) -> None:
        self._h_line.setValue(row)
        self._v_line.setValue(col)

    # ── Build UI ────────────────────────────────────────────────── #

    def _build_ui(self) -> None:
        bg, fg, cvs = _plane_colors(self.plane)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 1 ── Header ───────────────────────────────────────────── #
        hdr = QLabel(self._cfg["label"])
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setFixedHeight(28)
        hdr.setStyleSheet(
            f"background:{bg}; color:{fg};"
            "font-weight:bold; font-size:11px; letter-spacing:2px;"
        )
        self._hdr = hdr
        layout.addWidget(hdr)

        # 2 ── Canvas ───────────────────────────────────────────── #
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(cvs)
        self._glw.installEventFilter(self)

        self._vb = self._glw.addViewBox(row=0, col=0)
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)

        self._img = pg.ImageItem()
        self._vb.addItem(self._img)

        # Crosshair — mismos colores que el header del plano
        pen = pg.mkPen(fg, width=1, style=Qt.DashLine)
        self._h_line = pg.InfiniteLine(angle=0,  pen=pen)
        self._v_line = pg.InfiniteLine(angle=90, pen=pen)
        self._vb.addItem(self._h_line)
        self._vb.addItem(self._v_line)

        layout.addWidget(self._glw, stretch=1)

        # 3 ── Footer: slider + posición mm ─────────────────────── #
        footer = QWidget()
        footer.setStyleSheet(f"background:{bg};")
        self._footer = footer
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 5, 10, 5)
        fl.setSpacing(10)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.setStyleSheet(
            f"QSlider::groove:horizontal{{"
            f"  height:3px; background:rgba(255,255,255,30); border-radius:2px;}}"
            f"QSlider::handle:horizontal{{"
            f"  width:14px; height:14px; margin:-6px 0;"
            f"  background:{fg}; border-radius:7px;}}"
            f"QSlider::sub-page:horizontal{{"
            f"  background:{fg}; border-radius:2px; opacity:0.7;}}"
        )
        self._slider.valueChanged.connect(self._on_slider_changed)
        fl.addWidget(self._slider, stretch=1)

        self._pos_lbl = QLabel("—")
        self._pos_lbl.setFixedWidth(80)
        self._pos_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._pos_lbl.setStyleSheet(
            f"color:{fg}; font-size:10px; font-family:monospace;"
        )
        fl.addWidget(self._pos_lbl)

        layout.addWidget(footer)

    # ── Helpers internos ────────────────────────────────────────── #

    def _dim(self) -> int:
        z, y, x = self._volume.shape
        return {"axial": z, "coronal": y, "sagital": x}[self.plane]

    def _get_slice(self) -> np.ndarray:
        i = self._current_index
        if self.plane == "axial":
            return self._volume[i, :, :]
        elif self.plane == "coronal":
            return self._volume[:, i, :]
        else:                               # sagital
            return self._volume[:, :, i]

    def _physical_size(self) -> tuple[float, float]:
        """Devuelve (ancho_mm, alto_mm) de la imagen mostrada."""
        z, y, x = self._volume.shape
        sz, sy, sx = self._spacing
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
        slc = self._apply_window(self._get_slice())
        self._img.setImage(slc, autoLevels=False, levels=(0, 255))
        w_mm, h_mm = self._physical_size()
        self._img.setRect(QRectF(0.0, 0.0, w_mm, h_mm))
        self._update_label()

    def _fit_view(self) -> None:
        if self._volume is None:
            return
        w_mm, h_mm = self._physical_size()
        self._vb.setRange(rect=QRectF(0.0, 0.0, w_mm, h_mm), padding=0.02)

    def _update_label(self) -> None:
        sz, sy, sx = self._spacing
        pos_mm = {
            "axial":   self._current_index * sz,
            "sagital": self._current_index * sx,
            "coronal": self._current_index * sy,
        }[self.plane]
        axis = _AXIS_LABEL[self.plane]
        total = self._max_index + 1
        self._pos_lbl.setText(
            f"{axis} {self._current_index + 1}/{total}\n{pos_mm:.1f} mm"
        )

    # ── Theme ───────────────────────────────────────────────────── #

    def apply_theme(self) -> None:
        """Re-style header, canvas and footer to match the current theme."""
        bg, fg, cvs = _plane_colors(self.plane)
        self._hdr.setStyleSheet(
            f"background:{bg}; color:{fg};"
            "font-weight:bold; font-size:11px; letter-spacing:2px;"
        )
        self._glw.setBackground(cvs)
        self._footer.setStyleSheet(f"background:{bg};")
        self._slider.setStyleSheet(
            f"QSlider::groove:horizontal{{"
            f"  height:3px; background:rgba(255,255,255,30); border-radius:2px;}}"
            f"QSlider::handle:horizontal{{"
            f"  width:14px; height:14px; margin:-6px 0;"
            f"  background:{fg}; border-radius:7px;}}"
            f"QSlider::sub-page:horizontal{{"
            f"  background:{fg}; border-radius:2px; opacity:0.7;}}"
        )
        self._pos_lbl.setStyleSheet(
            f"color:{fg}; font-size:10px; font-family:monospace;"
        )
        # Update crosshair pens
        pen = pg.mkPen(fg, width=1, style=Qt.DashLine)
        self._h_line.setPen(pen)
        self._v_line.setPen(pen)

    # ── Eventos ─────────────────────────────────────────────────── #

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Scroll de rueda → avanzar/retroceder slice."""
        if obj is self._glw and event.type() == QEvent.Wheel:
            step = 1 if event.angleDelta().y() > 0 else -1
            new_idx = int(np.clip(
                self._current_index + step, 0, self._max_index
            ))
            self.set_index(new_idx)
            self.position_changed.emit(self.plane, self._current_index)
            return True
        return False

    def resizeEvent(self, event) -> None:   # noqa: N802
        super().resizeEvent(event)
        if self._volume is not None:
            self._fit_view()

    def _on_slider_changed(self, value: int) -> None:
        self.set_index(value)
        self.position_changed.emit(self.plane, self._current_index)


# ─────────────────────────────────────────────────────────────────────── #
# TriplanarDialog                                                          #
# ─────────────────────────────────────────────────────────────────────── #

class TriplanarDialog(QDialog):
    """
    Ventana flotante no-modal con los tres planos ortogonales MPR.

    Se abre con show() / raise_() y se oculta (no se destruye) al pulsar ✕,
    para preservar las posiciones de los sliders entre usos.

    Uso
    ---
        dlg = TriplanarDialog(parent=main_window)
        dlg.set_volume(series.volume, series.spacing)
        dlg.set_window(wc, ww)
        dlg.show()
    """

    #: Emitida cuando el usuario cierra la ventana, para que el botón
    #: de la toolbar actualice su estado checked.
    hidden = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vista Triplanar  —  Axial · Sagital · Coronal")
        self.setMinimumSize(860, 380)
        self.resize(1180, 500)
        self.setModal(False)

        self._spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._z: int = 0
        self._y: int = 0
        self._x: int = 0

        self._build_ui()

    # ── Public API ──────────────────────────────────────────────── #

    def set_volume(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        """Carga el volumen en los tres planos y centra los sliders."""
        self._spacing = spacing
        z, y, x = volume.shape
        self._z = z // 2
        self._y = y // 2
        self._x = x // 2
        for w in (self._axial, self._sagital, self._coronal):
            w.set_volume(volume, spacing)
        self._sync_crosshairs()

    def set_window(self, center: float, width: float) -> None:
        """Aplica ventana/nivel a los tres planos."""
        for w in (self._axial, self._sagital, self._coronal):
            w.set_window(center, width)

    # ── Build UI ────────────────────────────────────────────────── #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Splitter horizontal: los tres planos, redimensionables
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        self._splitter = splitter
        self._apply_splitter_style()

        self._axial   = _PlaneView("axial")
        self._sagital = _PlaneView("sagital")
        self._coronal = _PlaneView("coronal")

        for view in (self._axial, self._sagital, self._coronal):
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            splitter.addWidget(view)

        root.addWidget(splitter)

        # Conectar señales de posición para sincronizar crosshairs
        self._axial.position_changed.connect(self._on_axial_changed)
        self._sagital.position_changed.connect(self._on_sagital_changed)
        self._coronal.position_changed.connect(self._on_coronal_changed)

    def _apply_splitter_style(self) -> None:
        handle_bg  = "#0d1520" if _is_dark() else "#C8D4DC"
        handle_hov = "#4E6678" if _is_dark() else "#8B9BAA"
        self._splitter.setStyleSheet(
            f"QSplitter::handle        {{ background:{handle_bg}; }}"
            f"QSplitter::handle:hover  {{ background:{handle_hov}; }}"
        )

    # ── Theme ───────────────────────────────────────────────────── #

    def apply_theme(self) -> None:
        """Re-style all panes and splitter to match the current light/dark theme."""
        self._apply_splitter_style()
        for view in (self._axial, self._sagital, self._coronal):
            view.apply_theme()

    # ── Qt overrides ────────────────────────────────────────────── #

    def closeEvent(self, event) -> None:    # noqa: N802
        """Ocultar en lugar de destruir, para preservar el estado de sliders."""
        event.ignore()
        self.hide()
        self.hidden.emit()

    # ── Crosshair sync ──────────────────────────────────────────── #

    def _on_axial_changed(self, _plane: str, idx: int) -> None:
        self._z = idx
        self._sync_crosshairs(skip="axial")

    def _on_sagital_changed(self, _plane: str, idx: int) -> None:
        self._x = idx
        self._sync_crosshairs(skip="sagital")

    def _on_coronal_changed(self, _plane: str, idx: int) -> None:
        self._y = idx
        self._sync_crosshairs(skip="coronal")

    def _sync_crosshairs(self, skip: str = "") -> None:
        """
        Actualiza los crosshairs (excepto el plano *skip*) en mm físicos.

          plano      fila        columna
          ─────────  ──────────  ──────────
          axial      Y_mm        X_mm
          coronal    Z_mm        X_mm
          sagital    Z_mm        Y_mm
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
