"""CrossPlaneDialog — visor DICOM 3-D con planos MPR que se cruzan en el espacio.

Layout
------
  ┌──────────────────────────────────────────────────────────────┐
  │  [Header: "Visor DICOM 3D — SAG + COR + AX"]               │
  │                                                              │
  │       VTK render window (trackball camera)                   │
  │                                                              │
  │   plano SAGITAL (naranja) ╲  ╱ plano CORONAL (verde)        │
  │                            ╲╱                               │
  │                            /╲   plano AXIAL (azul)          │
  │                                                              │
  ├──────────────────────────────────────────────────────────────┤
  │  [SAG] ────────────── slider ────────────── X = 38.0 mm     │
  │  [COR] ────────────── slider ────────────── Y = 14.2 mm     │
  │  [AX ] ────────────── slider ────────────── Z = 25.1 mm     │
  └──────────────────────────────────────────────────────────────┘

Los tres planos son instancias de ``vtkImagePlaneWidget`` que comparten el mismo
``vtkRenderer``.  El usuario puede:
  • Girar la cámara con clic-arrastre izquierdo (trackball).
  • Mover cada plano con su slider inferior.
  • Aplicar zoom con la rueda del ratón.
  • Activar/desactivar cada plano individualmente con el botón ● / ○
    situado a la izquierda de cada fila de slider (visualización en 1, 2 o 3 planos).

API pública
-----------
set_volume(volume, spacing)
    Carga el volumen 3-D (shape Z×Y×X, valores HU) en los tres planos.

set_window(center, width)
    Aplica ventana/nivel a los tres planos (vtkLookupTable compartida).
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# ── Theme helper ─────────────────────────────────────────────────────────── #

def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


def _cpd_bg() -> str:
    """Return the header/footer background colour for the current theme."""
    return "#0d1520" if _is_dark() else "#D8E4EE"


# ── Paleta de colores por plano ───────────────────────────────────────────── #
_PLANE_CFG = {
    "sagital": {
        "label": "SAG",
        "fg":    "#ffcc80",          # naranja suave
        "vtk":   (1.0, 0.80, 0.50), # color RGB del borde del plano en VTK
    },
    "coronal": {
        "label": "COR",
        "fg":    "#69f0ae",          # verde menta
        "vtk":   (0.41, 0.94, 0.68),
    },
    "axial": {
        "label": "AX",
        "fg":    "#4fc3f7",          # azul claro
        "vtk":   (0.31, 0.76, 0.97),
    },
}


def _numpy_to_vtk_image(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
) -> vtk.vtkImageData:
    """Convierte numpy (Z,Y,X) float32 a vtkImageData."""
    from vtkmodules.util import numpy_support as ns

    z, y, x = volume.shape
    sz, sy, sx = spacing

    img = vtk.vtkImageData()
    img.SetDimensions(x, y, z)
    img.SetSpacing(sx, sy, sz)
    img.SetOrigin(0.0, 0.0, 0.0)

    flat = np.ascontiguousarray(volume, dtype=np.float32).ravel()
    vtk_arr = ns.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName("HU")
    img.GetPointData().SetScalars(vtk_arr)
    return img


# ─────────────────────────────────────────────────────────────────────────── #
# CrossPlaneDialog                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class CrossPlaneDialog(QDialog):
    """
    Ventana flotante no-modal con tres planos MPR intersectándose en 3-D.

    Cierra con hide() (no destroy) para preservar posiciones de sliders.
    Emite ``hidden`` al pulsar ✕, para que el botón de la toolbar pueda
    sincronizar su estado checked.

    Uso
    ---
        dlg = CrossPlaneDialog(parent=main_window)
        dlg.set_volume(series.volume, series.spacing)
        dlg.set_window(wc, ww)
        dlg.show()
    """

    hidden = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Visor DICOM 3D  —  SAG · COR · AX")
        self.setMinimumSize(620, 460)
        from PyQt5.QtWidgets import QApplication as _QApp
        _scr = _QApp.primaryScreen()
        if _scr is not None:
            _av = _scr.availableGeometry()
            self.resize(min(900, int(_av.width() * 0.88)), min(640, int(_av.height() * 0.85)))
        else:
            self.resize(900, 640)
        self.setModal(False)

        # Estado interno
        self._volume_loaded = False
        self._spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._dims:    tuple[int, int, int]        = (1, 1, 1)   # (z, y, x)
        self._wc: float = 40.0
        self._ww: float = 400.0

        # Lookup table compartida entre los tres planos
        self._lut: vtk.vtkLookupTable = self._build_lut(self._wc, self._ww)

        # Planos VTK (creados en _build_vtk_pipeline)
        self._plane_sag: vtk.vtkImagePlaneWidget | None = None
        self._plane_cor: vtk.vtkImagePlaneWidget | None = None
        self._plane_ax:  vtk.vtkImagePlaneWidget | None = None

        # Visibilidad por plano y botones de toggle (poblados en _build_ui)
        self._plane_vis: dict[str, bool]         = {
            "sagital": True,
            "coronal": True,
            "axial":   True,
        }
        self._vis_btns: dict[str, QPushButton]   = {}

        self._build_vtk_pipeline()
        self._build_ui()

    # ── VTK pipeline ─────────────────────────────────────────────── #

    def _build_vtk_pipeline(self) -> None:
        """Crea el interactor VTK, el renderer y los tres planos."""
        # Interactor embebido en Qt
        self._iren = QVTKRenderWindowInteractor()
        self._iren.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Renderer con fondo azul oscuro degradado
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.05, 0.07, 0.10)
        self._renderer.SetBackground2(0.02, 0.03, 0.06)
        self._renderer.SetGradientBackground(True)
        self._iren.GetRenderWindow().AddRenderer(self._renderer)

        # Estilo de interacción: trackball (girar con clic-arrastre)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self._iren.SetInteractorStyle(style)

        # Marcador de orientación (ejes R/A/S en esquina inferior-izquierda)
        axes_actor = vtk.vtkAxesActor()
        axes_actor.SetXAxisLabelText("R")
        axes_actor.SetYAxisLabelText("A")
        axes_actor.SetZAxisLabelText("S")
        self._axes_widget = vtk.vtkOrientationMarkerWidget()
        self._axes_widget.SetOrientationMarker(axes_actor)
        self._axes_widget.SetInteractor(self._iren)
        self._axes_widget.SetViewport(0.0, 0.0, 0.15, 0.15)
        self._axes_widget.EnabledOn()
        self._axes_widget.InteractiveOff()

        self._iren.Initialize()

    def _make_plane_widget(
        self,
        orientation: int,
        color_rgb: tuple[float, float, float],
    ) -> vtk.vtkImagePlaneWidget:
        """
        Crea un vtkImagePlaneWidget con la orientación dada.

        orientation:
            0 → YZ-plane (sagital, varía en X)
            1 → XZ-plane (coronal, varía en Y)
            2 → XY-plane (axial,   varía en Z)
        """
        plane = vtk.vtkImagePlaneWidget()
        plane.SetInteractor(self._iren)
        plane.SetCurrentRenderer(self._renderer)

        # Color del borde / cursor del plano
        plane.GetPlaneProperty().SetColor(*color_rgb)
        plane.GetPlaneProperty().SetLineWidth(1.5)
        plane.GetSelectedPlaneProperty().SetColor(*color_rgb)
        plane.GetSelectedPlaneProperty().SetOpacity(0.15)
        plane.GetCursorProperty().SetColor(*color_rgb)
        plane.GetMarginProperty().SetColor(*color_rgb)

        # Lookup table compartida (grayscale con ventana/nivel)
        plane.SetLookupTable(self._lut)

        plane.SetPlaneOrientation(orientation)
        plane.DisplayTextOn()        # coordenadas en la imagen
        plane.TextureInterpolateOn()

        # Desactivar la interacción directa sobre el plano para no entrar
        # en conflicto con el trackball; el usuario mueve los planos con
        # los sliders inferiores.
        plane.SetInteraction(0)

        return plane

    # ── Lookup table (grayscale W/L) ─────────────────────────────── #

    @staticmethod
    def _build_lut(center: float, width: float) -> vtk.vtkLookupTable:
        lut = vtk.vtkLookupTable()
        lo  = center - width / 2.0
        hi  = center + width / 2.0
        lut.SetRange(lo, hi)
        lut.SetValueRange(0.0, 1.0)
        lut.SetSaturationRange(0.0, 0.0)   # escala de grises
        lut.SetRampToLinear()
        lut.Build()
        return lut

    def _update_lut(self) -> None:
        lo = self._wc - self._ww / 2.0
        hi = self._wc + self._ww / 2.0
        self._lut.SetRange(lo, hi)
        self._lut.Build()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    # ── UI ───────────────────────────────────────────────────────── #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        hdr = QLabel("Visor DICOM 3D  —  SAG · COR · AX")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setFixedHeight(28)
        _hdr_clr = "#8B9BAA" if _is_dark() else "#4E6678"
        hdr.setStyleSheet(
            f"background:{_cpd_bg()}; color:{_hdr_clr}; font-weight:bold;"
            "font-size:11px; letter-spacing:2px;"
        )
        self._cpd_hdr = hdr
        root.addWidget(hdr)

        # VTK canvas
        root.addWidget(self._iren, stretch=1)

        # Footer con sliders
        footer = QWidget()
        footer.setStyleSheet(f"background:{_cpd_bg()};")
        self._cpd_footer = footer
        fl = QVBoxLayout(footer)
        fl.setContentsMargins(10, 6, 10, 8)
        fl.setSpacing(4)

        self._sliders: dict[str, QSlider]   = {}
        self._pos_lbls: dict[str, QLabel]   = {}

        for plane_name in ("sagital", "coronal", "axial"):
            cfg = _PLANE_CFG[plane_name]
            fg  = cfg["fg"]

            row = QWidget()
            hl  = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)

            # ── Botón de visibilidad ──────────────────────────────── #
            vis_btn = QPushButton("●")
            vis_btn.setCheckable(True)
            vis_btn.setChecked(True)
            vis_btn.setFixedSize(22, 22)
            vis_btn.setToolTip(f"Mostrar / ocultar plano {cfg['label']}")
            vis_btn.setStyleSheet(
                f"QPushButton {{"
                f"  color:{fg}; background:transparent; border:none;"
                f"  font-size:13px; font-weight:bold; padding:0px;"
                f"}}"
                f"QPushButton:!checked {{"
                f"  color:#555555;"
                f"}}"
            )
            vis_btn.toggled.connect(
                lambda chk, n=plane_name: self._toggle_plane(n, chk)
            )
            self._vis_btns[plane_name] = vis_btn
            hl.addWidget(vis_btn)

            tag = QLabel(cfg["label"])
            tag.setFixedWidth(28)
            tag.setAlignment(Qt.AlignCenter)
            tag.setStyleSheet(
                f"color:{fg}; font-size:9px; font-weight:bold;"
                "font-family:monospace; letter-spacing:1px;"
            )
            hl.addWidget(tag)

            sl = QSlider(Qt.Horizontal)
            sl.setMinimum(0)
            sl.setMaximum(0)
            sl.setValue(0)
            sl.setEnabled(False)
            sl.setStyleSheet(
                f"QSlider::groove:horizontal{{"
                f"  height:3px; background:rgba(255,255,255,25); border-radius:2px;}}"
                f"QSlider::handle:horizontal{{"
                f"  width:14px; height:14px; margin:-6px 0;"
                f"  background:{fg}; border-radius:7px;}}"
                f"QSlider::sub-page:horizontal{{"
                f"  background:{fg}; border-radius:2px; opacity:0.7;}}"
            )
            self._sliders[plane_name] = sl
            hl.addWidget(sl, stretch=1)

            pos_lbl = QLabel("—")
            pos_lbl.setFixedWidth(82)
            pos_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pos_lbl.setStyleSheet(
                f"color:{fg}; font-size:10px; font-family:monospace;"
            )
            self._pos_lbls[plane_name] = pos_lbl
            hl.addWidget(pos_lbl)

            fl.addWidget(row)

        root.addWidget(footer)

        # Conectar sliders
        self._sliders["sagital"].valueChanged.connect(self._on_sag_slider)
        self._sliders["coronal"].valueChanged.connect(self._on_cor_slider)
        self._sliders["axial"].valueChanged.connect(self._on_ax_slider)

    # ── Theme ────────────────────────────────────────────────────── #

    def apply_theme(self) -> None:
        """Re-style header and footer to match the current light/dark theme."""
        bg = _cpd_bg()
        _hdr_clr = "#8B9BAA" if _is_dark() else "#4E6678"
        self._cpd_hdr.setStyleSheet(
            f"background:{bg}; color:{_hdr_clr}; font-weight:bold;"
            "font-size:11px; letter-spacing:2px;"
        )
        self._cpd_footer.setStyleSheet(f"background:{bg};")
        for plane_name, btn in self._vis_btns.items():
            fg = _PLANE_CFG[plane_name]["fg"]
            btn.setStyleSheet(
                f"QPushButton {{"
                f"  color:{fg}; background:transparent; border:none;"
                f"  font-size:13px; font-weight:bold; padding:0px;"
                f"}}"
                f"QPushButton:!checked {{"
                f"  color:#555555;"
                f"}}"
            )

    # ── Plane visibility toggle ──────────────────────────────────── #

    def _toggle_plane(self, name: str, checked: bool) -> None:
        """Show or hide one MPR plane and update sliders and header."""
        self._plane_vis[name] = checked

        # Update button symbol (● visible / ○ hidden)
        btn = self._vis_btns.get(name)
        if btn is not None:
            btn.setText("●" if checked else "○")

        # Show/hide the VTK plane widget
        plane_map = {
            "sagital": self._plane_sag,
            "coronal": self._plane_cor,
            "axial":   self._plane_ax,
        }
        plane = plane_map.get(name)
        if plane is not None:
            if checked:
                plane.On()
            else:
                plane.Off()
            if self._volume_loaded:
                self._iren.GetRenderWindow().Render()

        # Enable / disable the position slider
        self._sliders[name].setEnabled(checked and self._volume_loaded)

        # Dim the position label when the plane is hidden
        fg = _PLANE_CFG[name]["fg"]
        self._pos_lbls[name].setStyleSheet(
            f"color:{'#555555' if not checked else fg};"
            "font-size:10px; font-family:monospace;"
        )
        if not checked:
            self._pos_lbls[name].setText("—")

        self._update_header_text()

    def _update_header_text(self) -> None:
        """Rebuild window title and header label to list only the visible planes."""
        visible = [
            _PLANE_CFG[n]["label"]
            for n in ("sagital", "coronal", "axial")
            if self._plane_vis.get(n, True)
        ]
        parts = " · ".join(visible) if visible else "sin planos activos"
        text  = f"Visor DICOM 3D  —  {parts}"
        self._cpd_hdr.setText(text)
        self.setWindowTitle(text)

    # ── Public API ───────────────────────────────────────────────── #

    def set_volume(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        """Carga el volumen en los tres planos y centra los sliders."""
        self._spacing = spacing
        z, y, x = volume.shape
        self._dims = (z, y, x)

        # Convertir a vtkImageData
        img_data = _numpy_to_vtk_image(volume, spacing)

        # Destruir planos previos si los hay
        for attr in ("_plane_sag", "_plane_cor", "_plane_ax"):
            old = getattr(self, attr, None)
            if old is not None:
                old.Off()
            setattr(self, attr, None)

        # Crear los tres planos
        self._plane_sag = self._make_plane_widget(0, _PLANE_CFG["sagital"]["vtk"])
        self._plane_cor = self._make_plane_widget(1, _PLANE_CFG["coronal"]["vtk"])
        self._plane_ax  = self._make_plane_widget(2, _PLANE_CFG["axial"]["vtk"])

        for plane in (self._plane_sag, self._plane_cor, self._plane_ax):
            plane.SetInputData(img_data)

        # Posición inicial: centro de cada eje
        idx_x = x // 2
        idx_y = y // 2
        idx_z = z // 2

        self._plane_sag.SetSliceIndex(idx_x)
        self._plane_cor.SetSliceIndex(idx_y)
        self._plane_ax.SetSliceIndex(idx_z)

        # Activar sólo los planos que el usuario marcó como visibles
        for plane_name, plane in (
            ("sagital", self._plane_sag),
            ("coronal", self._plane_cor),
            ("axial",   self._plane_ax),
        ):
            if self._plane_vis.get(plane_name, True):
                plane.On()
            else:
                plane.Off()

        # Configurar sliders (sólo habilitados si el plano es visible)
        for plane_name, sl in self._sliders.items():
            sl.setEnabled(self._plane_vis.get(plane_name, True))

        self._sliders["sagital"].blockSignals(True)
        self._sliders["sagital"].setMaximum(x - 1)
        self._sliders["sagital"].setValue(idx_x)
        self._sliders["sagital"].blockSignals(False)

        self._sliders["coronal"].blockSignals(True)
        self._sliders["coronal"].setMaximum(y - 1)
        self._sliders["coronal"].setValue(idx_y)
        self._sliders["coronal"].blockSignals(False)

        self._sliders["axial"].blockSignals(True)
        self._sliders["axial"].setMaximum(z - 1)
        self._sliders["axial"].setValue(idx_z)
        self._sliders["axial"].blockSignals(False)

        self._update_label("sagital", idx_x)
        self._update_label("coronal", idx_y)
        self._update_label("axial",   idx_z)

        self._renderer.ResetCamera()
        self._iren.GetRenderWindow().Render()
        self._volume_loaded = True

    def set_window(self, center: float, width: float) -> None:
        """Aplica ventana/nivel (window center / window width) a los tres planos."""
        self._wc = center
        self._ww = max(1.0, width)
        self._update_lut()

    # ── Slider handlers ──────────────────────────────────────────── #

    def _on_sag_slider(self, value: int) -> None:
        if self._plane_sag is not None:
            self._plane_sag.SetSliceIndex(value)
            if self._volume_loaded:
                self._iren.GetRenderWindow().Render()
        self._update_label("sagital", value)

    def _on_cor_slider(self, value: int) -> None:
        if self._plane_cor is not None:
            self._plane_cor.SetSliceIndex(value)
            if self._volume_loaded:
                self._iren.GetRenderWindow().Render()
        self._update_label("coronal", value)

    def _on_ax_slider(self, value: int) -> None:
        if self._plane_ax is not None:
            self._plane_ax.SetSliceIndex(value)
            if self._volume_loaded:
                self._iren.GetRenderWindow().Render()
        self._update_label("axial", value)

    # ── Label helper ─────────────────────────────────────────────── #

    def _update_label(self, plane: str, idx: int) -> None:
        sz, sy, sx = self._spacing
        z, y, x    = self._dims

        if plane == "sagital":
            total  = x
            pos_mm = idx * sx
            axis   = "X"
        elif plane == "coronal":
            total  = y
            pos_mm = idx * sy
            axis   = "Y"
        else:  # axial
            total  = z
            pos_mm = idx * sz
            axis   = "Z"

        if total == 0:
            return
        self._pos_lbls[plane].setText(
            f"{axis} {idx + 1}/{total}\n{pos_mm:.1f} mm"
        )

    # ── Qt overrides ─────────────────────────────────────────────── #

    def closeEvent(self, event) -> None:    # noqa: N802
        """Ocultar en lugar de destruir para preservar estado de sliders."""
        event.ignore()
        self.hide()
        self.hidden.emit()
