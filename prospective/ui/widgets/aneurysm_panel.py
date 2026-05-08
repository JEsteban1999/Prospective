"""Aneurysm detection & isolation panel — A-02-05.

Exposes signals so MainWindow can wire it to the VTK viewer.
"""
from __future__ import annotations

import logging

import vtk
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from prospective.io.mesh_exporter import MeshExporter
from prospective.processing.aneurysm_detector import (
    AneurysmCandidate,
    AneurysmDetector,
    DetectionResult,
)

logger = logging.getLogger(__name__)


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


# ──────────────────────────────────────────────────────────────────────────── #
# Background worker                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

class _DetectionWorker(QThread):
    finished = pyqtSignal(object)   # DetectionResult
    error    = pyqtSignal(str)

    def __init__(
        self,
        detector: AneurysmDetector,
        poly_data: vtk.vtkPolyData,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._detector  = detector
        self._poly_data = poly_data

    def run(self) -> None:
        try:
            result = self._detector.detect(self._poly_data)
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Aneurysm detection failed")
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────── #
# Panel widget                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

class AneurysmPanel(QWidget):
    """
    Self-contained aneurysm detection controls, meant for a QDockWidget.

    Signals
    -------
    candidate_highlighted(AneurysmCandidate | None)
        — user selected / deselected a row in the table
    candidate_isolated(AneurysmCandidate)
        — user clicked "Aislar"; show isolated mesh and focus 3D camera
    export_requested(vtkPolyData)
        — user clicked "Exportar STL"
    """

    candidate_highlighted = pyqtSignal(object)   # AneurysmCandidate | None  — browse only
    candidate_isolated    = pyqtSignal(object)   # AneurysmCandidate        — visual focus
    candidate_confirmed   = pyqtSignal(object)   # AneurysmCandidate        — proceed to morpho
    export_requested      = pyqtSignal(object)   # vtkPolyData
    #: Emitted when the user applies or resets a 3D crop.
    #: MainWindow connects this to update self._vessel_poly and the 3D viewer.
    mesh_cropped          = pyqtSignal(object)   # vtkPolyData
    #: Emitted to show/hide the interactive vtkBoxWidget2 in the MPR 3D view.
    box_widget_requested          = pyqtSignal(bool)   # True=show, False=hide
    #: Emitted when the user clicks "Aplicar recorte" in interactive-box mode.
    #: MainWindow reads the current box bounds from VTK and calls apply_external_bounds().
    interactive_crop_apply_requested = pyqtSignal()
    #: Emitted when the user toggles the candidate highlight visibility.
    #: True = visible (show magenta highlight), False = hidden.
    candidate_visibility_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mesh: vtk.vtkPolyData | None = None
        self._original_mesh: vtk.vtkPolyData | None = None  # saved for restore after crop
        self._candidates: list[AneurysmCandidate] = []
        self._worker: _DetectionWorker | None = None
        self._pre_smooth_iters: int = 0   # Laplacian passes before curvature; set by preset buttons

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_session_state(self) -> dict:
        return {
            "percentile":           self._pct_slider.value(),
            "gauss_percentile":     self._gauss_slider.value(),
            "min_r_mm":             self._min_r_spin.value(),
            "max_r_mm":             self._max_r_spin.value(),
            "min_pts":              self._min_pts_spin.value(),
            "min_pos_gauss_frac":   self._pgf_spin.value(),    # v4
            "min_sphericity":       self._sph_spin.value(),    # v5
        }

    def restore_session_state(self, d: dict) -> None:
        self._pct_slider.setValue(int(d.get("percentile", 75)))
        self._gauss_slider.setValue(int(d.get("gauss_percentile", 85)))
        self._min_r_spin.setValue(float(d.get("min_r_mm", 1.0)))
        self._max_r_spin.setValue(float(d.get("max_r_mm", 20.0)))
        self._min_pts_spin.setValue(int(d.get("min_pts", 8)))
        self._pgf_spin.setValue(float(d.get("min_pos_gauss_frac", 0.50)))   # v4
        self._sph_spin.setValue(float(d.get("min_sphericity", 0.28)))       # v5

    def set_mesh(self, poly_data: vtk.vtkPolyData) -> None:
        """Called when a new segmentation mesh is available.

        Saves the original mesh for restore after cropping and resets the crop
        UI so each new segmentation starts clean.
        """
        self._mesh          = poly_data
        self._original_mesh = poly_data   # keep reference for "Restablecer"
        self._btn_detect.setEnabled(True)
        self._lbl_status.setText("Malla lista — recorte opcional, luego pulse Detectar.")
        # Reset crop state
        self._crop_mode_combo.setCurrentIndex(0)   # "Sin recorte"
        self._btn_apply_crop.setEnabled(False)
        self._btn_reset_crop.setEnabled(False)
        # Clear detection results
        self._candidates.clear()
        self._table.setRowCount(0)
        self._set_action_buttons_enabled(False)

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        scroll.setWidget(container)

        # ── Recorte de malla (opcional) ───────────────────────────────── #
        crop_grp = QGroupBox("Recorte de malla  (opcional)")
        crop_grp.setToolTip(
            "Recorta la malla segmentada antes de la detección para eliminar\n"
            "tejido no vascular (músculo, hueso residual, artefactos).\n\n"
            "• Caja 3D  — define los límites X/Y/Z del volumen a conservar\n"
            "• Esfera   — conserva solo lo que queda dentro de la esfera\n\n"
            "Pulse 'Aplicar recorte' y revise el resultado en el visor 3D.\n"
            "'Restablecer' devuelve la malla original sin volver a segmentar."
        )
        cl = QVBoxLayout()
        cl.setSpacing(5)
        cl.setContentsMargins(6, 4, 6, 6)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_row.addWidget(QLabel("Modo:"))
        self._crop_mode_combo = QComboBox()
        self._crop_mode_combo.addItems([
            "Sin recorte",
            "Caja 3D (spinboxes)",
            "Esfera",
            "↔ Caja 3D (interactiva en visor)",
        ])
        self._crop_mode_combo.setToolTip(
            "Sin recorte                 — la malla pasa íntegra al detector\n"
            "Caja 3D (spinboxes)         — introduce límites X/Y/Z manualmente\n"
            "Esfera                      — conserva solo lo dentro de la esfera\n"
            "↔ Caja 3D (interactiva)     — arrastra handles directamente en el visor 3D\n"
            "                              (recomendado: ve a la pestaña MPR / Planif. 3D)"
        )
        self._crop_mode_combo.currentIndexChanged.connect(self._on_crop_mode_changed)
        mode_row.addWidget(self._crop_mode_combo, 1)
        cl.addLayout(mode_row)

        # Stacked pages: 0=hint · 1=box · 2=sphere
        self._crop_stack = QStackedWidget()

        # ── Page 0: no-crop hint ──────────────────────────────────────── #
        hint_w = QWidget()
        hint_lay = QVBoxLayout(hint_w)
        hint_lay.setContentsMargins(0, 2, 0, 2)
        hint_lbl = QLabel(
            "Seleccione 'Caja 3D' o 'Esfera' para activar\n"
            "el recorte antes de la detección."
        )
        hint_lbl.setAlignment(Qt.AlignCenter)
        hint_lbl.setProperty("role", "muted")
        hint_lbl.setWordWrap(True)
        hint_lay.addWidget(hint_lbl)
        self._crop_stack.addWidget(hint_w)   # index 0

        # ── Page 1: box controls ──────────────────────────────────────── #
        box_w  = QWidget()
        bf     = QFormLayout(box_w)
        bf.setLabelAlignment(Qt.AlignRight)
        bf.setSpacing(3)
        bf.setContentsMargins(0, 4, 0, 0)

        def _make_range_row(
            label: str,
        ) -> tuple["QWidget", "QDoubleSpinBox", "QDoubleSpinBox"]:
            """Return (row_widget, min_spinbox, max_spinbox)."""
            row = QWidget()
            rlay = QHBoxLayout(row)
            rlay.setContentsMargins(0, 0, 0, 0)
            rlay.setSpacing(4)
            mn = QDoubleSpinBox()
            mx = QDoubleSpinBox()
            for sb in (mn, mx):
                sb.setRange(-999.0, 999.0)
                sb.setSingleStep(1.0)
                sb.setDecimals(1)
                sb.setSuffix(" mm")
            rlay.addWidget(QLabel("de"))
            rlay.addWidget(mn, 1)
            rlay.addWidget(QLabel("a"))
            rlay.addWidget(mx, 1)
            bf.addRow(label, row)
            return row, mn, mx

        _, self._crop_xmin, self._crop_xmax = _make_range_row("X:")
        _, self._crop_ymin, self._crop_ymax = _make_range_row("Y:")
        _, self._crop_zmin, self._crop_zmax = _make_range_row("Z:")
        self._crop_stack.addWidget(box_w)    # index 1

        # ── Page 2: sphere controls ───────────────────────────────────── #
        sph_w = QWidget()
        sf    = QFormLayout(sph_w)
        sf.setLabelAlignment(Qt.AlignRight)
        sf.setSpacing(3)
        sf.setContentsMargins(0, 4, 0, 0)

        def _sph_spin(lo: float = -999.0, hi: float = 999.0,
                      step: float = 1.0, val: float = 0.0) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(1)
            sb.setSuffix(" mm")
            sb.setValue(val)
            return sb

        self._crop_cx = _sph_spin()
        self._crop_cy = _sph_spin()
        self._crop_cz = _sph_spin()
        self._crop_r  = _sph_spin(lo=1.0, hi=500.0, step=5.0, val=30.0)
        self._crop_cx.setToolTip("Coordenada X del centro de la esfera (mm)")
        self._crop_cy.setToolTip("Coordenada Y del centro de la esfera (mm)")
        self._crop_cz.setToolTip("Coordenada Z del centro de la esfera (mm)")
        self._crop_r.setToolTip(
            "Radio de la esfera en mm.\n"
            "El valor inicial es el 75 % de la semi-diagonal del bounding box."
        )
        sf.addRow("Centro X:", self._crop_cx)
        sf.addRow("Centro Y:", self._crop_cy)
        sf.addRow("Centro Z:", self._crop_cz)
        sf.addRow("Radio:",    self._crop_r)
        self._crop_stack.addWidget(sph_w)    # index 2

        cl.addWidget(self._crop_stack)

        # Action buttons
        crop_btn_row = QWidget()
        cbl = QHBoxLayout(crop_btn_row)
        cbl.setContentsMargins(0, 4, 0, 0)
        cbl.setSpacing(6)

        self._btn_apply_crop = QPushButton("Aplicar recorte")
        self._btn_apply_crop.setEnabled(False)
        self._btn_apply_crop.setToolTip(
            "Aplica el recorte a la malla y actualiza el visor 3D.\n"
            "La detección de aneurismas se ejecutará sobre la malla recortada."
        )
        self._btn_apply_crop.clicked.connect(self._apply_crop)
        cbl.addWidget(self._btn_apply_crop)

        self._btn_reset_crop = QPushButton("Restablecer")
        self._btn_reset_crop.setEnabled(False)
        self._btn_reset_crop.setToolTip(
            "Restaura la malla original completa sin necesidad de resegmentar."
        )
        self._btn_reset_crop.clicked.connect(self._reset_crop)
        cbl.addWidget(self._btn_reset_crop)

        cl.addWidget(crop_btn_row)
        crop_grp.setLayout(cl)
        layout.addWidget(crop_grp)

        # ── Parámetros ────────────────────────────────────────────────── #
        params_grp = QGroupBox("Parámetros de detección")
        pf = QFormLayout()
        pf.setLabelAlignment(Qt.AlignRight)

        # Curvature percentile slider
        pct_row = QWidget()
        pct_lay = QHBoxLayout(pct_row)
        pct_lay.setContentsMargins(0, 0, 0, 0)
        pct_lay.setSpacing(4)

        self._pct_slider = QSlider(Qt.Horizontal)
        self._pct_slider.setRange(20, 99)
        self._pct_slider.setValue(75)
        self._pct_slider.setToolTip(
            "Umbral mínimo de curvatura media por región (gate secundario).\n"
            "Las regiones que superen la curvatura gaussiana pero tengan\n"
            "curvatura media baja son descartadas aquí.\n"
            "Bajar (p60–p70) para datos XA con mallas más finas."
        )
        self._pct_slider.valueChanged.connect(
            lambda v: self._pct_lbl.setText(f"p{v}")
        )
        pct_lay.addWidget(self._pct_slider, stretch=1)

        self._pct_lbl = QLabel("p75")
        self._pct_lbl.setFixedWidth(30)
        pct_lay.addWidget(self._pct_lbl)
        pf.addRow("Gate curv.M:", pct_row)

        # Gaussian curvature percentile — PRIMARY gate
        gauss_row = QWidget()
        gauss_lay = QHBoxLayout(gauss_row)
        gauss_lay.setContentsMargins(0, 0, 0, 0)
        gauss_lay.setSpacing(4)

        self._gauss_slider = QSlider(Qt.Horizontal)
        self._gauss_slider.setRange(20, 99)
        self._gauss_slider.setValue(85)
        self._gauss_slider.setToolTip(
            "Umbral primario de curvatura gaussiana (κ_gauss = κ₁ × κ₂).\n"
            "• Pared de vaso: κ_gauss ≈ 0 (eje axial plano)\n"
            "• Domo aneurismático: κ_gauss >> 0 (esférico)\n"
            "• Bifurcación: κ_gauss mezclado (positivo en ápex, negativo en base)\n\n"
            "Este slider controla qué tan esférica debe ser la superficie.\n"
            "Reducir a p75–p80 para detectar más candidatos."
        )
        self._gauss_slider.valueChanged.connect(
            lambda v: self._gauss_lbl.setText(f"p{v}")
        )
        gauss_lay.addWidget(self._gauss_slider, stretch=1)

        self._gauss_lbl = QLabel("p85")
        self._gauss_lbl.setFixedWidth(30)
        gauss_lay.addWidget(self._gauss_lbl)
        pf.addRow("Umbral gauss.:", gauss_row)

        # Min radius
        self._min_r_spin = QDoubleSpinBox()
        self._min_r_spin.setRange(0.3, 10.0)
        self._min_r_spin.setSingleStep(0.25)
        self._min_r_spin.setValue(1.0)
        self._min_r_spin.setSuffix(" mm")
        self._min_r_spin.setToolTip(
            "Radio mínimo del candidato (excluye ruido).\n"
            "Para XA/DSA con mallas finas, prueba 0.3–0.8 mm."
        )
        pf.addRow("Radio mín.:", self._min_r_spin)

        # Max radius
        self._max_r_spin = QDoubleSpinBox()
        self._max_r_spin.setRange(3.0, 30.0)
        self._max_r_spin.setSingleStep(1.0)
        self._max_r_spin.setValue(20.0)
        self._max_r_spin.setSuffix(" mm")
        self._max_r_spin.setToolTip(
            "Radio máximo del candidato (excluye segmentos vasculares grandes)."
        )
        pf.addRow("Radio máx.:", self._max_r_spin)

        # Min points
        self._min_pts_spin = QSpinBox()
        self._min_pts_spin.setRange(4, 200)
        self._min_pts_spin.setValue(8)
        self._min_pts_spin.setToolTip(
            "Mínimo de vértices por componente para considerarlo candidato.\n"
            "Bajar a 4–8 para meshes XA/DSA con pocos vértices por región.\n"
            "Subir a 50+ para reducir falsos positivos en CTA."
        )
        pf.addRow("Puntos mín.:", self._min_pts_spin)

        # ── Fracción Gaussiana+ mínima (v4 hard gate) ─────────────────── #
        self._pgf_spin = QDoubleSpinBox()
        self._pgf_spin.setRange(0.0, 0.95)
        self._pgf_spin.setSingleStep(0.05)
        self._pgf_spin.setDecimals(2)
        self._pgf_spin.setValue(0.50)
        self._pgf_spin.setToolTip(
            "Fracción mínima de vértices con curvatura gaussiana POSITIVA.\n\n"
            "κ_G = κ₁ × κ₂ > 0 cuando ambas curvaturas principales son del mismo\n"
            "signo → superficie esférica/convexa (domo de saco).\n\n"
            "  Domo aneurismático (CTA): 0.85 – 1.00\n"
            "  Bifurcación / silla:      0.30 – 0.55  ← filtrado aquí\n"
            "  Pared vascular:           0.05 – 0.25  ← filtrado aquí\n\n"
            "CTA limpio → 0.60   XA/3DRA → 0.40   Máx. sensib. → 0.20"
        )
        pf.addRow("Frac. Gauss+ mín.:", self._pgf_spin)

        # ── Esfericidad mínima (v5 hard gate) ─────────────────────────── #
        self._sph_spin = QDoubleSpinBox()
        self._sph_spin.setRange(0.0, 0.80)
        self._sph_spin.setSingleStep(0.05)
        self._sph_spin.setDecimals(2)
        self._sph_spin.setValue(0.28)
        self._sph_spin.setToolTip(
            "Esfericidad mínima del candidato: min(dim) / max(dim) del bounding box.\n\n"
            "  1.0 = cubo perfecto (esfera)  ·  0.0 = completamente plano/alargado\n\n"
            "  Domo aneurismático:            0.50 – 0.90\n"
            "  Bifurcación ligeramente oval:  0.35 – 0.60\n"
            "  Curva de sifón carotídeo:      0.15 – 0.35  ← filtrado aquí\n"
            "  Franja vascular elongada:      0.05 – 0.25  ← filtrado aquí\n\n"
            "CTA limpio → 0.35   XA/3DRA → 0.20   Máx. sensib. → 0.10"
        )
        pf.addRow("Esfericidad mín.:", self._sph_spin)

        params_grp.setLayout(pf)
        layout.addWidget(params_grp)

        # ── Preajustes rápidos ────────────────────────────────────────── #
        preset_row = QWidget()
        preset_lay = QHBoxLayout(preset_row)
        preset_lay.setContentsMargins(0, 0, 0, 0)
        preset_lay.setSpacing(4)

        lbl_preset = QLabel("Modo:")
        lbl_preset.setStyleSheet(
            "color:#9B9B9B; font-size:10px;" if _is_dark()
            else "color:#6B6B6B; font-size:10px;"
        )
        preset_lay.addWidget(lbl_preset)

        self._btn_preset_cta = QPushButton("CTA")
        self._btn_preset_cta.setToolTip(
            "Parámetros para CTA cerebral estándar.\n"
            "Malla limpia e isola­da alrededor del aneurisma."
        )
        self._btn_preset_cta.clicked.connect(self._apply_preset_cta)
        preset_lay.addWidget(self._btn_preset_cta)

        self._btn_preset_xa = QPushButton("XA / 3DRA")
        self._btn_preset_xa.setToolTip(
            "Parámetros para angiografía rotacional (árbol vascular completo).\n"
            "Umbrales más bajos para encontrar candidatos en mallas ruidosas."
        )
        self._btn_preset_xa.clicked.connect(self._apply_preset_xa)
        preset_lay.addWidget(self._btn_preset_xa)

        self._btn_preset_max = QPushButton("Máx. sensib.")
        self._btn_preset_max.setToolTip(
            "Sensibilidad máxima — detecta todo lo que tenga curvatura elevada.\n"
            "Útil para explorar datos nuevos.  Esperar muchos falsos positivos."
        )
        self._btn_preset_max.clicked.connect(self._apply_preset_max)
        preset_lay.addWidget(self._btn_preset_max)
        self._apply_preset_styles()

        preset_lay.addStretch()
        layout.addWidget(preset_row)

        # ── Botón detectar ────────────────────────────────────────────── #
        self._btn_detect = QPushButton("Detectar candidatos")
        self._btn_detect.setEnabled(False)
        self._btn_detect.setMinimumHeight(30)
        self._btn_detect.setObjectName("btn_purple")
        self._btn_detect.clicked.connect(self._run_detection)
        layout.addWidget(self._btn_detect)

        self._lbl_status = QLabel("Requiere segmentación previa.")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        self._lbl_status.setProperty("role", "muted")
        self._lbl_status.setWordWrap(True)
        layout.addWidget(self._lbl_status)

        # ── Tabla de candidatos ───────────────────────────────────────── #
        table_grp = QGroupBox("Candidatos detectados")
        tl = QVBoxLayout()

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["#", "Diám.(mm)", "Gauss+%", "Esfer.", "Score", "Tipo"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 28)
        self._table.setColumnWidth(1, 68)
        self._table.setColumnWidth(2, 58)
        self._table.setColumnWidth(3, 46)
        self._table.setColumnWidth(4, 44)
        self._table.setMinimumHeight(130)
        self._table.setMaximumHeight(260)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        tl.addWidget(self._table)

        # "Ocultar bifurcaciones" filter checkbox
        self._chk_hide_bifurc = QCheckBox("Ocultar bifurcaciones")
        self._chk_hide_bifurc.setChecked(False)
        self._chk_hide_bifurc.setToolTip(
            "Oculta de la tabla los candidatos clasificados como 'Bifurc.' (gris).\n"
            "Útil para revisar rápidamente solo los Sacos y candidatos Mixtos.\n"
            "Los candidatos siguen siendo parte del resultado — solo se ocultan visualmente."
        )
        self._chk_hide_bifurc.stateChanged.connect(self._apply_bifurc_filter)
        tl.addWidget(self._chk_hide_bifurc)

        # Diagnostic info label
        self._lbl_diag = QLabel("")
        self._lbl_diag.setWordWrap(True)
        self._lbl_diag.setProperty("role", "muted")
        tl.addWidget(self._lbl_diag)

        table_grp.setLayout(tl)
        layout.addWidget(table_grp)

        # ── Acciones sobre candidato ──────────────────────────────────── #
        action_grp = QGroupBox("Candidato seleccionado")
        af = QVBoxLayout()

        self._lbl_candidate_info = QLabel("—")
        self._lbl_candidate_info.setWordWrap(True)
        self._lbl_candidate_info.setProperty("role", "muted")
        af.addWidget(self._lbl_candidate_info)

        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(6)

        self._btn_isolate = QPushButton("Aislar")
        self._btn_isolate.setEnabled(False)
        self._btn_isolate.setToolTip(
            "Muestra solo la malla del candidato seleccionado en el visor 3D."
        )
        self._btn_isolate.clicked.connect(self._on_isolate)
        btn_lay.addWidget(self._btn_isolate)

        # Visibility toggle — show/hide the magenta highlight without
        # deselecting the candidate or affecting the table.
        from prospective.ui.icons import I as _I
        self._btn_visibility = QPushButton(f"{_I.EYE} Ocultar")
        self._btn_visibility.setCheckable(True)
        self._btn_visibility.setChecked(False)   # starts unchecked = "show" state
        self._btn_visibility.setEnabled(False)
        self._btn_visibility.setToolTip(
            "Oculta o muestra el resaltado magenta del candidato en el visor 3D\n"
            "sin perder la selección en la tabla."
        )
        self._btn_visibility.clicked.connect(self._on_toggle_visibility)
        btn_lay.addWidget(self._btn_visibility)

        self._btn_export = QPushButton("Exportar STL")
        self._btn_export.setEnabled(False)
        self._btn_export.setToolTip("Exporta la malla del candidato como STL.")
        self._btn_export.clicked.connect(self._on_export)
        btn_lay.addWidget(self._btn_export)

        af.addWidget(btn_row)
        action_grp.setLayout(af)
        layout.addWidget(action_grp)

        # ── Confirmar candidato ───────────────────────────────────────── #
        # This is the ONE button that advances to the morphometrics step.
        # Browsing (row clicks, "Aislar") is non-destructive and never
        # advances the workflow on its own.
        self._btn_confirm = QPushButton("✔  Confirmar candidato  →  Morfometría")
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.setMinimumHeight(34)
        self._btn_confirm.setObjectName("btn_purple")
        self._btn_confirm.setToolTip(
            "Confirma este candidato como el aneurisma a tratar.\n\n"
            "Ejecutará el análisis morfométrico completo y avanzará\n"
            "automáticamente al paso de Morfometría."
        )
        if _is_dark():
            self._btn_confirm.setStyleSheet(
                "QPushButton{"
                "  background:#4E6678; color:#ffffff; border:none;"
                "  border-radius:9px; font-weight:bold; font-size:11px;"
                "  padding: 0 12px;"
                "}"
                "QPushButton:hover { background:#8B9BAA; }"
                "QPushButton:disabled{"
                "  background:#1C2E3E; color:#7A8E9E; border:1px solid #2A3E52;"
                "}"
            )
        else:
            self._btn_confirm.setStyleSheet(
                "QPushButton{"
                "  background:#4E6678; color:#ffffff; border:none;"
                "  border-radius:9px; font-weight:bold; font-size:11px;"
                "  padding: 0 12px;"
                "}"
                "QPushButton:hover { background:#8B9BAA; }"
                "QPushButton:disabled{"
                "  background:#C8D4DC; color:#8B9BAA; border:1px solid #B0BEC5;"
                "}"
            )
        self._btn_confirm.clicked.connect(self._on_confirm)
        layout.addWidget(self._btn_confirm)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Crop helpers                                                         #
    # ------------------------------------------------------------------ #

    def _on_crop_mode_changed(self, index: int) -> None:
        """Switch the stacked widget and populate bounds from the current mesh."""
        # index 3 = interactive box → does NOT show a stacked page (page 0 = hint)
        stack_idx = index if index <= 2 else 0
        self._crop_stack.setCurrentIndex(stack_idx)
        has_mesh = self._mesh is not None
        self._btn_apply_crop.setEnabled(index > 0 and has_mesh)

        if index == 3:
            # Activate the interactive box widget in the VTK view
            self.box_widget_requested.emit(True)
            self._btn_apply_crop.setText("Aplicar recorte de caja")
            self._btn_apply_crop.setToolTip(
                "Lee los límites actuales del cuadro interactivo en el visor 3D\n"
                "y recorta la malla a ese volumen."
            )
        else:
            # Deactivate the interactive box widget if it was active
            self.box_widget_requested.emit(False)
            self._btn_apply_crop.setText("Aplicar recorte")
            self._btn_apply_crop.setToolTip(
                "Aplica el recorte a la malla y actualiza el visor 3D.\n"
                "La detección de aneurismas se ejecutará sobre la malla recortada."
            )
            if index > 0 and has_mesh:
                self._populate_crop_bounds()

    def _populate_crop_bounds(self) -> None:
        """Fill spinboxes with sensible defaults derived from the mesh bounds."""
        if self._mesh is None:
            return
        import math as _math
        b = self._mesh.GetBounds()   # (xmin, xmax, ymin, ymax, zmin, zmax)
        self._crop_xmin.setValue(b[0])
        self._crop_xmax.setValue(b[1])
        self._crop_ymin.setValue(b[2])
        self._crop_ymax.setValue(b[3])
        self._crop_zmin.setValue(b[4])
        self._crop_zmax.setValue(b[5])

        # Sphere: centre = bbox centroid; radius = 75 % of semi-diagonal
        cx = (b[0] + b[1]) / 2.0
        cy = (b[2] + b[3]) / 2.0
        cz = (b[4] + b[5]) / 2.0
        semi_diag = _math.sqrt(
            (b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2
        ) / 2.0
        self._crop_cx.setValue(cx)
        self._crop_cy.setValue(cy)
        self._crop_cz.setValue(cz)
        self._crop_r.setValue(max(10.0, round(semi_diag * 0.75, 1)))

    def _apply_crop(self) -> None:
        """Clip the mesh with the current mode/parameters and emit mesh_cropped."""
        from prospective.processing.mesh_crop import clip_box, clip_sphere

        if self._mesh is None:
            return

        mode = self._crop_mode_combo.currentIndex()
        if mode == 1:       # ── Caja 3D (spinboxes) ──
            result = clip_box(
                self._mesh,
                self._crop_xmin.value(), self._crop_xmax.value(),
                self._crop_ymin.value(), self._crop_ymax.value(),
                self._crop_zmin.value(), self._crop_zmax.value(),
            )
        elif mode == 2:     # ── Esfera ──
            result = clip_sphere(
                self._mesh,
                (self._crop_cx.value(), self._crop_cy.value(), self._crop_cz.value()),
                self._crop_r.value(),
            )
        elif mode == 3:     # ── Caja 3D interactiva — delegado a MainWindow ──
            # MainWindow reads the VTK box bounds and calls apply_external_bounds().
            self.interactive_crop_apply_requested.emit()
            return
        else:
            return

        n_result = result.GetNumberOfPoints()
        if n_result == 0:
            QMessageBox.warning(
                self,
                "Recorte vacío",
                "El recorte eliminó todos los vértices de la malla.\n\n"
                "Amplíe los límites o reduzca la restricción e intente de nuevo.",
            )
            return

        # Update internal mesh; keep _original_mesh intact for restore
        self._mesh = result
        self._btn_reset_crop.setEnabled(True)
        self._btn_detect.setEnabled(True)

        # Clear stale detection results — they were from the un-cropped mesh
        self._candidates.clear()
        self._table.setRowCount(0)
        self._set_action_buttons_enabled(False)
        self._lbl_diag.setText("")

        n_orig = (
            self._original_mesh.GetNumberOfPoints()
            if self._original_mesh is not None
            else n_result
        )
        pct = n_result / n_orig * 100 if n_orig > 0 else 100.0
        self._lbl_status.setText(
            f"Malla recortada: {n_result:,} vértices ({pct:.0f}% del original).\n"
            "Revise el visor 3D y pulse 'Detectar candidatos'."
        )
        self.mesh_cropped.emit(result)   # → MainWindow updates _vessel_poly + viewer

    def _reset_crop(self) -> None:
        """Restore the original (un-cropped) mesh."""
        if self._original_mesh is None:
            return
        self._mesh = self._original_mesh
        self._btn_reset_crop.setEnabled(False)
        self._candidates.clear()
        self._table.setRowCount(0)
        self._set_action_buttons_enabled(False)
        self._lbl_diag.setText("")
        self._lbl_status.setText(
            "Malla restaurada. Pulse 'Detectar candidatos'."
        )
        self.mesh_cropped.emit(self._original_mesh)

    def apply_external_bounds(
        self,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        zmin: float, zmax: float,
    ) -> None:
        """Apply a crop box defined externally (e.g. by the VTK interactive widget).

        Called by MainWindow after it reads the vtkBoxWidget2 bounds.
        Delegates to the same clip_box pipeline used by the spinbox mode.
        """
        from prospective.processing.mesh_crop import clip_box
        import logging as _log
        _logger = _log.getLogger(__name__)

        if self._mesh is None:
            return

        # ── Pre-check: verify box intersects the mesh bounds ── #
        mb = self._mesh.GetBounds()   # (mx0,mx1, my0,my1, mz0,mz1)
        _logger.info(
            "Interactive crop — box: X[%.1f,%.1f] Y[%.1f,%.1f] Z[%.1f,%.1f] | "
            "mesh: X[%.1f,%.1f] Y[%.1f,%.1f] Z[%.1f,%.1f]",
            xmin, xmax, ymin, ymax, zmin, zmax,
            mb[0], mb[1], mb[2], mb[3], mb[4], mb[5],
        )
        no_overlap = (
            xmax < mb[0] or xmin > mb[1] or
            ymax < mb[2] or ymin > mb[3] or
            zmax < mb[4] or zmin > mb[5]
        )
        if no_overlap:
            QMessageBox.warning(
                self,
                "Recorte vacío",
                "El cuadro interactivo no contiene ningún vértice de la malla.\n\n"
                "Amplíe el cuadro hasta que rodee la zona de interés\n"
                "visible en el visor 3D e inténtelo de nuevo.",
            )
            return

        result = clip_box(self._mesh, xmin, xmax, ymin, ymax, zmin, zmax)
        n_result = result.GetNumberOfPoints()
        if n_result == 0:
            QMessageBox.warning(
                self,
                "Recorte vacío",
                "El recorte eliminó todos los vértices de la malla.\n\n"
                "Amplíe el cuadro interactivo e intente de nuevo.",
            )
            return

        self._mesh = result
        self._btn_reset_crop.setEnabled(True)
        self._btn_detect.setEnabled(True)
        self._candidates.clear()
        self._table.setRowCount(0)
        self._set_action_buttons_enabled(False)
        self._lbl_diag.setText("")

        n_orig = (
            self._original_mesh.GetNumberOfPoints()
            if self._original_mesh is not None else n_result
        )
        pct = n_result / n_orig * 100 if n_orig > 0 else 100.0
        self._lbl_status.setText(
            f"Malla recortada (caja interactiva): {n_result:,} vértices "
            f"({pct:.0f}% del original).\n"
            "Revise el visor 3D y pulse 'Detectar candidatos'."
        )
        # Hide the box widget — crop is applied, no longer needed
        self.box_widget_requested.emit(False)
        self._crop_mode_combo.blockSignals(True)
        self._crop_mode_combo.setCurrentIndex(0)   # back to "Sin recorte"
        self._crop_mode_combo.blockSignals(False)
        self._btn_apply_crop.setEnabled(False)

        self.mesh_cropped.emit(result)

    # ------------------------------------------------------------------ #
    # Modality / preset helpers                                            #
    # ------------------------------------------------------------------ #

    def set_modality(self, modality: str) -> None:
        """
        Auto-configure detection parameters based on the DICOM modality.

        For XA / RF / 3DRA (rotational angiography) the segmentation
        produces a full vascular tree instead of an isolated aneurysm
        region.  The curvature distribution is very different, so we
        use more permissive thresholds by default.
        """
        _XA = {"XA", "RF", "DX", "CR", "DR"}
        if modality.upper() in _XA:
            self._apply_preset_xa()
        # CT / MR → keep defaults (or already restored from session)

    def _apply_preset_styles(self) -> None:
        """Apply theme-aware styles to the preset buttons."""
        dark = _is_dark()
        base = (
            f"QPushButton{{border:1px solid {'#363636' if dark else '#E5E5E5'};"
            "border-radius:5px;padding:2px 8px;font-size:10px;}}"
            f"QPushButton:hover{{border-color:{'#A8B8C6' if dark else '#8B9BAA'};"
            f"color:{'#A8B8C6' if dark else '#8B9BAA'};}}"
            f"QPushButton:pressed{{background:{'#1C303F' if dark else '#DDE5EC'};}}"
        )
        cta_bg, cta_fg   = ("#2A2A2A", "#EBEBEB") if dark else ("#F7F7F7", "#0D0D0D")
        xa_bg,  xa_fg    = ("#182434", "#A8B8C6") if dark else ("#DDE5EC", "#4E6678")
        xa_bdr           = "#8B9BAA" if dark else "#8B9BAA"
        max_bg, max_fg   = ("#1a1400", "#e3b341") if dark else ("#fff8c5", "#d29922")
        max_bdr          = "#3d2e00" if dark else "#d4a72c"

        self._btn_preset_cta.setStyleSheet(
            base + f"QPushButton{{background:{cta_bg};color:{cta_fg};}}"
        )
        self._btn_preset_xa.setStyleSheet(
            base + f"QPushButton{{background:{xa_bg};color:{xa_fg};border-color:{xa_bdr};}}"
        )
        self._btn_preset_max.setStyleSheet(
            base + f"QPushButton{{background:{max_bg};color:{max_fg};border-color:{max_bdr};}}"
        )

    def apply_theme(self) -> None:
        """Re-apply theme-dependent colours. Call after toggling the UI theme."""
        self._apply_preset_styles()
        # Refresh confirm button disabled state colour
        if _is_dark():
            self._btn_confirm.setStyleSheet(
                "QPushButton{"
                "  background:#4E6678; color:#ffffff; border:none;"
                "  border-radius:9px; font-weight:bold; font-size:11px;"
                "  padding: 0 12px;"
                "}"
                "QPushButton:hover { background:#8B9BAA; }"
                "QPushButton:disabled{"
                "  background:#1C2E3E; color:#7A8E9E; border:1px solid #2A3E52;"
                "}"
            )
        else:
            self._btn_confirm.setStyleSheet(
                "QPushButton{"
                "  background:#4E6678; color:#ffffff; border:none;"
                "  border-radius:9px; font-weight:bold; font-size:11px;"
                "  padding: 0 12px;"
                "}"
                "QPushButton:hover { background:#8B9BAA; }"
                "QPushButton:disabled{"
                "  background:#C8D4DC; color:#8B9BAA; border:1px solid #B0BEC5;"
                "}"
            )

    def _apply_preset_cta(self) -> None:
        """Standard CTA preset — clean isolated mesh, high specificity."""
        self._gauss_slider.setValue(85)
        self._pct_slider.setValue(75)
        self._min_r_spin.setValue(1.0)
        self._max_r_spin.setValue(20.0)
        self._min_pts_spin.setValue(8)
        self._pgf_spin.setValue(0.60)   # v4: strict — CTA domes have pgf 0.85+
        self._sph_spin.setValue(0.35)   # v5: moderate — CTA domes are mostly round
        self._pre_smooth_iters = 0      # CTA mesh already smooth — no pre-smoothing

    def _apply_preset_xa(self) -> None:
        """XA / 3DRA preset — full vascular tree, noisier mesh, with pre-smoothing."""
        from prospective.ui.icons import I as _I
        self._gauss_slider.setValue(60)
        self._pct_slider.setValue(40)
        self._min_r_spin.setValue(1.5)   # 3 mm diam minimum — filters noise spikes
        self._max_r_spin.setValue(20.0)
        self._min_pts_spin.setValue(4)
        self._pgf_spin.setValue(0.55)    # v6: raised back up — pre-smoothing makes
                                         #     real domes reach pgf 0.75–0.90,
                                         #     noise bumps drop to ~0.20–0.40
        self._sph_spin.setValue(0.25)    # v5: moderate — XA domes can appear oval
        self._pre_smooth_iters = 25      # v6: 25 Laplacian passes denoise surface
                                         #     before curvature (transparent to user)
        self._lbl_status.setText(
            f"{_I.HINT} Modo XA/3DRA — suavizado previo activo para mayor precisión."
        )

    def _apply_preset_max(self) -> None:
        """Maximum sensitivity — finds all curved regions (many false positives)."""
        self._gauss_slider.setValue(30)
        self._pct_slider.setValue(20)
        self._min_r_spin.setValue(0.3)
        self._max_r_spin.setValue(30.0)
        self._min_pts_spin.setValue(4)
        self._pgf_spin.setValue(0.20)   # v4: very permissive — expect many FPs
        self._sph_spin.setValue(0.10)   # v5: minimal shape restriction
        self._pre_smooth_iters = 15     # v6: light smoothing even in max-sensib mode

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #

    def _run_detection(self) -> None:
        if self._mesh is None:
            return

        detector = AneurysmDetector(
            gauss_percentile          = float(self._gauss_slider.value()),
            mean_curv_gate_percentile = float(self._pct_slider.value()),
            min_radius_mm             = self._min_r_spin.value(),
            max_radius_mm             = self._max_r_spin.value(),
            min_points                = self._min_pts_spin.value(),
            min_positive_gauss_frac   = self._pgf_spin.value(),   # v4 hard gate
            min_sphericity            = self._sph_spin.value(),   # v5 hard gate
            pre_smooth_iterations     = self._pre_smooth_iters,   # v6: denoise before curvature
        )

        self._btn_detect.setEnabled(False)
        self._btn_detect.setText("Analizando…")
        self._lbl_status.setText("Calculando curvatura gaussiana y media…")
        self._lbl_diag.setText("")
        self._table.setRowCount(0)
        self._set_action_buttons_enabled(False)

        self._worker = _DetectionWorker(detector, self._mesh, self)
        self._worker.finished.connect(self._on_detection_done)
        self._worker.error.connect(self._on_detection_error)
        self._worker.start()

    def _on_detection_done(self, result: DetectionResult) -> None:
        candidates = result.candidates
        self._candidates = candidates
        self._btn_detect.setEnabled(True)
        self._btn_detect.setText("Detectar candidatos")

        # Build diagnostic string — also include actionable guidance
        n_total   = result.n_regions_total
        n_size    = result.n_failed_size
        n_curv    = result.n_failed_mean_curv
        n_pgf     = result.n_failed_pgf
        n_compact = result.n_failed_compact
        n_sph     = result.n_failed_sphericity
        n_pts     = result.n_failed_points   # direct counter; avoids truncation artefacts
        n_noise   = result.n_removed_components
        diag_parts = [
            f"Componentes de ruido eliminados: {n_noise}"                 if n_noise > 0  else "",
            f"Regiones encontradas: {n_total}",
            f"  ↓ puntos insuficientes: {n_pts}"                          if n_pts > 0    else "",
            f"  ↓ tamaño fuera de rango: {n_size}"                        if n_size > 0   else "",
            f"  ↓ curvatura media baja: {n_curv}"                         if n_curv > 0   else "",
            f"  ↓ Gauss+ insuficiente (bifurcaciones): {n_pgf}"           if n_pgf > 0    else "",
            f"  ↓ compacidad insuficiente (segmentos): {n_compact}"       if n_compact > 0 else "",
            f"  ↓ esfericidad insuficiente (elongadas): {n_sph}"          if n_sph > 0    else "",
            f"  fusionados (duplicados): {result.n_merged}"                if result.n_merged > 0 else "",
            f"  → Candidatos finales: {len(candidates)}",
        ]
        self._lbl_diag.setText("\n".join(p for p in diag_parts if p))

        if not candidates:
            # Build targeted guidance based on which filter eliminated most
            dominant = max(
                ("tamaño",        n_size),
                ("curvatura",     n_curv),
                ("Gauss+ frac.",  n_pgf),
                ("compacidad",    n_compact),
                ("esfericidad",   n_sph),
                key=lambda t: t[1],
            )
            if n_pgf > 0 and n_pgf == dominant[1]:
                tip = (
                    "↓ Regiones eliminadas por Fracción Gauss+ insuficiente.\n"
                    "• Baje 'Frac. Gauss+ mín.' (ej. 0.35–0.40) para datos XA/ruidosos.\n"
                    "• Use el preajuste 'XA / 3DRA' o 'Máx. sensib.'."
                )
            elif n_sph > 0 and n_sph == dominant[1]:
                tip = (
                    "↓ Regiones eliminadas por esfericidad insuficiente.\n"
                    "• Baje 'Esfericidad mín.' (ej. 0.15–0.20) para candidatos más alargados.\n"
                    "• Use el preajuste 'XA / 3DRA' si trabaja con árbol vascular completo."
                )
            elif n_compact > 0 and n_compact == dominant[1]:
                tip = (
                    "↓ Regiones eliminadas por compacidad insuficiente.\n"
                    "• La malla puede contener muchos segmentos vasculares elongados.\n"
                    "• Use 'Máx. sensib.' para ver todas las regiones candidatas."
                )
            elif n_size > 0 and n_size >= n_curv:
                tip = (
                    "↓ Muchas regiones eliminadas por tamaño.\n"
                    "• Baje 'Radio mín.' (ej. 0.3–0.5 mm) para capturar bultos pequeños.\n"
                    "• Use el preajuste 'XA / 3DRA' si trabaja con árbol vascular completo."
                )
            elif n_curv > 0:
                tip = (
                    "↓ Regiones eliminadas por curvatura media baja.\n"
                    "• Baje 'Gate curv.M' (ej. p30–p40).\n"
                    "• Use el preajuste 'XA / 3DRA' o 'Máx. sensib.'."
                )
            else:
                tip = (
                    "No se encontraron zonas de curvatura elevada.\n"
                    "• Baje 'Umbral gauss.' a p40–p50.\n"
                    "• Verifique que la malla tiene polígonos (segmentación válida)."
                )
            self._lbl_status.setText(tip)
            return

        self._lbl_status.setText(f"{len(candidates)} candidato(s) encontrado(s).")
        self._populate_table(candidates)

    def _on_detection_error(self, msg: str) -> None:
        self._btn_detect.setEnabled(True)
        self._btn_detect.setText("Detectar candidatos")
        self._lbl_status.setText("")
        QMessageBox.critical(self, "Error de detección", msg)

    # ------------------------------------------------------------------ #
    # Table                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _candidate_type(c: AneurysmCandidate) -> tuple[str, str]:
        """
        Return (label, hex_color) describing the morphological type.

        Uses positive_gauss_frac (fraction of region vertices with Gauss > 0),
        sphericity and diameter.  Thresholds are relaxed for larger structures
        (≥ 3 mm) because XA/noisy meshes always have lower pgf even on real domes.

        Labels:
          "Saco"    (red)    — morphology consistent with aneurysm dome
          "Mixto"   (yellow) — could be aneurysm at a bifurcation
          "Bifurc." (grey)   — likely normal bifurcation or noise artifact
        """
        pgf  = c.positive_gauss_frac
        sph  = c.sphericity
        diam = c.diameter_mm

        # For larger structures lower the threshold — a smooth 5–10 mm dome on
        # a noisy XA mesh may have pgf 0.45–0.65 due to surface roughness,
        # while a clean CTA dome would have pgf 0.80–0.95.
        if diam >= 4.0:
            saco_pgf  = 0.50   # relaxed for clinically relevant size
            mixto_pgf = 0.35
        elif diam >= 2.5:
            saco_pgf  = 0.65
            mixto_pgf = 0.45
        else:
            saco_pgf  = 0.80   # strict for tiny fragments (CTA-only range)
            mixto_pgf = 0.55

        if pgf >= saco_pgf and sph >= 0.55:
            return "Saco",    "#f85149"   # red — aneurysm suspect
        elif pgf >= mixto_pgf or (diam >= 4.0 and sph >= 0.60):
            return "Mixto",   "#e3b341"   # yellow — worth reviewing
        else:
            return "Bifurc.", "#9B9B9B"   # grey — likely normal structure

    def _populate_table(self, candidates: list[AneurysmCandidate]) -> None:
        from PyQt5.QtGui import QColor, QFont
        bold_font = QFont()
        bold_font.setBold(True)

        self._table.setRowCount(len(candidates))
        for row, c in enumerate(candidates):
            tipo_label, tipo_color = self._candidate_type(c)
            is_clinical = c.diameter_mm >= 3.0   # clinically relevant size

            values = [
                (str(c.index),                             Qt.AlignCenter, ""),
                (f"{c.diameter_mm:.1f}",                   Qt.AlignCenter, ""),
                (f"{c.positive_gauss_frac * 100:.0f}%",   Qt.AlignCenter, ""),
                (f"{c.sphericity:.2f}",                    Qt.AlignCenter, ""),
                (f"{c.score * 100:.0f}%",                  Qt.AlignCenter, ""),
                (tipo_label,                               Qt.AlignCenter, tipo_color),
            ]
            for col, (text, align, color) in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(align)
                if color:
                    item.setForeground(QColor(color))
                # Bold row for clinically relevant diameter (≥ 3 mm)
                if is_clinical:
                    item.setFont(bold_font)
                self._table.setItem(row, col, item)

        # Apply active hide-bifurcations filter, then land on first visible row
        self._apply_bifurc_filter()
        self._select_first_visible_row()

    def _on_selection_changed(self) -> None:
        rows = self._table.selectedItems()
        if not rows:
            self._lbl_candidate_info.setText("—")
            self._set_action_buttons_enabled(False)
            self.candidate_highlighted.emit(None)
            return

        row = self._table.currentRow()
        if row < 0 or row >= len(self._candidates):
            return

        c = self._candidates[row]
        # Interpret positive_gauss_frac
        pgf = c.positive_gauss_frac
        pgf_interp = "saco" if pgf > 0.80 else ("mixto" if pgf > 0.55 else "bifurcación")
        # Interpret cv_gauss (curvature uniformity)
        cv = c.cv_gauss
        cv_interp = "uniforme" if cv < 0.8 else ("moderado" if cv < 1.5 else "variable")
        # Interpret normal_isotropy
        ni = c.normal_isotropy
        ni_interp = "hemisférico" if ni > 0.25 else ("moderado" if ni > 0.10 else "concentrado")
        self._lbl_candidate_info.setText(
            f"Candidato #{c.index}  |  Ø {c.diameter_mm:.1f} mm\n"
            f"Centro: ({c.centroid[0]:.1f}, {c.centroid[1]:.1f}, {c.centroid[2]:.1f}) mm\n"
            f"Curv. media: {c.mean_curvature:.4f}  |  Gauss: {c.gauss_curvature:.4f}\n"
            f"Gauss+: {pgf:.0%} ({pgf_interp})  |  CV Gauss: {cv:.2f} ({cv_interp})\n"
            f"Compacidad: {c.compactness:.2f}  |  Esfericidad: {c.sphericity:.2f}\n"
            f"Isotropía normal: {ni:.2f} ({ni_interp})  |  Puntos: {c.n_points}\n"
            f"Puntuación: {c.score * 100:.0f}%"
        )
        self._set_action_buttons_enabled(True)
        self.candidate_highlighted.emit(c)   # emit full AneurysmCandidate

    # ------------------------------------------------------------------ #
    # Actions                                                              #
    # ------------------------------------------------------------------ #

    def _on_isolate(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._candidates):
            return
        self.candidate_isolated.emit(self._candidates[row])

    def _on_confirm(self) -> None:
        """Emit candidate_confirmed — the signal that triggers morphometrics + step advance."""
        row = self._table.currentRow()
        if row < 0 or row >= len(self._candidates):
            return
        self.candidate_confirmed.emit(self._candidates[row])

    def _on_export(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._candidates):
            return
        self.export_requested.emit(self._candidates[row].poly_data)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _apply_bifurc_filter(self) -> None:
        """Show/hide 'Bifurc.' rows according to the checkbox, then fix selection.

        If the currently selected row becomes hidden, the selection moves to the
        first visible row so the action buttons always reflect a visible candidate.
        """
        hide = self._chk_hide_bifurc.isChecked()
        for row in range(self._table.rowCount()):
            tipo_item = self._table.item(row, 5)   # column 5 = Tipo
            if tipo_item is None:
                continue
            is_bifurc = tipo_item.text() == "Bifurc."
            self._table.setRowHidden(row, hide and is_bifurc)

        # If the active selection is now hidden, move to the first visible row
        cur = self._table.currentRow()
        if cur >= 0 and self._table.isRowHidden(cur):
            self._select_first_visible_row()

    def _select_first_visible_row(self) -> None:
        """Select the first non-hidden table row.

        If every row is hidden (or the table is empty), clears the selection and
        resets the info panel so no stale candidate remains "active".
        """
        for r in range(self._table.rowCount()):
            if not self._table.isRowHidden(r):
                self._table.selectRow(r)
                return
        # Nothing visible — reset to empty state
        self._table.clearSelection()
        self._lbl_candidate_info.setText("—")
        self._set_action_buttons_enabled(False)
        self.candidate_highlighted.emit(None)

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        self._btn_isolate.setEnabled(enabled)
        self._btn_export.setEnabled(enabled)
        self._btn_confirm.setEnabled(enabled)
        self._btn_visibility.setEnabled(enabled)
        if enabled:
            # Reset to "visible" state each time a new candidate is selected
            from prospective.ui.icons import I as _I
            self._btn_visibility.setChecked(False)
            self._btn_visibility.setText(f"{_I.EYE} Ocultar")

    def _on_toggle_visibility(self) -> None:
        """Toggle the magenta candidate highlight in the 3D viewer."""
        from prospective.ui.icons import I as _I
        hidden = self._btn_visibility.isChecked()
        self._btn_visibility.setText(
            f"{_I.EYE_HIDDEN} Mostrar" if hidden else f"{_I.EYE} Ocultar"
        )
        self.candidate_visibility_changed.emit(not hidden)
