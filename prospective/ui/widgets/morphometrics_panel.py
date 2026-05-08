"""Morphometric analysis panel — F-03.

Receives an isolated aneurysm vtkPolyData (from AneurysmPanel),
runs MorphometricAnalyzer in a background thread, and displays
all clinical metrics with colour-coded risk indicators.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import vtk
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from prospective.processing.morphometrics import MorphometricAnalyzer, MorphometricResult
from prospective.ui.widgets.phases_panel import PHASESPanel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# Background worker                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

class _AnalysisWorker(QThread):
    finished = pyqtSignal(object)   # MorphometricResult
    error    = pyqtSignal(str)

    def __init__(self, poly_data: vtk.vtkPolyData, parent=None) -> None:
        super().__init__(parent)
        self._poly_data = poly_data
        self._analyzer  = MorphometricAnalyzer()

    def run(self) -> None:
        try:
            result = self._analyzer.analyze(self._poly_data)
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Morphometric analysis failed")
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────── #
# Colour helpers                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

_RISK_STYLE = {
    "Bajo":     "color:#56d364; font-weight:bold;",
    "Moderado": "color:#e3b341; font-weight:bold;",
    "Alto":     "color:#f85149; font-weight:bold;",
}


def _value_label(text: str, style: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    # Must expand horizontally so QFormLayout gives the field column all remaining
    # space regardless of the initial placeholder text ("—").  Without this the
    # column is sized to the sizeHint of "—" (≈15 px) and real values get clipped.
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    if style:
        lbl.setStyleSheet(style)
    return lbl


# ──────────────────────────────────────────────────────────────────────────── #
# Panel                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

class MorphometricsPanel(QWidget):
    """
    Dock panel that shows morphometric measurements for a selected
    aneurysm candidate.

    Usage
    -----
    Call set_mesh(poly_data) with the isolated aneurysm mesh to trigger
    analysis.  Results are displayed immediately after the background
    computation completes.

    Signals
    -------
    analysis_done(MorphometricResult) — emitted when analysis completes
    """

    analysis_done = pyqtSignal(object)   # MorphometricResult

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: MorphometricResult | None = None
        self._worker: _AnalysisWorker | None = None
        self._pending_mesh: vtk.vtkPolyData | None = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_mesh(self, poly_data: vtk.vtkPolyData | None) -> None:
        """
        Load a mesh for analysis.  Does NOT run analysis automatically —
        the user must press "Analizar" (or call analyze()).

        Pass None to clear the panel.
        """
        if poly_data is None:
            self._pending_mesh = None
            self._clear_values()
            self._btn_analyze.setEnabled(False)
            self._lbl_status.setText("Sin malla cargada.")
            return

        self._pending_mesh = poly_data
        n_pts  = poly_data.GetNumberOfPoints()
        n_tris = poly_data.GetNumberOfPolys()
        self._clear_values()
        self._btn_analyze.setEnabled(True)
        self._lbl_status.setText(
            f"Malla lista ({n_pts:,} verts · {n_tris:,} tris).\n"
            "Pulse Analizar para calcular las métricas."
        )

    def analyze(self) -> None:
        """Run the morphometric analysis on the currently loaded mesh."""
        if self._pending_mesh is None:
            return
        self._btn_analyze.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._lbl_status.setText("Calculando métricas…")

        self._worker = _AnalysisWorker(self._pending_mesh, self)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ── Analizar button + status ──────────────────────────────────── #
        self._btn_analyze = QPushButton("Analizar malla")
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.setMinimumHeight(30)
        self._btn_analyze.setObjectName("btn_success")
        self._btn_analyze.clicked.connect(self.analyze)
        outer.addWidget(self._btn_analyze)

        self._lbl_status = QLabel("Cargue una segmentación o seleccione un candidato.")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        self._lbl_status.setProperty("role", "muted")
        self._lbl_status.setWordWrap(True)
        outer.addWidget(self._lbl_status)

        # ── Scrollable metrics area ───────────────────────────────────── #
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content = QWidget()
        vl = QVBoxLayout(content)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(6)

        # ── Volumetric group ─────────────────────────────────────────── #
        vol_grp = QGroupBox("Volumetría")
        vf = QFormLayout()
        vf.setLabelAlignment(Qt.AlignLeft)
        vf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._lbl_volume      = _value_label("—")
        self._lbl_area        = _value_label("—")
        self._lbl_eq_diam     = _value_label("—")
        vf.addRow("Volumen:", self._lbl_volume)
        vf.addRow("Área sup.:", self._lbl_area)
        vf.addRow("Ø esfera eq.:", self._lbl_eq_diam)
        vol_grp.setLayout(vf)
        vl.addWidget(vol_grp)

        # ── Geometry group ───────────────────────────────────────────── #
        geom_grp = QGroupBox("Geometría")
        gf = QFormLayout()
        gf.setLabelAlignment(Qt.AlignLeft)
        gf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._lbl_max_diam    = _value_label("—")
        self._lbl_neck_diam   = _value_label("—")
        self._lbl_dome_h      = _value_label("—")
        self._lbl_bbox        = _value_label("—")
        gf.addRow("Ø máximo:", self._lbl_max_diam)
        gf.addRow("Ø cuello:", self._lbl_neck_diam)
        gf.addRow("Altura domo:", self._lbl_dome_h)
        gf.addRow("Caja (L×W×H):", self._lbl_bbox)
        geom_grp.setLayout(gf)
        vl.addWidget(geom_grp)

        # ── Clinical ratios group ────────────────────────────────────── #
        ratio_grp = QGroupBox("Índices clínicos")
        rf = QFormLayout()
        rf.setLabelAlignment(Qt.AlignLeft)
        rf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._lbl_dnr     = _value_label("—")
        self._lbl_ar      = _value_label("—")
        self._lbl_compact = _value_label("—")
        self._lbl_risk    = _value_label("—")
        rf.addRow("Domo/cuello (DNR):", self._lbl_dnr)
        rf.addRow("Aspect ratio (AR):", self._lbl_ar)
        rf.addRow("Esfericidad (Wadell):", self._lbl_compact)
        rf.addRow("Riesgo ruptura:", self._lbl_risk)
        ratio_grp.setLayout(rf)
        vl.addWidget(ratio_grp)

        # ── Shape-complexity indices group (Ankyras-style) ───────────── #
        shape_grp = QGroupBox("Índices de forma")
        sf = QFormLayout()
        sf.setLabelAlignment(Qt.AlignLeft)
        sf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._lbl_bf  = _value_label("—")
        self._lbl_ui  = _value_label("—")
        self._lbl_ei  = _value_label("—")
        self._lbl_nsi = _value_label("—")
        sf.addRow("Bottleneck Factor (BF):", self._lbl_bf)
        sf.addRow("Undulation Index (UI):", self._lbl_ui)
        sf.addRow("Ellipticity Index (EI):", self._lbl_ei)
        sf.addRow("Non-Sphericity (NSI):", self._lbl_nsi)
        shape_grp.setLayout(sf)
        vl.addWidget(shape_grp)

        # ── Size Ratio (requires parent artery diameter) ─────────────── #
        sr_grp = QGroupBox("Size Ratio (SR)")
        srl = QVBoxLayout()
        srl.setSpacing(4)
        sr_form = QFormLayout()
        sr_form.setLabelAlignment(Qt.AlignLeft)
        sr_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._spin_parent_diam = QDoubleSpinBox()
        self._spin_parent_diam.setRange(0.0, 20.0)
        self._spin_parent_diam.setDecimals(2)
        self._spin_parent_diam.setSuffix(" mm")
        self._spin_parent_diam.setSpecialValueText("—")
        self._spin_parent_diam.setValue(0.0)
        self._spin_parent_diam.setToolTip(
            "Diámetro de la arteria madre en la zona del cuello.\n"
            "SR = Ø_máx / Ø_arteria.  SR ≥ 3.0 → riesgo alto."
        )
        self._spin_parent_diam.valueChanged.connect(self._on_parent_diam_changed)
        self._lbl_sr = _value_label("—")
        sr_form.addRow("Ø arteria madre:", self._spin_parent_diam)
        sr_form.addRow("Size Ratio (SR):", self._lbl_sr)
        srl.addLayout(sr_form)
        sr_grp.setLayout(srl)
        vl.addWidget(sr_grp)

        # ── Risk legend ──────────────────────────────────────────────── #
        legend = QLabel(
            "<small style='color:#9B9B9B'>"
            "DNR &gt;1.6 / AR &gt;1.3 → riesgo moderado<br>"
            "DNR &gt;2.0 / AR &gt;1.6 / UI &gt;0.25 → riesgo alto<br>"
            "BF &gt;1.5 → cuello ancho (considerar stent)<br>"
            "SR ≥ 2.0 → moderado · SR ≥ 3.0 → alto<br>"
            "<i style='color:#9B9B9B'>Heurístico — no es diagnóstico clínico</i>"
            "</small>"
        )
        legend.setWordWrap(True)
        vl.addWidget(legend)

        # ── PHASES score calculator ───────────────────────────────────── #
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        try:
            from prospective.ui.themes import is_dark as _is_dark_morpho
            _sep_clr = "#363636" if _is_dark_morpho() else "#D8D8D8"
        except Exception:
            _sep_clr = "#363636"
        sep.setStyleSheet(
            f"QFrame{{border:none;border-top:1px solid {_sep_clr};margin-top:4px;}}"
        )
        vl.addWidget(sep)

        self._phases_panel = PHASESPanel()
        vl.addWidget(self._phases_panel)

        vl.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

        # ── Export button ────────────────────────────────────────────── #
        self._btn_export = QPushButton("Exportar informe CSV")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_csv)
        outer.addWidget(self._btn_export)

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _on_done(self, result: MorphometricResult) -> None:
        self._result = result
        self._btn_analyze.setEnabled(True)
        self._lbl_status.setText("Análisis completado.")
        self._populate(result)
        self._btn_export.setEnabled(True)
        self._phases_panel.set_max_diameter(result.max_diameter_mm)
        self.analysis_done.emit(result)

    def _on_error(self, msg: str) -> None:
        self._btn_analyze.setEnabled(self._pending_mesh is not None)
        self._lbl_status.setText("")
        QMessageBox.critical(self, "Error morfométrico", msg)

    # ------------------------------------------------------------------ #
    # Populate / clear                                                     #
    # ------------------------------------------------------------------ #

    def _populate(self, r: MorphometricResult) -> None:
        self._lbl_volume.setText(f"{r.volume_mm3:.1f} mm³")
        self._lbl_area.setText(f"{r.surface_area_mm2:.1f} mm²")
        self._lbl_eq_diam.setText(f"{r.eq_sphere_diam_mm:.2f} mm")

        self._lbl_max_diam.setText(f"{r.max_diameter_mm:.2f} mm")
        self._lbl_neck_diam.setText(f"{r.neck_diameter_mm:.2f} mm")
        self._lbl_dome_h.setText(f"{r.dome_height_mm:.2f} mm")
        self._lbl_bbox.setText(
            f"{r.bbox_l_mm:.1f} × {r.bbox_w_mm:.1f} × {r.bbox_h_mm:.1f} mm"
        )

        # ── DNR / AR colour-coding ────────────────────────────────────── #
        dnr_style = (
            "color:#f85149; font-weight:bold;" if r.dome_to_neck_ratio >= 2.0
            else "color:#e3b341; font-weight:bold;" if r.dome_to_neck_ratio >= 1.6
            else ""
        )
        ar_style = (
            "color:#f85149; font-weight:bold;" if r.aspect_ratio >= 1.6
            else "color:#e3b341; font-weight:bold;" if r.aspect_ratio >= 1.3
            else ""
        )
        self._lbl_dnr.setText(f"{r.dome_to_neck_ratio:.3f}")
        self._lbl_dnr.setStyleSheet(dnr_style)
        self._lbl_ar.setText(f"{r.aspect_ratio:.3f}")
        self._lbl_ar.setStyleSheet(ar_style)
        self._lbl_compact.setText(f"{r.compactness:.4f}")
        self._lbl_risk.setText(r.rupture_risk_label)
        self._lbl_risk.setStyleSheet(_RISK_STYLE.get(r.rupture_risk_label, ""))

        # ── Shape-complexity indices ──────────────────────────────────── #
        bf_style = (
            "color:#f85149; font-weight:bold;" if r.bottleneck_factor >= 2.0
            else "color:#e3b341; font-weight:bold;" if r.bottleneck_factor >= 1.5
            else ""
        )
        ui_style = (
            "color:#f85149; font-weight:bold;" if r.undulation_index >= 0.25
            else "color:#e3b341; font-weight:bold;" if r.undulation_index >= 0.10
            else ""
        )
        self._lbl_bf.setText(f"{r.bottleneck_factor:.3f}")
        self._lbl_bf.setStyleSheet(bf_style)
        self._lbl_ui.setText(f"{r.undulation_index:.4f}")
        self._lbl_ui.setStyleSheet(ui_style)
        self._lbl_ei.setText(f"{r.ellipticity_index:.4f}")
        self._lbl_ei.setStyleSheet("")
        self._lbl_nsi.setText(f"{r.non_sphericity_idx:.4f}")
        self._lbl_nsi.setStyleSheet("")

        # ── Size Ratio (recompute with current parent diam) ───────────── #
        self._update_sr_display()

    def _on_parent_diam_changed(self, value: float) -> None:
        """Recompute SR instantly when the user enters the parent artery diam."""
        if self._result is not None:
            self._result.size_ratio = (
                self._result.max_diameter_mm / value if value >= 0.5 else 0.0
            )
            self._update_sr_display()
            # Refresh risk label (SR may change classification)
            self._lbl_risk.setText(self._result.rupture_risk_label)
            self._lbl_risk.setStyleSheet(
                _RISK_STYLE.get(self._result.rupture_risk_label, "")
            )

    def _update_sr_display(self) -> None:
        if self._result is None or self._result.size_ratio <= 0:
            self._lbl_sr.setText("—")
            self._lbl_sr.setStyleSheet("")
            return
        sr = self._result.size_ratio
        sr_style = (
            "color:#f85149; font-weight:bold;" if sr >= 3.0
            else "color:#e3b341; font-weight:bold;" if sr >= 2.0
            else ""
        )
        self._lbl_sr.setText(f"{sr:.3f}")
        self._lbl_sr.setStyleSheet(sr_style)

    def _clear_values(self) -> None:
        for lbl in (
            self._lbl_volume, self._lbl_area, self._lbl_eq_diam,
            self._lbl_max_diam, self._lbl_neck_diam, self._lbl_dome_h,
            self._lbl_bbox, self._lbl_dnr, self._lbl_ar, self._lbl_compact,
            self._lbl_risk, self._lbl_bf, self._lbl_ui, self._lbl_ei,
            self._lbl_nsi, self._lbl_sr,
        ):
            lbl.setText("—")
            lbl.setStyleSheet("")
        self._result = None
        self._btn_export.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def _export_csv(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar informe morfométrico", "morfometria_aneurisma.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        rows = self._result.to_dict()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Métrica", "Valor"])
            for k, v in rows.items():
                writer.writerow([k, v])
        logger.info("Morphometric report exported: %s", path)
        QMessageBox.information(
            self, "Exportación completada",
            f"Informe guardado en:\n{path}",
        )
