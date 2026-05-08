"""Perforator risk panel — automated detection of at-risk perforating arteries.

Analyses the full vessel mesh around the aneurysm neck to:
  • Map three concentric risk zones (high / medium / low) on the surface
  • Detect branching-point candidates via vertex-valence anomaly detection
  • Emit colour-coded VTK actors to the planning renderer

Signals
-------
overlay_ready(list)   — list[vtkActor] to add to the planning renderer
overlay_cleared()     — remove previously emitted actors from renderer
"""
from __future__ import annotations

import logging

import vtk
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from prospective.processing.perforator_risk import (
    PerforatorCandidate,
    PerforatorRiskResult,
    compute_perforator_risk,
    neck_origin_from_morpho,
)
from prospective.rendering.perforator_risk_actor import build_risk_actors

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────── #
# Constants                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

_COL_IDX  = 0
_COL_DIST = 1
_COL_RISK = 2
_COL_POS  = 3
_HEADERS  = ["#", "Dist. (mm)", "Riesgo", "Posición (mm)"]

_ZONE_HEX = {1: "#f85149", 2: "#e3b341", 3: "#3fb950"}


class PerforatorRiskPanel(QWidget):
    """
    Self-contained panel for perforator-artery risk detection.

    Usage (MainWindow wiring)::

        panel.set_vessel_mesh(poly_data)          # after segmentation
        panel.set_morpho_data(result, aneurysm_poly)  # after morphometrics
        panel.overlay_ready.connect(planning_win.set_perforator_overlay)
        panel.overlay_cleared.connect(planning_win.clear_perforator_overlay)
    """

    overlay_ready   = pyqtSignal(list)   # list[vtkActor]
    overlay_cleared = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._vessel_poly:   vtk.vtkPolyData | None  = None
        self._morpho_result                          = None
        self._aneurysm_poly: vtk.vtkPolyData | None  = None
        self._result:        PerforatorRiskResult | None = None
        self._actors:        list | None             = None

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_vessel_mesh(self, poly_data: vtk.vtkPolyData | None) -> None:
        """Call from MainWindow when segmentation produces a vessel mesh."""
        self._vessel_poly = poly_data
        self._update_compute_btn()

    def set_morpho_data(
        self,
        result,
        aneurysm_poly: vtk.vtkPolyData | None,
    ) -> None:
        """Call from MainWindow after morphometric analysis completes."""
        self._morpho_result  = result
        self._aneurysm_poly  = aneurysm_poly
        self._update_compute_btn()

        if result is not None:
            cx, cy, cz = result.centroid
            self._lbl_status.setText(
                f"Cuello: Ø {result.neck_diameter_mm:.1f} mm  ·  "
                f"centroide ({cx:.1f}, {cy:.1f}, {cz:.1f})"
            )
            self._lbl_status.setStyleSheet("")
        else:
            self._lbl_status.setText("Sin morfometría disponible.")
            self._lbl_status.setStyleSheet("")

    # ------------------------------------------------------------------ #
    # UI construction                                                       #
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

        # ── Status ───────────────────────────────────────────────────── #
        self._lbl_status = QLabel(
            "Realice primero la segmentación y el análisis morfométrico."
        )
        self._lbl_status.setProperty("role", "muted")
        self._lbl_status.setWordWrap(True)
        layout.addWidget(self._lbl_status)

        # ── Zone radii ───────────────────────────────────────────────── #
        radii_grp = QGroupBox("Radios de zona de riesgo (mm)")
        rf = QFormLayout()
        rf.setLabelAlignment(Qt.AlignRight)
        rf.setSpacing(4)

        self._spin_high = _spin(1.0, 20.0, 3.0, 0.5,
                                "Zona de riesgo ALTO: distancia ≤ este valor desde el cuello.")
        self._spin_med  = _spin(1.0, 30.0, 5.0, 0.5,
                                "Zona de riesgo MEDIO.")
        self._spin_low  = _spin(1.0, 40.0, 8.0, 0.5,
                                "Zona de riesgo BAJO (límite exterior del mapa de colores).")

        rf.addRow(
            "<span style='color:#f85149'>●</span> Riesgo alto (&lt;):",
            self._spin_high,
        )
        rf.addRow(
            "<span style='color:#e3b341'>●</span> Riesgo medio (&lt;):",
            self._spin_med,
        )
        rf.addRow(
            "<span style='color:#3fb950'>●</span> Riesgo bajo (&lt;):",
            self._spin_low,
        )
        # Use rich-text labels via QLabel rows
        for i in range(rf.rowCount()):
            lbl = rf.labelForField(
                rf.itemAt(i, QFormLayout.FieldRole).widget()
            )
            if lbl:
                lbl.setTextFormat(Qt.RichText)

        radii_grp.setLayout(rf)
        layout.addWidget(radii_grp)

        # ── Action buttons ───────────────────────────────────────────── #
        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(6)

        from prospective.ui.icons import I as _I
        self._btn_compute = QPushButton(f"{_I.STEP_DETECT} Detectar perforantes")
        self._btn_compute.setMinimumHeight(30)
        self._btn_compute.setObjectName("btn_primary")
        self._btn_compute.setEnabled(False)
        self._btn_compute.clicked.connect(self._on_compute)
        br.addWidget(self._btn_compute)

        self._btn_clear = QPushButton("Limpiar")
        self._btn_clear.setObjectName("btn_muted")
        self._btn_clear.setEnabled(False)
        self._btn_clear.clicked.connect(self._on_clear)
        br.addWidget(self._btn_clear)

        layout.addWidget(btn_row)

        # ── Result summary ───────────────────────────────────────────── #
        self._lbl_result = QLabel("—")
        self._lbl_result.setProperty("role", "muted")
        self._lbl_result.setWordWrap(True)
        self._lbl_result.setTextFormat(Qt.RichText)
        layout.addWidget(self._lbl_result)

        # ── Candidates table ─────────────────────────────────────────── #
        tbl_grp = QGroupBox("Candidatos detectados")
        tbl_lay = QVBoxLayout()
        tbl_lay.setContentsMargins(4, 4, 4, 4)

        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMaximumHeight(200)
        tbl_lay.addWidget(self._table)
        tbl_grp.setLayout(tbl_lay)
        layout.addWidget(tbl_grp)

        # ── Legend ───────────────────────────────────────────────────── #
        legend = QLabel(
            "<span style='color:#f85149'>■</span> Alto riesgo &nbsp;"
            "<span style='color:#e3b341'>■</span> Riesgo medio &nbsp;"
            "<span style='color:#3fb950'>■</span> Riesgo bajo &nbsp;"
            "<span style='color:#506080'>■</span> Sin riesgo"
        )
        legend.setStyleSheet("font-size:10px;")
        legend.setTextFormat(Qt.RichText)
        layout.addWidget(legend)

        # ── Disclaimer ───────────────────────────────────────────────── #
        disc = QLabel(
            "⚠ Detección automática experimental basada en análisis "
            "topológico de la malla vascular. La identificación final de "
            "arterias perforantes requiere revisión neuroquirúrgica experta."
        )
        disc.setProperty("role", "muted")
        disc.setStyleSheet("font-style:italic;")
        disc.setWordWrap(True)
        layout.addWidget(disc)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _on_compute(self) -> None:
        if self._vessel_poly is None or self._morpho_result is None:
            return

        r_high = self._spin_high.value()
        r_med  = self._spin_med.value()
        r_low  = self._spin_low.value()

        if not (r_high < r_med < r_low):
            self._lbl_result.setText(
                "⚠ Los radios deben ser estrictamente crecientes: "
                "alto &lt; medio &lt; bajo."
            )
            return

        # Clear previous overlay before computing new one
        if self._actors is not None:
            self.overlay_cleared.emit()
            self._actors = None

        try:
            if self._aneurysm_poly is not None:
                neck_pt = neck_origin_from_morpho(
                    self._morpho_result, self._aneurysm_poly
                )
            else:
                neck_pt = tuple(float(v) for v in self._morpho_result.centroid)

            result = compute_perforator_risk(
                vessel_poly = self._vessel_poly,
                neck_origin = neck_pt,
                zone_radii  = (r_high, r_med, r_low),
            )
        except Exception as exc:
            logger.exception("Perforator risk computation failed")
            self._lbl_result.setText(f"<span style='color:#f85149'>Error: {exc}</span>")
            return

        self._result = result
        self._populate_table(result.candidates)

        n      = len(result.candidates)
        n_high = sum(1 for c in result.candidates if c.risk_level == 1)
        n_med  = sum(1 for c in result.candidates if c.risk_level == 2)
        n_low  = sum(1 for c in result.candidates if c.risk_level == 3)

        if n == 0:
            self._lbl_result.setText(
                "No se detectaron candidatos en la zona de búsqueda. "
                "Pruebe a aumentar los radios de zona."
            )
            self._lbl_result.setStyleSheet("")
        else:
            self._lbl_result.setText(
                f"{n} candidato(s): "
                f"<span style='color:#f85149'>{n_high} alto</span> &nbsp;"
                f"<span style='color:#e3b341'>{n_med} medio</span> &nbsp;"
                f"<span style='color:#3fb950'>{n_low} bajo</span>"
            )
            self._lbl_result.setStyleSheet("font-size:10px;")

        try:
            actors_named = build_risk_actors(result)
            self._actors = list(actors_named)
        except Exception as exc:
            logger.exception("Failed to build perforator actors")
            self._lbl_result.setText(
                f"<span style='color:#f85149'>Error al construir actores: {exc}</span>"
            )
            self._actors = []

        self._btn_clear.setEnabled(True)
        self.overlay_ready.emit(self._actors or [])

    def _on_clear(self) -> None:
        self._result = None
        self._actors = None
        self._table.setRowCount(0)
        self._lbl_result.setText("—")
        self._lbl_result.setStyleSheet("")
        self._btn_clear.setEnabled(False)
        self.overlay_cleared.emit()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _update_compute_btn(self) -> None:
        ready = (
            self._vessel_poly is not None
            and self._morpho_result is not None
        )
        self._btn_compute.setEnabled(ready)

    def _populate_table(self, candidates: list[PerforatorCandidate]) -> None:
        self._table.setRowCount(0)
        for c in candidates:
            row = self._table.rowCount()
            self._table.insertRow(row)

            idx_item  = QTableWidgetItem(f"P{c.index + 1}")
            dist_item = QTableWidgetItem(f"{c.distance_to_neck_mm:.1f}")
            risk_item = QTableWidgetItem(c.risk_label)
            pos_item  = QTableWidgetItem(
                f"({c.position[0]:.1f}, {c.position[1]:.1f}, {c.position[2]:.1f})"
            )

            color_hex = _ZONE_HEX.get(c.risk_level, "#9B9B9B")
            risk_item.setForeground(QColor(color_hex))

            for col, item in enumerate([idx_item, dist_item, risk_item, pos_item]):
                item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(row, col, item)

        self._table.resizeColumnsToContents()

    # ------------------------------------------------------------------ #
    # Session state                                                         #
    # ------------------------------------------------------------------ #

    def get_session_state(self) -> dict:
        return {
            "r_high": self._spin_high.value(),
            "r_med":  self._spin_med.value(),
            "r_low":  self._spin_low.value(),
        }

    def restore_session_state(self, d: dict) -> None:
        self._spin_high.setValue(float(d.get("r_high", 3.0)))
        self._spin_med.setValue(float(d.get("r_med",  5.0)))
        self._spin_low.setValue(float(d.get("r_low",  8.0)))


# ──────────────────────────────────────────────────────────────────────────── #
# Module helper                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

def _spin(
    lo: float,
    hi: float,
    val: float,
    step: float,
    tip: str,
) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(val)
    sb.setSingleStep(step)
    sb.setSuffix(" mm")
    sb.setToolTip(tip)
    return sb
