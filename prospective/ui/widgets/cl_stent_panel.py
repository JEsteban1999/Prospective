"""Centreline-based stent deployment panel — Feature 4.

Workflow
--------
1. A :class:`CenterlineResult` is pushed in via :meth:`set_centerline`.
2. Two range spinboxes let the user pick start / end arc-length positions
   (dual handles on the arc range, with the total arc length as the maximum).
3. The user selects a stent diameter from the catalogue *or* enters a custom
   value; a coverage indicator shows stent_Ø / vessel_Ø.
4. "Desplegar stent" runs :func:`deploy_stent_on_centerline` in a background
   thread and emits ``stent_deployed(DeployedStentResult)`` when done.
5. "Retirar stent" removes the current deployment and resets the UI.

Signals
-------
``stent_deployed(object)``    — DeployedStentResult ready for rendering
``stent_retracted()``         — remove the deployed stent actor
"""
from __future__ import annotations

import logging

from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QDoubleSpinBox, QComboBox, QCheckBox,
    QProgressBar, QSizePolicy, QMessageBox, QSlider,
)

logger = logging.getLogger(__name__)


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


# Standard stent diameters (mm) shown in the quick-pick combo
_STENT_DIAMETERS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


# ──────────────────────────────────────────────────────────────────────────── #
# Background worker                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

class _DeployWorker(QObject):
    finished = pyqtSignal(object)   # DeployedStentResult
    error    = pyqtSignal(str)

    def __init__(self, centerline, diameter, start_arc, end_arc, braid):
        super().__init__()
        self._cl    = centerline
        self._diam  = diameter
        self._s0    = start_arc
        self._s1    = end_arc
        self._braid = braid

    @pyqtSlot()
    def run(self):
        try:
            from prospective.processing.stent_deployment import deploy_stent_on_centerline
            result = deploy_stent_on_centerline(
                self._cl,
                stent_diameter_mm=self._diam,
                start_arc_mm=self._s0,
                end_arc_mm=self._s1,
                braid=self._braid,
            )
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Stent deployment failed")
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────── #
# Panel                                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

class CLStentPanel(QWidget):
    """Dock panel for centreline-based stent deployment."""

    stent_deployed  = pyqtSignal(object)   # DeployedStentResult
    stent_retracted = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._centerline = None
        self._result     = None
        self._thread     = None
        self._worker     = None
        self._total_arc  = 0.0
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_centerline(self, centerline) -> None:
        """
        Called after centreline extraction completes.
        Updates arc-length spinbox ranges and the diameter recommendation.
        """
        self._centerline = centerline
        self._total_arc  = float(centerline.arc_length_mm)

        self._spin_start.setMaximum(self._total_arc)
        self._spin_start.setValue(0.0)
        self._spin_end.setMaximum(self._total_arc)
        self._spin_end.setValue(self._total_arc)

        # Recommend stent diameter = mean vessel diameter (from CL radii)
        mean_diam = centerline.mean_radius_mm * 2.0
        self._spin_diam.setValue(round(mean_diam * 2) / 2.0)  # round to 0.5 mm

        self._btn_deploy.setEnabled(True)
        self._update_coverage()

    def clear_centerline(self) -> None:
        self._centerline = None
        self._total_arc  = 0.0
        self._btn_deploy.setEnabled(False)
        self._on_retract()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        _mc = "#9B9B9B" if _is_dark() else "#6B6B6B"
        guide = QLabel(
            f"<small style='color:{_mc}'>"
            "Extrae primero la línea central; luego selecciona el segmento "
            "a cubrir y el diámetro del stent."
            "</small>"
        )
        guide.setWordWrap(True)
        layout.addWidget(guide)

        # ── Arc-length range ──────────────────────────────────────────── #
        grp_arc = QGroupBox("Segmento (mm de arco)")
        al = QVBoxLayout(grp_arc)
        al.setSpacing(4)

        row_start = QHBoxLayout()
        row_start.addWidget(QLabel("Inicio:"))
        self._spin_start = QDoubleSpinBox()
        self._spin_start.setRange(0.0, 999.0)
        self._spin_start.setValue(0.0)
        self._spin_start.setSingleStep(0.5)
        self._spin_start.setDecimals(1)
        self._spin_start.setSuffix(" mm")
        self._spin_start.valueChanged.connect(self._on_range_changed)
        row_start.addWidget(self._spin_start)
        al.addLayout(row_start)

        row_end = QHBoxLayout()
        row_end.addWidget(QLabel("Fin:"))
        self._spin_end = QDoubleSpinBox()
        self._spin_end.setRange(0.0, 999.0)
        self._spin_end.setValue(0.0)
        self._spin_end.setSingleStep(0.5)
        self._spin_end.setDecimals(1)
        self._spin_end.setSuffix(" mm")
        self._spin_end.valueChanged.connect(self._on_range_changed)
        row_end.addWidget(self._spin_end)
        al.addLayout(row_end)

        self._lbl_seg_len = QLabel("Longitud segmento: — mm")
        self._lbl_seg_len.setProperty("role", "muted")
        al.addWidget(self._lbl_seg_len)

        layout.addWidget(grp_arc)

        # ── Stent diameter ────────────────────────────────────────────── #
        grp_diam = QGroupBox("Diámetro del stent")
        dl = QVBoxLayout(grp_diam)
        dl.setSpacing(4)

        row_quick = QHBoxLayout()
        row_quick.addWidget(QLabel("Rápido:"))
        self._combo_diam = QComboBox()
        for d in _STENT_DIAMETERS:
            self._combo_diam.addItem(f"{d:.1f} mm", d)
        self._combo_diam.currentIndexChanged.connect(self._on_quick_diam)
        row_quick.addWidget(self._combo_diam, 1)
        dl.addLayout(row_quick)

        row_custom = QHBoxLayout()
        row_custom.addWidget(QLabel("Personalizado:"))
        self._spin_diam = QDoubleSpinBox()
        self._spin_diam.setRange(1.0, 12.0)
        self._spin_diam.setSingleStep(0.5)
        self._spin_diam.setValue(4.0)
        self._spin_diam.setDecimals(2)
        self._spin_diam.setSuffix(" mm")
        self._spin_diam.valueChanged.connect(self._update_coverage)
        row_custom.addWidget(self._spin_diam)
        dl.addLayout(row_custom)

        self._lbl_coverage = QLabel("Cobertura: —")
        self._lbl_coverage.setStyleSheet("font-weight:bold;")
        dl.addWidget(self._lbl_coverage)

        layout.addWidget(grp_diam)

        # ── Options ───────────────────────────────────────────────────── #
        grp_opt = QGroupBox("Opciones")
        ol = QVBoxLayout(grp_opt)
        self._chk_braid = QCheckBox("Mostrar trenzado metálico")
        self._chk_braid.setChecked(True)
        ol.addWidget(self._chk_braid)
        layout.addWidget(grp_opt)

        # ── Buttons ───────────────────────────────────────────────────── #
        row_btns = QHBoxLayout()
        from prospective.ui.icons import I as _I
        self._btn_deploy = QPushButton(f"{_I.STENT} Desplegar stent")
        self._btn_deploy.setEnabled(False)
        self._btn_deploy.clicked.connect(self._on_deploy)
        self._btn_retract = QPushButton("✕ Retirar")
        self._btn_retract.setEnabled(False)
        self._btn_retract.clicked.connect(self._on_retract)
        row_btns.addWidget(self._btn_deploy, 3)
        row_btns.addWidget(self._btn_retract, 1)
        layout.addLayout(row_btns)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        self._progress.setFormat("Generando malla…")
        layout.addWidget(self._progress)

        # ── Results ───────────────────────────────────────────────────── #
        grp_res = QGroupBox("Resultado")
        rl = QVBoxLayout(grp_res)
        rl.setSpacing(3)

        def _r(label):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            lbl = QLabel("—")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet("font-weight:bold;")
            h.addWidget(lbl)
            rl.addLayout(h)
            return lbl

        self._lbl_res_len    = _r("Longitud stent:")
        self._lbl_res_diam   = _r("Ø stent:")
        self._lbl_res_vessel = _r("Ø vaso medio:")
        self._lbl_res_cov    = _r("Cobertura:")

        layout.addWidget(grp_res)
        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _on_range_changed(self) -> None:
        s0 = self._spin_start.value()
        s1 = self._spin_end.value()
        if s1 < s0:
            # Enforce s1 >= s0
            self.sender().blockSignals(True)
            if self.sender() is self._spin_start:
                self._spin_end.setValue(s0)
            else:
                self._spin_start.setValue(s1)
            self.sender().blockSignals(False)
        length = abs(self._spin_end.value() - self._spin_start.value())
        self._lbl_seg_len.setText(f"Longitud segmento: {length:.1f} mm")
        self._update_coverage()

    def _on_quick_diam(self, idx: int) -> None:
        d = self._combo_diam.itemData(idx)
        if d is not None:
            self._spin_diam.setValue(float(d))

    def _update_coverage(self) -> None:
        if self._centerline is None:
            self._lbl_coverage.setText("Cobertura: —")
            return
        mean_vessel_r = self._centerline.mean_radius_mm
        stent_r       = self._spin_diam.value() / 2.0
        if mean_vessel_r < 1e-6:
            self._lbl_coverage.setText("Cobertura: —")
            return
        ratio = stent_r / mean_vessel_r
        pct   = ratio * 100.0
        if ratio < 0.80:
            color = "#f85149"
            label = f"⚠ Infradimensionado ({pct:.0f}%)"
        elif ratio > 1.15:
            color = "#d29922"
            label = f"⚠ Sobredimensionado ({pct:.0f}%)"
        else:
            color = "#3fb950"
            label = f"✔ Adecuado ({pct:.0f}%)"
        self._lbl_coverage.setText(
            f"<span style='color:{color}'>{label}</span>"
        )

    def _on_deploy(self) -> None:
        if self._centerline is None:
            return
        self._btn_deploy.setEnabled(False)
        self._btn_retract.setEnabled(False)
        self._progress.setVisible(True)

        self._thread = QThread()
        self._worker = _DeployWorker(
            self._centerline,
            self._spin_diam.value(),
            self._spin_start.value(),
            self._spin_end.value(),
            self._chk_braid.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    @pyqtSlot(object)
    def _on_done(self, result) -> None:
        self._result = result
        self._progress.setVisible(False)
        self._btn_deploy.setEnabled(True)
        self._btn_retract.setEnabled(True)
        self._fill_results(result)
        self.stent_deployed.emit(result)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._btn_deploy.setEnabled(self._centerline is not None)
        QMessageBox.critical(self, "Error — Despliegue de stent", msg)

    def _on_retract(self) -> None:
        self._result = None
        self._btn_retract.setEnabled(False)
        for lbl in (self._lbl_res_len, self._lbl_res_diam,
                    self._lbl_res_vessel, self._lbl_res_cov):
            lbl.setText("—")
        self.stent_retracted.emit()

    def _fill_results(self, result) -> None:
        self._lbl_res_len.setText(f"{result.length_mm:.1f} mm")
        self._lbl_res_diam.setText(f"{result.nominal_diameter_mm:.2f} mm")
        self._lbl_res_vessel.setText(f"{result.mean_vessel_diameter_mm:.2f} mm")
        r = result.coverage_ratio
        if r < 0.80:
            color = "#f85149"
        elif r > 1.15:
            color = "#d29922"
        else:
            color = "#3fb950"
        self._lbl_res_cov.setText(
            f"<span style='color:{color}'>{r:.2f}</span>"
        )
