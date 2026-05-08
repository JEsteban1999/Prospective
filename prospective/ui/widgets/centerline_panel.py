"""Centreline planning panel — Feature 1.

Workflow
--------
1. User clicks "Marcar origen" → planning window enters centerline-pick mode
   (first click on vessel = source point, highlighted in green).
2. User clicks "Marcar destino" → second click = target point (red).
3. Click "Extraer línea central" → background thread runs the EDT Dijkstra.
4. Results displayed: arc length, tortuosity, min/max/mean radius.
5. Centreline tube actor added to the 3D scene with radius colormap.
6. "Limpiar" removes the actor and resets the panel.

Signals emitted
---------------
``pick_source_requested()``      — planning window should enter pick mode for source
``pick_target_requested()``      — planning window should enter pick mode for target
``centerline_ready(object)``     — CenterlineResult (after successful extraction)
``centerline_cleared()``         — actor should be removed from scene
"""
from __future__ import annotations

import logging

from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QDoubleSpinBox, QSpinBox, QProgressBar, QSizePolicy,
    QMessageBox,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# Background workers                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

class _XSWorker(QObject):
    """Background worker for cross-section analysis."""
    progress = pyqtSignal(float)
    finished = pyqtSignal(object)   # CrossSectionResult
    error    = pyqtSignal(str)

    def __init__(self, centerline_result, poly_data, n_samples):
        super().__init__()
        self._cl      = centerline_result
        self._poly    = poly_data
        self._n       = n_samples

    @pyqtSlot()
    def run(self):
        try:
            from prospective.processing.cross_section import compute_cross_sections
            result = compute_cross_sections(
                self._cl, self._poly, self._n,
                progress_cb=lambda f: self.progress.emit(f),
            )
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Cross-section analysis failed")
            self.error.emit(str(exc))

class _Worker(QObject):
    progress = pyqtSignal(float)
    finished = pyqtSignal(object)    # CenterlineResult
    error    = pyqtSignal(str)

    def __init__(self, poly_data, source_mm, target_mm, voxel_size):
        super().__init__()
        self._poly   = poly_data
        self._src    = source_mm
        self._tgt    = target_mm
        self._vs     = voxel_size

    @pyqtSlot()
    def run(self):
        try:
            from prospective.processing.centerline import CenterlineExtractor
            extractor = CenterlineExtractor(
                voxel_size_mm=self._vs,
                progress_cb=lambda f: self.progress.emit(f),
            )
            result = extractor.extract(self._poly, self._src, self._tgt)
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("Centerline extraction failed")
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────── #
# Panel                                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

class CenterlinePanel(QWidget):
    """Dock panel for interactive centreline extraction."""

    pick_source_requested = pyqtSignal()
    pick_target_requested = pyqtSignal()
    centerline_ready      = pyqtSignal(object)   # CenterlineResult
    centerline_cleared    = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._source_mm: tuple | None = None
        self._target_mm: tuple | None = None
        self._vessel_poly = None
        self._result      = None
        self._thread      = None
        self._worker      = None

        # Cross-section state
        self._xs_result  = None
        self._xs_thread  = None
        self._xs_worker  = None

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API (called by planning window)                               #
    # ------------------------------------------------------------------ #

    def set_vessel_mesh(self, poly_data) -> None:
        self._vessel_poly = poly_data
        self._update_extract_btn()

    def set_source_point(self, pt: tuple) -> None:
        self._source_mm = pt
        self._lbl_src.setText(f"({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})")
        self._lbl_src.setStyleSheet("color: #3fb950;")   # green
        self._btn_src.setText("✔ Origen")
        self._update_extract_btn()

    def set_target_point(self, pt: tuple) -> None:
        self._target_mm = pt
        self._lbl_tgt.setText(f"({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})")
        self._lbl_tgt.setStyleSheet("color: #f85149;")   # red
        self._btn_tgt.setText("✔ Destino")
        self._update_extract_btn()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Guide label ───────────────────────────────────────────────── #
        guide = QLabel(
            "<small style='color:#9B9B9B'>"
            "Marca origen y destino sobre el vaso; la línea central "
            "sigue el eje medial por mínimo coste (radio máximo)."
            "</small>"
        )
        guide.setWordWrap(True)
        layout.addWidget(guide)

        # ── Endpoint pickers ──────────────────────────────────────────── #
        grp_pts = QGroupBox("Puntos extremos")
        gl = QVBoxLayout(grp_pts)
        gl.setSpacing(4)

        row_src = QHBoxLayout()
        from prospective.ui.icons import I as _I
        self._btn_src = QPushButton(f"{_I.SEED} Marcar origen")
        self._btn_src.setCheckable(True)
        self._btn_src.clicked.connect(self._on_pick_source)
        self._lbl_src = QLabel("—")
        self._lbl_src.setProperty("role", "muted")
        row_src.addWidget(self._btn_src)
        row_src.addWidget(self._lbl_src, 1)
        gl.addLayout(row_src)

        row_tgt = QHBoxLayout()
        self._btn_tgt = QPushButton(f"{_I.STEP_DETECT} Marcar destino")
        self._btn_tgt.setCheckable(True)
        self._btn_tgt.clicked.connect(self._on_pick_target)
        self._lbl_tgt = QLabel("—")
        self._lbl_tgt.setProperty("role", "muted")
        row_tgt.addWidget(self._btn_tgt)
        row_tgt.addWidget(self._lbl_tgt, 1)
        gl.addLayout(row_tgt)

        layout.addWidget(grp_pts)

        # ── Settings ──────────────────────────────────────────────────── #
        grp_cfg = QGroupBox("Configuración")
        cl = QHBoxLayout(grp_cfg)
        cl.addWidget(QLabel("Resolución (mm):"))
        self._spin_vs = QDoubleSpinBox()
        self._spin_vs.setRange(0.3, 2.0)
        self._spin_vs.setSingleStep(0.1)
        self._spin_vs.setValue(0.8)
        self._spin_vs.setDecimals(1)
        self._spin_vs.setToolTip(
            "Tamaño de vóxel para la voxelización.\n"
            "Menor = más preciso pero más lento.\n"
            "0.8 mm recomendado para vasculatura cerebral."
        )
        cl.addWidget(self._spin_vs)
        cl.addStretch()
        layout.addWidget(grp_cfg)

        # ── Extract / Clear buttons ───────────────────────────────────── #
        row_btns = QHBoxLayout()
        self._btn_extract = QPushButton("⚙  Extraer línea central")
        self._btn_extract.setEnabled(False)
        self._btn_extract.clicked.connect(self._on_extract)
        self._btn_clear = QPushButton("✕ Limpiar")
        self._btn_clear.setEnabled(False)
        self._btn_clear.clicked.connect(self._on_clear)
        row_btns.addWidget(self._btn_extract, 3)
        row_btns.addWidget(self._btn_clear, 1)
        layout.addLayout(row_btns)

        # ── Progress bar ──────────────────────────────────────────────── #
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Calculando… %p%")
        layout.addWidget(self._progress)

        # ── Results ───────────────────────────────────────────────────── #
        grp_res = QGroupBox("Resultados")
        rl = QVBoxLayout(grp_res)
        rl.setSpacing(3)

        def _row(label):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            lbl = QLabel("—")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet("font-weight: bold;")
            h.addWidget(lbl)
            rl.addLayout(h)
            return lbl

        self._lbl_arc      = _row("Longitud de arco:")
        self._lbl_chord    = _row("Longitud recta:")
        self._lbl_tort     = _row("Tortuosidad:")
        self._lbl_ti       = _row("Índice tortuosidad:")
        self._lbl_r_mean   = _row("Ø medio:")
        self._lbl_r_min    = _row("Ø mínimo:")
        self._lbl_r_max    = _row("Ø máximo:")

        layout.addWidget(grp_res)

        # ── Cross-section analysis ────────────────────────────────────── #
        grp_xs = QGroupBox("Sección transversal")
        xl = QVBoxLayout(grp_xs)
        xl.setSpacing(4)

        row_xs_cfg = QHBoxLayout()
        row_xs_cfg.addWidget(QLabel("Muestras:"))
        self._spin_xs = QSpinBox()
        self._spin_xs.setRange(10, 100)
        self._spin_xs.setValue(40)
        self._spin_xs.setToolTip("Número de planos de corte a lo largo de la línea central.")
        row_xs_cfg.addWidget(self._spin_xs)
        row_xs_cfg.addStretch()
        xl.addLayout(row_xs_cfg)

        row_xs_btn = QHBoxLayout()
        self._btn_xs = QPushButton(f"{_I.ANGLE_MEAS} Analizar secciones")
        self._btn_xs.setEnabled(False)
        self._btn_xs.clicked.connect(self._on_xs_analyze)
        self._btn_xs_clear = QPushButton("✕")
        self._btn_xs_clear.setFixedWidth(28)
        self._btn_xs_clear.setEnabled(False)
        self._btn_xs_clear.setToolTip("Limpiar análisis de secciones")
        self._btn_xs_clear.clicked.connect(self._on_xs_clear)
        row_xs_btn.addWidget(self._btn_xs, 1)
        row_xs_btn.addWidget(self._btn_xs_clear)
        xl.addLayout(row_xs_btn)

        self._xs_progress = QProgressBar()
        self._xs_progress.setRange(0, 100)
        self._xs_progress.setValue(0)
        self._xs_progress.setVisible(False)
        self._xs_progress.setFormat("Analizando… %p%")
        xl.addWidget(self._xs_progress)

        # Summary labels
        def _xs_row(label):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            lbl = QLabel("—")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet("font-weight: bold;")
            h.addWidget(lbl)
            xl.addLayout(h)
            return lbl

        self._lbl_xs_d_mean   = _xs_row("Ø medio:")
        self._lbl_xs_d_median = _xs_row("Ø mediana:")
        self._lbl_xs_d_min    = _xs_row("Ø mínimo:")
        self._lbl_xs_d_max    = _xs_row("Ø máximo:")
        self._lbl_xs_stenosis = _xs_row("Estenosis:")

        # Embedded chart
        from prospective.ui.widgets.cross_section_chart import CrossSectionChart
        self._xs_chart = CrossSectionChart(self)
        self._xs_chart.setMinimumHeight(160)
        xl.addWidget(self._xs_chart)

        layout.addWidget(grp_xs)
        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _on_pick_source(self, checked: bool) -> None:
        if checked:
            self._btn_tgt.setChecked(False)
            self.pick_source_requested.emit()
        else:
            # user un-toggled manually — nothing to do
            pass

    def _on_pick_target(self, checked: bool) -> None:
        if checked:
            self._btn_src.setChecked(False)
            self.pick_target_requested.emit()

    def _on_extract(self) -> None:
        if self._vessel_poly is None:
            QMessageBox.warning(self, "Sin malla", "Cargue primero la malla vascular.")
            return
        if self._source_mm is None or self._target_mm is None:
            QMessageBox.warning(self, "Puntos faltantes",
                                "Marque origen y destino antes de extraer.")
            return

        self._btn_extract.setEnabled(False)
        self._btn_clear.setEnabled(False)
        self._progress.setValue(0)
        self._progress.setVisible(True)

        # Run in background thread
        self._thread = QThread()
        self._worker = _Worker(
            self._vessel_poly,
            self._source_mm,
            self._target_mm,
            self._spin_vs.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_clear(self) -> None:
        self._result    = None
        self._source_mm = None
        self._target_mm = None
        self._lbl_src.setText("—"); self._lbl_src.setStyleSheet("")
        self._lbl_tgt.setText("—"); self._lbl_tgt.setStyleSheet("")
        self._btn_src.setText(f"{_I.SEED} Marcar origen");  self._btn_src.setChecked(False)
        self._btn_tgt.setText(f"{_I.STEP_DETECT} Marcar destino"); self._btn_tgt.setChecked(False)
        for lbl in (self._lbl_arc, self._lbl_chord, self._lbl_tort,
                    self._lbl_ti, self._lbl_r_mean, self._lbl_r_min, self._lbl_r_max):
            lbl.setText("—")
        self._progress.setVisible(False)
        self._btn_clear.setEnabled(False)
        self._update_extract_btn()
        self._on_xs_clear()
        self.centerline_cleared.emit()

    @pyqtSlot(float)
    def _on_progress(self, frac: float) -> None:
        self._progress.setValue(int(frac * 100))

    @pyqtSlot(object)
    def _on_done(self, result) -> None:
        self._result = result
        self._progress.setVisible(False)
        self._btn_clear.setEnabled(True)
        self._btn_xs.setEnabled(True)
        self._update_extract_btn()
        self._fill_results(result)
        self.centerline_ready.emit(result)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._update_extract_btn()
        QMessageBox.critical(self, "Error — Línea central", msg)

    # ------------------------------------------------------------------ #
    # Cross-section slots                                                  #
    # ------------------------------------------------------------------ #

    def _on_xs_analyze(self) -> None:
        if self._result is None or self._vessel_poly is None:
            return
        self._btn_xs.setEnabled(False)
        self._btn_xs_clear.setEnabled(False)
        self._xs_progress.setValue(0)
        self._xs_progress.setVisible(True)

        self._xs_thread = QThread()
        self._xs_worker = _XSWorker(
            self._result, self._vessel_poly, self._spin_xs.value()
        )
        self._xs_worker.moveToThread(self._xs_thread)
        self._xs_thread.started.connect(self._xs_worker.run)
        self._xs_worker.progress.connect(
            lambda f: self._xs_progress.setValue(int(f * 100))
        )
        self._xs_worker.finished.connect(self._on_xs_done)
        self._xs_worker.error.connect(self._on_xs_error)
        self._xs_worker.finished.connect(self._xs_thread.quit)
        self._xs_worker.error.connect(self._xs_thread.quit)
        self._xs_thread.start()

    @pyqtSlot(object)
    def _on_xs_done(self, result) -> None:
        self._xs_result = result
        self._xs_progress.setVisible(False)
        self._btn_xs.setEnabled(True)
        self._btn_xs_clear.setEnabled(True)
        self._fill_xs_results(result)
        self._xs_chart.plot(result)

    @pyqtSlot(str)
    def _on_xs_error(self, msg: str) -> None:
        self._xs_progress.setVisible(False)
        self._btn_xs.setEnabled(self._result is not None)
        QMessageBox.critical(self, "Error — Sección transversal", msg)

    def _on_xs_clear(self) -> None:
        self._xs_result = None
        self._btn_xs.setEnabled(self._result is not None)
        self._btn_xs_clear.setEnabled(False)
        self._xs_progress.setVisible(False)
        for lbl in (self._lbl_xs_d_mean, self._lbl_xs_d_median,
                    self._lbl_xs_d_min, self._lbl_xs_d_max,
                    self._lbl_xs_stenosis):
            lbl.setText("—")
        self._xs_chart.clear()

    def _fill_xs_results(self, result) -> None:
        self._lbl_xs_d_mean.setText(f"{result.mean_diameter_mm:.2f} mm")
        self._lbl_xs_d_median.setText(f"{result.median_diameter_mm:.2f} mm")
        self._lbl_xs_d_min.setText(f"{result.min_diameter_mm:.2f} mm")
        self._lbl_xs_d_max.setText(f"{result.max_diameter_mm:.2f} mm")
        ratio = result.stenosis_ratio
        pct   = (1.0 - ratio) * 100.0
        if pct < 20:
            color = "#3fb950"
            text  = f"<span style='color:{color}'>Sin estenosis ({pct:.0f}%)</span>"
        elif pct < 50:
            color = "#d29922"
            text  = f"<span style='color:{color}'>Leve ({pct:.0f}%)</span>"
        else:
            color = "#f85149"
            text  = f"<span style='color:{color}'>Significativa ({pct:.0f}%)</span>"
        self._lbl_xs_stenosis.setText(text)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _update_extract_btn(self) -> None:
        ready = (
            self._vessel_poly is not None
            and self._source_mm is not None
            and self._target_mm is not None
            and (self._thread is None or not self._thread.isRunning())
        )
        self._btn_extract.setEnabled(ready)

    def _fill_results(self, result) -> None:
        self._lbl_arc.setText(f"{result.arc_length_mm:.1f} mm")
        self._lbl_chord.setText(f"{result.chord_length_mm:.1f} mm")

        t = result.tortuosity
        color = "#f85149" if t >= 1.25 else "#d29922" if t >= 1.10 else "#3fb950"
        self._lbl_tort.setText(f"<span style='color:{color}'>{t:.3f}</span>")

        self._lbl_ti.setText(f"{result.tortuosity_index * 100:.1f} %")
        self._lbl_r_mean.setText(f"{result.mean_radius_mm * 2:.2f} mm")
        self._lbl_r_min.setText(f"{result.min_radius_mm  * 2:.2f} mm")
        self._lbl_r_max.setText(f"{result.max_radius_mm  * 2:.2f} mm")
