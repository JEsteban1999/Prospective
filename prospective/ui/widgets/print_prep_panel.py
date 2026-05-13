"""3D-print preparation panel — Feature 7.

Workflow
--------
1. MainWindow calls :meth:`set_mesh` when a segmentation mesh is ready.
2. User tweaks parameters (target size, smoothing, etc.) and picks a print bed.
3. "Preparar" triggers background processing via :class:`_PrepWorker`.
4. Results: dimensions, volume, watertight indicator, warnings, bed-fit check.
5. "Exportar STL" writes the prepared mesh to disk.

Signals
-------
``prep_ready(object)``   — emitted with the :class:`PrintPrepResult` on success
``prep_failed(str)``     — emitted with an error message on failure
"""
from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QFileDialog, QMessageBox,
    QSizePolicy, QFormLayout, QTextEdit,
)

from prospective.processing.mesh_prep import (
    prepare_mesh_for_print,
    PrintPrepResult,
    PRINT_BED_PRESETS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# Background worker                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

class _PrepWorker(QObject):
    finished = pyqtSignal(object)   # PrintPrepResult
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, poly_data, kwargs: dict) -> None:
        super().__init__()
        self._poly_data = poly_data
        self._kwargs    = kwargs

    def run(self) -> None:
        try:
            result = prepare_mesh_for_print(
                self._poly_data,
                progress_cb=self.progress.emit,
                **self._kwargs,
            )
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("PrintPrep worker error")
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────── #
# Panel widget                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

class PrintPrepPanel(QWidget):
    """Dock panel for 3D-print mesh preparation."""

    prep_ready  = pyqtSignal(object)   # PrintPrepResult
    prep_failed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mesh:   Optional[object] = None   # vtkPolyData
        self._result: Optional[PrintPrepResult] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_PrepWorker] = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_mesh(self, poly_data) -> None:
        """Called by MainWindow when a segmentation mesh is available."""
        self._mesh = poly_data
        self._btn_prepare.setEnabled(poly_data is not None)
        self._btn_export.setEnabled(False)
        self._result = None
        self._clear_results()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Parameters group ─────────────────────────────────────────── #
        grp_params = QGroupBox("Parámetros de preparación")
        form = QFormLayout(grp_params)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(4)

        # Target size
        self._spin_size = QDoubleSpinBox()
        self._spin_size.setRange(0.0, 500.0)
        self._spin_size.setValue(80.0)
        self._spin_size.setSuffix(" mm")
        self._spin_size.setDecimals(1)
        self._spin_size.setToolTip(
            "Dimensión máxima del modelo impreso. 0 = sin escalar."
        )
        form.addRow("Tamaño máximo:", self._spin_size)

        # Smooth iterations
        self._spin_smooth = QSpinBox()
        self._spin_smooth.setRange(0, 100)
        self._spin_smooth.setValue(20)
        form.addRow("Iteraciones suavizado:", self._spin_smooth)

        # Relaxation factor
        self._spin_relax = QDoubleSpinBox()
        self._spin_relax.setRange(0.01, 1.0)
        self._spin_relax.setValue(0.1)
        self._spin_relax.setSingleStep(0.05)
        self._spin_relax.setDecimals(2)
        self._spin_relax.setToolTip("Factor de relajación Laplaciano (0.01–1.0)")
        form.addRow("Factor relajación:", self._spin_relax)

        # Fill holes
        self._chk_fill = QCheckBox("Rellenar agujeros")
        self._chk_fill.setChecked(True)
        self._chk_fill.toggled.connect(self._on_fill_toggled)
        form.addRow("", self._chk_fill)

        self._spin_hole = QDoubleSpinBox()
        self._spin_hole.setRange(0.5, 50.0)
        self._spin_hole.setValue(5.0)
        self._spin_hole.setSuffix(" mm")
        self._spin_hole.setDecimals(1)
        self._spin_hole.setToolTip("Perímetro máximo de agujero a rellenar")
        form.addRow("Tamaño máx. agujero:", self._spin_hole)

        # Subdivide
        self._chk_sub = QCheckBox("Subdivisión lineal (×4 triángulos)")
        self._chk_sub.setChecked(False)
        self._chk_sub.setToolTip(
            "Duplica la resolución. Útil para impresoras de alta resolución (SLA). "
            "Aumenta el tiempo de procesado."
        )
        form.addRow("", self._chk_sub)

        layout.addWidget(grp_params)

        # ── Print bed selector ────────────────────────────────────────── #
        grp_bed = QGroupBox("Cama de impresión")
        bed_layout = QVBoxLayout(grp_bed)
        bed_layout.setSpacing(4)

        self._combo_bed = QComboBox()
        for name in PRINT_BED_PRESETS:
            self._combo_bed.addItem(name)
        self._combo_bed.currentTextChanged.connect(self._on_bed_changed)
        bed_layout.addWidget(self._combo_bed)

        # Custom bed dimensions
        self._custom_bed_widget = QWidget()
        cbw_layout = QHBoxLayout(self._custom_bed_widget)
        cbw_layout.setContentsMargins(0, 0, 0, 0)
        cbw_layout.setSpacing(4)
        for attr, label in [("_spin_bed_x", "X:"), ("_spin_bed_y", "Y:"), ("_spin_bed_z", "Z:")]:
            cbw_layout.addWidget(QLabel(label))
            sp = QDoubleSpinBox()
            sp.setRange(0, 1000)
            sp.setValue(200)
            sp.setSuffix(" mm")
            sp.setDecimals(0)
            setattr(self, attr, sp)
            cbw_layout.addWidget(sp)
        self._custom_bed_widget.setVisible(False)
        bed_layout.addWidget(self._custom_bed_widget)

        # Bed fit label
        self._lbl_bed_fit = QLabel("")
        self._lbl_bed_fit.setAlignment(Qt.AlignCenter)
        bed_layout.addWidget(self._lbl_bed_fit)
        layout.addWidget(grp_bed)

        # ── Action buttons ────────────────────────────────────────────── #
        btn_row = QHBoxLayout()
        from prospective.ui.icons import I as _I
        self._btn_prepare = QPushButton(f"{_I.SETTINGS} Preparar malla")
        self._btn_prepare.setEnabled(False)
        self._btn_prepare.clicked.connect(self._on_prepare)
        self._btn_export = QPushButton(f"{_I.SAVE} Exportar STL")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(self._btn_prepare, 2)
        btn_row.addWidget(self._btn_export, 1)
        layout.addLayout(btn_row)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # ── Results group ─────────────────────────────────────────────── #
        grp_result = QGroupBox("Resultado")
        res_layout = QFormLayout(grp_result)
        res_layout.setLabelAlignment(Qt.AlignRight)
        res_layout.setSpacing(4)

        self._lbl_dims    = QLabel("—")
        self._lbl_scale   = QLabel("—")
        self._lbl_volume  = QLabel("—")
        self._lbl_surface = QLabel("—")
        self._lbl_wt      = QLabel("—")

        res_layout.addRow("Dimensiones:", self._lbl_dims)
        res_layout.addRow("Escala:",      self._lbl_scale)
        res_layout.addRow("Volumen:",     self._lbl_volume)
        res_layout.addRow("Superficie:",  self._lbl_surface)
        res_layout.addRow("Hermeticidad:", self._lbl_wt)
        layout.addWidget(grp_result)

        # Warnings text area
        self._txt_warnings = QTextEdit()
        self._txt_warnings.setReadOnly(True)
        self._txt_warnings.setMaximumHeight(80)
        self._txt_warnings.setPlaceholderText("Sin advertencias")
        self._txt_warnings.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self._txt_warnings)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _on_fill_toggled(self, checked: bool) -> None:
        self._spin_hole.setEnabled(checked)

    def _on_bed_changed(self, name: str) -> None:
        is_custom = (name == "Personalizado")
        self._custom_bed_widget.setVisible(is_custom)
        self._update_bed_fit()

    def _on_prepare(self) -> None:
        if self._mesh is None:
            return
        # Prevent double-start
        if self._thread and self._thread.isRunning():
            return

        kwargs = {
            "target_size_mm":     self._spin_size.value(),
            "smooth_iterations":  self._spin_smooth.value(),
            "smooth_relaxation":  self._spin_relax.value(),
            "fill_holes":         self._chk_fill.isChecked(),
            "hole_size":          self._spin_hole.value(),
            "subdivide":          self._chk_sub.isChecked(),
        }

        self._btn_prepare.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._clear_results()

        self._thread = QThread()
        self._worker = _PrepWorker(self._mesh, kwargs)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_prep_done)
        self._worker.error.connect(self._on_prep_error)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_prep_done(self, result: PrintPrepResult) -> None:
        self._result = result
        self._progress.setVisible(False)
        self._btn_prepare.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._fill_results(result)
        self._update_bed_fit()
        self.prep_ready.emit(result)

    def _on_prep_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._btn_prepare.setEnabled(True)
        QMessageBox.critical(self, "Error al preparar malla", msg)
        self.prep_failed.emit(msg)

    def _on_export(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar STL para impresión", "aneurysm_print.stl", "STL (*.stl)"
        )
        if not path:
            return
        try:
            self._result.export_stl(path)
            QMessageBox.information(self, "Exportado", f"Guardado en:\n{path}")
        except OSError as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _clear_results(self) -> None:
        for lbl in (self._lbl_dims, self._lbl_scale, self._lbl_volume,
                    self._lbl_surface, self._lbl_wt):
            lbl.setText("—")
        self._lbl_bed_fit.setText("")
        self._txt_warnings.clear()

    def _fill_results(self, r: PrintPrepResult) -> None:
        dx, dy, dz = r.dimensions_mm
        self._lbl_dims.setText(f"{dx:.1f} × {dy:.1f} × {dz:.1f} mm")
        self._lbl_scale.setText(f"×{r.scale_factor:.4f}")
        self._lbl_volume.setText(f"{r.volume_cm3:.3f} cm³")
        self._lbl_surface.setText(f"{r.surface_area_cm2:.2f} cm²")

        if r.is_watertight:
            self._lbl_wt.setText("✔ Hermética")
            self._lbl_wt.setStyleSheet("color: #3fb950;")
        else:
            self._lbl_wt.setText(f"✖ No hermética ({r.open_edge_count} bordes)")
            self._lbl_wt.setStyleSheet("color: #f85149;")

        if r.warnings:
            self._txt_warnings.setPlainText("\n".join(r.warnings))
        else:
            self._txt_warnings.setPlainText("")

    def _update_bed_fit(self) -> None:
        if self._result is None:
            self._lbl_bed_fit.setText("")
            return
        bed_name = self._combo_bed.currentText()
        if bed_name == "Personalizado":
            bed = (
                self._spin_bed_x.value(),
                self._spin_bed_y.value(),
                self._spin_bed_z.value(),
            )
        else:
            bed = PRINT_BED_PRESETS.get(bed_name, (0, 0, 0))

        fits = self._result.fits_in_bed(bed)
        if fits:
            self._lbl_bed_fit.setText("✔ Cabe en la cama seleccionada")
            self._lbl_bed_fit.setStyleSheet("color: #3fb950;")
        else:
            dx, dy, dz = self._result.dimensions_mm
            bx, by, bz = bed
            self._lbl_bed_fit.setText(
                f"✖ No cabe ({dx:.0f}×{dy:.0f}×{dz:.0f} > {bx:.0f}×{by:.0f}×{bz:.0f} mm)"
            )
            self._lbl_bed_fit.setStyleSheet("color: #f85149;")

    # ------------------------------------------------------------------ #
    # Session persistence                                                  #
    # ------------------------------------------------------------------ #

    def get_session_state(self) -> dict:
        """Return all preparation parameters as a serialisable dict."""
        return {
            "size":   self._spin_size.value(),
            "smooth": self._spin_smooth.value(),
            "relax":  self._spin_relax.value(),
            "fill":   self._chk_fill.isChecked(),
            "hole":   self._spin_hole.value(),
            "sub":    self._chk_sub.isChecked(),
            "bed":    self._combo_bed.currentText(),
        }

    def restore_session_state(self, state: dict) -> None:
        """Restore preparation parameters from a previously saved state dict."""
        if not state:
            return
        self._spin_size.setValue(float(state.get("size", 80.0)))
        self._spin_smooth.setValue(int(state.get("smooth", 20)))
        self._spin_relax.setValue(float(state.get("relax", 0.1)))
        self._chk_fill.setChecked(bool(state.get("fill", True)))
        self._spin_hole.setValue(float(state.get("hole", 5.0)))
        self._chk_sub.setChecked(bool(state.get("sub", False)))
        bed = state.get("bed", "")
        if bed:
            idx = self._combo_bed.findText(bed)
            if idx >= 0:
                self._combo_bed.setCurrentIndex(idx)
