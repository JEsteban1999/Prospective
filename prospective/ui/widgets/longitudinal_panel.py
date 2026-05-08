"""Longitudinal comparison panel — Feature 6.

Workflow
--------
1. Each time a morphometric analysis completes, MainWindow can call
   :meth:`add_snapshot` to append a new time-point to the series.
2. The user can also manually import a CSV snapshot (previously exported).
3. The session table shows all snapshots with their date, label, key values,
   and a colour-coded risk badge.
4. Two combo boxes select the primary and optional secondary metric for the
   trend chart.
5. The "Exportar CSV" button writes the full series to disk.
6. A delta table below the chart shows absolute and percentage change between
   consecutive sessions.

Signal emitted
--------------
``snapshot_count_changed(int)``  — fired whenever the series grows/shrinks
"""
from __future__ import annotations

import logging
from datetime import date

from PyQt5.QtCore import Qt, QDate, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QDateEdit, QLineEdit,
    QFileDialog, QMessageBox, QSizePolicy, QSplitter,
)


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True

from prospective.processing.longitudinal import (
    LongitudinalSeries,
    MorphometricSnapshot,
    TRACKED_METRICS,
    METRIC_LABELS,
)

logger = logging.getLogger(__name__)


class LongitudinalPanel(QWidget):
    """Dock panel for longitudinal / multi-session morphometric comparison."""

    snapshot_count_changed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series = LongitudinalSeries()
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_snapshot(
        self,
        result,
        session_date: date | None = None,
        label: str = "",
    ) -> None:
        """
        Append a morphometric result as a new time-point.

        Parameters
        ----------
        result       : MorphometricResult
        session_date : date (defaults to today if None)
        label        : free-text session label
        """
        snap = MorphometricSnapshot.from_morpho_result(
            result,
            session_date=session_date,
            label=label or f"Sesión {len(self._series) + 1}",
        )
        self._series.add_snapshot(snap)
        self._refresh()

    def set_series(self, series: LongitudinalSeries) -> None:
        """Replace the current series (e.g. loaded from CSV)."""
        self._series = series
        self._refresh()

    def get_series(self) -> LongitudinalSeries:
        return self._series

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Top action bar ────────────────────────────────────────────── #
        top = QHBoxLayout()
        from prospective.ui.icons import I as _I
        self._btn_add_current = QPushButton(f"{_I.ANNOTATION} Añadir sesión actual")
        self._btn_add_current.setToolTip(
            "Agrega el resultado morfométrico actual como nueva sesión."
        )
        self._btn_add_current.setEnabled(False)
        self._btn_add_current.clicked.connect(self._on_add_current)

        self._btn_import = QPushButton(f"{_I.FOLDER} Importar CSV")
        self._btn_import.clicked.connect(self._on_import)

        self._btn_export = QPushButton("⬇ Exportar CSV")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)

        self._btn_clear = QPushButton("✕ Limpiar")
        self._btn_clear.clicked.connect(self._on_clear)

        top.addWidget(self._btn_add_current, 2)
        top.addWidget(self._btn_import, 1)
        top.addWidget(self._btn_export, 1)
        top.addWidget(self._btn_clear, 1)
        layout.addLayout(top)

        # ── Date + label for manual add ───────────────────────────────── #
        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("Fecha:"))
        self._date_edit = QDateEdit()
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setDate(QDate.currentDate())
        self._date_edit.setDisplayFormat("yyyy-MM-dd")
        date_row.addWidget(self._date_edit)
        date_row.addWidget(QLabel("Etiqueta:"))
        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("Ej: Baseline, 6 meses…")
        date_row.addWidget(self._label_edit, 2)
        layout.addLayout(date_row)

        # ── Splitter: session table (top) | chart + delta (bottom) ───── #
        splitter = QSplitter(Qt.Vertical)

        # Session table
        self._session_table = QTableWidget(0, 5)
        self._session_table.setHorizontalHeaderLabels([
            "Fecha", "Etiqueta", "Ø max (mm)", "AR", "Riesgo",
        ])
        self._session_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._session_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._session_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._session_table.verticalHeader().setVisible(False)
        self._session_table.setAlternatingRowColors(True)
        self._session_table.setMinimumHeight(80)
        splitter.addWidget(self._session_table)

        # Chart area
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(4)

        # Metric selectors
        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("Métrica 1:"))
        self._combo_m1 = QComboBox()
        for key in TRACKED_METRICS:
            self._combo_m1.addItem(METRIC_LABELS.get(key, key), key)
        self._combo_m1.currentIndexChanged.connect(self._on_metric_changed)
        metric_row.addWidget(self._combo_m1, 2)

        metric_row.addWidget(QLabel("Métrica 2:"))
        self._combo_m2 = QComboBox()
        self._combo_m2.addItem("(ninguna)", None)
        for key in TRACKED_METRICS:
            self._combo_m2.addItem(METRIC_LABELS.get(key, key), key)
        self._combo_m2.currentIndexChanged.connect(self._on_metric_changed)
        metric_row.addWidget(self._combo_m2, 2)
        chart_layout.addLayout(metric_row)

        # Chart
        from prospective.ui.widgets.longitudinal_chart import LongitudinalChart
        self._chart = LongitudinalChart(self)
        self._chart.setMinimumHeight(180)
        chart_layout.addWidget(self._chart)

        # Delta table
        grp_delta = QGroupBox("Cambios entre sesiones")
        dl = QVBoxLayout(grp_delta)
        self._delta_table = QTableWidget(0, 0)
        self._delta_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._delta_table.verticalHeader().setVisible(False)
        self._delta_table.setAlternatingRowColors(True)
        self._delta_table.setMinimumHeight(60)
        dl.addWidget(self._delta_table)
        chart_layout.addWidget(grp_delta)

        splitter.addWidget(chart_widget)
        splitter.setSizes([150, 350])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------ #
    # Current result cache (set by MainWindow)                             #
    # ------------------------------------------------------------------ #

    def set_current_result(self, result) -> None:
        """MainWindow calls this whenever a new morphometric result is ready."""
        self._current_result = result
        self._btn_add_current.setEnabled(result is not None)

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _on_add_current(self) -> None:
        result = getattr(self, "_current_result", None)
        if result is None:
            QMessageBox.information(
                self, "Sin datos",
                "Ejecuta el análisis morfométrico primero."
            )
            return
        qd    = self._date_edit.date()
        d     = date(qd.year(), qd.month(), qd.day())
        label = self._label_edit.text().strip()
        self.add_snapshot(result, session_date=d, label=label)
        self._label_edit.clear()

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Importar serie longitudinal", "", "CSV (*.csv)"
        )
        if not path:
            return
        series = LongitudinalSeries.from_csv(path)
        if not series.snapshots:
            QMessageBox.warning(
                self, "Importar CSV",
                "No se pudieron leer snapshots del archivo seleccionado."
            )
            return
        # Merge into current series
        for snap in series.snapshots:
            self._series.add_snapshot(snap)
        self._refresh()

    def _on_export(self) -> None:
        if not self._series.snapshots:
            QMessageBox.information(self, "Sin datos", "No hay sesiones para exportar.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar serie longitudinal", "longitudinal.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._series.export_csv(path)
            QMessageBox.information(self, "Exportado", f"Guardado en:\n{path}")
        except OSError as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))

    def _on_clear(self) -> None:
        if not self._series.snapshots:
            return
        reply = QMessageBox.question(
            self, "Limpiar serie",
            "¿Eliminar todas las sesiones?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._series.clear()
            self._refresh()

    def _on_metric_changed(self) -> None:
        self._update_chart()

    # ------------------------------------------------------------------ #
    # Refresh helpers                                                      #
    # ------------------------------------------------------------------ #

    def _refresh(self) -> None:
        self._update_session_table()
        self._update_chart()
        self._update_delta_table()
        n = len(self._series)
        self._btn_export.setEnabled(n > 0)
        self.snapshot_count_changed.emit(n)

    def _update_session_table(self) -> None:
        snaps = self._series.snapshots
        self._session_table.setRowCount(len(snaps))
        for row, snap in enumerate(snaps):
            # Semantic risk badge colours — slightly muted on light theme so
            # they don't clash with the white table background.
            if _is_dark():
                risk_colors = {
                    "Bajo":     (QColor("#3fb950"), QColor("#000000")),
                    "Moderado": (QColor("#d29922"), QColor("#000000")),
                    "Alto":     (QColor("#f85149"), QColor("#ffffff")),
                }
            else:
                risk_colors = {
                    "Bajo":     (QColor(195, 235, 200), QColor("#0d4a1a")),
                    "Moderado": (QColor(255, 235, 185), QColor("#5a3a00")),
                    "Alto":     (QColor(255, 210, 210), QColor("#7a0f0f")),
                }
            rc, tc = risk_colors.get(snap.rupture_risk, (QColor("#888888"), QColor("#ffffff")))

            def _item(text, center=True):
                it = QTableWidgetItem(str(text))
                if center:
                    it.setTextAlignment(Qt.AlignCenter)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                return it

            self._session_table.setItem(row, 0, _item(snap.session_date.isoformat()))
            self._session_table.setItem(row, 1, _item(snap.label, center=False))
            self._session_table.setItem(row, 2, _item(f"{snap.max_diameter_mm:.2f}"))
            self._session_table.setItem(row, 3, _item(f"{snap.aspect_ratio:.3f}"))
            risk_item = _item(snap.rupture_risk)
            risk_item.setBackground(rc)
            risk_item.setForeground(tc)
            self._session_table.setItem(row, 4, risk_item)

    def _update_chart(self) -> None:
        m1 = self._combo_m1.currentData()
        m2 = self._combo_m2.currentData()
        if not m1:
            self._chart.clear()
            return
        self._chart.plot(self._series, primary_metric=m1,
                         secondary_metric=m2 if m2 else None)

    def _update_delta_table(self) -> None:
        snaps = self._series.snapshots
        n     = len(snaps)
        if n < 2:
            self._delta_table.setRowCount(0)
            self._delta_table.setColumnCount(0)
            return

        metrics = TRACKED_METRICS
        # Columns: one per transition ("S1→S2", "S2→S3", …)
        cols = [f"S{i}→S{i+1}" for i in range(1, n)]
        self._delta_table.setColumnCount(len(cols))
        self._delta_table.setHorizontalHeaderLabels(cols)
        self._delta_table.setRowCount(len(metrics))
        self._delta_table.setVerticalHeaderLabels(
            [METRIC_LABELS.get(k, k) for k in metrics]
        )
        self._delta_table.verticalHeader().setVisible(True)
        self._delta_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._delta_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

        from PyQt5.QtGui import QColor
        for row_i, key in enumerate(metrics):
            deltas = self._series.deltas(key)
            pcts   = self._series.percent_change(key)
            for col_i, (delta, pct) in enumerate(zip(deltas, pcts)):
                text = f"{delta:+.2f}\n({pct:+.1f}%)"
                it   = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignCenter)
                it.setFlags(Qt.ItemIsEnabled)
                if abs(delta) < 1e-9:
                    color = QColor(54, 54, 54, 180) if _is_dark() else QColor(210, 215, 220, 120)
                elif delta > 0:
                    # Growth → red tint
                    color = QColor(45, 26, 26, 200) if _is_dark() else QColor(255, 220, 220, 180)
                else:
                    # Reduction → green tint
                    color = QColor(26, 45, 26, 200) if _is_dark() else QColor(210, 240, 215, 180)
                it.setBackground(color)
                self._delta_table.setItem(row_i, col_i, it)
