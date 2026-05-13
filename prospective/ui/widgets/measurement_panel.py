"""Measurement / caliper panel — Feature 3.

Workflow
--------
1. User clicks "Nueva medición" → planning window enters *ruler-pick mode*
   (cursor changes, next two clicks on the 3D scene define endpoints A and B).
2. After both points are picked the ruler actor is added to the scene and
   the measurement appears as a row in the list.
3. Each row shows: index, distance, optional label, visibility toggle,
   and a delete button.
4. "Exportar CSV" dumps all measurements to a comma-separated file.

Signals emitted
---------------
``ruler_requested()``              — planning window should enter ruler-pick mode
``ruler_visibility(int, bool)``    — show/hide ruler index *n*
``ruler_deleted(int)``             — remove ruler index *n* from scene
``ruler_label_changed(int, str)``  — rename ruler *n*
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from typing import Any

from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QLineEdit, QCheckBox, QFileDialog,
    QMessageBox, QSizePolicy,
)

logger = logging.getLogger(__name__)


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True



# ──────────────────────────────────────────────────────────────────────────── #
# Data record                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class Measurement:
    """A single caliper measurement."""
    index:    int
    pt_a:     tuple[float, float, float]
    pt_b:     tuple[float, float, float]
    distance: float          # mm, pre-computed
    label:    str   = ""
    visible:  bool  = True


# ──────────────────────────────────────────────────────────────────────────── #
# Panel                                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

class MeasurementPanel(QWidget):
    """Dock panel for interactive 3-D measurements."""

    ruler_requested      = pyqtSignal()
    ruler_visibility     = pyqtSignal(int, bool)    # index, visible
    ruler_deleted        = pyqtSignal(int)           # index
    ruler_label_changed  = pyqtSignal(int, str)      # index, new_label

    _COL_IDX   = 0
    _COL_DIST  = 1
    _COL_LABEL = 2
    _COL_VIS   = 3
    _COL_DEL   = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._measurements: list[Measurement] = []
        self._next_index = 1
        self._picking = False
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_measurement(
        self,
        pt_a: tuple[float, float, float],
        pt_b: tuple[float, float, float],
        distance: float,
        label: str = "",
    ) -> int:
        """
        Register a completed measurement and update the list.

        Returns the assigned index.
        """
        idx = self._next_index
        self._next_index += 1
        m = Measurement(idx, pt_a, pt_b, distance, label or f"M{idx}")
        self._measurements.append(m)
        self._append_row(m)
        self._update_summary()
        logger.info("Measurement %d added: %.2f mm", idx, distance)

        # Reset pick button state
        from prospective.ui.icons import I as _I
        self._btn_new.setChecked(False)
        self._btn_new.setText(f"{_I.MEASURE} Nueva medición")
        self._picking = False
        return idx

    def set_picking(self, picking: bool) -> None:
        """Called by planning window to sync button state."""
        from prospective.ui.icons import I as _I
        self._picking = picking
        self._btn_new.setChecked(picking)
        self._btn_new.setText(
            f"{_I.WAIT} Haz clic en el punto A…" if picking else f"{_I.MEASURE} Nueva medición"
        )

    def get_measurements(self) -> list[Measurement]:
        return list(self._measurements)

    def clear_all(self) -> None:
        for m in list(self._measurements):
            self.ruler_deleted.emit(m.index)
        self._measurements.clear()
        self._table.setRowCount(0)
        self._update_summary()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Guide ─────────────────────────────────────────────────────── #
        _mc = "#9B9B9B" if _is_dark() else "#6B6B6B"
        guide = QLabel(
            f"<small style='color:{_mc}'>"
            "Haz clic en <b>Nueva medición</b>, luego selecciona dos puntos "
            "en la escena 3D. La distancia euclídea se calcula automáticamente."
            "</small>"
        )
        guide.setWordWrap(True)
        layout.addWidget(guide)

        # ── Action buttons ─────────────────────────────────────────────── #
        row_btns = QHBoxLayout()
        from prospective.ui.icons import I as _I
        self._btn_new = QPushButton(f"{_I.MEASURE} Nueva medición")
        self._btn_new.setCheckable(True)
        self._btn_new.setToolTip("Inicia una medición de distancia — haz clic en dos puntos de la escena 3D")
        self._btn_new.clicked.connect(self._on_new)
        self._btn_clear = QPushButton("✕ Limpiar todo")
        self._btn_clear.setToolTip("Elimina todas las mediciones de la sesión")
        self._btn_clear.clicked.connect(self._on_clear_all)
        self._btn_export = QPushButton("⬇ CSV")
        self._btn_export.setToolTip("Exportar mediciones a CSV")
        self._btn_export.clicked.connect(self._on_export)
        row_btns.addWidget(self._btn_new, 3)
        row_btns.addWidget(self._btn_export, 1)
        row_btns.addWidget(self._btn_clear, 1)
        layout.addLayout(row_btns)

        # ── Table ──────────────────────────────────────────────────────── #
        self._table = QTableWidget(0, 5)
        self._table.installEventFilter(self)
        self._table.setHorizontalHeaderLabels(["#", "Distancia", "Etiqueta", _I.EYE, "✕"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(100)
        layout.addWidget(self._table)

        # ── Summary ────────────────────────────────────────────────────── #
        grp_sum = QGroupBox("Resumen")
        sl = QVBoxLayout(grp_sum)
        sl.setSpacing(3)

        def _s_row(label):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            lbl = QLabel("—")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet("font-weight:bold;")
            h.addWidget(lbl)
            sl.addLayout(h)
            return lbl

        self._lbl_count = _s_row("Mediciones:")
        self._lbl_min   = _s_row("Mínima:")
        self._lbl_max   = _s_row("Máxima:")
        self._lbl_mean  = _s_row("Media:")

        layout.addWidget(grp_sum)
        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _on_new(self, checked: bool) -> None:
        from prospective.ui.icons import I as _I
        if checked:
            self._picking = True
            self._btn_new.setText(f"{_I.WAIT} Haz clic en el punto A…")
            self.ruler_requested.emit()
        else:
            self._picking = False
            self._btn_new.setText(f"{_I.MEASURE} Nueva medición")

    def _on_clear_all(self) -> None:
        if not self._measurements:
            return
        reply = QMessageBox.question(
            self, "Limpiar mediciones",
            "¿Eliminar todas las mediciones?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.clear_all()

    def _on_export(self) -> None:
        if not self._measurements:
            QMessageBox.information(self, "Sin datos", "No hay mediciones para exportar.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar mediciones", "mediciones.csv",
            "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["#", "Etiqueta", "xA", "yA", "zA",
                             "xB", "yB", "zB", "Distancia_mm"])
                for m in self._measurements:
                    w.writerow([
                        m.index, m.label,
                        f"{m.pt_a[0]:.3f}", f"{m.pt_a[1]:.3f}", f"{m.pt_a[2]:.3f}",
                        f"{m.pt_b[0]:.3f}", f"{m.pt_b[1]:.3f}", f"{m.pt_b[2]:.3f}",
                        f"{m.distance:.4f}",
                    ])
            QMessageBox.information(self, "Exportado", f"Guardado en:\n{path}")
        except OSError as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))

    # ------------------------------------------------------------------ #
    # Event filter (keyboard Delete on table)                              #
    # ------------------------------------------------------------------ #

    def eventFilter(self, obj: object, event: object) -> bool:
        """Delete key on the measurements table removes the selected row."""
        if obj is self._table and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self._delete_current_row()
                return True
        return super().eventFilter(obj, event)

    def _delete_current_row(self) -> None:
        """Remove the currently selected measurement row via keyboard."""
        row = self._table.currentRow()
        if row < 0:
            return
        item = self._table.item(row, self._COL_IDX)
        if item is None:
            return
        idx = int(item.text())
        self._measurements = [m for m in self._measurements if m.index != idx]
        self._table.removeRow(row)
        self._update_summary()
        self.ruler_deleted.emit(idx)

    # ------------------------------------------------------------------ #
    # Table helpers                                                         #
    # ------------------------------------------------------------------ #

    def _append_row(self, m: Measurement) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        # #
        idx_item = QTableWidgetItem(str(m.index))
        idx_item.setTextAlignment(Qt.AlignCenter)
        idx_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self._table.setItem(row, self._COL_IDX, idx_item)

        # Distance
        dist_item = QTableWidgetItem(f"{m.distance:.2f} mm")
        dist_item.setTextAlignment(Qt.AlignCenter)
        dist_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self._table.setItem(row, self._COL_DIST, dist_item)

        # Editable label
        label_item = QTableWidgetItem(m.label)
        label_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
        self._table.setItem(row, self._COL_LABEL, label_item)
        self._table.itemChanged.connect(self._on_label_changed)

        # Visibility checkbox
        vis_widget = QWidget()
        vis_layout = QHBoxLayout(vis_widget)
        vis_layout.setAlignment(Qt.AlignCenter)
        vis_layout.setContentsMargins(0, 0, 0, 0)
        chk = QCheckBox()
        chk.setChecked(True)
        chk.setProperty("ruler_index", m.index)
        chk.toggled.connect(self._on_vis_toggled)
        vis_layout.addWidget(chk)
        self._table.setCellWidget(row, self._COL_VIS, vis_widget)

        # Delete button
        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(28)
        del_btn.setProperty("ruler_index", m.index)
        del_btn.clicked.connect(self._on_delete_clicked)
        self._table.setCellWidget(row, self._COL_DEL, del_btn)

    def _on_label_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != self._COL_LABEL:
            return
        row = item.row()
        idx_item = self._table.item(row, self._COL_IDX)
        if idx_item is None:
            return
        ruler_idx = int(idx_item.text())
        new_label = item.text()
        for m in self._measurements:
            if m.index == ruler_idx:
                m.label = new_label
                break
        self.ruler_label_changed.emit(ruler_idx, new_label)

    def _on_vis_toggled(self, checked: bool) -> None:
        chk = self.sender()
        if chk is None:
            return
        idx = chk.property("ruler_index")
        if idx is not None:
            self.ruler_visibility.emit(int(idx), checked)

    def _on_delete_clicked(self) -> None:
        btn = self.sender()
        if btn is None:
            return
        idx = btn.property("ruler_index")
        if idx is None:
            return
        idx = int(idx)
        # Remove from internal list
        self._measurements = [m for m in self._measurements if m.index != idx]
        # Remove row
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._COL_IDX)
            if item and int(item.text()) == idx:
                self._table.removeRow(row)
                break
        self._update_summary()
        self.ruler_deleted.emit(idx)

    def _update_summary(self) -> None:
        ms = self._measurements
        if not ms:
            for lbl in (self._lbl_count, self._lbl_min, self._lbl_max, self._lbl_mean):
                lbl.setText("—")
            return
        dists = [m.distance for m in ms]
        self._lbl_count.setText(str(len(ms)))
        self._lbl_min.setText(f"{min(dists):.2f} mm")
        self._lbl_max.setText(f"{max(dists):.2f} mm")
        self._lbl_mean.setText(f"{sum(dists)/len(dists):.2f} mm")
