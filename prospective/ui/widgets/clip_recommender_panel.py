"""Clip selection assistant panel — Feature 8.

Displays a ranked table of clip recommendations based on the current
morphometric analysis.  The user can select any row and send the clip
directly to the Clip Planning panel.

Signal emitted
--------------
``clip_selected(object)``  — emits the :class:`ClipSpec` chosen by the user
"""
from __future__ import annotations

import logging

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSizePolicy, QComboBox, QSpinBox,
    QFormLayout, QTextEdit,
)

from prospective.processing.clip_recommender import (
    ClipRecommendation,
    recommend_from_morpho,
    recommend_clips,
    WIDE_NECK_THRESHOLD_MM,
    DEEP_DOME_AR_THRESHOLD,
)
from prospective.models.clip_library import ClipShape, CLIP_CATALOGUE

logger = logging.getLogger(__name__)

# Score colour thresholds
_COLOR_EXCELLENT = "#3fb950"   # green
_COLOR_GOOD      = "#d29922"   # amber
_COLOR_ACCEPTABLE= "#A8B8C6"   # violet
_COLOR_MARGINAL  = "#888888"   # grey


def _score_color(score: float) -> str:
    if score >= 75:
        return _COLOR_EXCELLENT
    if score >= 55:
        return _COLOR_GOOD
    if score >= 35:
        return _COLOR_ACCEPTABLE
    return _COLOR_MARGINAL


class ClipRecommenderPanel(QWidget):
    """
    Panel showing ranked clip recommendations for the current aneurysm.
    """

    clip_selected = pyqtSignal(object)   # ClipSpec

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._morpho_result = None
        self._recommendations: list[ClipRecommendation] = []
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_morpho_result(self, result) -> None:
        """
        Called by MainWindow when a morphometric analysis completes.
        Triggers automatic re-scoring.
        """
        self._morpho_result = result
        self._update_context_labels(result)
        self._run_recommendation()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Context summary ───────────────────────────────────────────── #
        grp_ctx = QGroupBox("Morfometría actual")
        ctx_form = QFormLayout(grp_ctx)
        ctx_form.setLabelAlignment(Qt.AlignRight)
        ctx_form.setSpacing(3)
        self._lbl_neck   = QLabel("—")
        self._lbl_ar     = QLabel("—")
        self._lbl_ctx    = QLabel("—")   # wide-neck / deep-dome / standard
        ctx_form.addRow("Cuello (mm):", self._lbl_neck)
        ctx_form.addRow("Aspect Ratio:", self._lbl_ar)
        ctx_form.addRow("Contexto:", self._lbl_ctx)
        layout.addWidget(grp_ctx)

        # ── Filter controls ───────────────────────────────────────────── #
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Resultados:"))
        self._spin_n = QSpinBox()
        self._spin_n.setRange(1, 20)
        self._spin_n.setValue(8)
        self._spin_n.valueChanged.connect(self._run_recommendation)
        filter_row.addWidget(self._spin_n)
        filter_row.addStretch()
        self._lbl_count = QLabel("0 clips compatibles")
        filter_row.addWidget(self._lbl_count)
        layout.addLayout(filter_row)

        # ── Recommendations table ─────────────────────────────────────── #
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            "Puntuación", "Clip", "Forma", "Hoja (mm)", "Cobertura", "Fabricante",
        ])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (0, 2, 3, 4, 5):
            self._table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(160)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

        # ── Detail / reasons box ──────────────────────────────────────── #
        grp_detail = QGroupBox("Detalles del clip seleccionado")
        dl = QVBoxLayout(grp_detail)
        self._txt_reasons = QTextEdit()
        self._txt_reasons.setReadOnly(True)
        self._txt_reasons.setMaximumHeight(80)
        self._txt_reasons.setPlaceholderText("Selecciona un clip para ver detalles…")
        dl.addWidget(self._txt_reasons)
        layout.addWidget(grp_detail)

        # ── Action button ─────────────────────────────────────────────── #
        btn_row = QHBoxLayout()
        from prospective.ui.icons import I as _I
        self._btn_send = QPushButton(f"{_I.CLIPS} Enviar clip al planificador")
        self._btn_send.setEnabled(False)
        self._btn_send.setToolTip(
            "Envía el clip seleccionado al panel de planificación de clips."
        )
        self._btn_send.clicked.connect(self._on_send)
        btn_row.addWidget(self._btn_send)
        layout.addLayout(btn_row)

        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Slots / helpers                                                       #
    # ------------------------------------------------------------------ #

    def _update_context_labels(self, result) -> None:
        neck = result.neck_diameter_mm
        ar   = result.aspect_ratio
        self._lbl_neck.setText(f"{neck:.2f} mm")
        self._lbl_ar.setText(f"{ar:.3f}")

        if neck >= WIDE_NECK_THRESHOLD_MM:
            ctx = "Cuello ancho → fenestrado preferido"
            self._lbl_ctx.setStyleSheet("color: #f85149;")
        elif ar >= DEEP_DOME_AR_THRESHOLD:
            ctx = "Domo profundo (AR alto) → angulado/bayoneta"
            self._lbl_ctx.setStyleSheet("color: #d29922;")
        else:
            ctx = "Estándar → recto/curvo"
            self._lbl_ctx.setStyleSheet("color: #3fb950;")
        self._lbl_ctx.setText(ctx)

    def _run_recommendation(self) -> None:
        if self._morpho_result is None:
            return
        n = self._spin_n.value()
        self._recommendations = recommend_from_morpho(self._morpho_result, n=n)
        self._populate_table()

    def _populate_table(self) -> None:
        recs = self._recommendations
        self._table.setRowCount(len(recs))
        self._lbl_count.setText(f"{len(recs)} clips compatibles")

        for row, rec in enumerate(recs):
            color = _score_color(rec.score)

            def _item(text: str, center: bool = True) -> QTableWidgetItem:
                it = QTableWidgetItem(str(text))
                if center:
                    it.setTextAlignment(Qt.AlignCenter)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                return it

            # Score badge with colour
            score_it = _item(f"{rec.score:.0f}  {rec.score_label}")
            score_it.setForeground(QColor(color))
            bold = QFont()
            bold.setBold(True)
            score_it.setFont(bold)
            self._table.setItem(row, 0, score_it)

            self._table.setItem(row, 1, _item(rec.clip.name, center=False))
            self._table.setItem(row, 2, _item(rec.clip.shape.value))
            self._table.setItem(row, 3, _item(f"{rec.clip.blade_length_mm:.0f}"))
            cov_it = _item(rec.coverage_label)
            cov_it.setForeground(QColor(_COLOR_EXCELLENT if rec.coverage_ratio >= 1.2 else _COLOR_GOOD))
            self._table.setItem(row, 4, cov_it)
            self._table.setItem(row, 5, _item(rec.clip.manufacturer, center=False))

        # Auto-select best
        if recs:
            self._table.selectRow(0)

    def _on_selection_changed(self) -> None:
        self._on_row_changed(self._table.currentRow())

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._recommendations):
            self._txt_reasons.clear()
            self._btn_send.setEnabled(False)
            return
        rec = self._recommendations[row]
        lines = [
            f"Hoja: {rec.clip.blade_length_mm:.0f} mm  |  "
            f"Margen: {rec.safety_margin_mm:.1f} mm  |  "
            f"Fuerza cierre: {rec.clip.closing_force_g:.0f} g",
            "",
        ] + rec.reasons
        self._txt_reasons.setPlainText("\n".join(lines))
        self._btn_send.setEnabled(True)

    def _on_send(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._recommendations):
            return
        clip = self._recommendations[row].clip
        logger.info("ClipRecommender: sending clip '%s' to planner", clip.name)
        self.clip_selected.emit(clip)
