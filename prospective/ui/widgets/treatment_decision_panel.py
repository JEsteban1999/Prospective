"""Treatment Decision Panel — CLIP vs ENDOVASCULAR recommendation.

Displays the multi-factor therapeutic strategy recommendation based on
aneurysm morphometry and optional clinical context inputs.

Integrates into Step 5 (Planificación) of the workflow stepper.

Signal
------
``decision_updated(TreatmentDecision)`` — emitted whenever the scores change.
"""
from __future__ import annotations

import logging

from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QBrush,
)
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from prospective.processing.treatment_decision import (
    LOCATIONS,
    LOCATION_UNKNOWN,
    TreatmentDecision,
    compute_decision,
)

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────── #
_CLIP_COLOR  = "#8B9BAA"   # violet — surgical     (brand colour)
_ENDO_COLOR  = "#3fb950"   # green — endovascular (theme-independent)
_MDT_COLOR   = "#d29922"   # amber — MDT          (theme-independent)
_SURV_COLOR  = "#9B9B9B"   # grey — surveillance  (theme-independent)


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


def _palette() -> dict:
    """Return theme-aware colour strings."""
    dark = _is_dark()
    return {
        "neutral_bg": "#2A2A2A" if dark else "#E5E5E5",
        "text":       "#EBEBEB" if dark else "#0D0D0D",
        "muted":      "#9B9B9B" if dark else "#6B6B6B",
        "border":     "#363636" if dark else "#E5E5E5",
        "combo_bg":   "#1F1F1F" if dark else "#FFFFFF",
        "combo_list": "#2A2A2A" if dark else "#FFFFFF",
    }


def _rec_color(key: str) -> str:
    return {
        "clip":         _CLIP_COLOR,
        "endo":         _ENDO_COLOR,
        "mdt":          _MDT_COLOR,
        "surveillance": _SURV_COLOR,
    }.get(key, _palette()["muted"])


def _conf_color(conf: str) -> str:
    m = _palette()["muted"]
    return {"Alta": _ENDO_COLOR, "Moderada": _MDT_COLOR, "Baja": m}.get(conf, m)


def _dir_color(direction: str) -> str:
    return {"clip": _CLIP_COLOR, "endo": _ENDO_COLOR}.get(direction, _palette()["muted"])


# ──────────────────────────────────────────────────────────────────────────── #
# Score gauge widget (QPainter)                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class _ScoreGauge(QWidget):
    """Horizontal balance bar — left=endo (green), right=clip (blue).

    The fill extends from the centre outward in the winning direction.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._clip_pct = 50
        self._endo_pct = 50
        self._rec_key  = "mdt"

    def set_data(self, clip_pct: int, endo_pct: int, rec_key: str) -> None:
        self._clip_pct = clip_pct
        self._endo_pct = endo_pct
        self._rec_key  = rec_key
        self.update()

    # ------------------------------------------------------------------ #

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h   = self.width(), self.height()
        cx     = w // 2
        bar_h  = 20
        bar_y  = (h - bar_h) // 2
        radius = bar_h // 2

        # ── Background bar ──────────────────────────────────────────── #
        bg_path = QPainterPath()
        bg_path.addRoundedRect(QRectF(0, bar_y, w, bar_h), radius, radius)
        p.fillPath(bg_path, QColor(_palette()["neutral_bg"]))

        # ── Endo fill (left half, fills inward from left edge) ──────── #
        endo_w = int(cx * self._endo_pct / 100)
        if endo_w > 0:
            endo_path = QPainterPath()
            endo_path.addRoundedRect(
                QRectF(cx - endo_w, bar_y, endo_w + radius, bar_h),
                radius, radius,
            )
            # Intersect with left half to avoid overflow
            left_rect = QPainterPath()
            left_rect.addRect(QRectF(0, bar_y, cx, bar_h))
            p.fillPath(endo_path.intersected(left_rect), QColor(_ENDO_COLOR))

        # ── Clip fill (right half, fills inward from right edge) ─────── #
        clip_w = int(cx * self._clip_pct / 100)
        if clip_w > 0:
            clip_path = QPainterPath()
            clip_path.addRoundedRect(
                QRectF(cx - radius, bar_y, clip_w + radius, bar_h),
                radius, radius,
            )
            right_rect = QPainterPath()
            right_rect.addRect(QRectF(cx, bar_y, cx, bar_h))
            p.fillPath(clip_path.intersected(right_rect), QColor(_CLIP_COLOR))

        # ── Centre tick ─────────────────────────────────────────────── #
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(cx, bar_y - 2, cx, bar_y + bar_h + 2)

        # ── Percentage labels ────────────────────────────────────────── #
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)

        # Endo label — left of centre
        p.setPen(QPen(QColor(_ENDO_COLOR)))
        p.drawText(
            QRectF(4, 0, cx - 8, h),
            Qt.AlignVCenter | Qt.AlignLeft,
            f"ENDO  {self._endo_pct}%",
        )

        # Clip label — right of centre
        p.setPen(QPen(QColor(_CLIP_COLOR)))
        p.drawText(
            QRectF(cx + 8, 0, cx - 12, h),
            Qt.AlignVCenter | Qt.AlignRight,
            f"{self._clip_pct}%  CLIP",
        )

        p.end()


# ──────────────────────────────────────────────────────────────────────────── #
# Recommendation badge                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

class _RecommendationBadge(QWidget):
    """Coloured rounded-rect card with icon + recommendation text."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(72)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._icon  = "—"
        self._text  = "—"
        self._conf  = "—"
        self._color = QColor(_palette()["muted"])

    def set_data(self, icon: str, text: str, conf: str, rec_key: str) -> None:
        self._icon  = icon
        self._text  = text
        self._conf  = conf
        self._color = QColor(_rec_color(rec_key))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h    = self.width(), self.height()
        radius  = 8

        # Background fill (semi-transparent tinted)
        bg = QColor(self._color)
        bg.setAlpha(28)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), radius, radius)
        p.fillPath(path, bg)

        # Border
        p.setPen(QPen(self._color, 1.5))
        p.setBrush(QBrush(Qt.NoBrush))
        p.drawRoundedRect(QRectF(0.75, 0.75, w - 1.5, h - 1.5), radius, radius)

        # Icon
        icon_f = QFont()
        icon_f.setPointSize(22)
        p.setFont(icon_f)
        p.setPen(QPen(self._color))
        p.drawText(QRectF(12, 0, 44, h), Qt.AlignVCenter | Qt.AlignLeft, self._icon)

        # Recommendation text
        txt_f = QFont()
        txt_f.setPointSize(11)
        txt_f.setBold(True)
        p.setFont(txt_f)
        p.setPen(QPen(self._color))
        p.drawText(QRectF(64, 8, w - 80, 28), Qt.AlignLeft | Qt.AlignVCenter, self._text)

        # Confidence text
        conf_f = QFont()
        conf_f.setPointSize(9)
        p.setFont(conf_f)
        p.setPen(QPen(QColor(_palette()["muted"])))
        p.drawText(
            QRectF(64, 36, w - 80, 20),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Confianza: {self._conf}",
        )

        p.end()


# ──────────────────────────────────────────────────────────────────────────── #
# Main panel                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

class TreatmentDecisionPanel(QWidget):
    """Step-5 panel — shows CLIP vs ENDOVASCULAR recommendation."""

    decision_updated = pyqtSignal(object)   # TreatmentDecision

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._morpho   = None
        self._decision: TreatmentDecision | None = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_morpho_result(self, result) -> None:
        """Called by MainWindow when morphometric analysis completes."""
        self._morpho = result
        self._recompute()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        body = QWidget()
        lay  = QVBoxLayout(body)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # ── Placeholder (shown when no morpho loaded) ─────────────────── #
        self._placeholder = QLabel(
            "Ejecuta el análisis morfométrico (paso 4)\npara obtener la recomendación terapéutica."
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color:{_palette()['muted']}; font-size:11px; padding:24px;"
        )
        lay.addWidget(self._placeholder)

        # ── Content container (hidden until morpho loaded) ────────────── #
        self._content = QWidget()
        self._content.setVisible(False)
        c_lay = QVBoxLayout(self._content)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(10)

        # Score gauge
        self._gauge = _ScoreGauge()
        c_lay.addWidget(self._gauge)

        # Recommendation badge
        self._badge = _RecommendationBadge()
        c_lay.addWidget(self._badge)

        # Metrics summary
        self._grp_metrics = self._build_metrics_group()
        c_lay.addWidget(self._grp_metrics)

        # Clinical context inputs
        self._grp_context = self._build_context_group()
        c_lay.addWidget(self._grp_context)

        # Factors breakdown
        self._grp_factors = self._build_factors_group()
        c_lay.addWidget(self._grp_factors)

        # Notes
        self._grp_notes = self._build_notes_group()
        c_lay.addWidget(self._grp_notes)

        c_lay.addStretch()
        lay.addWidget(self._content)
        lay.addStretch()

        scroll.setWidget(body)
        root.addWidget(scroll)

        # ── Disclaimer ────────────────────────────────────────────────── #
        disclaimer = QLabel(
            "⚠  Herramienta de soporte de decisión clínica. "
            "No sustituye el criterio del equipo neurovascular tratante."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(
            f"color:{_palette()['muted']}; font-size:9px; padding:6px 10px 6px 10px;"
            f"border-top:1px solid {_palette()['border']};"
        )
        root.addWidget(disclaimer)

    # ── Sub-groups ──────────────────────────────────────────────────────── #

    def _build_metrics_group(self) -> QGroupBox:
        grp = QGroupBox("Métricas morfométricas")
        grp.setStyleSheet(self._grp_qss())
        form = QFormLayout(grp)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setSpacing(4)

        def _val(text: str = "—") -> QLabel:
            lbl = QLabel(text)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            return lbl

        self._m_neck = _val()
        self._m_ar   = _val()
        self._m_dnr  = _val()
        self._m_diam = _val()
        self._m_bf   = _val()
        self._m_ui   = _val()
        self._m_vol  = _val()

        for lbl, widget in [
            ("Cuello (mm):",     self._m_neck),
            ("Aspect Ratio:",    self._m_ar),
            ("DNR:",             self._m_dnr),
            ("Ø máx (mm):",      self._m_diam),
            ("BF:",              self._m_bf),
            ("UI:",              self._m_ui),
            ("Volumen (mm³):",   self._m_vol),
        ]:
            form.addRow(lbl, widget)
        return grp

    def _build_context_group(self) -> QGroupBox:
        grp = QGroupBox("Datos clínicos adicionales")
        grp.setStyleSheet(self._grp_qss())
        form = QFormLayout(grp)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setSpacing(6)

        self._combo_location = QComboBox()
        self._combo_location.addItems(LOCATIONS)
        self._combo_location.setCurrentText(LOCATION_UNKNOWN)
        self._combo_location.currentIndexChanged.connect(self._recompute)
        self._combo_location.setStyleSheet(
            f"QComboBox{{background:{_palette()['combo_bg']};"
            f"border:1px solid {_palette()['border']};"
            f"border-radius:8px;color:{_palette()['text']};padding:3px 6px;}}"
            f"QComboBox QAbstractItemView{{"
            f"background:{_palette()['combo_list']};color:{_palette()['text']};"
            f"selection-background-color:#8B9BAA;selection-color:#ffffff;}}"
        )

        self._chk_ruptured = QCheckBox("Aneurisma roto (HSA)")
        self._chk_ruptured.setStyleSheet(f"color:{_palette()['text']}; font-size:11px;")
        self._chk_ruptured.stateChanged.connect(self._recompute)

        form.addRow("Localización:", self._combo_location)
        form.addRow("",              self._chk_ruptured)
        return grp

    def _build_factors_group(self) -> QGroupBox:
        grp = QGroupBox("Factores determinantes")
        grp.setStyleSheet(self._grp_qss())
        self._factors_layout = QVBoxLayout(grp)
        self._factors_layout.setSpacing(3)
        self._factors_layout.setContentsMargins(6, 8, 6, 6)
        return grp

    def _build_notes_group(self) -> QGroupBox:
        grp = QGroupBox("Notas clínicas")
        grp.setStyleSheet(self._grp_qss())
        self._notes_layout = QVBoxLayout(grp)
        self._notes_layout.setSpacing(4)
        self._notes_layout.setContentsMargins(6, 8, 6, 6)
        return grp

    # ------------------------------------------------------------------ #
    # Recompute & refresh                                                  #
    # ------------------------------------------------------------------ #

    def _recompute(self) -> None:
        if self._morpho is None:
            return

        loc      = self._combo_location.currentText()
        ruptured = self._chk_ruptured.isChecked()

        self._decision = compute_decision(self._morpho, location=loc, ruptured=ruptured)
        self._refresh_ui()
        self.decision_updated.emit(self._decision)

    def _refresh_ui(self) -> None:
        d = self._decision
        m = self._morpho
        if d is None or m is None:
            return

        self._placeholder.setVisible(False)
        self._content.setVisible(True)

        # Gauge
        self._gauge.set_data(d.clip_pct, d.endo_pct, d.recommendation_key)

        # Badge
        self._badge.set_data(d.icon, d.recommendation, d.confidence, d.recommendation_key)

        # Metrics
        self._m_neck.setText(f"{m.neck_diameter_mm:.2f}")
        self._m_neck.setStyleSheet(
            f"color:{'#f85149' if m.neck_diameter_mm > 5 else _palette()['text']}; background:transparent;"
        )
        self._m_ar.setText(f"{m.aspect_ratio:.3f}")
        self._m_ar.setStyleSheet(
            f"color:{'#f85149' if m.aspect_ratio > 2.0 else _palette()['text']}; background:transparent;"
        )
        self._m_dnr.setText(f"{m.dome_to_neck_ratio:.3f}")
        self._m_diam.setText(f"{m.max_diameter_mm:.2f}")
        self._m_bf.setText(f"{m.bottleneck_factor:.3f}")
        self._m_ui.setText(f"{m.undulation_index:.4f}")
        self._m_vol.setText(f"{m.volume_mm3:.1f}")

        for lbl in (self._m_dnr, self._m_diam, self._m_bf, self._m_ui, self._m_vol):
            lbl.setStyleSheet(f"color:{_palette()['text']}; background:transparent;")

        # Factors
        _clear_layout(self._factors_layout)
        for factor in d.factors:
            row = self._make_factor_row(factor)
            self._factors_layout.addWidget(row)
        if not d.factors:
            self._factors_layout.addWidget(
                QLabel("Sin factores determinantes.")
            )

        # Notes
        _clear_layout(self._notes_layout)
        self._grp_notes.setVisible(bool(d.notes))
        for note in d.notes:
            lbl = QLabel(note)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                f"color:{_MDT_COLOR}; font-size:10px; background:transparent;"
            )
            self._notes_layout.addWidget(lbl)

    def _make_factor_row(self, factor: "DecisionFactor") -> QWidget:
        w   = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(6)

        # Direction indicator
        from prospective.ui.icons import I as _I
        if factor.direction == "clip":
            icon_text  = _I.CUT
            icon_color = _CLIP_COLOR
        elif factor.direction == "endo":
            icon_text  = _I.COIL
            icon_color = _ENDO_COLOR
        else:
            icon_text  = "~"
            icon_color = _palette()["muted"]

        icon_lbl = QLabel(icon_text)
        icon_lbl.setFixedWidth(18)
        icon_lbl.setStyleSheet(
            f"font-size:12px; color:{icon_color}; background:transparent;"
        )
        row.addWidget(icon_lbl)

        # Factor name
        name_lbl = QLabel(factor.name)
        name_lbl.setWordWrap(False)
        name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        name_lbl.setStyleSheet(f"font-size:10px; color:{_palette()['text']}; background:transparent;")
        name_lbl.setToolTip(factor.detail)
        row.addWidget(name_lbl)

        # Points badge
        if factor.points > 0:
            pts_lbl = QLabel(f"+{factor.points}")
            pts_lbl.setStyleSheet(
                f"font-size:9px; font-weight:bold; color:{icon_color}; background:transparent;"
            )
            pts_lbl.setFixedWidth(28)
            row.addWidget(pts_lbl)

        return w

    # ------------------------------------------------------------------ #
    # Style helper                                                         #
    # ------------------------------------------------------------------ #

    def _grp_qss(self) -> str:  # noqa: PLR6301
        return f"""
            QGroupBox {{
                font-weight: bold;
                color: {_palette()['text']};
                border: 1px solid {_palette()['border']};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                left: 8px;
            }}
            QLabel {{
                color: {_palette()['text']};
                background: transparent;
                font-size: 10px;
            }}
        """


# ──────────────────────────────────────────────────────────────────────────── #
# Utility                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _clear_layout(layout) -> None:
    """Remove and delete all widgets from a layout."""
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
