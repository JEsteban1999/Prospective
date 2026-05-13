"""PHASES score calculator — intracranial aneurysm 5-year rupture risk.

PHASES (Greving JP et al., Lancet Neurol. 2014):
  P — Population background
  H — Hypertension (treated or untreated)
  A — Age ≥ 70 years
  S — Size of aneurysm (mm)
  E — Earlier SAH from a different aneurysm
  S — Site of aneurysm

The panel auto-populates the Size score when connected to a
MorphometricResult via set_max_diameter().

Reference
---------
Greving JP, Wermer MJ, Brown RD Jr et al.  Development of the PHASES score
for prediction of risk of rupture of intracranial aneurysms: a pooled
analysis of six prospective cohort studies.
Lancet Neurol. 2014; 13(1):59–66.  doi:10.1016/S1474-4422(13)70263-1
"""
from __future__ import annotations

import logging
from typing import ClassVar

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# PHASES score tables (Greving 2014)                                            #
# ──────────────────────────────────────────────────────────────────────────── #

# Points per factor
_P_SCORES: dict[str, int] = {
    "North America / Europe (otras)": 0,
    "Japón":                          3,
    "Finlandia":                      5,
}

_SITE_SCORES: dict[str, int] = {
    "ACI (arteria carótida interna)":  0,
    "ACM (arteria cerebral media)":    2,
    "ACA / ACP / basilar / vertebral": 4,
}

_SIZE_THRESHOLDS: list[tuple[float, int]] = [
    (7.0,  0),   # < 7.0 mm
    (10.0, 3),   # 7.0 – 9.9 mm
    (20.0, 5),   # 10.0 – 19.9 mm
    (1e9,  6),   # ≥ 20 mm
]

# 5-year rupture risk by total score (index = score, value = % risk)
# Extrapolated from Table 2 of Greving 2014
_RISK_BY_SCORE: list[float] = [
    0.4,   # 0
    0.5,   # 1
    0.7,   # 2
    0.9,   # 3
    1.3,   # 4
    1.7,   # 5
    2.4,   # 6
    3.2,   # 7
    4.3,   # 8
    5.9,   # 9
    7.8,   # 10
    10.2,  # 11
    13.0,  # 12
    17.0,  # 13+
]


def _risk_for_score(score: int) -> float:
    idx = min(score, len(_RISK_BY_SCORE) - 1)
    return _RISK_BY_SCORE[idx]


def _size_score(diam_mm: float) -> int:
    for threshold, pts in _SIZE_THRESHOLDS:
        if diam_mm < threshold:
            return pts
    return 6


# ──────────────────────────────────────────────────────────────────────────── #
# Panel widget                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

class PHASESPanel(QWidget):
    """Interactive PHASES score calculator.

    Call set_max_diameter(mm) to auto-fill the size component when
    morphometric analysis finishes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._max_diam_mm: float = 0.0
        self._build_ui()
        self._recalculate()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_max_diameter(self, diam_mm: float) -> None:
        """Auto-populate the Size component from a morphometrics result."""
        self._max_diam_mm = diam_mm
        self._update_size_label()
        self._recalculate()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        grp = QGroupBox("Score PHASES  (Greving 2014)")
        gl = QVBoxLayout()
        gl.setSpacing(6)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        # ── P: Population ─────────────────────────────────────────────── #
        self._cmb_pop = QComboBox()
        for label in _P_SCORES:
            self._cmb_pop.addItem(label)
        self._cmb_pop.currentIndexChanged.connect(self._recalculate)
        form.addRow("P — Población:", self._cmb_pop)

        # ── H: Hypertension ───────────────────────────────────────────── #
        self._chk_htn = QCheckBox("Hipertensión arterial")
        self._chk_htn.stateChanged.connect(self._recalculate)
        form.addRow("H — HTA:", self._chk_htn)

        # ── A: Age ────────────────────────────────────────────────────── #
        self._chk_age = QCheckBox("Edad ≥ 70 años")
        self._chk_age.stateChanged.connect(self._recalculate)
        form.addRow("A — Edad:", self._chk_age)

        # ── S: Size (auto-filled from morphometrics) ──────────────────── #
        self._lbl_size_info = QLabel("—  (analizar morfometría)")
        self._lbl_size_info.setProperty("role", "muted")
        self._lbl_size_pts = QLabel("0 pts")
        self._lbl_size_pts.setStyleSheet("")
        size_row = QHBoxLayout()
        size_row.addWidget(self._lbl_size_info, stretch=1)
        size_row.addWidget(self._lbl_size_pts)
        size_widget = QWidget()
        size_widget.setLayout(size_row)
        form.addRow("S — Tamaño:", size_widget)

        # ── E: Earlier SAH ────────────────────────────────────────────── #
        self._chk_sah = QCheckBox("HSA previa (otro aneurisma)")
        self._chk_sah.stateChanged.connect(self._recalculate)
        form.addRow("E — HSA previa:", self._chk_sah)

        # ── S: Site ───────────────────────────────────────────────────── #
        self._cmb_site = QComboBox()
        for label in _SITE_SCORES:
            self._cmb_site.addItem(label)
        self._cmb_site.currentIndexChanged.connect(self._recalculate)
        form.addRow("S — Localización:", self._cmb_site)

        gl.addLayout(form)

        # ── Separator ─────────────────────────────────────────────────── #
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        try:
            from prospective.ui.themes import is_dark, current_theme as _ct
            _sep_clr = "#363636" if is_dark() else "#E5E5E5"
        except Exception:
            _sep_clr = "#363636"
        sep.setStyleSheet(f"border-top: 1px solid {_sep_clr};")
        gl.addWidget(sep)

        # ── Result ────────────────────────────────────────────────────── #
        result_layout = QFormLayout()
        result_layout.setLabelAlignment(Qt.AlignLeft)

        self._lbl_total = QLabel("0")
        self._lbl_total.setStyleSheet("font-size:20px; font-weight:bold;")
        self._lbl_total.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        result_layout.addRow("Score total:", self._lbl_total)

        self._lbl_risk5 = QLabel("—")
        self._lbl_risk5.setStyleSheet("font-size:14px; font-weight:bold;")
        self._lbl_risk5.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        result_layout.addRow("Riesgo 5 años:", self._lbl_risk5)

        self._lbl_interp = QLabel("")
        self._lbl_interp.setWordWrap(True)
        self._lbl_interp.setProperty("role", "muted")
        result_layout.addRow("", self._lbl_interp)

        gl.addLayout(result_layout)

        # ── Disclaimer ────────────────────────────────────────────────── #
        try:
            from prospective.ui.themes import is_dark
            _disc_clr = "#9B9B9B" if is_dark() else "#6B6B6B"
        except Exception:
            _disc_clr = "#5a7a9a"
        disc = QLabel(
            f"<small style='color:{_disc_clr}'>"
            "Solo para investigación — no sustituye juicio clínico.<br>"
            "Ref: Greving et al. Lancet Neurol. 2014;13(1):59-66."
            "</small>"
        )
        disc.setWordWrap(True)
        gl.addWidget(disc)

        grp.setLayout(gl)
        outer.addWidget(grp)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _update_size_label(self) -> None:
        if self._max_diam_mm <= 0:
            self._lbl_size_info.setText("—  (analizar morfometría)")
            self._lbl_size_pts.setText("0 pts")
            return
        pts = _size_score(self._max_diam_mm)
        self._lbl_size_info.setText(f"{self._max_diam_mm:.1f} mm")
        self._lbl_size_info.setStyleSheet("")
        self._lbl_size_pts.setText(f"{pts} pts")

    def _recalculate(self) -> None:
        pop_label = self._cmb_pop.currentText()
        p_pts  = _P_SCORES.get(pop_label, 0)
        h_pts  = 1 if self._chk_htn.isChecked() else 0
        a_pts  = 1 if self._chk_age.isChecked() else 0
        s_pts  = _size_score(self._max_diam_mm)
        e_pts  = 4 if self._chk_sah.isChecked() else 0
        site_label = self._cmb_site.currentText()
        s2_pts = _SITE_SCORES.get(site_label, 0)

        total = p_pts + h_pts + a_pts + s_pts + e_pts + s2_pts
        risk  = _risk_for_score(total)

        self._lbl_total.setText(str(total))
        self._lbl_risk5.setText(f"{risk:.1f} %")
        self._lbl_size_pts.setText(f"{s_pts} pts")

        if risk >= 7.0:
            color = "#f85149"
            interp = "Riesgo alto de ruptura — considerar tratamiento."
        elif risk >= 2.0:
            color = "#e3b341"
            interp = "Riesgo moderado — seguimiento estrecho recomendado."
        else:
            color = "#56d364"
            interp = "Riesgo bajo — vigilancia periódica."

        self._lbl_risk5.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{color};"
        )
        self._lbl_interp.setText(interp)
        self._lbl_total.setStyleSheet(
            f"font-size:20px; font-weight:bold; color:{color};"
        )

        logger.debug(
            "PHASES score=%d  risk5=%.1f%%  (P=%d H=%d A=%d S=%d E=%d S2=%d)",
            total, risk, p_pts, h_pts, a_pts, s_pts, e_pts, s2_pts,
        )

    # ------------------------------------------------------------------ #
    # Session persistence                                                  #
    # ------------------------------------------------------------------ #

    def get_session_state(self) -> dict:
        """Return all PHASES inputs as a serialisable dict."""
        return {
            "pop_idx":  self._cmb_pop.currentIndex(),
            "htn":      self._chk_htn.isChecked(),
            "age":      self._chk_age.isChecked(),
            "sah":      self._chk_sah.isChecked(),
            "site_idx": self._cmb_site.currentIndex(),
        }

    def restore_session_state(self, state: dict) -> None:
        """Restore PHASES inputs from a previously saved state dict."""
        if not state:
            return
        # Block signals to avoid multiple _recalculate() calls during restore
        for w in (self._cmb_pop, self._cmb_site,
                  self._chk_htn, self._chk_age, self._chk_sah):
            w.blockSignals(True)
        self._cmb_pop.setCurrentIndex(int(state.get("pop_idx", 0)))
        self._cmb_site.setCurrentIndex(int(state.get("site_idx", 0)))
        self._chk_htn.setChecked(bool(state.get("htn", False)))
        self._chk_age.setChecked(bool(state.get("age", False)))
        self._chk_sah.setChecked(bool(state.get("sah", False)))
        for w in (self._cmb_pop, self._cmb_site,
                  self._chk_htn, self._chk_age, self._chk_sah):
            w.blockSignals(False)
        self._recalculate()
