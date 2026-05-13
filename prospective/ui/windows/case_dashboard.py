"""CaseDashboard — scrollable grid of recent clinical cases.

This window is NOT the root post-login screen (that is WelcomeWindow).
CaseDashboard is opened lazily when the user clicks "CASOS EXISTENTES"
in WelcomeWindow's navigation bar.

Flow
----
  Login → WelcomeWindow → [CASOS EXISTENTES] → **CaseDashboard**
                        → [case click / CASO NUEVO] → MainWindow

Signals
-------
  open_case(dicom_path: str, study_id: int)
      Emitted when the user selects an existing case or successfully creates
      a new one.  ``dicom_path`` may be an empty string if no DICOM directory
      was linked to the study.

  open_session(file_path: str)
      Emitted when the user clicks "Abrir →" on a planning-session row
      inside a case card.  ``file_path`` is the absolute path to a
      ``.prospective`` JSON file.

Notes
-----
  * Do NOT call ``showMaximized()`` (or ``show()``) inside ``__init__``.
    WelcomeWindow connects ``open_case`` / ``open_session`` *after* creating
    the instance and calls ``show()`` itself.  Calling show inside __init__
    would make the window interactive before those signals are connected.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from prospective.ui.glass_utils import GlassCard, enable_acrylic

logger = logging.getLogger(__name__)

# ── Palette ────────────────────────────────────────────────────────────────── #
_RES       = Path(__file__).resolve().parents[3] / "resources"
_LOGO_PATH = str(_RES / "logo.png")
_HEADER_H  = 64


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


def _pal() -> dict:
    """Return colour tokens for the current theme."""
    if _is_dark():
        return {
            "bg":       "#1F1F1F",
            "card":     "#2A2A2A",
            "card_hov": "#363636",
            "border":   "rgba(139,155,170,65)",
            "txt":      "#EBEBEB",
            "muted":    "#9B9B9B",
            "accent":   "#A8B8C6",
            "success":  "#4dce6a",
            "warning":  "#ffaa33",
            "danger":   "#E85A6A",
        }
    return {
        "bg":       "#FFFFFF",
        "card":     "#F7F7F7",
        "card_hov": "#DDE5EC",
        "border":   "#E5E5E5",
        "txt":      "#0D0D0D",
        "muted":    "#6B6B6B",
        "accent":   "#8B9BAA",
        "success":  "#15803D",
        "warning":  "#92400E",
        "danger":   "#D7263D",
    }


# (All palette values are resolved dynamically via _pal() — no module-level
#  colour constants so that every widget reacts correctly to theme changes.)


def _tinted_pixmap(path: str, size: int, color: str) -> QPixmap:
    """Load *path*, scale to *size*, tint every non-transparent pixel to *color*."""
    pix = QPixmap(path)
    if pix.isNull():
        return pix
    pix = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    tinted = QPixmap(pix.size())
    tinted.fill(Qt.transparent)
    p = QPainter(tinted)
    try:
        p.drawPixmap(0, 0, pix)
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(tinted.rect(), QColor(color))
    finally:
        p.end()
    return tinted


def _logo_pix(size: int) -> QPixmap:
    """Return a theme-appropriate tinted logo at *size* px."""
    color = "#ffffff" if _is_dark() else "#4E6678"
    return _tinted_pixmap(_LOGO_PATH, size, color)


# ── Utilities ─────────────────────────────────────────────────────────────── #

def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* chars, adding '…' if needed."""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _fmt_date(raw) -> str:
    """Convert any date-like value to 'DD/MM/YYYY' display string."""
    if not raw:
        return "—"
    if hasattr(raw, "strftime"):
        return raw.strftime("%d/%m/%Y")
    s = str(raw)[:10]
    parts = s.split("-")
    if len(parts) == 3:
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return s


# ── Background widget with gradient header ─────────────────────────────────  #

class _BgWidget(QWidget):
    def paintEvent(self, event):  # type: ignore[override]
        pal = _pal()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(pal["bg"]))
        # Subtle cloud-grey shimmer at the top (glass feel on both themes)
        g = QLinearGradient(0, 0, 0, 200)
        if _is_dark():
            g.setColorAt(0.0, QColor(139, 155, 170, 22))
            g.setColorAt(0.6, QColor(139, 155, 170,  6))
        else:
            g.setColorAt(0.0, QColor(139, 155, 170, 14))
            g.setColorAt(0.6, QColor(139, 155, 170,  4))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(self.rect(), g)
        p.end()


# ── Case card ─────────────────────────────────────────────────────────────── #

class _CaseCard(GlassCard):
    """Clickable card that represents one recent clinical case.

    Inherits GlassCard for glassmorphism painting (shadow, specular static,
    specular dynamic cursor-tracking, angular gradient border).
    Hover colours are handled by overriding _resolve_colors() rather than
    toggling QSS stylesheets.
    """

    clicked                  = pyqtSignal(str, int)       # dicom_path, study_id
    edit_requested           = pyqtSignal(int, int)       # patient_id, study_id
    session_open_requested   = pyqtSignal(str)            # .prospective file path
    session_delete_requested = pyqtSignal(int, str, str)  # session_id, file_path, label

    # card geometry — width is fixed; height grows with session count
    CARD_W = 252   # slightly narrower so 3+ cards fit on 860px screens

    def __init__(self, case: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, radius=14, highlight=True)
        self._dicom_path = case.get("dicom_path") or ""
        self._study_id   = case.get("study_id")   or 0
        self._patient_id = case.get("patient_id") or 0
        self._hovered    = False
        self.setFixedWidth(self.CARD_W)
        self.setCursor(Qt.PointingHandCursor)
        self._build(case)

    # ── Theme-aware hover colours ─────────────────────────────────────── #

    def _resolve_colors(self) -> tuple:   # type: ignore[override]
        """Return (bg, border) for the current theme + hover state."""
        dark = _is_dark()
        if dark:
            if self._hovered:
                return QColor(54, 58, 66, 230), QColor(168, 184, 198, 160)
            return QColor(32, 36, 44, 210), QColor(139, 155, 170, 75)
        else:
            if self._hovered:
                return QColor(221, 229, 236, 230), QColor(139, 155, 170, 130)
            return QColor(247, 248, 250, 215), QColor(200, 210, 218, 140)

    # ------------------------------------------------------------------ #

    def _build(self, c: dict) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 11, 14, 9)
        root.setSpacing(3)

        # ── Top row: status dot · date · DICOM badge ─────────────────── #
        top = QHBoxLayout()
        top.setSpacing(5)

        pal = _pal()
        tipo = (c.get("tipo_aneurisma") or "").strip()
        dot_color = pal["danger"] if tipo else pal["muted"]
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{dot_color}; font-size:9px; background:transparent;")
        top.addWidget(dot)

        date_lbl = QLabel(_fmt_date(c.get("study_date") or c.get("created_at")))
        date_lbl.setStyleSheet(f"color:{pal['muted']}; font-size:9px; background:transparent;")
        top.addWidget(date_lbl)
        top.addStretch()

        if c.get("dicom_path"):
            dcm = QLabel("DICOM ✔")
            dcm.setStyleSheet(f"color:{pal['success']}; font-size:9px; background:transparent;")
            top.addWidget(dcm)

        root.addLayout(top)

        # ── Patient name ──────────────────────────────────────────────── #
        name_lbl = QLabel(_truncate(c.get("patient_name") or "—", 32))
        name_lbl.setFont(QFont("Inter",11, QFont.Bold))
        name_lbl.setStyleSheet(f"color:{pal['txt']}; background:transparent;")
        root.addWidget(name_lbl)

        nhc_lbl = QLabel(f"NHC: {c.get('hospital_id') or '—'}")
        nhc_lbl.setStyleSheet(f"color:{pal['muted']}; font-size:9px; background:transparent;")
        root.addWidget(nhc_lbl)

        root.addSpacing(4)

        # ── Diagnóstico principal ─────────────────────────────────────── #
        dx = (c.get("dx_principal") or "").strip()
        if dx:
            dx_lbl = QLabel(_truncate(dx, 46))
            dx_lbl.setStyleSheet(f"color:{pal['txt']}; font-size:10px; background:transparent;")
            dx_lbl.setWordWrap(True)
            root.addWidget(dx_lbl)

        # ── Región + lateralidad ──────────────────────────────────────── #
        region = c.get("region_anatomica") or ""
        lat    = c.get("lateralidad") or ""
        if region or lat:
            rl = " · ".join(p for p in (region, lat) if p)
            rl_lbl = QLabel(_truncate(rl, 50))
            rl_lbl.setStyleSheet(f"color:{pal['muted']}; font-size:9px; background:transparent;")
            root.addWidget(rl_lbl)

        # ── Planning sessions ─────────────────────────────────────────── #
        root.addSpacing(6)
        self._build_sessions_section(root)

        # ── Footer ────────────────────────────────────────────────────── #
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{pal['border']};")
        sep.setFixedHeight(1)
        root.addWidget(sep)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 4, 0, 0)
        footer.setSpacing(4)

        if tipo:
            tipo_lbl = QLabel(_truncate(tipo, 18))
            tipo_lbl.setStyleSheet(
                f"color:{pal['warning']}; font-size:9px;"
                f" font-style:italic; background:transparent;"
            )
            footer.addWidget(tipo_lbl)

        footer.addStretch()

        edit_btn = QPushButton("✏")
        edit_btn.setFixedSize(26, 26)
        edit_btn.setToolTip("Editar datos del caso")
        edit_btn.setStyleSheet(
            f"QPushButton{{background:transparent; border:none;"
            f" color:{pal['muted']}; font-size:13px; padding:0; min-height:0;}}"
            f"QPushButton:hover{{color:{pal['accent']};}}"
        )
        edit_btn.clicked.connect(
            lambda: self.edit_requested.emit(self._patient_id, self._study_id)
        )
        footer.addWidget(edit_btn)

        open_lbl = QLabel("Abrir →")
        open_lbl.setStyleSheet(
            f"color:{pal['accent']}; font-size:10px; font-weight:bold; background:transparent;"
        )
        footer.addWidget(open_lbl)

        root.addLayout(footer)

    # ------------------------------------------------------------------ #

    def _build_sessions_section(self, root: "QVBoxLayout") -> None:
        """Render planning sessions for this study inside the card.

        Each session gets its own row:
            [★/📄]  [nombre / fecha]     [Abrir →]  [✕]
        Disabled (grey) if the .prospective file is missing on disk.
        The ✕ button emits session_delete_requested(session_id, file_path, label).
        A "＋ Nueva sesión" button always appears at the bottom.
        """
        from pathlib import Path
        from prospective.db import DatabaseManager

        sessions: list = []
        if self._study_id:
            try:
                sessions = DatabaseManager.instance().get_planning_sessions(
                    self._study_id
                )
            except Exception:
                pass

        pal = _pal()

        # ── Separator ────────────────────────────────────────────────── #
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{pal['border']};")
        sep.setFixedHeight(1)
        root.addWidget(sep)

        # ── Section header: "SESIONES (N)" ────────────────────────────── #
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 4, 0, 2)
        hdr.setSpacing(4)

        sess_lbl = QLabel("SESIONES")
        sess_lbl.setStyleSheet(
            f"color:{pal['muted']}; font-size:8px; font-weight:bold;"
            f" letter-spacing:1px; background:transparent;"
        )
        hdr.addWidget(sess_lbl)

        if sessions:
            badge = QLabel(str(len(sessions)))
            badge.setStyleSheet(
                f"background:{pal['accent']}; color:#fff; font-size:9px;"
                f" font-weight:bold; padding:1px 6px; border-radius:8px;"
                f" min-width:16px; min-height:16px;"
            )
            badge.setAlignment(Qt.AlignCenter)
            hdr.addWidget(badge)

        hdr.addStretch()
        root.addLayout(hdr)

        # ── One row per session ────────────────────────────────────────── #
        for ps in sessions:
            from prospective.ui.icons import I as _I
            is_final = getattr(ps, "is_final", False)
            icon     = "★" if is_final else _I.DOC
            raw_label = (ps.label or "").strip()
            name = _truncate(raw_label, 18) if raw_label else _fmt_date(ps.created_at)
            has_file = bool(ps.file_path and Path(ps.file_path).exists())

            row_frame = QFrame()
            row_frame.setFixedHeight(28)
            row_frame.setStyleSheet(
                "QFrame{background:transparent; border:none; border-radius:5px;}"
                "QFrame:hover{background:rgba(139,155,170,12);}"
            )
            row_lay = QHBoxLayout(row_frame)
            row_lay.setContentsMargins(2, 0, 2, 0)
            row_lay.setSpacing(4)

            icon_color = pal["warning"] if is_final else pal["muted"]
            icon_lbl = QLabel(icon)
            icon_lbl.setStyleSheet(f"color:{icon_color}; font-size:10px; background:transparent;")
            icon_lbl.setFixedWidth(16)
            row_lay.addWidget(icon_lbl)

            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(
                f"color:{pal['txt'] if has_file else pal['muted']}; font-size:10px;"
                " background:transparent;"
            )
            row_lay.addWidget(name_lbl, 1)

            open_btn = QPushButton("Abrir →")
            open_btn.setFixedHeight(24)
            open_btn.setFixedWidth(68)
            open_btn.setEnabled(has_file)
            open_btn.setToolTip(
                ps.file_path if has_file
                else "Archivo .prospective no encontrado en disco"
            )
            open_btn.setStyleSheet(
                f"QPushButton{{background:transparent; border:1px solid {pal['border']};"
                f" color:{pal['accent'] if has_file else pal['muted']}; font-size:9px;"
                f" padding:0 4px; border-radius:5px; min-height:0;}}"
                f"QPushButton:hover{{background:rgba(139,155,170,15);"
                f" border-color:{pal['accent']};}}"
                f"QPushButton:disabled{{color:{pal['muted']}; border-color:{pal['border']};}}"
            )
            if has_file:
                _path = ps.file_path
                open_btn.clicked.connect(
                    lambda _=False, p=_path: self.session_open_requested.emit(p)
                )
            row_lay.addWidget(open_btn)

            del_btn = QPushButton("✕")
            del_btn.setFixedSize(22, 22)
            del_btn.setToolTip("Eliminar esta sesión del proyecto")
            del_btn.setStyleSheet(
                f"QPushButton{{background:transparent; border:none;"
                f" color:{pal['muted']}; font-size:10px; font-weight:bold; padding:0;"
                f" min-height:0;}}"
                f"QPushButton:hover{{color:{pal['danger']};"
                f" background:rgba(232,90,106,12); border-radius:5px;}}"
            )
            _sid  = ps.id
            _fp   = ps.file_path or ""
            _name = name
            del_btn.clicked.connect(
                lambda _=False, sid=_sid, fp=_fp, lbl=_name:
                    self.session_delete_requested.emit(sid, fp, lbl)
            )
            row_lay.addWidget(del_btn)

            root.addWidget(row_frame)

        if not sessions:
            empty_lbl = QLabel("Sin sesiones guardadas")
            empty_lbl.setStyleSheet(
                f"color:{pal['muted']}; font-size:9px; font-style:italic;"
                " background:transparent; padding:2px 2px;"
            )
            root.addWidget(empty_lbl)

        root.addSpacing(2)
        new_btn = QPushButton("＋  Nueva sesión")
        new_btn.setFixedHeight(26)
        new_btn.setToolTip("Abre el caso para iniciar una nueva sesión de planificación")
        new_btn.setStyleSheet(
            f"QPushButton{{background:transparent;"
            f" border:1px solid {pal['border']}; color:{pal['muted']};"
            f" font-size:10px; padding:0 6px; border-radius:5px; text-align:left;"
            f" min-height:0;}}"
            f"QPushButton:hover{{border-color:{pal['success']}; color:{pal['success']};}}"
        )
        new_btn.clicked.connect(
            lambda: self.clicked.emit(self._dicom_path, self._study_id)
        )
        root.addWidget(new_btn)
        root.addSpacing(3)

    # ── Mouse events ─────────────────────────────────────────────────── #

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._dicom_path, self._study_id)
        super().mousePressEvent(event)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = False
        super().leaveEvent(event)   # GlassCard clears cursor_pos + calls update()


# ── Dashboard ─────────────────────────────────────────────────────────────── #

class CaseDashboard(QMainWindow):
    """Post-login dashboard — shows recent cases and lets users create new ones.

    Usage::

        dashboard = CaseDashboard(username="dr_garcia")
        dashboard.open_case.connect(lambda path: launch_main_window(path))
        dashboard.show()
    """

    #: Emitted when a case is chosen.  ``dicom_path`` may be "".
    open_case    = pyqtSignal(str, int)   # dicom_path, study_id
    #: Emitted when the user opens an existing planning session from a card.
    open_session = pyqtSignal(str)        # .prospective file path

    def __init__(
        self,
        username: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._username = username
        self._first_show = True       # showMaximized on first appearance

        self.setWindowTitle("PROSPECTIVE — Panel de casos")
        self.setMinimumSize(780, 540)

        self._build_ui()
        self._load_cases()
        # NOTE: do NOT call showMaximized() here — the caller (WelcomeWindow or
        # tests) is responsible for calling show().  Showing inside __init__
        # would make the window visible before the caller connects open_case /
        # open_session signals, creating a race condition where a fast click
        # dispatches to an unconnected signal and the event is silently dropped.

    def showEvent(self, event) -> None:    # noqa: N802
        super().showEvent(event)
        # Maximise on first show so the card grid has plenty of room.
        if self._first_show:
            self._first_show = False
            self.showMaximized()
        # Apply Windows DWM Acrylic blur to the dashboard window on first show.
        # enable_acrylic is a no-op on non-Windows platforms.
        try:
            enable_acrylic(self)
        except Exception:
            pass

    def resizeEvent(self, event) -> None:    # noqa: N802
        """Rebuild card grid when window width changes enough to alter column count."""
        super().resizeEvent(event)
        # Only rebuild when the effective column count has changed to avoid
        # unnecessary redraws on every pixel of resize.
        avail_w = max(self.width() - 96, _CaseCard.CARD_W * 2)
        new_cols = max(2, min(4, avail_w // (_CaseCard.CARD_W + 16)))
        if new_cols != getattr(self, "_current_cols", None):
            self._current_cols = new_cols
            self._load_cases()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        pal = _pal()
        bg = _BgWidget()
        self.setCentralWidget(bg)

        root = QVBoxLayout(bg)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────── #
        header = QWidget()
        header.setFixedHeight(_HEADER_H)
        header.setStyleSheet(
            f"background:{pal['card']}; border-bottom:1px solid {pal['border']};"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(28, 0, 28, 0)
        hl.setSpacing(10)

        # SkullApp logo — tinted to contrast with the current theme's header bg
        # (white on dark card, brand-purple on light card)
        logo_img = QLabel()
        logo_img.setStyleSheet("background:transparent;")
        _lpix = _logo_pix(70)  # 70×30 px — fits the 64 px header with comfortable margin
        if not _lpix.isNull():
            logo_img.setPixmap(_lpix)
        else:
            logo_img.setText("💀")
            logo_img.setStyleSheet("font-size:22px; background:transparent;")
        hl.addWidget(logo_img)
        hl.addSpacing(6)

        logo_lbl = QLabel("PROSPECTIVE™")
        logo_lbl.setFont(QFont("Inter",16, QFont.Bold))
        logo_lbl.setStyleSheet(f"color:{pal['accent']}; background:transparent;")
        hl.addWidget(logo_lbl)

        sub_lbl = QLabel("Hybrid Neurovascular Planning Software Platform")
        sub_lbl.setStyleSheet(
            f"color:{pal['muted']}; font-size:11px; background:transparent;"
        )
        hl.addWidget(sub_lbl)

        hl.addStretch()

        if self._username:
            from prospective.ui.icons import I as _I
            user_lbl = QLabel(f"{_I.USER}  {self._username}")
            user_lbl.setStyleSheet(
                f"color:{pal['txt']}; font-size:11px; background:transparent;"
            )
            hl.addWidget(user_lbl)

            sep2 = QLabel("|")
            sep2.setStyleSheet(f"color:{pal['border']}; background:transparent;")
            hl.addWidget(sep2)

        self._btn_nuevo = QPushButton("＋  Nuevo Caso")
        self._btn_nuevo.setFixedHeight(34)
        if _is_dark():
            self._btn_nuevo.setStyleSheet(
                f"QPushButton{{background:#1a3a1a; border:1px solid {pal['success']};"
                f" color:{pal['success']}; padding:0 18px; border-radius:9px;"
                f" font-weight:bold; font-size:11px;}}"
                f"QPushButton:hover{{background:#265226; color:#fff;}}"
            )
        else:
            self._btn_nuevo.setStyleSheet(
                f"QPushButton{{background:#e6f4ea; border:1px solid {pal['success']};"
                f" color:{pal['success']}; padding:0 18px; border-radius:9px;"
                f" font-weight:bold; font-size:11px;}}"
                f"QPushButton:hover{{background:{pal['success']}; color:#fff;}}"
            )
        self._btn_nuevo.clicked.connect(self._open_new_case)
        hl.addWidget(self._btn_nuevo)

        root.addWidget(header)

        # ── Scrollable body ─────────────────────────────────────────── #
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            f"QScrollArea{{background:{pal['bg']}; border:none;}}"
            f"QScrollBar:vertical{{background:{pal['card']}; width:8px; border-radius:8px;}}"
            f"QScrollBar::handle:vertical{{background:{pal['border']}; border-radius:8px;}}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical{{height:0;}}"
        )

        self._body = QWidget()
        self._body.setStyleSheet(f"background:{pal['bg']};")
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(32, 28, 32, 32)
        self._body_lay.setSpacing(20)

        scroll.setWidget(self._body)
        root.addWidget(scroll, stretch=1)

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    def _load_cases(self) -> None:
        """Fetch recent cases from the DB and rebuild the card grid."""
        from prospective.db import DatabaseManager
        db = DatabaseManager.instance()
        cases = db.get_recent_cases(created_by=self._username, limit=12)

        # Clear previous content
        while self._body_lay.count():
            item = self._body_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # ── Section header ─────────────────────────────────────────── #
        pal = _pal()
        hdr_w = QWidget()
        hdr_lay = QHBoxLayout(hdr_w)
        hdr_lay.setContentsMargins(0, 0, 0, 0)
        hdr_lay.setSpacing(8)

        sec_title = QLabel("Casos Recientes")
        sec_title.setFont(QFont("Inter",15, QFont.Bold))
        sec_title.setStyleSheet(f"color:{pal['txt']}; background:transparent;")
        hdr_lay.addWidget(sec_title)

        if self._username:
            usr_lbl = QLabel(f"— {self._username}")
            usr_lbl.setStyleSheet(
                f"color:{pal['muted']}; font-size:13px; background:transparent;"
            )
            hdr_lay.addWidget(usr_lbl)

        hdr_lay.addStretch()

        n = len(cases)
        cnt_lbl = QLabel(f"{n} caso{'s' if n != 1 else ''}")
        cnt_lbl.setStyleSheet(
            f"color:{pal['muted']}; font-size:11px; background:transparent;"
        )
        hdr_lay.addWidget(cnt_lbl)

        self._body_lay.addWidget(hdr_w)

        # ── Content ────────────────────────────────────────────────── #
        if not cases:
            self._body_lay.addWidget(self._make_empty_state())
        else:
            self._body_lay.addWidget(self._make_grid(cases))

        self._body_lay.addStretch()

    # ------------------------------------------------------------------ #

    def _make_empty_state(self) -> QWidget:
        pal = _pal()
        w = QWidget()
        w.setMinimumHeight(240)
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(16)

        from prospective.ui.icons import I as _I
        icon_lbl = QLabel(_I.FOLDER)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet("font-size:52px; background:transparent;")
        lay.addWidget(icon_lbl)

        msg_lbl = QLabel(
            "No hay casos recientes.\n"
            "Cree su primer caso clínico para comenzar."
        )
        msg_lbl.setAlignment(Qt.AlignCenter)
        msg_lbl.setStyleSheet(
            f"color:{pal['muted']}; font-size:14px; background:transparent;"
        )
        msg_lbl.setWordWrap(True)
        lay.addWidget(msg_lbl)

        btn = QPushButton("＋  Crear primer caso")
        btn.setMinimumWidth(180)
        btn.setMaximumWidth(260)
        btn.setFixedHeight(40)
        if _is_dark():
            btn.setStyleSheet(
                f"QPushButton{{background:#1a3a1a; border:1px solid {pal['success']};"
                f" color:{pal['success']}; border-radius:10px;"
                f" font-size:13px; font-weight:bold;}}"
                f"QPushButton:hover{{background:#265226; color:#fff;}}"
            )
        else:
            btn.setStyleSheet(
                f"QPushButton{{background:#e6f4ea; border:1px solid {pal['success']};"
                f" color:{pal['success']}; border-radius:10px;"
                f" font-size:13px; font-weight:bold;}}"
                f"QPushButton:hover{{background:{pal['success']}; color:#fff;}}"
            )
        btn.clicked.connect(self._open_new_case)
        lay.addWidget(btn, alignment=Qt.AlignCenter)

        return w

    def _make_grid(self, cases: list) -> QWidget:
        grid_w = QWidget()
        grid_w.setStyleSheet(f"background:{_pal()['bg']};")
        grid = QGridLayout(grid_w)
        grid.setSpacing(16)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # Dynamic column count: fit as many CARD_W-wide cards as the current
        # viewport allows, with a minimum of 2 and a maximum of 4.
        avail_w = max(self.width() - 96, _CaseCard.CARD_W * 2)
        cols = max(2, min(4, avail_w // (_CaseCard.CARD_W + 16)))

        for i, case in enumerate(cases):
            card = _CaseCard(case)
            card.clicked.connect(self._on_card_clicked)
            card.edit_requested.connect(self._on_card_edit)
            card.session_open_requested.connect(self._on_session_open)
            card.session_delete_requested.connect(self._on_session_delete)
            grid.addWidget(card, i // cols, i % cols, Qt.AlignTop)

        # Pad last row so cards stay left-aligned
        remainder = len(cases) % cols
        if remainder:
            last_row = len(cases) // cols
            for j in range(remainder, cols):
                pad = QWidget()
                pad.setFixedWidth(_CaseCard.CARD_W)
                pad.setStyleSheet("background:transparent;")
                grid.addWidget(pad, last_row, j, Qt.AlignTop)

        return grid_w

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _on_card_clicked(self, dicom_path: str, study_id: int) -> None:
        logger.info("Dashboard: case selected  study_id=%d  dicom=%r", study_id, dicom_path)
        self.open_case.emit(dicom_path, study_id)

    def _on_card_edit(self, patient_id: int, study_id: int) -> None:
        """Open the case form in edit mode; reload cards on save."""
        from PyQt5.QtWidgets import QDialog
        from prospective.ui.widgets.nuevo_caso_dialog import NuevoCasoDialog

        dlg = NuevoCasoDialog(
            created_by=self._username,
            patient_id=patient_id,
            study_id=study_id,
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            logger.info("Case edited — patient_id=%d  study_id=%d", patient_id, study_id)
            self._load_cases()   # refresh cards with updated data

    def _on_session_open(self, file_path: str) -> None:
        """Open an existing planning session selected from a case card."""
        from pathlib import Path

        if not Path(file_path).exists():
            QMessageBox.warning(
                self,
                "Archivo no encontrado",
                f"No se encontró el archivo de sesión:\n{file_path}",
            )
            return
        logger.info("Dashboard: session selected  path=%r", file_path)
        self.open_session.emit(file_path)

    def _on_session_delete(
        self, session_id: int, file_path: str, label: str
    ) -> None:
        """Delete a planning session after double confirmation.

        Step 1 — ask whether to remove the DB record.
        Step 2 — if the .prospective file exists on disk, ask whether to also
                  delete it (and the companion _mesh.vtp if present).
        """
        from pathlib import Path
        from prospective.db import DatabaseManager

        # ── Step 1: confirm removal from project ──────────────────────── #
        reply = QMessageBox.question(
            self,
            "Eliminar sesión",
            f"¿Eliminar la sesión <b>{label}</b> del proyecto?<br><br>"
            f"El registro se eliminará de la base de datos.<br>"
            f"El archivo guardado en disco <u>no</u> se borrará en este paso.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        DatabaseManager.instance().delete_planning_session(session_id)
        logger.info("Session deleted from DB — id=%d  label=%r", session_id, label)

        # ── Step 2: offer to delete the file from disk ────────────────── #
        p = Path(file_path) if file_path else None
        if p and p.exists():
            reply2 = QMessageBox.question(
                self,
                "Archivo en disco",
                f"¿Eliminar también el archivo del disco?<br><br>"
                f"<span style='color:#7a98c0; font-size:9pt;'>{file_path}</span>",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply2 == QMessageBox.Yes:
                try:
                    p.unlink(missing_ok=True)
                    # Remove companion mesh file if present (_mesh.vtp)
                    mesh = p.parent / f"{p.stem}_mesh.vtp"
                    if mesh.exists():
                        mesh.unlink(missing_ok=True)
                    logger.info("Session file deleted from disk: %s", file_path)
                except Exception as exc:
                    logger.warning("Could not delete session file: %s", exc)
                    QMessageBox.warning(
                        self,
                        "Error al eliminar archivo",
                        f"No se pudo eliminar el archivo:\n{exc}",
                    )

        self._load_cases()  # Refresh all case cards

    def _open_new_case(self) -> None:
        """Open the NuevoCasoDialog; on success emit open_case."""
        from PyQt5.QtWidgets import QDialog
        from prospective.ui.widgets.nuevo_caso_dialog import NuevoCasoDialog

        dlg = NuevoCasoDialog(created_by=self._username, parent=self)
        if dlg.exec() == QDialog.Accepted:
            logger.info(
                "New case created — patient_id=%s  study_id=%s  dicom=%r",
                dlg.patient_id,
                dlg.study_id,
                dlg.dicom_path,
            )
            self.open_case.emit(dlg.dicom_path or "", dlg.study_id or 0)
