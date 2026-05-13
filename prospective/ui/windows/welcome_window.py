"""WelcomeWindow — cinematic home screen shown after login.

Layout (all overlaid on a looping surgical video background):
  ┌───────────────────────────────────────────────────────────────────────┐
  │ NavBar 52px:  BIENVENIDO, DR. X   [items ...]              [My  ▼]   │
  ├───────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │                   (shutterstock_1090107869.mov)                       │
  │                                                                       │
  │                                          [SkullApp logo – 80px]      │
  └───────────────────────────────────────────────────────────────────────┘

Navigation mapping
------------------
  CASO NUEVO        → NuevoCasoDialog (modal) → MainWindow
  CASOS EXISTENTES  → CaseDashboard window (lazy-created)
  SKULLCLOUD        → "Próximamente" toast
  3D                → opens active MainWindow or prompts to select a case
  AR/VR             → "Próximamente" toast
  ANALITICA         → "Próximamente" toast
  NOSOTROS          → About QMessageBox
  PQRS              → _PQRSDialog
  My ▼              → QMenu with user info + logout
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QLineEdit,
)

logger = logging.getLogger(__name__)

# ── Resources ─────────────────────────────────────────────────────────────── #
_RES        = Path(__file__).resolve().parents[3] / "resources"
_LOGO_PATH  = str(_RES / "logo.png")
_VIDEO_PATH = str(_RES / "shutterstock_1090107869.mp4")

# ── Layout ────────────────────────────────────────────────────────────────── #
_NAV_H    = 52   # navigation bar height in pixels
_BANNER_H = 38   # recent-case banner height in pixels
_LOGO_SZ  = 130  # SkullApp logo display size in pixels

# ── Colours ───────────────────────────────────────────────────────────────── #
_WHITE    = "#ffffff"
_ACCENT   = "#8B9BAA"
_BG_DARK  = "#07101e"
_NAV_BG   = "rgba(0,0,0,195)"          # semi-transparent black nav bar
_TOAST_BG = "rgba(10,18,28,215)"       # dark slate toast background

# ── Navigation items (label, action_key) ──────────────────────────────────── #
_NAV_ITEMS = [
    ("CASO NUEVO",       "new_case"),
    ("CASOS EXISTENTES", "cases"),
    ("SKULLCLOUD",       "cloud"),
    ("3D",               "3d"),
    ("AR/VR",            "arvr"),
    ("ANALITICA",        "analytics"),
    ("NOSOTROS",         "about"),
    ("PQRS",             "pqrs"),
]


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _imageio_available() -> bool:
    try:
        import imageio   # noqa: F401
        import numpy     # noqa: F401
        return True
    except ImportError:
        return False


def _white_pixmap(path: str, size: int) -> QPixmap:
    """Load *path*, scale to *size*, and tint every non-transparent pixel white."""
    pix = QPixmap(path)
    if pix.isNull():
        return pix
    pix = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    white = QPixmap(pix.size())
    white.fill(Qt.transparent)
    p = QPainter(white)
    try:
        p.drawPixmap(0, 0, pix)
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(white.rect(), QColor(_WHITE))
    finally:
        p.end()   # always end the painter, even if an exception occurs
    return white


# ──────────────────────────────────────────────────────────────────────────── #
# PQRS dialog                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

def _pqrs_is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


def _pqrs_style() -> str:
    """Return QSS for _PQRSDialog adapted to the current UI theme."""
    if _pqrs_is_dark():
        return """
            QDialog  { background:#07101e; color:#d0d9ea; }
            QLabel   { color:#d0d9ea; background:transparent; border:none; }
            QLineEdit, QTextEdit, QComboBox {
                background:#0f1e3a;
                border:1px solid rgba(139,155,170,65);
                border-radius:8px;
                color:#d0d9ea;
                padding:4px 8px;
                font-size:11px;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color:#8B9BAA;
            }
            QPushButton {
                background:#182434;
                border:1px solid rgba(139,155,170,80);
                border-radius:8px;
                color:#d0d9ea;
                padding:6px 20px;
            }
            QPushButton:hover { background:#4E6678; border-color:#4E6678; color:#fff; }
        """
    return """
        QDialog  { background:#F4F7FA; color:#0D0D0D; }
        QLabel   { color:#0D0D0D; background:transparent; border:none; }
        QLineEdit, QTextEdit, QComboBox {
            background:#FFFFFF;
            border:1px solid #D0D5DC;
            border-radius:8px;
            color:#0D0D0D;
            padding:4px 8px;
            font-size:11px;
        }
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border-color:#8B9BAA;
        }
        QPushButton {
            background:#E8EEF4;
            border:1px solid #C8D0D8;
            border-radius:8px;
            color:#0D0D0D;
            padding:6px 20px;
        }
        QPushButton:hover { background:#4E6678; border-color:#4E6678; color:#fff; }
    """


class _PQRSDialog(QDialog):
    """PQRS form — Peticiones, Quejas, Reclamos, Sugerencias."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PQRS — Peticiones, Quejas, Reclamos y Sugerencias")
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(_pqrs_style())
        self._build()
        # Apply Windows DWM Acrylic blur-behind so the system frosted-glass
        # effect shows through the dialog titlebar and any transparent regions.
        try:
            from prospective.ui.glass_utils import enable_acrylic
            enable_acrylic(self)
        except Exception:
            pass

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(28, 22, 28, 22)

        # Accent colour: smoke-gray in dark (good contrast on #07101e),
        # deeper teal-blue in light (4.9:1 on #F4F7FA — meets AA).
        _accent_c = "#8B9BAA" if _pqrs_is_dark() else "#4E6678"
        title = QLabel("Contáctenos")
        title.setFont(QFont("Inter", 14, QFont.Bold))
        title.setStyleSheet(f"color:{_accent_c};")
        lay.addWidget(title)

        note = QLabel(
            "Su mensaje será atendido por el equipo de PROSPECTIVE "
            "(Fundación Universitaria Navarra UNINAVARRA)."
        )
        note.setStyleSheet(f"color:{_accent_c}; font-size:11px;")
        note.setWordWrap(True)
        lay.addWidget(note)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)

        self._cmb_tipo = QComboBox()
        self._cmb_tipo.addItems(["Petición", "Queja", "Reclamo", "Sugerencia"])
        form.addRow("Tipo:", self._cmb_tipo)

        self._fld_name = QLineEdit()
        self._fld_name.setPlaceholderText("Su nombre completo")
        form.addRow("Nombre:", self._fld_name)

        self._fld_email = QLineEdit()
        self._fld_email.setPlaceholderText("correo@ejemplo.com")
        form.addRow("Correo:", self._fld_email)

        self._txt_msg = QTextEdit()
        self._txt_msg.setPlaceholderText("Describa su solicitud con el mayor detalle posible…")
        self._txt_msg.setFixedHeight(110)
        form.addRow("Mensaje:", self._txt_msg)

        lay.addLayout(form)
        lay.addSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_row.addSpacing(8)

        btn_send = QPushButton("Enviar")
        btn_send.setStyleSheet(
            "QPushButton{background:#4E6678;border:none;border-radius:8px;"
            "color:#fff;font-weight:bold;padding:6px 24px;}"
            "QPushButton:hover{background:#8B9BAA;}"
        )
        btn_send.setDefault(True)
        btn_send.clicked.connect(self._send)
        btn_row.addWidget(btn_send)
        lay.addLayout(btn_row)

        # Tab order and Enter-to-submit on single-line fields
        self.setTabOrder(self._cmb_tipo, self._fld_name)
        self.setTabOrder(self._fld_name, self._fld_email)
        self.setTabOrder(self._fld_email, self._txt_msg)
        for _f in (self._fld_name, self._fld_email):
            _f.returnPressed.connect(self._send)

    def _send(self) -> None:
        if not self._txt_msg.toPlainText().strip():
            QMessageBox.warning(self, "Campo requerido", "El mensaje no puede estar vacío.")
            return
        QMessageBox.information(
            self,
            "¡Mensaje enviado!",
            "Gracias por contactarnos.\n\n"
            "Su solicitud ha sido registrada y será atendida a la brevedad.",
        )
        self.accept()


# ──────────────────────────────────────────────────────────────────────────── #
# WelcomeWindow                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class WelcomeWindow(QWidget):
    """
    Full-screen cinematic home screen with looping video background and
    a top navigation bar that dispatches to every major app section.

    Usage (from app.py)
    -------------------
        welcome = WelcomeWindow(username=_username, full_name=_full_name)
        welcome.destroyed.connect(app.quit)
        welcome.showMaximized()
    """

    def __init__(
        self,
        username:  str = "",
        full_name: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent, Qt.Window)

        self._username  = username
        self._full_name = full_name or username

        # ── Video state ──────────────────────────────────────────────── #
        self._reader    = None
        self._viter     = None
        self._vtimer: QTimer | None = None
        self._frame_ms  = 40

        # ── Sub-window references ─────────────────────────────────────── #
        self._dashboard     = None        # CaseDashboard — created lazily
        self._main_win      = None        # active MainWindow (at most one)
        self._is_closing    = False       # guard against re-entrant close/logout
        self._navigate_back = False       # set by case_closed signal → show dashboard on return

        self.setWindowTitle("PROSPECTIVE")
        self.setMinimumSize(800, 520)
        # WA_DeleteOnClose ensures that the C++ QObject is actually destroyed
        # when close() is called.  Without this flag, close() only hides the
        # widget and the destroyed() signal (connected to app.quit in app.py)
        # is never emitted → app.exec_() hangs indefinitely after the window
        # is closed.
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # ── Layer 0: video background ─────────────────────────────────── #
        self._bg_lbl = QLabel(self)
        self._bg_lbl.setStyleSheet(f"background:{_BG_DARK};")
        self._bg_lbl.setAlignment(Qt.AlignCenter)

        # ── Layer 1: navigation bar ───────────────────────────────────── #
        self._nav_bar = self._build_nav()

        # ── Layer 2: recent case banner (below nav bar) ───────────────── #
        self._recent_banner: QWidget | None = None
        self._recent_session_path: str = ""
        self._build_recent_banner()

        # ── Layer 3: SkullApp logo (bottom-right) ─────────────────────── #
        self._logo_lbl = self._build_logo()

    # ------------------------------------------------------------------ #
    # Widget construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_nav(self) -> QWidget:
        """Build the top navigation bar and return it."""
        bar = QWidget(self)
        bar.setAttribute(Qt.WA_StyledBackground, True)
        bar.setStyleSheet(f"background:{_NAV_BG}; border:none;")

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(24, 0, 20, 0)
        lay.setSpacing(0)

        # ── Greeting (left side) ────────────────────────────────────── #
        display = self._full_name.upper() if self._full_name else self._username.upper()
        greet = QLabel(f"BIENVENIDO,  {display}")
        greet.setFont(QFont("Inter", 13, QFont.Bold))
        greet.setStyleSheet("color:#ffffff; background:transparent; border:none;")
        lay.addWidget(greet)

        lay.addStretch()

        # ── Nav buttons (right side) ────────────────────────────────── #
        for label, key in _NAV_ITEMS:
            btn = self._nav_btn(label)
            btn.clicked.connect(lambda _c, k=key: self._on_nav(k))
            lay.addWidget(btn)

        lay.addSpacing(12)

        # ── My ▼ (rightmost) ────────────────────────────────────────── #
        self._my_btn = self._nav_btn("My  ▼", accent=True)
        self._my_btn.clicked.connect(self._show_my_menu)
        lay.addWidget(self._my_btn)

        return bar

    @staticmethod
    def _nav_btn(label: str, accent: bool = False) -> QPushButton:
        color = _ACCENT if accent else _WHITE
        btn = QPushButton(label)
        btn.setFixedHeight(36)
        btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;"
            f"color:{color};font-size:13px;font-weight:600;padding:0 13px;}}"
            f"QPushButton:hover{{color:{_ACCENT};}}"
        )
        return btn

    def _build_recent_banner(self) -> None:
        """Query the DB for the last planning session and show a resume strip."""
        try:
            from prospective.db import DatabaseManager
            db   = DatabaseManager.instance()
            sess = db.get_latest_planning_session()   # returns None if no sessions
        except Exception:
            sess = None

        if sess is None:
            return

        # Build the banner widget
        banner = QWidget(self)
        banner.setAttribute(Qt.WA_StyledBackground, True)
        banner.setStyleSheet(
            "background: rgba(10, 6, 30, 185);"
            "border-bottom: 1px solid rgba(139,155,170,60);"
        )
        lay = QHBoxLayout(banner)
        lay.setContentsMargins(24, 0, 14, 0)
        lay.setSpacing(10)

        # Build a human-readable label: prefer patient name, fallback to label/case#
        _patient_name = ""
        try:
            _patient_name = sess.study.patient.full_name if sess.study and sess.study.patient else ""
        except Exception:
            pass
        _raw_label = getattr(sess, "label", "") or ""
        # Strip autosave prefixes so they're not shown verbatim to the user
        _clean_label = _raw_label.replace("autosave_", "").replace("_", " ").strip()
        if _patient_name and _patient_name not in ("—", ""):
            _display = f"Paciente: <b>{_patient_name}</b>"
        elif _clean_label:
            _display = f"<b>{_clean_label}</b>"
        else:
            _display = f"Caso #{getattr(sess, 'study_id', '?')}"
        _lbl = QLabel(f"⏱  Continuar:  {_display}", banner)
        _lbl.setStyleSheet(
            "color: #A8B8C6; font-size: 11px; background: transparent; border: none;"
        )
        lay.addWidget(_lbl)
        lay.addStretch()

        # Resume button
        _btn = QPushButton("▶  Abrir sesión", banner)
        _btn.setFixedHeight(26)
        _btn.setMinimumWidth(120)
        _btn.setStyleSheet(
            "QPushButton{background:#4E6678;border:none;border-radius:8px;"
            "color:#fff;font-weight:bold;font-size:11px;padding:0 14px;}"
            "QPushButton:hover{background:#8B9BAA;}"
        )
        _path = getattr(sess, "file_path", "") or ""
        _sid  = getattr(sess, "study_id", 0) or 0
        _btn.clicked.connect(lambda: self._resume_recent(_path, _sid))
        lay.addWidget(_btn)

        # Dismiss button
        _x = QPushButton("✕", banner)
        _x.setFixedSize(22, 22)
        _x.setStyleSheet(
            "QPushButton{background:transparent;border:none;color:#888;font-size:13px;}"
            "QPushButton:hover{color:#fff;}"
        )
        _x.clicked.connect(self._dismiss_recent_banner)
        lay.addWidget(_x)

        self._recent_banner = banner
        self._recent_session_path = _path
        banner.show()

    def _resume_recent(self, session_path: str, study_id: int) -> None:
        """Open the most recent session directly from the banner."""
        self._dismiss_recent_banner()
        if session_path:
            import os
            if os.path.isfile(session_path):
                self._open_main_for_session(session_path)
                return
        # File missing — fall back to opening the case by study_id
        if study_id:
            self._open_main_for_case("", study_id)

    def _dismiss_recent_banner(self) -> None:
        if self._recent_banner is not None:
            self._recent_banner.deleteLater()
            self._recent_banner = None

    def _build_logo(self) -> QLabel:
        lbl = QLabel(self)
        lbl.setStyleSheet("background:transparent; border:none;")
        pix = _white_pixmap(_LOGO_PATH, _LOGO_SZ)
        if not pix.isNull():
            lbl.setPixmap(pix)
        else:
            lbl.setText("SkullApp")
            lbl.setStyleSheet("color:#ffffff; font-size:16px; font-weight:bold;")
        lbl.adjustSize()
        return lbl

    # ------------------------------------------------------------------ #
    # Qt lifecycle                                                         #
    # ------------------------------------------------------------------ #

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._reposition_overlays()
        # Restart video when the window becomes visible (deferred so it has
        # its final geometry before the first frame is rendered).
        if self._vtimer is None or not self._vtimer.isActive():
            QTimer.singleShot(0, self._start_video)

    def hideEvent(self, event) -> None:
        """Pause the video while the window is hidden to save CPU."""
        super().hideEvent(event)
        self._stop_video()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlays()

    def closeEvent(self, event) -> None:
        """Stop video and close all child sub-windows before exiting."""
        if self._is_closing:          # prevent re-entrant close
            event.accept()
            return
        self._is_closing = True
        self._stop_video()
        # Safely close MainWindow — RuntimeError means the C++ object is already gone
        if self._main_win is not None:
            try:
                self._main_win.destroyed.disconnect(self._on_main_window_closed)
                self._main_win.close()
            except (RuntimeError, Exception):
                pass
            self._main_win = None
        # Safely close CaseDashboard
        if self._dashboard is not None:
            try:
                self._dashboard.destroyed.disconnect(self._on_dashboard_destroyed)
                self._dashboard.close()
            except (RuntimeError, Exception):
                pass
            self._dashboard = None
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # Overlay positioning                                                  #
    # ------------------------------------------------------------------ #

    def _reposition_overlays(self) -> None:
        sw, sh = self.width(), self.height()
        if sw == 0 or sh == 0:
            return
        self._bg_lbl.setGeometry(0, 0, sw, sh)
        self._nav_bar.setGeometry(0, 0, sw, _NAV_H)
        self._nav_bar.raise_()
        # Recent-case banner sits immediately below the nav bar
        if self._recent_banner is not None:
            self._recent_banner.setGeometry(0, _NAV_H, sw, _BANNER_H)
            self._recent_banner.raise_()
        lw, lh = self._logo_lbl.width(), self._logo_lbl.height()
        self._logo_lbl.move(sw - lw - 20, sh - lh - 18)
        self._logo_lbl.raise_()

    # ------------------------------------------------------------------ #
    # Video playback (imageio/numpy frame-by-frame — same engine as login) #
    # ------------------------------------------------------------------ #

    def _start_video(self) -> None:
        if not _imageio_available() or not os.path.isfile(_VIDEO_PATH):
            logger.debug("WelcomeWindow: video unavailable (%s)", _VIDEO_PATH)
            return
        # Prevent double-start: if the timer is already running, do nothing.
        # (Can happen if showEvent fires twice before the deferred singleShot runs.)
        if self._vtimer is not None and self._vtimer.isActive():
            return
        self._open_reader()

    def _open_reader(self) -> None:
        import imageio
        try:
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass
            self._reader   = imageio.get_reader(_VIDEO_PATH, "ffmpeg")
            meta           = self._reader.get_meta_data()
            fps            = float(meta.get("fps") or 25.0)
            self._frame_ms = max(1, int(1000.0 / fps))
            self._viter    = iter(self._reader)
        except Exception as exc:
            logger.warning("WelcomeWindow: cannot open video reader — %s", exc)
            return

        if self._vtimer is None:
            self._vtimer = QTimer(self)
            self._vtimer.setInterval(self._frame_ms)
            self._vtimer.timeout.connect(self._next_frame)
        self._vtimer.start()

    def _next_frame(self) -> None:
        try:
            frame = next(self._viter)
        except StopIteration:
            self._open_reader()   # seamless loop
            return
        except Exception as exc:
            logger.warning("WelcomeWindow: frame error — %s", exc)
            if self._vtimer:
                self._vtimer.stop()
            return
        self._render_frame(frame)

    def _render_frame(self, frame) -> None:
        import numpy as np
        from PyQt5.QtGui import QImage, QPixmap as _QPixmap

        h, w, ch = frame.shape
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        img = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
        tw  = self._bg_lbl.width()
        th  = self._bg_lbl.height()
        pix = _QPixmap.fromImage(img).scaled(
            tw, th, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation
        )
        if pix.width() > tw or pix.height() > th:
            ox = (pix.width()  - tw) // 2
            oy = (pix.height() - th) // 2
            pix = pix.copy(ox, oy, tw, th)
        self._bg_lbl.setPixmap(pix)

    def _stop_video(self) -> None:
        if self._vtimer is not None:
            self._vtimer.stop()
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None   # release ffmpeg subprocess reference
        self._viter = None        # release iterator (holds ref to reader)

    # ------------------------------------------------------------------ #
    # Navigation dispatch                                                  #
    # ------------------------------------------------------------------ #

    def _on_nav(self, key: str) -> None:
        _dispatch = {
            "new_case":  self._on_new_case,
            "cases":     self._on_show_cases,
            "cloud":     lambda: self._show_coming_soon("SkullCloud"),
            "3d":        self._on_3d,
            "arvr":      lambda: self._show_coming_soon("AR / VR"),
            "analytics": lambda: self._show_coming_soon("Analítica"),
            "about":     self._on_about,
            "pqrs":      self._on_pqrs,
        }
        handler = _dispatch.get(key)
        if handler:
            handler()

    # ── CASO NUEVO ────────────────────────────────────────────────────── #

    def _on_new_case(self) -> None:
        from prospective.ui.widgets.nuevo_caso_dialog import NuevoCasoDialog
        dlg = NuevoCasoDialog(created_by=self._username, parent=self)
        if dlg.exec() == dlg.Accepted and dlg.study_id:
            self._open_main_for_case(dlg.dicom_path or "", dlg.study_id)

    # ── CASOS EXISTENTES ──────────────────────────────────────────────── #

    def _on_show_cases(self) -> None:
        """Open (or raise) the CaseDashboard window."""
        if self._dashboard is None:
            from prospective.ui.windows.case_dashboard import CaseDashboard
            self._dashboard = CaseDashboard(username=self._username)
            self._dashboard.open_case.connect(self._open_main_for_case)
            self._dashboard.open_session.connect(self._open_main_for_session)
            self._dashboard.destroyed.connect(self._on_dashboard_destroyed)
        self._dashboard.show()
        self._dashboard.raise_()
        self._dashboard.activateWindow()

    # ── 3D ────────────────────────────────────────────────────────────── #

    def _on_3d(self) -> None:
        """Bring active MainWindow to front, or prompt user to select a case."""
        if self._main_win is not None:
            try:
                # RuntimeError is raised by PyQt5 when the underlying C++ object
                # has already been deleted (window closed outside our tracking).
                self._main_win.show()
                self._main_win.raise_()
                self._main_win.activateWindow()
                return
            except RuntimeError:
                # C++ object destroyed; clear the stale Python reference
                self._main_win = None
            except Exception:
                self._main_win = None

        # No active planning session — guide the user
        ans = QMessageBox.question(
            self,
            "Planificación 3D",
            "La planificación 3D requiere un caso clínico activo.\n\n"
            "¿Desea abrir la ventana de casos para seleccionar uno?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans == QMessageBox.Yes:
            self._on_show_cases()

    # ── NOSOTROS ──────────────────────────────────────────────────────── #

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "Acerca de PROSPECTIVE",
            "<b>PROSPECTIVE™ v0.1.0</b><br>"
            "Preoperative Planning Platform for Cerebral Aneurysm Clipping<br><br>"
            "<b>Fundación Universitaria Navarra UNINAVARRA</b> &mdash; 2026<br>"
            "Python 3.11 &nbsp;·&nbsp; PyQt5 &nbsp;·&nbsp; VTK &nbsp;·&nbsp; PyQtGraph<br><br>"
            "Desarrollado por el equipo de Neurocirugía e Ingeniería Biomédica.<br>"
            "<i style='color:#888;'>Documento Confidencial</i>",
        )

    # ── PQRS ──────────────────────────────────────────────────────────── #

    def _on_pqrs(self) -> None:
        _PQRSDialog(self).exec()

    # ── Próximamente toast ────────────────────────────────────────────── #

    def _show_coming_soon(self, feature: str) -> None:
        """Show a translucent overlay notification for 2 seconds."""
        toast = QLabel(f"✦   {feature}  —  Próximamente", self)
        toast.setAlignment(Qt.AlignCenter)
        toast.setStyleSheet(
            f"background:{_TOAST_BG};"
            "color:#A8B8C6;"
            "border:1px solid rgba(139,155,170,120);"
            "border-radius:16px;"
            "padding:16px 36px;"
            "font-size:16px;"
            "font-weight:bold;"
        )
        toast.adjustSize()
        toast.move(
            (self.width()  - toast.width())  // 2,
            (self.height() - toast.height()) // 2,
        )
        toast.show()
        toast.raise_()
        QTimer.singleShot(2200, toast.deleteLater)

    # ── My ▼ ──────────────────────────────────────────────────────────── #

    def _show_my_menu(self) -> None:
        from prospective.auth import AuthManager
        user = AuthManager.instance().current_user

        menu = QMenu(self)
        if _pqrs_is_dark():
            menu.setStyleSheet("""
                QMenu {
                    background: #0f1e3a;
                    border: 1px solid rgba(139,155,170,80);
                    border-radius: 10px;
                    color: #d0d9ea;
                    padding: 4px 0;
                }
                QMenu::item          { padding: 8px 22px 8px 16px; font-size: 13px; }
                QMenu::item:selected { background: #4E6678; color: #ffffff; border-radius: 14px; }
                QMenu::item:disabled { color: #5a5a6a; }
                QMenu::separator     { height: 1px; background: rgba(139,155,170,50);
                                       margin: 4px 10px; }
            """)
        else:
            menu.setStyleSheet("""
                QMenu {
                    background: #FFFFFF;
                    border: 1px solid #D0D5DC;
                    border-radius: 10px;
                    color: #0D0D0D;
                    padding: 4px 0;
                }
                QMenu::item          { padding: 8px 22px 8px 16px; font-size: 13px; }
                QMenu::item:selected { background: #DDE5EC; color: #0D0D0D; border-radius: 14px; }
                QMenu::item:disabled { color: #9B9B9B; }
                QMenu::separator     { height: 1px; background: #E5E5E5;
                                       margin: 4px 10px; }
            """)

        from prospective.ui.icons import I as _I
        if user:
            name_act = menu.addAction(f"  {_I.USER}  {user.full_name or user.username}")
            name_act.setEnabled(False)
            role_act = menu.addAction(f"       Rol: {user.role.capitalize()}")
            role_act.setEnabled(False)
            menu.addSeparator()

        menu.addAction(f"{_I.LOCK}  Cerrar sesión", self._on_logout)

        # Pop the menu just below the "My" button
        pos = self._my_btn.mapToGlobal(self._my_btn.rect().bottomLeft())
        menu.exec_(pos)

    def _on_logout(self) -> None:
        if self._is_closing:
            return
        from prospective.auth import AuthManager
        AuthManager.instance().logout()
        self.close()   # → closeEvent → sub-windows close → destroyed → app.quit

    # ------------------------------------------------------------------ #
    # Sub-window management                                                #
    # ------------------------------------------------------------------ #

    def _open_main_for_case(self, dicom_path: str, study_id: int) -> None:
        """Hide welcome/dashboard, open MainWindow for the selected case."""
        from prospective.ui.main_window import MainWindow
        self.hide()
        if self._dashboard is not None:
            self._dashboard.hide()
        try:
            win = MainWindow()
            win.show()
            win.preload_dicom(dicom_path, study_id)
            win.case_closed.connect(self._on_case_closed_navigate_back)
            win.destroyed.connect(self._on_main_window_closed)
            self._main_win = win
        except Exception as exc:
            logger.exception("Failed to open MainWindow for case: %s", exc)
            # Restore visibility so the user is not stuck with a blank screen
            self._main_win = None
            self.show()
            if self._dashboard is not None:
                self._dashboard.show()

    def _open_main_for_session(self, session_path: str) -> None:
        """Hide welcome/dashboard, open MainWindow restoring a planning session."""
        from prospective.ui.main_window import MainWindow
        self.hide()
        if self._dashboard is not None:
            self._dashboard.hide()
        try:
            win = MainWindow()
            win.show()
            # auto_load_dicom=True: seamless resume — DICOM is reloaded silently
            # without interrupting the user with a modal dialog.
            win._load_session_from_path(session_path, auto_load_dicom=True)
            win.case_closed.connect(self._on_case_closed_navigate_back)
            win.destroyed.connect(self._on_main_window_closed)
            self._main_win = win
        except Exception as exc:
            logger.exception("Failed to open MainWindow for session: %s", exc)
            self._main_win = None
            self.show()
            if self._dashboard is not None:
                self._dashboard.show()

    def _on_case_closed_navigate_back(self) -> None:
        """Slot for MainWindow.case_closed — user clicked '⌂ Casos', not the X.

        Sets a flag so _on_main_window_closed knows to re-show the dashboard.
        """
        self._navigate_back = True

    def _on_main_window_closed(self) -> None:
        """MainWindow was destroyed — decide what to show next.

        * If the user explicitly pressed '⌂ Casos' (case_closed signal was
          emitted first) → re-show WelcomeWindow **and** CaseDashboard.
        * Otherwise (plain window close / OS close button) → re-show only
          WelcomeWindow so the user can then close it to quit the application.
          This avoids the zombie-process scenario where both WelcomeWindow and
          CaseDashboard end up hidden and the app has no quit path.
        """
        self._main_win = None
        navigating = self._navigate_back
        self._navigate_back = False

        # Always bring WelcomeWindow back — it is the lifecycle anchor that
        # drives app.quit() via welcome.destroyed.connect(app.quit).
        self.show()
        self.raise_()
        self.activateWindow()

        # Additionally re-open the dashboard when the user chose to navigate back
        if navigating and self._dashboard is not None:
            try:
                self._dashboard.show()
                self._dashboard.raise_()
                self._dashboard.activateWindow()
            except RuntimeError:
                self._dashboard = None

    def _on_dashboard_destroyed(self) -> None:
        self._dashboard = None
        # Safety net: if WelcomeWindow ended up hidden (e.g. user somehow closed
        # the dashboard while the WelcomeWindow was not visible and MainWindow is
        # already gone), make sure WelcomeWindow is visible so there is always a
        # quit path via its X button → destroyed → app.quit().
        if self._main_win is None and not self.isVisible():
            self.show()
            self.raise_()
            self.activateWindow()
