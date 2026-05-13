"""Login window — cinematic full-screen landing page.

Video background (looping) with a dark header bar (title + LOGIN / SIGNUP buttons)
and the SkullApp logo in the bottom-left corner.  Clicking LOGIN reveals a
semi-transparent credential card on the right side of the screen.

On first run (no users) the card is auto-shown with the admin-creation form.
Closing the window exits the application.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap, QLinearGradient
from PyQt5.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget, QFrame,
    QGraphicsOpacityEffect,
)

from prospective.ui.widgets.avatar_widget import AvatarWidget
from prospective.ui.glass_utils import GlassCard
from prospective.auth.auth_manager import AuthManager
from prospective.db.models import User

logger = logging.getLogger(__name__)

# ── Resources ─────────────────────────────────────────────────────────────── #
_RES       = Path(__file__).resolve().parents[3] / "resources"
_LOGO_PATH = str(_RES / "logo.png")
_VIDEO_MOV = str(_RES / "Multimedia1.mov")

# ── Palette ───────────────────────────────────────────────────────────────── #
_BLACK      = "#000000"
_WHITE      = "#ffffff"
_ACCENT     = "#8B9BAA"
_MUTED      = "#8B9BAA"
_INPUT_BG   = "rgba(8,12,22,190)"
_INPUT_BORD = "rgba(139,155,170,70)"
_INPUT_FOCUS= "#8B9BAA"
_RED        = "#f87171"
_BTN_BG     = "#4E6678"
_BTN_HOVER  = "#8B9BAA"
_HEADER_H   = 56   # px

# ── QSS tokens ────────────────────────────────────────────────────────────── #
_INPUT_QSS = (
    f"QLineEdit{{background:{_INPUT_BG}; border:1px solid {_INPUT_BORD};"
    f"border-radius:11px; color:{_WHITE}; padding:0 12px;"
    f"font-size:13px; selection-background-color:#4E6678;}}"
    f"QLineEdit:focus{{border:1px solid {_INPUT_FOCUS}; background:rgba(10,18,30,210);}}"
    f"QLineEdit::placeholder{{color:rgba(139,155,170,140);}}"
)
_BTN_QSS = (
    f"QPushButton{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    f"stop:0 #3B5268,stop:1 #5A7A94);"
    f"border:none; border-radius:11px; color:#ffffff;"
    f"font-weight:700; font-size:13px; letter-spacing:0.3px;"
    f"padding:8px 28px;}}"
    f"QPushButton:hover{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    f"stop:0 #4E6678,stop:1 #7A9AB4);}}"
    f"QPushButton:pressed{{background:#2A3F52;}}"
    f"QPushButton:disabled{{background:rgba(78,102,120,60); color:rgba(255,255,255,80);}}"
)


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _tinted_pixmap(path: str, size: int, color: str) -> QPixmap:
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


def _imageio_available() -> bool:
    try:
        import imageio  # noqa: F401
        import numpy    # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_video() -> str | None:
    if os.path.isfile(_VIDEO_MOV):
        return _VIDEO_MOV
    return None


def _lbl(text: str, size: int = 11, color: str = _WHITE,
         bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    f = QFont("Inter", size)
    f.setBold(bold)
    lbl.setFont(f)
    lbl.setStyleSheet(f"color:{color}; background:transparent; border:none;")
    return lbl


def _field_row(icon: str, placeholder: str, pw: bool = False) -> tuple[QWidget, QLineEdit]:
    """Return (row_widget, QLineEdit) with a decorative left icon inside the field."""
    container = QWidget()
    container.setStyleSheet("background:transparent;")
    lay = QHBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)

    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setMinimumHeight(42)
    if pw:
        edit.setEchoMode(QLineEdit.Password)
    edit.setStyleSheet(
        f"QLineEdit{{background:{_INPUT_BG}; border:1px solid {_INPUT_BORD};"
        f"border-radius:11px; color:{_WHITE}; padding:0 12px 0 38px;"
        f"font-size:13px; selection-background-color:#4E6678;}}"
        f"QLineEdit:focus{{border:1px solid {_INPUT_FOCUS}; background:rgba(10,18,30,210);}}"
    )
    lay.addWidget(edit)

    # Floating icon label overlaid on the left edge of the field
    icon_lbl = QLabel(icon, container)
    icon_lbl.setStyleSheet(
        f"color:rgba(139,155,170,160); font-size:14px; background:transparent; border:none;"
    )
    icon_lbl.setFixedWidth(30)
    icon_lbl.setAlignment(Qt.AlignCenter)
    # Position it inside the edit field (will be moved in showEvent of parent)
    icon_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
    # Use a deferred move so geometry is settled
    def _place_icon():
        icon_lbl.move(8, (edit.height() - icon_lbl.height()) // 2)
    QTimer.singleShot(0, _place_icon)

    return container, edit


# ─────────────────────────────────────────────────────────────────────────── #
# LoginDialog                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class LoginDialog(QDialog):
    """
    Cinematic full-screen login/first-run window.

    Layout (z-ordered):
      0 — QLabel ``_bg_lbl``  : looping video frames (below header)
      1 — QWidget ``_header`` : solid-black navbar with title + buttons
      1 — QWidget ``_logo_w`` : SkullApp logo overlay (bottom-left)
      2 — GlassCard ``_card`` : semi-transparent credential card (centered)
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._auth    = AuthManager.instance()
        self._reader  = None
        self._viter   = None
        self._vtimer: QTimer | None = None
        self._frame_ms = 40
        self._vpath    = _resolve_video()

        self.setWindowTitle("PROSPECTIVE")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        screen = QApplication.primaryScreen().geometry()
        sw, sh = screen.width(), screen.height()
        self.setFixedSize(sw, sh)
        self.move(screen.topLeft())

        # ── Layer 0: video background ─────────────────────────────────── #
        self._bg_lbl = QLabel(self)
        self._bg_lbl.setGeometry(0, _HEADER_H, sw, sh - _HEADER_H)
        self._bg_lbl.setStyleSheet("background:#000;")

        # ── Layer 1: header bar ───────────────────────────────────────── #
        self._build_header(sw)

        # ── Layer 1: SkullApp logo (bottom-left) ─────────────────────── #
        self._build_logo(sh)

        # ── Layer 2: credential card (centered) ──────────────────────── #
        has_users = self._auth.has_any_user()
        card_w = 380
        card_h = 520 if has_users else 580
        card_x = (sw - card_w) // 2
        card_y = _HEADER_H + (sh - _HEADER_H - card_h) // 2
        self._card = GlassCard(self, radius=18)
        self._card.setGeometry(card_x, card_y, card_w, card_h)
        self._card.hide()
        self._build_card()

        # ── Start looping video ───────────────────────────────────────── #
        self._start_video()

        if not has_users:
            QTimer.singleShot(500, self._show_card)

    # ── Header ──────────────────────────────────────────────────────────── #

    def _build_header(self, sw: int) -> None:
        hdr = QWidget(self)
        hdr.setGeometry(0, 0, sw, _HEADER_H)
        hdr.setStyleSheet(f"background:{_BLACK};")
        hdr.setAttribute(Qt.WA_StyledBackground, True)

        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(28, 0, 28, 0)
        lay.setSpacing(0)

        title = QLabel("PROSPECTIVE")
        title.setFont(QFont("Inter", 18, QFont.Bold))
        title.setStyleSheet(f"color:{_WHITE}; background:transparent;")
        lay.addWidget(title)
        lay.addStretch()

        self._btn_login_nav = QPushButton("LOGIN")
        self._btn_login_nav.setFixedHeight(34)
        self._btn_login_nav.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;"
            f"color:{_WHITE};font-size:13px;font-weight:bold;padding:0 20px;"
            f"min-height:0;}}"
            f"QPushButton:hover{{color:{_ACCENT};}}"
        )
        self._btn_login_nav.clicked.connect(self._show_card)
        lay.addWidget(self._btn_login_nav)
        lay.addSpacing(6)

        btn_su = QPushButton("SIGNUP")
        btn_su.setFixedHeight(34)
        btn_su.setStyleSheet(
            f"QPushButton{{background:transparent;"
            f"border:1px solid rgba(255,255,255,140);"
            f"border-radius:8px;color:{_WHITE};"
            f"font-size:13px;font-weight:bold;padding:0 18px; min-height:0;}}"
            f"QPushButton:hover{{border-color:{_ACCENT};color:{_ACCENT};}}"
        )
        btn_su.clicked.connect(self._on_signup)
        lay.addWidget(btn_su)

    # ── SkullApp logo ────────────────────────────────────────────────────── #

    def _build_logo(self, sh: int) -> None:
        lbl_img = QLabel(self)
        lbl_img.setStyleSheet("background:transparent;")
        pix = _tinted_pixmap(_LOGO_PATH, 130, _WHITE)
        if not pix.isNull():
            lbl_img.setPixmap(pix)
        else:
            lbl_img.setText("💀")
            lbl_img.setStyleSheet("font-size:96px; background:transparent;")
        lbl_img.adjustSize()
        lbl_img.move(20, sh - lbl_img.height() - 16)

    # ── Credential card ──────────────────────────────────────────────────── #

    def _build_card(self) -> None:
        outer = QVBoxLayout(self._card)
        outer.setContentsMargins(32, 14, 32, 28)
        outer.setSpacing(0)

        # Close button
        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_x = QPushButton("✕")
        btn_x.setFixedSize(28, 28)
        btn_x.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;"
            f"color:{_MUTED};font-size:14px; min-height:0;}}"
            f"QPushButton:hover{{color:{_WHITE};}}"
        )
        btn_x.clicked.connect(self._card.hide)
        close_row.addWidget(btn_x)
        outer.addLayout(close_row)

        if self._auth.has_any_user():
            self._fill_login_form(outer)
        else:
            self._fill_first_run_form(outer)

    def _fill_login_form(self, lay: QVBoxLayout) -> None:
        # Avatar centrado
        av_row = QHBoxLayout()
        av_row.addStretch()
        av = AvatarWidget(
            size=64,
            bg_color="rgba(10,18,30,200)",
            icon_color=_MUTED,
            border_color=_INPUT_BORD,
        )
        av_row.addWidget(av)
        av_row.addStretch()
        lay.addLayout(av_row)
        lay.addSpacing(16)

        lay.addWidget(_lbl("Bienvenido de nuevo", 17, bold=True))
        lay.addSpacing(4)
        lay.addWidget(_lbl("Inicia sesión en PROSPECTIVE", 10, color=_MUTED))
        lay.addSpacing(24)

        # Separador sutil
        _sep_line(lay)
        lay.addSpacing(20)

        # Username field
        lay.addWidget(_lbl("Usuario", 9, color=_MUTED))
        lay.addSpacing(4)
        usr_w, self._username_edit = _field_row("👤", "Nombre de usuario")
        lay.addWidget(usr_w)
        lay.addSpacing(12)

        # Password field
        lay.addWidget(_lbl("Contraseña", 9, color=_MUTED))
        lay.addSpacing(4)
        pw_w, self._password_edit = _field_row("🔒", "••••••••", pw=True)
        self._password_edit.returnPressed.connect(self._do_login)
        lay.addWidget(pw_w)
        lay.addSpacing(6)

        # Error label
        self._err_lbl = QLabel("")
        self._err_lbl.setWordWrap(True)
        self._err_lbl.setStyleSheet(
            f"color:{_RED}; font-size:10px; background:transparent; border:none;"
        )
        lay.addWidget(self._err_lbl)
        lay.addSpacing(16)

        # CTA
        btn_in = QPushButton("Iniciar sesión")
        btn_in.setMinimumHeight(44)
        btn_in.setStyleSheet(_BTN_QSS)
        btn_in.clicked.connect(self._do_login)
        lay.addWidget(btn_in)
        lay.addSpacing(10)

        # Forgot password
        btn_fp = QPushButton("¿Olvidaste tu contraseña?")
        btn_fp.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;"
            f"color:{_MUTED};font-size:10px;text-decoration:underline; min-height:0;}}"
            f"QPushButton:hover{{color:{_ACCENT};}}"
        )
        btn_fp.clicked.connect(self._on_forgot)
        lay.addWidget(btn_fp, alignment=Qt.AlignCenter)
        lay.addStretch()

        # Branding footer
        _sep_line(lay)
        lay.addSpacing(8)
        foot = _lbl("Hybrid Neurovascular Planning Platform  ·  UNINAVARRA", 8, color=_MUTED)
        foot.setAlignment(Qt.AlignCenter)
        lay.addWidget(foot)

        self._username_edit.setFocus()

    def _fill_first_run_form(self, lay: QVBoxLayout) -> None:
        # Header
        lay.addWidget(_lbl("¡Bienvenido a PROSPECTIVE!", 15, bold=True))
        lay.addSpacing(6)
        lay.addWidget(_lbl(
            "No hay usuarios registrados.\n"
            "Crea la cuenta de administrador para comenzar.",
            10, color=_MUTED,
        ))
        lay.addSpacing(20)
        _sep_line(lay)
        lay.addSpacing(18)

        lay.addWidget(_lbl("Usuario", 9, color=_MUTED))
        lay.addSpacing(4)
        usr_w, self._new_username = _field_row("👤", "admin")
        lay.addWidget(usr_w)
        lay.addSpacing(12)

        lay.addWidget(_lbl("Nombre completo", 9, color=_MUTED))
        lay.addSpacing(4)
        fn_w, self._new_fullname = _field_row("🏥", "Ej: Dr. García López")
        lay.addWidget(fn_w)
        lay.addSpacing(12)

        lay.addWidget(_lbl("Contraseña (mín. 8 caracteres)", 9, color=_MUTED))
        lay.addSpacing(4)
        pw1_w, self._new_password = _field_row("🔒", "contraseña", pw=True)
        lay.addWidget(pw1_w)
        lay.addSpacing(12)

        lay.addWidget(_lbl("Confirmar contraseña", 9, color=_MUTED))
        lay.addSpacing(4)
        pw2_w, self._new_password2 = _field_row("🔒", "repetir contraseña", pw=True)
        self._new_password2.returnPressed.connect(self._do_create_admin)
        lay.addWidget(pw2_w)
        lay.addSpacing(6)

        self._create_err_lbl = QLabel("")
        self._create_err_lbl.setWordWrap(True)
        self._create_err_lbl.setStyleSheet(
            f"color:{_RED}; font-size:10px; background:transparent; border:none;"
        )
        lay.addWidget(self._create_err_lbl)
        lay.addSpacing(16)

        btn = QPushButton("Crear cuenta de administrador")
        btn.setMinimumHeight(44)
        btn.setStyleSheet(_BTN_QSS)
        btn.clicked.connect(self._do_create_admin)
        lay.addWidget(btn)
        lay.addStretch()

        _sep_line(lay)
        lay.addSpacing(8)
        foot = _lbl("Hybrid Neurovascular Planning Platform  ·  UNINAVARRA", 8, color=_MUTED)
        foot.setAlignment(Qt.AlignCenter)
        lay.addWidget(foot)

        self._new_username.setFocus()

    # ── Video background ─────────────────────────────────────────────────── #

    def _start_video(self) -> None:
        if not self._vpath or not _imageio_available():
            logger.warning("LoginDialog: video not available — static background.")
            return
        self._open_reader()

    def _open_reader(self) -> None:
        try:
            import imageio
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass
            self._reader = imageio.get_reader(self._vpath)
            meta = self._reader.get_meta_data()
            fps  = float(meta.get("fps") or 25.0)
            self._frame_ms = max(1, int(1000.0 / fps))
            self._viter = iter(self._reader)
        except Exception as exc:
            logger.warning("LoginDialog: cannot open video reader — %s", exc)
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
            self._open_reader()
            return
        except Exception as exc:
            logger.warning("LoginDialog: frame error — %s", exc)
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

        img  = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
        tw   = self._bg_lbl.width()
        th   = self._bg_lbl.height()
        pix  = _QPixmap.fromImage(img).scaled(
            tw, th, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation
        )
        if pix.width() > tw or pix.height() > th:
            ox = (pix.width()  - tw) // 2
            oy = (pix.height() - th) // 2
            pix = pix.copy(ox, oy, tw, th)
        self._bg_lbl.setPixmap(pix)

    def _stop_video(self) -> None:
        if self._vtimer:
            self._vtimer.stop()
        if self._reader:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        self._viter = None

    # ── Slots ────────────────────────────────────────────────────────────── #

    def _show_card(self) -> None:
        self._card.show()
        self._card.raise_()
        target = (
            self._username_edit if hasattr(self, "_username_edit") else
            self._new_username  if hasattr(self, "_new_username")  else
            None
        )
        if target:
            target.setFocus()

    def _do_login(self) -> None:
        u = self._username_edit.text().strip()
        p = self._password_edit.text()
        if not u or not p:
            self._err_lbl.setText("⚠  Introduce usuario y contraseña.")
            return
        ok, msg = self._auth.login(u, p)
        if ok:
            self._stop_video()
            self.accept()
        else:
            self._err_lbl.setText(f"⚠  {msg}")
            self._password_edit.clear()
            self._password_edit.setFocus()

    def _do_create_admin(self) -> None:
        u   = self._new_username.text().strip()
        fn  = self._new_fullname.text().strip()
        pw1 = self._new_password.text()
        pw2 = self._new_password2.text()
        if not u:
            self._create_err_lbl.setText("⚠  El nombre de usuario es obligatorio.")
            return
        if pw1 != pw2:
            self._create_err_lbl.setText("⚠  Las contraseñas no coinciden.")
            return
        if len(pw1) < 8:
            self._create_err_lbl.setText(
                "⚠  La contraseña debe tener al menos 8 caracteres."
            )
            return
        user, err = self._auth.create_user(
            username=u, password=pw1,
            full_name=fn, role=User.ROLE_ADMIN,
        )
        if user is None:
            self._create_err_lbl.setText(f"⚠  {err}")
            return
        ok, msg = self._auth.login(u, pw1)
        if ok:
            self._stop_video()
            self.accept()
        else:
            self._create_err_lbl.setText(f"⚠  {msg}")

    def _on_signup(self) -> None:
        from prospective.ui.widgets.signup_dialog import SignUpDialog
        dlg = SignUpDialog(self)
        dlg.exec()

    def _on_forgot(self) -> None:
        QMessageBox.information(
            self, "Recuperar contraseña",
            "<b>¿Olvidaste tu contraseña?</b><br><br>"
            "Dado que PROSPECTIVE funciona de forma local, el restablecimiento "
            "de contraseña debe realizarlo un <b>administrador del sistema</b>.<br><br>"
            "Contacta al administrador de tu institución para que la restablezca "
            "desde <i>Menú → Gestión de usuarios</i>.",
        )

    # ── Window events ────────────────────────────────────────────────────── #

    def reject(self) -> None:
        self._stop_video()
        super().reject()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._stop_video()
        super().closeEvent(event)


# ── Helpers de módulo ─────────────────────────────────────────────────────── #

def _sep_line(layout: QVBoxLayout) -> None:
    """Añade una línea separadora sutil al layout."""
    line = QWidget()
    line.setFixedHeight(1)
    line.setStyleSheet("background: rgba(139,155,170,45); border:none;")
    layout.addWidget(line)
