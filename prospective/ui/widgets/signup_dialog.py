"""Sign-up dialog — cinematic full-screen registration form.

Misma estética que LoginDialog:
  - Ventana sin bordes que ocupa la pantalla completa
  - Vídeo quirúrgico en bucle como fondo
  - GlassCard centrada con el formulario de registro profesional
  - Scroll interno en la tarjeta para los múltiples campos

La lógica de negocio (validación, copia de ficheros, llamada a AuthManager)
es idéntica a la versión anterior.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from prospective.ui.glass_utils import GlassCard
from prospective.ui.widgets.avatar_widget import AvatarWidget
from prospective.auth.auth_manager import AuthManager

logger = logging.getLogger(__name__)

# ── Resources ──────────────────────────────────────────────────────────────── #
_RES       = Path(__file__).resolve().parents[3] / "resources"
_LOGO_PATH = str(_RES / "logo.png")
_VIDEO_1   = str(_RES / "Multimedia1.mov")
_VIDEO_2   = str(_RES / "shutterstock_1090107869.mp4")

# ── Layout ─────────────────────────────────────────────────────────────────── #
_HEADER_H  = 56
_CARD_W    = 530
_CARD_MARG = 20    # margen vertical sobre/bajo la tarjeta

# ── Palette (glass-on-glass) ──────────────────────────────────────────────── #
_FG        = "#d0d9ea"
_FG_MUTED  = "#8B9BAA"
_INPUT_BG  = "rgba(10,14,20,160)"
_INPUT_BD  = "rgba(139,155,170,65)"
_INPUT_FO  = "#8B9BAA"
_BTN_VIO   = "#4E6678"
_BTN_HVR   = "#8B9BAA"
_BTN_PRESS = "#1F3347"
_RED       = "#f87171"

# ── Storage ────────────────────────────────────────────────────────────────── #
_USERS_DIR  = Path.home() / ".prospective" / "users"
_PHOTOS_DIR = _USERS_DIR / "photos"
_CV_DIR     = _USERS_DIR / "cv"

# ── Listas de opciones ─────────────────────────────────────────────────────── #
_SPECIALTIES = [
    "Neurocirugía",
    "Neurorradiología intervencionista",
    "Neurología",
    "Radiología",
    "Anestesiología",
    "Medicina interna",
    "Otra",
]
_POSITIONS = [
    "Neurocirujano/a",
    "Neurorradiólogo/a",
    "Residente de Neurocirugía",
    "Residente de Radiología",
    "Fellow",
    "Médico adjunto",
    "Estudiante de medicina",
    "Investigador/a",
    "Otro",
]

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


# ── QSS reutilizables ─────────────────────────────────────────────────────── #
_INPUT_QSS = (
    f"QLineEdit{{background:{_INPUT_BG};border:1px solid {_INPUT_BD};"
    f"border-radius:11px;color:{_FG};padding:6px 12px;font-size:12px;"
    f"min-height:38px; selection-background-color:{_BTN_VIO};}}"
    f"QLineEdit:focus{{border:1px solid {_INPUT_FO}; background:rgba(10,18,30,210);}}"
)
_COMBO_QSS = (
    f"QComboBox{{background:{_INPUT_BG};border:1px solid {_INPUT_BD};"
    f"border-radius:11px;color:{_FG};padding:6px 28px 6px 12px;font-size:12px;"
    f"min-height:38px;}}"
    f"QComboBox:focus{{border:1px solid {_INPUT_FO};}}"
    f"QComboBox::drop-down{{border:none;width:22px;"
    f"subcontrol-origin:padding;subcontrol-position:right center;}}"
    f"QComboBox QAbstractItemView{{background:#0c1828;color:{_FG};"
    f"selection-background-color:{_BTN_VIO};selection-color:#fff;"
    f"outline:none;padding:4px;border-radius:8px;}}"
)
_GRP_QSS = (
    "QGroupBox{color:#8B9BAA;font-weight:bold;font-size:10px;"
    "border:1px solid rgba(139,155,170,45);border-radius:12px;"
    "background:transparent;margin-top:14px;padding-top:8px;"
    "letter-spacing:0.8px;}"
    "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;"
    "padding:2px 8px;left:10px;}"
    f"QLabel{{color:{_FG};background:transparent;font-size:12px;}}"
)
_SCROLL_QSS = (
    "QScrollArea{background:transparent;border:none;}"
    "QScrollBar:vertical{background:transparent;width:6px;margin:0;}"
    "QScrollBar::handle:vertical{background:rgba(139,155,170,90);"
    "border-radius:3px;min-height:24px;}"
    "QScrollBar::handle:vertical:hover{background:rgba(139,155,170,160);}"
    "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
)
_BTN_PRIMARY_QSS = (
    f"QPushButton{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    f"stop:0 #3B5268,stop:1 #5A7A94);"
    f"border:none;border-radius:11px;color:#fff;"
    f"font-weight:700;font-size:13px;letter-spacing:0.3px;"
    f"padding:8px 28px;}}"
    f"QPushButton:hover{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    f"stop:0 {_BTN_VIO},stop:1 #7A9AB4);}}"
    f"QPushButton:pressed{{background:#2A3F52;}}"
)
_BTN_SEC_QSS = (
    f"QPushButton{{background:rgba(5,3,15,140);"
    f"border:1px solid rgba(139,155,170,65);"
    f"border-radius:11px;color:{_FG};padding:0 20px;font-size:13px;}}"
    f"QPushButton:hover{{background:rgba(30,42,52,180);"
    f"border-color:#8B9BAA;}}"
)


# ── Helpers de módulo ─────────────────────────────────────────────────────── #

def _imageio_available() -> bool:
    try:
        import imageio  # noqa: F401
        import numpy    # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_video() -> str | None:
    for p in (_VIDEO_1, _VIDEO_2):
        if os.path.isfile(p):
            return p
    return None


def _ensure_dirs() -> None:
    _PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    _CV_DIR.mkdir(parents=True, exist_ok=True)


def _vrow(label: str, widget) -> QWidget:
    """Label-above-field compact row — gives the field the full available width."""
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    vl = QVBoxLayout(w)
    vl.setContentsMargins(0, 0, 0, 0)
    vl.setSpacing(4)
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color:{_FG_MUTED}; background:transparent; font-size:11px;")
    vl.addWidget(lbl)
    vl.addWidget(widget)
    return w


# ────────────────────────────────────────────────────────────────────────────── #
# SignUpDialog                                                                   #
# ────────────────────────────────────────────────────────────────────────────── #

class SignUpDialog(QDialog):
    """
    Ventana de registro profesional en pantalla completa.

    Layout (z-ordered):
      0 — QLabel ``_bg_lbl``  : frames del vídeo en bucle
      1 — QWidget ``_header`` : barra negra superior con título y botón de volver
      1 — QLabel  ``_logo``   : logo SkullApp (esquina inferior izquierda)
      2 — GlassCard ``_card`` : formulario scrolleable centrado en la pantalla
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._photo_src: str = ""
        self._cv_src:    str = ""
        self._auth       = AuthManager.instance()

        # Vídeo
        self._reader   = None
        self._viter    = None
        self._vtimer: QTimer | None = None
        self._frame_ms = 40
        self._vpath    = _resolve_video()

        self.setWindowTitle("PROSPECTIVE — Crear cuenta")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        screen = QApplication.primaryScreen().geometry()
        sw, sh = screen.width(), screen.height()
        self.setFixedSize(sw, sh)
        self.move(screen.topLeft())

        # ── Capa 0: fondo de vídeo ────────────────────────────────────────── #
        self._bg_lbl = QLabel(self)
        self._bg_lbl.setGeometry(0, _HEADER_H, sw, sh - _HEADER_H)
        self._bg_lbl.setStyleSheet("background:#000;")

        # ── Capa 1: barra de navegación superior ──────────────────────────── #
        self._build_header(sw)

        # ── Capa 1: logo SkullApp (esquina inferior izquierda) ────────────── #
        self._build_logo(sh)

        # ── Capa 2: GlassCard centrada ────────────────────────────────────── #
        card_h = min(sh - _HEADER_H - _CARD_MARG * 2, 760)
        card_x = (sw - _CARD_W) // 2
        card_y = _HEADER_H + _CARD_MARG
        self._card = GlassCard(self, radius=16)
        self._card.setGeometry(card_x, card_y, _CARD_W, card_h)
        self._build_card()

        # ── Arrancar vídeo ────────────────────────────────────────────────── #
        self._start_video()

    # ── Construcción de capas ─────────────────────────────────────────────── #

    def _build_header(self, sw: int) -> None:
        hdr = QWidget(self)
        hdr.setGeometry(0, 0, sw, _HEADER_H)
        hdr.setStyleSheet("background:#000000;")
        hdr.setAttribute(Qt.WA_StyledBackground, True)

        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(28, 0, 28, 0)
        lay.setSpacing(0)

        title = QLabel("PROSPECTIVE")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color:#ffffff; background:transparent;")
        lay.addWidget(title)
        lay.addStretch()

        back_btn = QPushButton("← Volver al Login")
        back_btn.setFixedHeight(34)
        back_btn.setStyleSheet(
            "QPushButton{background:transparent;border:none;"
            "color:#ffffff;font-size:13px;font-weight:bold;padding:0 20px;}"
            "QPushButton:hover{color:#8B9BAA;}"
        )
        back_btn.clicked.connect(self.reject)
        lay.addWidget(back_btn)

    def _build_logo(self, sh: int) -> None:
        # Logo sits on the dark video background → white tint for maximum contrast
        # Size 130 matches LoginDialog so both screens look identical.
        lbl = QLabel(self)
        lbl.setStyleSheet("background:transparent;")
        pix = _tinted_pixmap(_LOGO_PATH, 130, "#ffffff")
        if not pix.isNull():
            lbl.setPixmap(pix)
        else:
            lbl.setText("💀")
            lbl.setStyleSheet("font-size:96px; background:transparent;")
        lbl.adjustSize()
        lbl.move(20, sh - lbl.height() - 16)

    def _build_card(self) -> None:
        outer = QVBoxLayout(self._card)
        outer.setContentsMargins(28, 10, 22, 18)
        outer.setSpacing(0)

        # ── Botón cerrar (✕) ─────────────────────────────────────────────── #
        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_x = QPushButton("✕")
        btn_x.setFixedSize(28, 28)
        btn_x.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;"
            f"color:{_FG_MUTED};font-size:14px;}}"
            f"QPushButton:hover{{color:#ffffff;}}"
        )
        btn_x.clicked.connect(self.reject)
        close_row.addWidget(btn_x)
        outer.addLayout(close_row)

        # ── Cabecera de la tarjeta ────────────────────────────────────────── #
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(14)

        self._photo_preview = AvatarWidget(
            size=52,
            bg_color="rgba(5,3,15,160)",
            icon_color=_FG_MUTED,
            border_color="rgba(139,155,170,65)",
        )
        hdr_row.addWidget(self._photo_preview)

        title_col = QVBoxLayout()
        title_col.setSpacing(3)
        lbl_title = QLabel("Crear cuenta profesional")
        lbl_title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        lbl_title.setStyleSheet("color:#ffffff; background:transparent;")
        lbl_sub = QLabel("Regístrate para acceder a PROSPECTIVE")
        lbl_sub.setStyleSheet(
            f"color:{_FG_MUTED}; font-size:11px; background:transparent;"
        )
        title_col.addWidget(lbl_title)
        title_col.addWidget(lbl_sub)
        hdr_row.addLayout(title_col)
        hdr_row.addStretch()
        outer.addLayout(hdr_row)
        outer.addSpacing(12)

        # Separador
        _sep(outer)
        outer.addSpacing(8)

        # ── Área scrolleable con el formulario ────────────────────────────── #
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet(_SCROLL_QSS)
        scroll.viewport().setStyleSheet("background: transparent;")
        scroll.viewport().setAttribute(Qt.WA_TranslucentBackground)

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body.setAttribute(Qt.WA_TranslucentBackground)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 6, 0)
        bl.setSpacing(12)

        # ── Sección 1: Datos personales ──────────────────────────────────── #
        grp1 = QGroupBox("Datos personales")
        grp1.setStyleSheet(_GRP_QSS)
        g1l = QVBoxLayout(grp1)
        g1l.setContentsMargins(12, 14, 12, 12)
        g1l.setSpacing(8)

        self._f_fullname    = self._field("Nombre completo")
        self._f_national_id = self._field("Cédula o ID")
        self._f_photo_btn   = self._file_btn("Seleccionar foto…", self._pick_photo)

        g1l.addWidget(_vrow("Nombre completo *", self._f_fullname))
        g1l.addWidget(_vrow("Cédula o ID", self._f_national_id))
        g1l.addWidget(_vrow("Foto de perfil", self._f_photo_btn))
        bl.addWidget(grp1)

        # ── Sección 2: Datos profesionales ───────────────────────────────── #
        grp2 = QGroupBox("Datos profesionales")
        grp2.setStyleSheet(_GRP_QSS)
        g2l = QVBoxLayout(grp2)
        g2l.setContentsMargins(12, 14, 12, 12)
        g2l.setSpacing(8)

        self._f_prof_id    = self._field("ID Profesional")
        self._f_specialty  = QComboBox()
        self._f_specialty.addItems(_SPECIALTIES)
        self._f_specialty.setStyleSheet(_COMBO_QSS)
        self._f_university = self._field("Universidad afiliada")
        self._f_hospital   = self._field("Hospital / Centro")
        self._f_position   = QComboBox()
        self._f_position.addItems(_POSITIONS)
        self._f_position.setStyleSheet(_COMBO_QSS)
        self._f_orcid      = self._field("0000-0002-1825-0097")
        self._f_cv_btn     = self._file_btn("Adjuntar CV…", self._pick_cv)

        g2l.addWidget(_vrow("ID Profesional", self._f_prof_id))
        g2l.addWidget(_vrow("Especialidad", self._f_specialty))
        g2l.addWidget(_vrow("Universidad", self._f_university))
        g2l.addWidget(_vrow("Hospital / Centro", self._f_hospital))
        g2l.addWidget(_vrow("Cargo", self._f_position))
        g2l.addWidget(_vrow("ORCID", self._f_orcid))
        g2l.addWidget(_vrow("CV", self._f_cv_btn))
        bl.addWidget(grp2)

        # ── Sección 3: Credenciales ───────────────────────────────────────── #
        grp3 = QGroupBox("Credenciales de acceso")
        grp3.setStyleSheet(_GRP_QSS)
        g3l = QVBoxLayout(grp3)
        g3l.setContentsMargins(12, 14, 12, 12)
        g3l.setSpacing(8)

        self._f_username = self._field("Nombre de usuario")
        self._f_pw1      = self._field("Contraseña", password=True)
        self._f_pw2      = self._field("Confirmar contraseña", password=True)
        self._f_pw2.returnPressed.connect(self._do_register)

        g3l.addWidget(_vrow("Usuario *", self._f_username))
        g3l.addWidget(_vrow("Contraseña *", self._f_pw1))
        g3l.addWidget(_vrow("Confirmar contraseña *", self._f_pw2))
        bl.addWidget(grp3)

        # ── Aviso pendiente de aprobación ─────────────────────────────────── #
        notice = QLabel(
            "⚠  Tu cuenta quedará <b>pendiente de aprobación</b> hasta que "
            "un administrador la active."
        )
        notice.setWordWrap(True)
        notice.setStyleSheet(
            "background:rgba(210,148,0,16); border:1px solid rgba(210,148,0,75);"
            "border-radius:11px; padding:10px 12px; color:#ffcc44; font-size:11px;"
        )
        bl.addWidget(notice)

        # ── Label de error ────────────────────────────────────────────────── #
        self._error_lbl = QLabel("")
        self._error_lbl.setWordWrap(True)
        self._error_lbl.setStyleSheet(
            f"color:{_RED}; font-size:10px; background:transparent; border:none;"
        )
        bl.addWidget(self._error_lbl)
        bl.addStretch()

        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=1)
        outer.addSpacing(10)

        # Separador sobre botones
        _sep(outer)
        outer.addSpacing(12)

        # ── Botones de acción ─────────────────────────────────────────────── #
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        cancel_btn = QPushButton("Cancelar")
        cancel_btn.setMinimumHeight(42)
        cancel_btn.setStyleSheet(_BTN_SEC_QSS)
        cancel_btn.clicked.connect(self.reject)

        register_btn = QPushButton("Enviar solicitud de registro")
        register_btn.setMinimumHeight(42)
        register_btn.setStyleSheet(_BTN_PRIMARY_QSS)
        register_btn.clicked.connect(self._do_register)

        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(register_btn)
        outer.addLayout(btn_row)

        # ── Branding footer ───────────────────────────────────────────────── #
        outer.addSpacing(10)
        foot = QLabel("Hybrid Neurovascular Planning Platform  ·  UNINAVARRA")
        foot.setAlignment(Qt.AlignCenter)
        foot.setStyleSheet(
            f"color:rgba(139,155,170,120); font-size:9px; background:transparent; border:none;"
        )
        outer.addWidget(foot)

    # ── Reproducción de vídeo ─────────────────────────────────────────────── #

    def _start_video(self) -> None:
        if not self._vpath or not _imageio_available():
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
            self._reader  = imageio.get_reader(self._vpath)
            meta          = self._reader.get_meta_data()
            fps           = float(meta.get("fps") or 25.0)
            self._frame_ms = max(1, int(1000.0 / fps))
            self._viter   = iter(self._reader)
        except Exception as exc:
            logger.warning("SignUpDialog: cannot open video reader — %s", exc)
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
            logger.warning("SignUpDialog: frame error — %s", exc)
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
        tw, th = self._bg_lbl.width(), self._bg_lbl.height()
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
            self._reader = None   # release ffmpeg subprocess reference
        self._viter = None        # release iterator (holds ref to reader)

    # ── Helpers de campo ──────────────────────────────────────────────────── #

    @staticmethod
    def _field(placeholder: str = "", password: bool = False) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setMinimumHeight(38)
        if password:
            edit.setEchoMode(QLineEdit.Password)
        edit.setStyleSheet(_INPUT_QSS)
        return edit

    @staticmethod
    def _file_btn(label: str, slot) -> QPushButton:
        from prospective.ui.icons import I as _I
        btn = QPushButton(f"{_I.ATTACH}  {label}")
        btn.setMinimumHeight(38)
        btn.setFixedHeight(38)
        btn.setStyleSheet(
            f"QPushButton{{background:{_INPUT_BG};border:1px solid {_INPUT_BD};"
            f"border-radius:11px;color:{_FG_MUTED};padding:0px 12px;"
            f"text-align:left;font-size:12px;min-height:38px;}}"
            f"QPushButton:hover{{background:rgba(30,42,52,180);"
            f"color:{_FG};border-color:{_INPUT_FO};}}"
        )
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.clicked.connect(slot)
        return btn

    def _pick_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar foto de perfil", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
        )
        if not path:
            return
        self._photo_src = path
        self._f_photo_btn.setText(f"✔  {Path(path).name}")
        self._photo_preview.set_photo(path)

    def _pick_cv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Adjuntar Currículum Vitae", "",
            "Documentos (*.pdf *.doc *.docx)",
        )
        if not path:
            return
        self._cv_src = path
        self._f_cv_btn.setText(f"✔  {Path(path).name}")

    def _copy_file(self, src: str, dest_dir: Path, username: str) -> str:
        if not src:
            return ""
        _ensure_dirs()
        suffix = Path(src).suffix
        dest   = dest_dir / f"{username}{suffix}"
        shutil.copy2(src, dest)
        return str(dest)

    # ── Lógica de registro ────────────────────────────────────────────────── #

    def _do_register(self) -> None:
        username  = self._f_username.text().strip()
        full_name = self._f_fullname.text().strip()
        pw1       = self._f_pw1.text()
        pw2       = self._f_pw2.text()

        if not full_name:
            self._error_lbl.setText("El nombre completo es obligatorio.")
            self._f_fullname.setFocus()
            return
        if not username:
            self._error_lbl.setText("El nombre de usuario es obligatorio.")
            self._f_username.setFocus()
            return
        if not pw1:
            self._error_lbl.setText("La contraseña es obligatoria.")
            self._f_pw1.setFocus()
            return
        if pw1 != pw2:
            self._error_lbl.setText("Las contraseñas no coinciden.")
            self._f_pw2.setFocus()
            return
        if len(pw1) < 8:
            self._error_lbl.setText("La contraseña debe tener al menos 8 caracteres.")
            self._f_pw1.setFocus()
            return

        photo_path = self._copy_file(self._photo_src, _PHOTOS_DIR, username)
        cv_path    = self._copy_file(self._cv_src,    _CV_DIR,     username)

        user, err = self._auth.request_signup(
            username=username,
            password=pw1,
            full_name=full_name,
            national_id=self._f_national_id.text().strip(),
            professional_id=self._f_prof_id.text().strip(),
            specialty=self._f_specialty.currentText(),
            university=self._f_university.text().strip(),
            hospital=self._f_hospital.text().strip(),
            position=self._f_position.currentText(),
            orcid=self._f_orcid.text().strip(),
            cv_path=cv_path,
            photo_path=photo_path,
        )

        if user is None:
            self._error_lbl.setText(err)
            return

        self._stop_video()
        QMessageBox.information(
            self, "Solicitud enviada",
            f"<b>¡Solicitud enviada correctamente!</b><br><br>"
            f"Hola, <b>{full_name}</b>.<br>"
            f"Tu cuenta <b>{username}</b> ha sido registrada con éxito.<br><br>"
            f"Un administrador revisará tu solicitud y recibirás acceso "
            f"en cuanto sea aprobada.",
        )
        self.accept()

    # ── Eventos de ventana ────────────────────────────────────────────────── #

    def reject(self) -> None:            # noqa: A003
        self._stop_video()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_video()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)


# ── Separador interno ─────────────────────────────────────────────────────── #

def _sep(layout: QVBoxLayout) -> None:
    """Añade un separador violeta suave al layout."""
    line = QWidget()
    line.setFixedHeight(1)
    line.setStyleSheet("background: rgba(139,155,170,50);")
    layout.addWidget(line)
