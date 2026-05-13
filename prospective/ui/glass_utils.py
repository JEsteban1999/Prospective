"""
glass_utils.py — Glassmorphism + Liquid Glass primitives para PROSPECTIVE (PyQt5 / Windows).

Public API (extended)
---------------------
dialog_qss() -> str
    Returns a complete, theme-aware QSS stylesheet for utility dialogs
    (patient manager, user manager, form sub-dialogs, etc.).  Apply once
    with ``dialog.setStyleSheet(dialog_qss())``.  Inline per-widget
    setStyleSheet() calls win over this global sheet, so special-purpose
    buttons (approve/reject) can still override individual colours.

enable_acrylic(widget, tint_abgr=None)
    Aplica Windows DWM Acrylic blur-behind.

GlassCard(parent, *, bg=None, border=None, radius=18, highlight=True)
    QFrame pintado como tarjeta glassmorphism + Liquid Glass.

Ambos temas (dark y light) obtienen efectos glass:

  Dark  — tarjeta slate-navy translúcida, acrylic oscuro tintado con gris humo.
  Light — tarjeta blanca esmerilada (frosted white) con borde gris humo sutil,
           acrylic casi transparente con tinte claro.

Public API
----------
enable_acrylic(widget, tint_abgr=None)
    Aplica Windows DWM Acrylic blur-behind.  Si tint_abgr es None, elige
    automáticamente el tinte según el tema activo.

GlassCard(parent, *, bg=None, border=None, radius=18, highlight=True)
    QFrame pintado como tarjeta glassmorphism + Liquid Glass.
    Detecta el tema actual en cada paintEvent → reacciona a cambios de tema
    sin necesidad de recrear la tarjeta.

    Capas (abajo → arriba):
      1. Sombra suave (10 pasadas exponenciales)
      2. Relleno semi-transparente (slate-navy en dark; blanco-gris en light)
      3. Gradiente especular estático (top ~40 %)
      4. Línea brillante en top edge (Liquid Glass edge highlight)
      3b. Mancha especular dinámica que sigue al cursor (Liquid Glass)
      5. Borde angular (gradiente diagonal top-left brillante → bottom-right tenue)
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen,
    QRadialGradient,
)
from PyQt5.QtWidgets import QFrame


# ── Windows DWM: Acrylic blur-behind ─────────────────────────────────────────

class _ACCENT_POLICY(ctypes.Structure):
    _fields_ = [
        ("AccentState",    wt.INT),
        ("AccentFlags",    wt.INT),
        ("GradientColor",  wt.INT),   # 0xAABBGGRR
        ("AnimationId",    wt.INT),
    ]


class _WCA_DATA(ctypes.Structure):
    _fields_ = [
        ("Attribute",   wt.DWORD),
        ("Data",        ctypes.c_void_p),
        ("SizeOfData",  wt.DWORD),
    ]


_ACCENT_ENABLE_ACRYLIC = 4
_WCA_ACCENT_POLICY     = 19


def enable_acrylic(widget, tint_abgr: int | None = None) -> bool:
    """
    Aplica Windows Acrylic (blur-behind) a la ventana nativa del widget.

    Parameters
    ----------
    widget      : Cualquier QWidget cuyo winId() exponga un HWND real.
    tint_abgr   : Color 32-bit 0xAABBGGRR mezclado sobre el blur.
                  Si None (default), elige automáticamente según el tema activo:
                    dark  → 0x55_2A_20_1C (violeta oscuro, ~33 % opacidad)
                    light → 0x18_FF_FA_F7 (blanco-violeta, ~10 % opacidad)

    Devuelve True si OK, False si no es Windows 10+ o la llamada falló.
    """
    if sys.platform != "win32":
        return False
    if tint_abgr is None:
        try:
            from prospective.ui.themes import acrylic_tint
            tint_abgr = acrylic_tint()
        except Exception:
            tint_abgr = 0x55_2A_20_1C
    try:
        accent = _ACCENT_POLICY()
        accent.AccentState   = _ACCENT_ENABLE_ACRYLIC
        accent.AccentFlags   = 2          # bordes redondeados
        accent.GradientColor = tint_abgr
        data = _WCA_DATA()
        data.Attribute  = _WCA_ACCENT_POLICY
        data.Data       = ctypes.cast(ctypes.pointer(accent), ctypes.c_void_p)
        data.SizeOfData = ctypes.sizeof(accent)
        ctypes.windll.user32.SetWindowCompositionAttribute(
            int(widget.winId()), ctypes.byref(data)
        )
        return True
    except Exception:
        return False


# ── GlassCard ─────────────────────────────────────────────────────────────────

def _is_dark() -> bool:
    """Return current theme darkness without raising on import errors."""
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


class GlassCard(QFrame):
    """
    QFrame pintado como tarjeta Glassmorphism / Liquid Glass.

    Soporta ambos temas:
      dark  — relleno violeta-navy oscuro semi-transparente + borde violeta luminoso.
      light — relleno blanco esmerilado (frosted white) + borde violeta sutil.

    El color se recalcula en cada paintEvent, por lo que la tarjeta responde
    automáticamente a cambios de tema sin necesitar recrearse.

    Pasa bg / border explícitos para sobreescribir el color automático.

    Parameters
    ----------
    parent    : Widget padre.
    bg        : QColor de relleno explícito (None = automático según tema).
    border    : QColor de borde explícito (None = automático según tema).
    radius    : Radio de esquinas en px (default 18).
    highlight : Dibujar gradiente especular + edge highlight (default True).
    """

    def __init__(
        self,
        parent=None,
        *,
        bg:        QColor | None = None,
        border:    QColor | None = None,
        radius:    int  = 18,
        highlight: bool = True,
    ) -> None:
        super().__init__(parent)
        # None = use theme-default colours, computed dynamically in paintEvent
        self._bg_custom     = bg
        self._border_custom = border
        self._radius        = radius
        self._highlight     = highlight
        # Liquid Glass: posición del cursor dentro de la tarjeta
        self._cursor_pos: QPointF | None = None
        # WA_TranslucentBackground alone still lets Qt pre-fill the widget rect
        # with the parent's solid background before paintEvent, making the square
        # widget corners visible ("square shadow" artefact).  WA_NoSystemBackground
        # prevents that pre-fill so paintEvent starts with the parent's already-
        # composited pixels — giving truly transparent corners.
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFrameShape(QFrame.NoFrame)
        self.setMouseTracking(True)

    # ── Colores por tema ──────────────────────────────────────────────────────

    def _resolve_colors(self) -> tuple[QColor, QColor]:
        """Return (bg, border) appropriate for the current theme."""
        dark = _is_dark()
        if dark:
            bg     = self._bg_custom     or QColor(10,  15, 22,  160)  # slate-navy dark
            border = self._border_custom or QColor(139, 155, 170, 85)  # smoke grey
        else:
            bg     = self._bg_custom     or QColor(245, 248, 252, 200) # frosted white-grey
            border = self._border_custom or QColor(139, 155, 170, 50)  # smoke grey subtle
        return bg, border

    # ── Eventos de ratón — Liquid Glass specular tracking ─────────────────────

    def mouseMoveEvent(self, event) -> None:         # noqa: N802
        self._cursor_pos = QPointF(event.pos())
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:             # noqa: N802
        self._cursor_pos = None
        self.update()
        super().leaveEvent(event)

    # ── Pintado ───────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:             # noqa: N802
        dark = _is_dark()
        bg, border = self._resolve_colors()

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # _SPREADS must be ≤ the margin used in adjusted() below so the shadow
        # never reaches the rectangular widget boundary and gets clipped to a
        # straight edge (which would recreate the "square shadow" artefact).
        _SPREADS = 4
        _MARGIN  = _SPREADS + 2          # widget-edge margin; card is inset this far
        r   = QRectF(self.rect()).adjusted(_MARGIN, _MARGIN, -_MARGIN, -_MARGIN)
        rad = float(self._radius)

        # ── 1. Sombra suave ───────────────────────────────────────────────── #
        # Clip to a rounded region that stays INSIDE the widget bounds so Qt
        # never clips the shadow at the rectangular widget edge.
        shadow_clip = QPainterPath()
        shadow_clip.addRoundedRect(
            r.adjusted(-_SPREADS, -_SPREADS, _SPREADS, _SPREADS),
            rad + _SPREADS, rad + _SPREADS,
        )
        p.setClipPath(shadow_clip)

        shadow_alpha = 70 if dark else 35
        # Paint from outermost (lightest) to innermost (darkest).
        for spread in range(_SPREADS, 0, -1):
            t = (_SPREADS - spread + 1) / _SPREADS
            alpha = int(shadow_alpha * t ** 1.8)
            sr = r.adjusted(-spread, -spread, spread, spread)
            sp = QPainterPath()
            sp.addRoundedRect(sr, rad + spread, rad + spread)
            p.fillPath(sp, QBrush(QColor(0, 0, 0, alpha)))

        # ── 2. Relleno (clipped) ─────────────────────────────────────────── #
        card_path = QPainterPath()
        card_path.addRoundedRect(r, rad, rad)
        p.setClipPath(card_path)
        p.fillPath(card_path, QBrush(bg))

        # ── 3. Especular estático ─────────────────────────────────────────── #
        if self._highlight:
            hi = QLinearGradient(0, r.top(), 0, r.top() + r.height() * 0.40)
            if dark:
                hi.setColorAt(0.0, QColor(190, 210, 225, 55))  # slate-blue specular
                hi.setColorAt(0.5, QColor(190, 210, 225, 18))
                hi.setColorAt(1.0, QColor(190, 210, 225,  0))
            else:
                hi.setColorAt(0.0, QColor(255, 255, 255, 90))  # white
                hi.setColorAt(0.4, QColor(235, 240, 245, 30))  # neutral grey sutil
                hi.setColorAt(1.0, QColor(235, 240, 245,  0))
            p.fillPath(card_path, QBrush(hi))

            # ── 4. Línea brillante de 2 px en top edge ────────────────────── #
            edge = QPainterPath()
            edge.addRect(QRectF(r.left() + rad, r.top(), r.width() - 2 * rad, 2.0))
            edge_color = QColor(180, 205, 220, 180) if dark else QColor(255, 255, 255, 200)
            p.fillPath(edge, QBrush(edge_color))

        # ── 3b. Especular dinámico (Liquid Glass) ────────────────────────── #
        if self._cursor_pos is not None:
            cx = max(r.left() + 8.0, min(r.right()  - 8.0, self._cursor_pos.x()))
            cy = max(r.top()  + 8.0, min(r.bottom() - 8.0, self._cursor_pos.y()))
            spot = QRadialGradient(cx, cy, r.width() * 0.55)
            if dark:
                spot.setColorAt(0.00, QColor(175, 200, 218, 50))
                spot.setColorAt(0.30, QColor(175, 200, 218, 18))
                spot.setColorAt(1.00, QColor(175, 200, 218,  0))
            else:
                spot.setColorAt(0.00, QColor(139, 155, 170, 35))
                spot.setColorAt(0.30, QColor(139, 155, 170, 12))
                spot.setColorAt(1.00, QColor(139, 155, 170,  0))
            p.fillPath(card_path, QBrush(spot))

        # ── 5. Borde angular — brighter top-left, dimmer bottom-right ──── #
        # Simulates light reflecting on the chamfered edge of the glass pane.
        p.setClipping(False)
        bdr = QLinearGradient(r.topLeft(), r.bottomRight())
        if dark:
            bdr.setColorAt(0.00, QColor(200, 218, 232, 170))  # bright specular top-left
            bdr.setColorAt(0.45, QColor(139, 155, 170,  90))  # mid smoke
            bdr.setColorAt(1.00, QColor( 70,  90, 105,  35))  # dim shadow bottom-right
        else:
            bdr.setColorAt(0.00, QColor(255, 255, 255, 220))  # bright white top-left
            bdr.setColorAt(0.45, QColor(139, 155, 170,  70))  # mid smoke
            bdr.setColorAt(1.00, QColor(139, 155, 170,  22))  # dim bottom-right
        p.setPen(QPen(QBrush(bdr), 1.25))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r, rad, rad)
        p.end()


# ── Shared utility dialog stylesheet ─────────────────────────────────────────


def dialog_qss() -> str:
    """Return a complete, theme-aware QSS for all utility dialogs.

    Covers QDialog / QWidget backgrounds, QLabel, QLineEdit / QTextEdit /
    QComboBox inputs, QTableWidget + QHeaderView, QPushButton, QGroupBox,
    QTabWidget / QTabBar, QSplitter, QScrollBar, and QCheckBox.

    Inline ``setStyleSheet()`` calls on individual widgets take precedence
    over this sheet, so per-button colour overrides (approve/reject, etc.)
    continue to work as before.
    """
    dark = _is_dark()

    if dark:
        bg        = "#0B1220"
        bg_input  = "rgba(8,16,30,225)"
        bg_alt    = "rgba(14,24,42,200)"
        bg_header = "rgba(16,28,50,230)"
        bg_btn    = "rgba(18,30,52,220)"
        bg_hover  = "rgba(78,102,120,210)"
        bg_sel    = "rgba(78,102,120,170)"
        bg_tab_on = "rgba(58,82,100,160)"
        border    = "rgba(139,155,170,65)"
        border_h  = "#8B9BAA"
        txt       = "#D0D9EA"
        txt_muted = "#8B9BAA"
        grid      = "rgba(139,155,170,28)"
    else:
        bg        = "#F4F7FA"
        bg_input  = "#FFFFFF"
        bg_alt    = "#EEF2F7"
        bg_header = "#E4ECF4"
        bg_btn    = "#FFFFFF"
        bg_hover  = "#DDE5EC"
        bg_sel    = "rgba(78,102,120,100)"
        bg_tab_on = "rgba(78,102,120,55)"
        border    = "#C8D4DC"
        border_h  = "#8B9BAA"
        txt       = "#0D1520"
        txt_muted = "#4E6678"
        grid      = "#E4ECF2"

    return f"""
/* ── Base ─────────────────────────────────────────────────────── */
QDialog, QWidget {{
    background: {bg};
    color: {txt};
    font-family: 'Inter', 'Segoe UI Symbol', 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
}}

/* ── Labels ───────────────────────────────────────────────────── */
QLabel {{
    color: {txt};
    background: transparent;
    font-size: 13px;
}}
QLabel[role="muted"] {{
    color: {txt_muted};
    font-size: 11px;
}}
QLabel[role="title"] {{
    color: {txt};
    font-size: 14px;
    font-weight: bold;
}}

/* ── Text inputs ──────────────────────────────────────────────── */
QLineEdit, QTextEdit {{
    background: {bg_input};
    border: 1px solid {border};
    border-radius: 9px;
    color: {txt};
    padding: 5px 10px;
    font-size: 13px;
    selection-background-color: rgba(78,102,120,180);
    selection-color: #ffffff;
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {border_h};
}}
QLineEdit:read-only {{
    color: {txt_muted};
}}

/* ── Combo box ────────────────────────────────────────────────── */
QComboBox {{
    background: {bg_input};
    border: 1px solid {border};
    border-radius: 9px;
    color: {txt};
    padding: 5px 28px 5px 10px;
    font-size: 13px;
    min-height: 28px;
}}
QComboBox:focus {{
    border-color: {border_h};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
    subcontrol-origin: padding;
    subcontrol-position: right center;
}}
QComboBox QAbstractItemView {{
    background: {bg_input};
    border: 1px solid {border};
    color: {txt};
    selection-background-color: rgba(78,102,120,160);
    selection-color: #ffffff;
    outline: none;
    padding: 2px;
}}

/* ── Table ────────────────────────────────────────────────────── */
QTableWidget {{
    background: {bg_input};
    alternate-background-color: {bg_alt};
    border: 1px solid {border};
    border-radius: 10px;
    color: {txt};
    gridline-color: {grid};
    selection-background-color: {bg_sel};
    outline: none;
}}
QTableWidget::item {{
    padding: 5px 8px;
    border: none;
}}
QTableWidget::item:selected {{
    background: {bg_sel};
    color: #ffffff;
}}
QHeaderView::section {{
    background: {bg_header};
    color: {txt_muted};
    border: none;
    border-bottom: 1px solid {border};
    border-right: 1px solid {grid};
    padding: 7px 8px;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.3px;
}}
QHeaderView::section:last-child {{
    border-right: none;
}}

/* ── Buttons ──────────────────────────────────────────────────── */
QPushButton {{
    background: {bg_btn};
    border: 1px solid {border};
    border-radius: 9px;
    color: {txt};
    padding: 6px 16px;
    font-size: 13px;
}}
QPushButton:hover {{
    background: {bg_hover};
    border-color: {border_h};
    color: #ffffff;
}}
QPushButton:pressed {{
    background: rgba(40,65,95,230);
    color: #ffffff;
}}
QPushButton:disabled {{
    color: {txt_muted};
    border-color: {border};
    background: {bg_alt};
}}
QPushButton:default {{
    border-color: {border_h};
    font-weight: 700;
}}

/* ── Group box ────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {border};
    border-radius: 10px;
    color: {txt_muted};
    font-weight: 700;
    font-size: 11px;
    margin-top: 12px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 12px;
    color: {border_h};
}}

/* ── Tab widget ───────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 10px;
    background: {bg};
    top: -1px;
}}
QTabBar::tab {{
    background: {bg_btn};
    border: 1px solid {border};
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    color: {txt_muted};
    padding: 6px 10px;
    font-size: 11px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {bg_tab_on};
    color: {txt};
    border-color: {border_h};
}}
QTabBar::tab:hover:!selected {{
    color: {txt};
    background: {bg_alt};
}}

/* ── Splitter ─────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {border};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
QSplitter::handle:vertical {{
    height: 1px;
}}

/* ── Scroll bars ──────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {border};
    border-radius: 3px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {border_h};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {border};
    border-radius: 3px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {border_h};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Check box ────────────────────────────────────────────────── */
QCheckBox {{
    color: {txt};
    spacing: 7px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {border};
    border-radius: 5px;
    background: {bg_input};
}}
QCheckBox::indicator:checked {{
    background: #4E6678;
    border-color: #8B9BAA;
}}
QCheckBox::indicator:hover {{
    border-color: {border_h};
}}

/* ── Dialog button box ────────────────────────────────────────── */
QDialogButtonBox QPushButton {{
    min-width: 88px;
    padding: 6px 18px;
}}

/* ── Status / info labels ─────────────────────────────────────── */
QStatusBar {{
    background: {bg};
    color: {txt_muted};
    font-size: 11px;
}}
"""
