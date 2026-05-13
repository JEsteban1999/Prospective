"""Application-wide visual themes for PROSPECTIVE.

Two themes, both with Liquid Glass glassmorphism effects on dialogs/cards:

  dark  — Neutral dark per IDENTIDAD_VISUAL.md  bg #1F1F1F  (default)
  light — Clean light per IDENTIDAD_VISUAL.md   bg #FFFFFF

Both themes share the smoke-gray brand colour (#8B9BAA) and apply
glassmorphism / acrylic effects through glass_utils.GlassCard and
glass_utils.enable_acrylic().

Token sources
-------------
  Design bundle 2026-05: colors_and_type.css + styles.css + extras.css
  Primary: #8B9BAA (smoke gray, hue 210)
  Typography: Inter (bundled) → Segoe UI Symbol → Segoe UI, 13 px base
  Radii (iOS scale): 6 / 10 / 14 / 18 px
  Paleta DARK y LIGHT per IDENTIDAD_VISUAL.md §2
  Fonts loaded via QFontDatabase.addApplicationFont() in app.py

Usage
-----
    from prospective.ui.themes import apply_theme, toggle_theme, current_theme

    apply_theme("dark")    # or "light"
    toggle_theme()         # cycles dark → light → dark
    is_dark()              # True when current theme is dark
    acrylic_tint()         # 0xAABBGGRR tint for enable_acrylic()
"""
from __future__ import annotations

from PyQt5.QtWidgets import QApplication

# ──────────────────────────────────────────────────────────────────────────── #
# DARK theme — IDENTIDAD_VISUAL.md dark tokens                                 #
# ──────────────────────────────────────────────────────────────────────────── #
# background     #1F1F1F  foreground   #EBEBEB
# card           #2A2A2A  card_fg      #EBEBEB
# primary        #A8B8C6  primary_fg   #1C1C1C
# secondary      #363636  secondary_fg #EBEBEB
# muted          #363636  muted_fg     #9B9B9B
# accent         #363636  accent_fg    #EBEBEB
# destructive    #E85A6A  border       #363636
# ring           #8B9BAA

_DARK = """
    * {
        font-family: "Inter", "Segoe UI Symbol", "Segoe UI", "Helvetica Neue", "Arial Unicode MS", sans-serif;
        font-size: 13px;
    }
    QMainWindow, QDialog, QWidget {
        background-color: #1F1F1F;
        color: #EBEBEB;
    }
    QFrame { background-color: transparent; border: none; }
    QMenuBar {
        background-color: #2A2A2A;
        color: #EBEBEB;
        border-bottom: 1px solid #363636;
        padding: 2px 0;
        font-size: 13px;
    }
    QMenuBar::item { padding: 4px 10px; border-radius: 6px; }
    QMenuBar::item:selected { background-color: #4E6678; color: #ffffff; }
    QMenu {
        background-color: #2A2A2A;
        border: 1px solid #363636;
        border-radius: 14px;
        padding: 4px;
        font-size: 13px;
    }
    QMenu::item { padding: 6px 28px 6px 16px; border-radius: 6px; }
    QMenu::item:selected { background-color: #4E6678; color: #ffffff; }
    QMenu::separator { height: 1px; background: #363636; margin: 4px 8px; }
    QToolBar {
        background-color: #242424;
        border-bottom: 1px solid #2d2d2d;
        padding: 3px 8px;
        spacing: 4px;
    }
    QToolBar::handle { width: 0; height: 0; image: none; }
    QToolBar::separator { width: 1px; background: #333333; margin: 6px 4px; }
    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 5px 12px;
        color: #EBEBEB;
        font-size: 13px;
        font-weight: 500;
    }
    QToolButton:hover { background-color: #363636; }
    QToolButton:pressed { background-color: #4E6678; color: #ffffff; }
    QToolButton:checked { background-color: #1C303F; color: #A8B8C6; }
    QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; color: #EBEBEB; }
    QDockWidget::title {
        background: #1e2830;
        padding: 4px 10px;
        border-left: 2px solid #4E6678;
        font-size: 9px;
        font-weight: 700;
        color: #4a5e6e;
        letter-spacing: 1.5px;
    }
    QDockWidget::close-button, QDockWidget::float-button {
        background: transparent; border: none; width: 14px; height: 14px;
    }
    QGroupBox {
        border: none;
        border-top: 1px solid #2a3540;
        border-radius: 0;
        background: transparent;
        margin-top: 24px;
        padding: 8px 6px 6px 4px;
        color: #5a7080;
        font-weight: 700;
        font-size: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 8px;
        left: 4px;
        background: #1F1F1F;
        color: #5a7080;
    }
    QLabel { color: #EBEBEB; border: none; background: transparent; }
    QPushButton {
        background-color: #363636;
        border: 1px solid #4a4a4a;
        border-radius: 10px;
        padding: 5px 14px;
        color: #EBEBEB;
        font-size: 13px;
        font-weight: 500;
        min-height: 28px;
    }
    QPushButton:hover { background-color: #404040; border-color: #A8B8C6; }
    QPushButton:pressed { background-color: #4E6678; color: #ffffff; border-color: #4E6678; }
    QPushButton:disabled { color: #5a5a5a; border-color: #363636; background-color: #2A2A2A; }
    QPushButton:flat {
        background: transparent; border: none; color: #EBEBEB;
    }
    QPushButton:flat:hover { background: #363636; }
    QCheckBox { color: #EBEBEB; spacing: 6px; }
    QCheckBox::indicator {
        width: 16px; height: 16px;
        border: 1px solid #4a4a4a;
        border-radius: 6px;
        background: #2A2A2A;
    }
    QCheckBox::indicator:checked { background: #8B9BAA; border-color: #8B9BAA; }
    QCheckBox::indicator:hover { border-color: #A8B8C6; }
    QRadioButton { color: #EBEBEB; spacing: 6px; }
    QRadioButton::indicator {
        width: 16px; height: 16px;
        border: 1px solid #4a4a4a;
        border-radius: 14px;
        background: #2A2A2A;
    }
    QRadioButton::indicator:checked { background: #8B9BAA; border-color: #8B9BAA; }
    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #2A2A2A;
        border: 1px solid #363636;
        border-radius: 10px;
        padding: 4px 8px;
        color: #EBEBEB;
        selection-background-color: #8B9BAA;
        selection-color: #1C1C1C;
        font-size: 13px;
        min-height: 26px;
    }
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #8B9BAA;
    }
    QLineEdit:disabled, QTextEdit:disabled { color: #5a5a5a; border-color: #363636; }
    QListWidget, QListView {
        background-color: #1F1F1F;
        border: 1px solid #363636;
        border-radius: 14px;
        alternate-background-color: #252525;
        selection-background-color: rgba(139,155,170,120);
        selection-color: #ffffff;
        outline: none;
        font-size: 13px;
    }
    QListWidget::item, QListView::item { padding: 5px 8px; border-radius: 6px; }
    QListWidget::item:hover, QListView::item:hover { background-color: #2A2A2A; }
    QListWidget::item:selected, QListView::item:selected {
        background-color: rgba(139,155,170,120); color: #ffffff;
    }
    QTreeWidget, QTreeView {
        background-color: #1F1F1F;
        border: 1px solid #363636;
        border-radius: 14px;
        alternate-background-color: #252525;
        selection-background-color: rgba(139,155,170,120);
        selection-color: #ffffff;
        outline: none;
    }
    QTreeWidget::item:hover, QTreeView::item:hover { background-color: #2A2A2A; }
    QTreeWidget::item:selected, QTreeView::item:selected {
        background-color: rgba(139,155,170,120); color: #ffffff;
    }
    QHeaderView::section {
        background-color: #2A2A2A;
        color: #9B9B9B;
        border: none;
        border-bottom: 1px solid #363636;
        border-right: 1px solid #363636;
        padding: 5px 8px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    QComboBox {
        background-color: #2A2A2A;
        border: 1px solid #363636;
        border-radius: 10px;
        padding: 4px 28px 4px 8px;
        color: #EBEBEB;
        selection-background-color: #4E6678;
        min-height: 28px;
    }
    QComboBox:hover { border-color: #A8B8C6; }
    QComboBox:focus { border-color: #8B9BAA; }
    QComboBox::drop-down { border: none; width: 22px; subcontrol-origin: padding; subcontrol-position: right center; }
    QComboBox QAbstractItemView {
        background-color: #2A2A2A;
        color: #EBEBEB;
        border: 1px solid #363636;
        border-radius: 10px;
        selection-background-color: #4E6678;
        selection-color: #ffffff;
        outline: none;
        padding: 4px;
    }
    QDoubleSpinBox, QSpinBox {
        background-color: #2A2A2A;
        border: 1px solid #363636;
        border-radius: 10px;
        padding: 4px 6px;
        color: #EBEBEB;
        selection-background-color: #4E6678;
        min-height: 26px;
    }
    QDoubleSpinBox:hover, QSpinBox:hover { border-color: #A8B8C6; }
    QDoubleSpinBox:focus, QSpinBox:focus { border-color: #8B9BAA; }
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
    QSpinBox::up-button, QSpinBox::down-button {
        background-color: #363636; border: none; width: 16px;
    }
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
    QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #4E6678; }
    QDateEdit, QDateTimeEdit, QTimeEdit {
        background-color: #2A2A2A;
        border: 1px solid #363636;
        border-radius: 10px;
        padding: 3px 6px;
        color: #EBEBEB;
        selection-background-color: #4E6678;
        min-height: 24px;
    }
    QDateEdit:hover, QDateTimeEdit:hover, QTimeEdit:hover { border-color: #A8B8C6; }
    QDateEdit:focus, QDateTimeEdit:focus, QTimeEdit:focus { border-color: #8B9BAA; }
    QDateEdit::drop-down, QDateTimeEdit::drop-down, QTimeEdit::drop-down {
        border: none; width: 20px;
    }
    QDateEdit::up-button, QDateEdit::down-button,
    QDateTimeEdit::up-button, QDateTimeEdit::down-button,
    QTimeEdit::up-button, QTimeEdit::down-button {
        background-color: #363636; border: none; width: 16px;
    }
    QDateEdit::up-button:hover, QDateEdit::down-button:hover,
    QDateTimeEdit::up-button:hover, QDateTimeEdit::down-button:hover,
    QTimeEdit::up-button:hover, QTimeEdit::down-button:hover { background-color: #4E6678; }
    QCalendarWidget {
        background-color: #1F1F1F;
        color: #EBEBEB;
        border: 1px solid #363636;
        border-radius: 14px;
    }
    QCalendarWidget QWidget { background-color: #1F1F1F; color: #EBEBEB; }
    QCalendarWidget QWidget#qt_calendar_navigationbar {
        background-color: #2A2A2A;
        border-bottom: 1px solid #363636;
        border-radius: 14px 14px 0 0;
        padding: 4px 6px;
    }
    QCalendarWidget QToolButton {
        background-color: transparent;
        color: #EBEBEB;
        border: none;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 13px;
        font-weight: 600;
    }
    QCalendarWidget QToolButton:hover { background-color: #363636; color: #A8B8C6; }
    QCalendarWidget QToolButton:pressed { background-color: #4E6678; color: #ffffff; }
    QCalendarWidget QSpinBox {
        background-color: #2A2A2A;
        color: #EBEBEB;
        border: 1px solid #363636;
        border-radius: 6px;
        padding: 2px 4px;
    }
    QCalendarWidget QAbstractItemView:enabled {
        background-color: #1F1F1F;
        color: #EBEBEB;
        selection-background-color: #4E6678;
        selection-color: #ffffff;
        font-size: 13px;
    }
    QCalendarWidget QAbstractItemView:disabled {
        color: #555555;
    }
    QCalendarWidget QAbstractItemView::item:hover {
        background-color: #2A2A2A;
        border-radius: 6px;
    }
    QTabWidget::pane { border: none; border-top: 1px solid #252525; background: #1F1F1F; }
    QTabWidget::tab-bar { left: 0; }
    QTabBar { background: #242424; border-bottom: 1px solid #2b2b2b; }
    QTabBar::tab {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 7px 10px 5px;
        color: #565656;
        font-size: 11px;
        font-weight: 500;
        margin-right: 2px;
    }
    QTabBar::tab:selected { color: #C0CDD8; border-bottom: 2px solid #8B9BAA; font-weight: 600; }
    QTabBar::tab:hover:!selected { color: #EBEBEB; border-bottom: 2px solid #3a4e5e; }
    QStatusBar {
        background-color: #2A2A2A;
        border-top: 1px solid #363636;
        color: #9B9B9B;
        font-size: 13px;
        padding: 2px 8px;
    }
    QProgressBar {
        background-color: #363636;
        border: none;
        border-radius: 6px;
        height: 6px;
        text-align: center;
        color: transparent;
    }
    QProgressBar::chunk { background-color: #8B9BAA; border-radius: 6px; }
    QScrollBar:vertical { background: #1F1F1F; width: 8px; margin: 0; border-radius: 6px; }
    QScrollBar::handle:vertical {
        background: #363636; border-radius: 6px; min-height: 24px;
    }
    QScrollBar::handle:vertical:hover { background: #A8B8C6; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar:horizontal { background: #1F1F1F; height: 8px; margin: 0; border-radius: 6px; }
    QScrollBar::handle:horizontal {
        background: #363636; border-radius: 6px; min-width: 24px;
    }
    QScrollBar::handle:horizontal:hover { background: #A8B8C6; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
    QScrollArea { background: transparent; border: none; }
    QSplitter::handle { background: transparent; }
    QSplitter::handle:horizontal { width: 3px; border-left: 1px solid #282828; }
    QSplitter::handle:vertical   { height: 3px; border-top:  1px solid #282828; }
    QSplitter::handle:hover { background: #4E6678; border-color: #4E6678; }
    QToolTip {
        background-color: #2A2A2A;
        color: #EBEBEB;
        border: 1px solid #363636;
        border-radius: 10px;
        padding: 6px 10px;
        font-size: 13px;
    }
    QTableWidget, QTableView {
        background-color: #2A2A2A;
        color: #EBEBEB;
        gridline-color: #363636;
        border: 1px solid #363636;
        border-radius: 14px;
        alternate-background-color: #252525;
        selection-background-color: rgba(139,155,170,120);
        selection-color: #ffffff;
    }
    QTableWidget::item, QTableView::item { padding: 4px 8px; }
    QTableWidget::item:selected, QTableView::item:selected {
        background-color: rgba(139,155,170,120); color: #ffffff;
    }
    QTableWidget QHeaderView::section, QTableView QHeaderView::section {
        background-color: #2A2A2A;
        color: #9B9B9B;
        border: none;
        border-bottom: 1px solid #363636;
        border-right: 1px solid #363636;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    QSlider::groove:horizontal { height: 4px; background: #363636; border-radius: 2px; }
    QSlider::handle:horizontal {
        width: 14px; height: 14px; margin: -5px 0;
        background: #A8B8C6; border-radius: 7px;
    }
    QSlider::handle:horizontal:hover { background: #8B9BAA; }
    QSlider::sub-page:horizontal { background: #8B9BAA; border-radius: 2px; }
    QSlider::groove:horizontal:disabled { background: #2A2A2A; }
    QSlider::handle:horizontal:disabled { background: #363636; }
    QSlider::groove:vertical { width: 4px; background: #363636; border-radius: 2px; }
    QSlider::handle:vertical {
        width: 14px; height: 14px; margin: 0 -5px;
        background: #A8B8C6; border-radius: 7px;
    }
    QSlider::sub-page:vertical { background: #8B9BAA; border-radius: 2px; }
    /* Named button variants */
    QPushButton#btn_primary {
        background: #1C303F; border: 1px solid rgba(168,184,198,80);
        color: #EBEBEB; font-weight: 700;
    }
    QPushButton#btn_primary:hover { background: #4E6678; border-color: #A8B8C6; }
    QPushButton#btn_primary:disabled { background: #2A2A2A; color: #5a5a5a; border-color: #363636; }
    QPushButton#btn_success {
        background: #1a3a1a; border: 1px solid rgba(60,180,80,80);
        color: #EBEBEB; font-weight: 700;
    }
    QPushButton#btn_success:hover { background: #1f5020; border-color: rgba(60,200,80,140); }
    QPushButton#btn_success:disabled { background: #2A2A2A; color: #5a5a5a; border-color: #363636; }
    QPushButton#btn_purple {
        background: #1C303F; border: 1px solid rgba(168,184,198,80);
        color: #EBEBEB; font-weight: 700;
    }
    QPushButton#btn_purple:hover { background: #2A4A62; border-color: rgba(185,200,215,140); }
    QPushButton#btn_purple:disabled { background: #2A2A2A; color: #5a5a5a; border-color: #363636; }
    QPushButton#btn_muted {
        background: #252525; border: 1px solid #363636; color: #9B9B9B;
    }
    QPushButton#btn_muted:hover { background: #363636; color: #EBEBEB; border-color: #A8B8C6; }
    QPushButton#btn_danger {
        background: #3a1020; border: 1px solid rgba(232,90,106,90); color: #E85A6A;
    }
    QPushButton#btn_danger:hover { background: #501525; color: #ff8f9a; }
    /* Role labels */
    QLabel[role="muted"]   { color: #9B9B9B; font-size: 13px; }
    QLabel[role="section"] { color: #9B9B9B; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
    QLabel[role="help"] {
        color: #9B9B9B; font-size: 11px;
        background: #252525; border-radius: 6px; padding: 4px 6px;
    }
    QWidget[role="adv_section"] { background: #252525; border: 1px solid #363636; border-radius: 6px; }
    QToolButton[role="adv_toggle"] {
        background: transparent; border: none;
        color: #A8B8C6; font-size: 13px; text-align: left; padding: 2px 4px;
    }
    QToolButton[role="adv_toggle"]:hover { color: #A8B8C6; }
    /* QPushButton:checked for toggle groups */
    QPushButton:checked { background-color: #1C303F; color: #A8B8C6; border-color: #8B9BAA; }
"""

# ──────────────────────────────────────────────────────────────────────────── #
# LIGHT theme — IDENTIDAD_VISUAL.md light tokens                               #
# ──────────────────────────────────────────────────────────────────────────── #
# background     #FFFFFF  foreground   #0D0D0D
# card           #F7F7F7  card_fg      #0D0D0D
# primary        #8B9BAA  primary_fg   #1C1C1C
# secondary      #F2F2F2  secondary_fg #1C1C1C
# muted          #F2F2F2  muted_fg     #6B6B6B
# accent         #DDE5EC  accent_fg    #2E4A5F
# destructive    #D7263D  border       #E5E5E5
# ring           #8B9BAA

_LIGHT = """
    * {
        font-family: "Inter", "Segoe UI Symbol", "Segoe UI", "Helvetica Neue", "Arial Unicode MS", sans-serif;
        font-size: 13px;
    }
    QMainWindow, QDialog, QWidget {
        background-color: #F4F7FA;
        color: #0D0D0D;
    }
    QFrame { background-color: transparent; border: none; }
    QMenuBar {
        background-color: #FFFFFF;
        color: #0D0D0D;
        border-bottom: 1px solid #E8EDF2;
        padding: 2px 0;
        font-size: 13px;
    }
    QMenuBar::item { padding: 4px 10px; border-radius: 6px; }
    QMenuBar::item:selected { background-color: #8B9BAA; color: #ffffff; }
    QMenu {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 14px;
        padding: 4px;
        font-size: 13px;
    }
    QMenu::item { padding: 6px 28px 6px 16px; border-radius: 6px; }
    QMenu::item:selected { background-color: #8B9BAA; color: #ffffff; }
    QMenu::separator { height: 1px; background: #E5E5E5; margin: 4px 8px; }
    QToolBar {
        background-color: #FFFFFF;
        border-bottom: 1px solid #E8EDF2;
        padding: 3px 8px;
        spacing: 4px;
    }
    QToolBar::handle { width: 0; height: 0; image: none; }
    QToolBar::separator { width: 1px; background: #E2E8EE; margin: 6px 4px; }
    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 5px 12px;
        color: #0D0D0D;
        font-size: 13px;
        font-weight: 500;
    }
    QToolButton:hover { background-color: #DDE5EC; }
    QToolButton:pressed { background-color: #8B9BAA; color: #ffffff; }
    QToolButton:checked { background-color: #DDE5EC; color: #2E4A5F; }
    QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; color: #0D0D0D; }
    QDockWidget::title {
        background: #EBF1F7;
        padding: 4px 10px;
        border-left: 2px solid #8B9BAA;
        font-size: 9px;
        font-weight: 700;
        color: #8B9BAA;
        letter-spacing: 1.5px;
    }
    QDockWidget::close-button, QDockWidget::float-button {
        background: transparent; border: none; width: 14px; height: 14px;
    }
    QGroupBox {
        border: none;
        border-top: 1px solid #DDE5ED;
        border-radius: 0;
        background: transparent;
        margin-top: 24px;
        padding: 8px 6px 6px 4px;
        color: #7a92a6;
        font-weight: 700;
        font-size: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 8px;
        left: 4px;
        background: #F4F7FA;
        color: #7a92a6;
    }
    QLabel { color: #0D0D0D; border: none; background: transparent; }
    QPushButton {
        background-color: #F7F7F7;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        padding: 5px 14px;
        color: #0D0D0D;
        font-size: 13px;
        font-weight: 500;
        min-height: 28px;
    }
    QPushButton:hover { background-color: #DDE5EC; border-color: #8B9BAA; color: #2E4A5F; }
    QPushButton:pressed { background-color: #8B9BAA; color: #ffffff; border-color: #8B9BAA; }
    QPushButton:disabled { color: #AAAAAA; border-color: #E5E5E5; background-color: #F7F7F7; }
    QPushButton:flat {
        background: transparent; border: none; color: #0D0D0D;
    }
    QPushButton:flat:hover { background: #DDE5EC; }
    QCheckBox { color: #0D0D0D; spacing: 6px; }
    QCheckBox::indicator {
        width: 16px; height: 16px;
        border: 1px solid #E5E5E5;
        border-radius: 6px;
        background: #FFFFFF;
    }
    QCheckBox::indicator:checked { background: #8B9BAA; border-color: #8B9BAA; }
    QCheckBox::indicator:hover { border-color: #8B9BAA; }
    QRadioButton { color: #0D0D0D; spacing: 6px; }
    QRadioButton::indicator {
        width: 16px; height: 16px;
        border: 1px solid #E5E5E5;
        border-radius: 14px;
        background: #FFFFFF;
    }
    QRadioButton::indicator:checked { background: #8B9BAA; border-color: #8B9BAA; }
    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        padding: 4px 8px;
        color: #0D0D0D;
        selection-background-color: #8B9BAA;
        selection-color: #1C1C1C;
        font-size: 13px;
        min-height: 26px;
    }
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #8B9BAA;
    }
    QLineEdit:disabled, QTextEdit:disabled { color: #AAAAAA; border-color: #E5E5E5; }
    QListWidget, QListView {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 14px;
        alternate-background-color: #F7F7F7;
        selection-background-color: #DDE5EC;
        selection-color: #2E4A5F;
        outline: none;
        font-size: 13px;
    }
    QListWidget::item, QListView::item { padding: 5px 8px; border-radius: 6px; }
    QListWidget::item:hover, QListView::item:hover { background-color: #EBF0F5; }
    QListWidget::item:selected, QListView::item:selected {
        background-color: #DDE5EC; color: #2E4A5F;
    }
    QTreeWidget, QTreeView {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 14px;
        alternate-background-color: #F7F7F7;
        selection-background-color: #DDE5EC;
        selection-color: #2E4A5F;
        outline: none;
    }
    QTreeWidget::item:hover, QTreeView::item:hover { background-color: #EBF0F5; }
    QTreeWidget::item:selected, QTreeView::item:selected {
        background-color: #DDE5EC; color: #2E4A5F;
    }
    QHeaderView::section {
        background-color: #F7F7F7;
        color: #6B6B6B;
        border: none;
        border-bottom: 1px solid #E5E5E5;
        border-right: 1px solid #E5E5E5;
        padding: 5px 8px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    QComboBox {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        padding: 4px 28px 4px 8px;
        color: #0D0D0D;
        selection-background-color: #8B9BAA;
        min-height: 28px;
    }
    QComboBox:hover { border-color: #8B9BAA; }
    QComboBox:focus { border-color: #8B9BAA; }
    QComboBox::drop-down { border: none; width: 22px; subcontrol-origin: padding; subcontrol-position: right center; }
    QComboBox QAbstractItemView {
        background-color: #FFFFFF;
        color: #0D0D0D;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        selection-background-color: #8B9BAA;
        selection-color: #ffffff;
        outline: none;
        padding: 4px;
    }
    QDoubleSpinBox, QSpinBox {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        padding: 4px 6px;
        color: #0D0D0D;
        selection-background-color: #8B9BAA;
        min-height: 26px;
    }
    QDoubleSpinBox:hover, QSpinBox:hover { border-color: #8B9BAA; }
    QDoubleSpinBox:focus, QSpinBox:focus { border-color: #8B9BAA; }
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
    QSpinBox::up-button, QSpinBox::down-button {
        background-color: #F7F7F7; border: none; width: 16px;
    }
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
    QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #8B9BAA; }
    QDateEdit, QDateTimeEdit, QTimeEdit {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        padding: 3px 6px;
        color: #0D0D0D;
        selection-background-color: #8B9BAA;
        min-height: 24px;
    }
    QDateEdit:hover, QDateTimeEdit:hover, QTimeEdit:hover { border-color: #8B9BAA; }
    QDateEdit:focus, QDateTimeEdit:focus, QTimeEdit:focus { border-color: #8B9BAA; }
    QDateEdit::drop-down, QDateTimeEdit::drop-down, QTimeEdit::drop-down {
        border: none; width: 20px;
    }
    QDateEdit::up-button, QDateEdit::down-button,
    QDateTimeEdit::up-button, QDateTimeEdit::down-button,
    QTimeEdit::up-button, QTimeEdit::down-button {
        background-color: #F7F7F7; border: none; width: 16px;
    }
    QDateEdit::up-button:hover, QDateEdit::down-button:hover,
    QDateTimeEdit::up-button:hover, QDateTimeEdit::down-button:hover,
    QTimeEdit::up-button:hover, QTimeEdit::down-button:hover { background-color: #8B9BAA; }
    QCalendarWidget {
        background-color: #FFFFFF;
        color: #0D0D0D;
        border: 1px solid #E5E5E5;
        border-radius: 14px;
    }
    QCalendarWidget QWidget { background-color: #FFFFFF; color: #0D0D0D; }
    QCalendarWidget QWidget#qt_calendar_navigationbar {
        background-color: #F7F7F7;
        border-bottom: 1px solid #E5E5E5;
        border-radius: 14px 14px 0 0;
        padding: 4px 6px;
    }
    QCalendarWidget QToolButton {
        background-color: transparent;
        color: #0D0D0D;
        border: none;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 13px;
        font-weight: 600;
    }
    QCalendarWidget QToolButton:hover { background-color: #DDE5EC; color: #2E4A5F; }
    QCalendarWidget QToolButton:pressed { background-color: #8B9BAA; color: #ffffff; }
    QCalendarWidget QSpinBox {
        background-color: #FFFFFF;
        color: #0D0D0D;
        border: 1px solid #E5E5E5;
        border-radius: 6px;
        padding: 2px 4px;
    }
    QCalendarWidget QAbstractItemView:enabled {
        background-color: #FFFFFF;
        color: #0D0D0D;
        selection-background-color: #8B9BAA;
        selection-color: #ffffff;
        font-size: 13px;
    }
    QCalendarWidget QAbstractItemView:disabled {
        color: #AAAAAA;
    }
    QCalendarWidget QAbstractItemView::item:hover {
        background-color: #EBF0F5;
        border-radius: 6px;
    }
    QTabWidget::pane { border: none; border-top: 1px solid #E8EDF2; background: #F4F7FA; }
    QTabWidget::tab-bar { left: 0; }
    QTabBar { background: #FFFFFF; border-bottom: 1px solid #E8EDF2; }
    QTabBar::tab {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 7px 10px 5px;
        color: #9BAAB8;
        font-size: 11px;
        font-weight: 500;
        margin-right: 2px;
    }
    QTabBar::tab:selected { color: #4E6678; border-bottom: 2px solid #8B9BAA; font-weight: 600; }
    QTabBar::tab:hover:!selected { color: #2E4A5F; border-bottom: 2px solid #C4D4E0; }
    QStatusBar {
        background-color: #FFFFFF;
        border-top: 1px solid #E8EDF2;
        color: #6B6B6B;
        font-size: 13px;
        padding: 2px 8px;
    }
    QProgressBar {
        background-color: #F2F2F2;
        border: none;
        border-radius: 6px;
        height: 6px;
        text-align: center;
        color: transparent;
    }
    QProgressBar::chunk { background-color: #8B9BAA; border-radius: 6px; }
    QScrollBar:vertical { background: #F7F7F7; width: 8px; margin: 0; border-radius: 6px; }
    QScrollBar::handle:vertical {
        background: #E5E5E5; border-radius: 6px; min-height: 24px;
    }
    QScrollBar::handle:vertical:hover { background: #8B9BAA; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar:horizontal { background: #F7F7F7; height: 8px; margin: 0; border-radius: 6px; }
    QScrollBar::handle:horizontal {
        background: #E5E5E5; border-radius: 6px; min-width: 24px;
    }
    QScrollBar::handle:horizontal:hover { background: #8B9BAA; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
    QScrollArea { background: transparent; border: none; }
    QSplitter::handle { background: transparent; }
    QSplitter::handle:horizontal { width: 3px; border-left: 1px solid #E2E8EE; }
    QSplitter::handle:vertical   { height: 3px; border-top:  1px solid #E2E8EE; }
    QSplitter::handle:hover { background: #8B9BAA; border-color: #8B9BAA; }
    QToolTip {
        background-color: #FFFFFF;
        color: #0D0D0D;
        border: 1px solid #E5E5E5;
        border-radius: 10px;
        padding: 6px 10px;
        font-size: 13px;
    }
    QTableWidget, QTableView {
        background-color: #FFFFFF;
        color: #0D0D0D;
        gridline-color: #E5E5E5;
        border: 1px solid #E5E5E5;
        border-radius: 14px;
        alternate-background-color: #F7F7F7;
        selection-background-color: #DDE5EC;
        selection-color: #2E4A5F;
    }
    QTableWidget::item, QTableView::item { padding: 4px 8px; }
    QTableWidget::item:selected, QTableView::item:selected {
        background-color: #DDE5EC; color: #2E4A5F;
    }
    QTableWidget QHeaderView::section, QTableView QHeaderView::section {
        background-color: #F7F7F7;
        color: #6B6B6B;
        border: none;
        border-bottom: 1px solid #E5E5E5;
        border-right: 1px solid #E5E5E5;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
    }
    QSlider::groove:horizontal { height: 4px; background: #E5E5E5; border-radius: 2px; }
    QSlider::handle:horizontal {
        width: 14px; height: 14px; margin: -5px 0;
        background: #8B9BAA; border-radius: 7px;
    }
    QSlider::handle:horizontal:hover { background: #4E6678; }
    QSlider::sub-page:horizontal { background: #8B9BAA; border-radius: 2px; }
    QSlider::groove:horizontal:disabled { background: #F2F2F2; }
    QSlider::handle:horizontal:disabled { background: #E5E5E5; }
    QSlider::groove:vertical { width: 4px; background: #E5E5E5; border-radius: 2px; }
    QSlider::handle:vertical {
        width: 14px; height: 14px; margin: 0 -5px;
        background: #8B9BAA; border-radius: 7px;
    }
    QSlider::sub-page:vertical { background: #8B9BAA; border-radius: 2px; }
    /* Named button variants */
    QPushButton#btn_primary {
        background: #DDE5EC; border: 1px solid #8B9BAA;
        color: #2E4A5F; font-weight: 700;
    }
    QPushButton#btn_primary:hover { background: #C4D4E0; border-color: #4E6678; color: #2A4C66; }
    QPushButton#btn_primary:disabled { background: #F7F7F7; color: #AAAAAA; border-color: #E5E5E5; }
    QPushButton#btn_success {
        background: #DCFCE7; border: 1px solid #16A34A;
        color: #15803D; font-weight: 700;
    }
    QPushButton#btn_success:hover { background: #BBF7D0; border-color: #15803D; }
    QPushButton#btn_success:disabled { background: #F7F7F7; color: #AAAAAA; border-color: #E5E5E5; }
    QPushButton#btn_purple {
        background: #E8EFF7; border: 1px solid #4E6678;
        color: #2E4A5F; font-weight: 700;
    }
    QPushButton#btn_purple:hover { background: #DEE8F0; border-color: #2E4A5F; }
    QPushButton#btn_purple:disabled { background: #F7F7F7; color: #AAAAAA; border-color: #E5E5E5; }
    QPushButton#btn_muted {
        background: #F2F2F2; border: 1px solid #E5E5E5; color: #6B6B6B;
    }
    QPushButton#btn_muted:hover { background: #DDE5EC; color: #0D0D0D; border-color: #8B9BAA; }
    QPushButton#btn_danger {
        background: #FFF0EE; border: 1px solid #D7263D; color: #D7263D;
    }
    QPushButton#btn_danger:hover { background: #FFE4E6; color: #A40E26; }
    /* Role labels */
    QLabel[role="muted"]   { color: #6B6B6B; font-size: 13px; }
    QLabel[role="section"] { color: #6B6B6B; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
    QLabel[role="help"] {
        color: #6B6B6B; font-size: 11px;
        background: #EBF1F7; border: 1px solid #DDE5ED;
        border-radius: 6px; padding: 4px 6px;
    }
    QWidget[role="adv_section"] { background: #EBF1F7; border: 1px solid #DDE5ED; border-radius: 6px; }
    QToolButton[role="adv_toggle"] {
        background: transparent; border: none;
        color: #4E6678; font-size: 13px; text-align: left; padding: 2px 4px;
    }
    QToolButton[role="adv_toggle"]:hover { color: #2E4A5F; }
    /* QPushButton:checked for toggle groups */
    QPushButton:checked { background-color: #DDE5EC; color: #2E4A5F; border-color: #8B9BAA; }
"""

_themes: dict[str, str] = {"dark": _DARK, "light": _LIGHT}
_current: str = "dark"


def current_theme() -> str:
    """Return the name of the currently active theme ('dark' or 'light')."""
    return _current


def apply_theme(name: str) -> None:
    """Apply *name* theme to the running QApplication."""
    global _current
    if name not in _themes:
        # Graceful fallback: unknown themes (e.g. old 'glass' preference) → dark
        name = "dark"
    _current = name
    app = QApplication.instance()
    if app is not None:
        app.setStyleSheet(_themes[name])


def toggle_theme() -> str:
    """Cycle dark → light → dark; return the new theme name."""
    new = "light" if _current == "dark" else "dark"
    apply_theme(new)
    return new


def is_dark() -> bool:
    """Return True if the current theme has a dark background."""
    return _current == "dark"


def acrylic_tint() -> int:
    """Return the DWM acrylic tint colour (0xAABBGGRR) for the current theme.

    Dark: deep violet-navy tint (55 % alpha).
    Light: barely-visible frosted-white tint (18 % alpha).
    """
    return 0x55_2A_20_1C if is_dark() else 0x18_FF_FA_F7
