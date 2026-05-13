"""Main application window — PyQt5 QMainWindow with dockwidgets."""
from __future__ import annotations

import logging
from pathlib import Path

import vtk
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from prospective.dicom.loader import DICOMLoader
from prospective.dicom.series import DicomSeries
from prospective.io.mesh_exporter import MeshExporter
from prospective.io.session import SessionData, load_session, save_session, SESSION_EXT
from prospective.ui.viewers.mpr_viewer import MPRViewer
from prospective.ui.viewers.cross_plane_dialog import CrossPlaneDialog
from prospective.ui.widgets.aneurysm_panel import AneurysmPanel
from prospective.ui.widgets.clip_panel import ClipPanel
from prospective.ui.widgets.longitudinal_panel import LongitudinalPanel
from prospective.ui.widgets.print_prep_panel import PrintPrepPanel
from prospective.ui.widgets.clip_recommender_panel import ClipRecommenderPanel
from prospective.ui.widgets.treatment_decision_panel import TreatmentDecisionPanel
from prospective.ui.widgets.perforator_risk_panel import PerforatorRiskPanel
from prospective.ui.widgets.workflow_stepper import WorkflowStepper, WorkflowStepDef
from prospective.ui.widgets.morphometrics_panel import MorphometricsPanel
from prospective.ui.windows.planning_window import PlanningWindow
from prospective.auth import AuthManager
from prospective.db import DatabaseManager
from prospective.db.models import User
from prospective.ui.widgets.login_dialog import LoginDialog
from prospective.ui.widgets.patient_manager import PatientManagerDialog
from prospective.ui.widgets.user_manager import UserManagerDialog
from prospective.ui.icons import I as _I
from prospective.ui.widgets.coil_panel import CoilPanel
from prospective.ui.widgets.report_panel import ReportPanel
from prospective.ui.widgets.segmentation_panel import SegmentationPanel
from prospective.utils.window_presets import DEFAULT_PRESET, WINDOW_PRESETS
from prospective.audit.skull_chain import (
    SkullChain,
    ACT_LOGIN,
    ACT_PATIENT_LOADED,
    ACT_SERIES_LOADED,
    ACT_SEGMENTATION,
    ACT_MESH_EXPORTED,
    ACT_TREATMENT_DECISION,
    ACT_REPORT_GENERATED,
)

logger = logging.getLogger(__name__)

_RES       = Path(__file__).resolve().parents[2] / "resources"  # ui/ → prospective/ → Prospective/
_LOGO_PATH = str(_RES / "logo.png")
_ICON_PATH = str(_RES / "icon.ico")   # multi-resolution .ico for the native title bar


def _tinted_logo_pix(height: int) -> QPixmap:
    """Return the SkullApp logo tinted for the current theme, scaled to *height* px.

    Scales by height (not into a square bounding box) so the natural width of the
    logo is preserved regardless of its aspect ratio.  This prevents the image from
    appearing tiny in the toolbar when the logo is wider than it is tall.
    """
    try:
        from prospective.ui.themes import is_dark
        color = "#ffffff" if is_dark() else "#4E6678"
    except Exception:
        color = "#ffffff"
    pix = QPixmap(_LOGO_PATH)
    if pix.isNull():
        return pix
    # scaledToHeight keeps aspect ratio automatically
    pix = pix.scaledToHeight(height, Qt.SmoothTransformation)
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


# ──────────────────────────────────────────────────────────────────────────── #
# Background loader thread                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


class _LoadThread(QThread):
    finished = pyqtSignal(object)  # DicomSeries
    error = pyqtSignal(str)

    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self._path = path

    def run(self) -> None:
        try:
            series = DICOMLoader().load_directory(self._path)
            self.finished.emit(series)
        except Exception as exc:
            logger.exception("DICOM load failed")
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────── #
# Frameless window helpers                                                       #
# ──────────────────────────────────────────────────────────────────────────── #


class _DraggableToolBar(QToolBar):
    """QToolBar that acts as the drag handle for the frameless MainWindow.

    Clicking and dragging on the toolbar background moves the window.
    Double-clicking toggles maximised / normal state.
    Child widgets (buttons, labels) still receive their own events because Qt
    delivers mouse events to the deepest widget that accepts them — clicking a
    button does NOT trigger the toolbar's mousePressEvent.
    """

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(title, parent)
        self._drag_pos = None   # QPoint | None

    # ── Drag-to-move ─────────────────────────────────────────────────────── #

    def mousePressEvent(self, event) -> None:             # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_pos = (
                event.globalPos() - self.window().frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:              # noqa: N802
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            win = self.window()
            if not win.isMaximized():
                win.move(event.globalPos() - self._drag_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:           # noqa: N802
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:       # noqa: N802
        if event.button() == Qt.LeftButton:
            win = self.window()
            if win.isMaximized():
                win.showNormal()
            else:
                win.showMaximized()
        super().mouseDoubleClickEvent(event)


class _WinButtons(QWidget):
    """Minimal close / minimise / maximise buttons for the custom title bar.

    Three 30 × 30 flat buttons placed at the right end of the toolbar.
    The close button turns red on hover; the other two use a subtle tint.
    Colours adapt automatically to dark / light theme via ``apply_theme()``.
    """

    _BTN_SIZE = 30

    def __init__(self, window, parent=None) -> None:
        super().__init__(parent)
        self._win = window

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(2)

        self._btn_min   = QPushButton("−")
        self._btn_max   = QPushButton("□")
        self._btn_close = QPushButton("✕")

        for btn in (self._btn_min, self._btn_max, self._btn_close):
            btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFlat(True)
            lay.addWidget(btn)

        self._btn_min.setToolTip("Minimizar")
        self._btn_max.setToolTip("Maximizar / Restaurar")
        self._btn_close.setToolTip("Cerrar")

        self._btn_min.clicked.connect(window.showMinimized)
        self._btn_max.clicked.connect(self._toggle_max)
        self._btn_close.clicked.connect(window.close)

        self.apply_theme()

    def _toggle_max(self) -> None:
        if self._win.isMaximized():
            self._win.showNormal()
            self._btn_max.setText("□")
        else:
            self._win.showMaximized()
            self._btn_max.setText("❐")

    def apply_theme(self) -> None:
        """Update button colours to match the current dark / light theme."""
        try:
            from prospective.ui.themes import is_dark
            dark = is_dark()
        except Exception:
            dark = True

        if dark:
            fg          = "#C8D4DF"
            fg_hover    = "#FFFFFF"
            bg_hover    = "rgba(139,155,170,46)"
        else:
            fg          = "#4E6678"
            fg_hover    = "#1A2733"
            bg_hover    = "rgba(78,102,120,31)"

        _base = f"""
            QPushButton {{
                border: none; border-radius: 6px;
                color: {fg}; font-size: 14px; background: transparent;
            }}
            QPushButton:hover {{
                background: {bg_hover}; color: {fg_hover};
            }}
        """
        _close = f"""
            QPushButton {{
                border: none; border-radius: 6px;
                color: {fg}; font-size: 13px; background: transparent;
            }}
            QPushButton:hover {{
                background: rgba(232,80,70,217); color: #FFFFFF;
            }}
        """
        self._btn_min.setStyleSheet(_base)
        self._btn_max.setStyleSheet(_base)
        self._btn_close.setStyleSheet(_close)


# ──────────────────────────────────────────────────────────────────────────── #
# Level-3 layout helpers                                                         #
# ──────────────────────────────────────────────────────────────────────────── #


def _is_dark_mw() -> bool:
    """Theme probe for module-level layout helpers."""
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


class _ActivityBar(QWidget):
    """56 px vertical icon rail — VS Code-style workflow step navigator.

    Active step:  3 px left-edge accent stripe + highlighted icon.
    Done steps:   full-brightness icon (dimmed when locked).
    Locked steps: greyed icon, click disabled.

    Signals
    -------
    step_requested(int) — user clicked a non-locked step button
    """

    step_requested = pyqtSignal(int)
    _W = 56

    def __init__(self, steps: list, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(self._W)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._step_btns: list[QPushButton] = []

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # ── Logo ──────────────────────────────────────────────────────── #
        self._logo_lbl = QLabel()
        self._logo_lbl.setFixedHeight(44)
        self._logo_lbl.setAlignment(Qt.AlignCenter)
        self._logo_lbl.setStyleSheet("background: transparent;")
        vl.addWidget(self._logo_lbl)

        # ── Thin separator ─────────────────────────────────────────────── #
        self._sep = QWidget()
        self._sep.setFixedHeight(1)
        vl.addWidget(self._sep)
        vl.addSpacing(6)

        # ── Step buttons (one per workflow step) ───────────────────────── #
        for i, step in enumerate(steps):
            btn = QPushButton(step.icon)
            btn.setFixedSize(self._W, 50)
            btn.setCheckable(True)
            btn.setEnabled(i == 0)   # only step 0 unlocked at start
            btn.setToolTip(f"Paso {step.number}: {step.title}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, idx=i: self.step_requested.emit(idx))
            self._step_btns.append(btn)
            vl.addWidget(btn)

        vl.addStretch()
        self.apply_theme()

    # ── Public API ──────────────────────────────────────────────────────── #

    def set_active(self, idx: int) -> None:
        """Check button *idx*; uncheck all others (no signal re-emission)."""
        for i, btn in enumerate(self._step_btns):
            btn.blockSignals(True)
            btn.setChecked(i == idx)
            btn.blockSignals(False)

    def sync_from_stepper(self, stepper) -> None:
        """Mirror each step's lock/unlock state from *stepper*'s header."""
        from prospective.ui.widgets.workflow_stepper import StepStatus
        for i, btn in enumerate(self._step_btns):
            status = stepper._header.get_status(i)
            btn.setEnabled(status != StepStatus.LOCKED)

    # ── Theme ────────────────────────────────────────────────────────────── #

    def apply_theme(self) -> None:
        dark = _is_dark_mw()
        bg  = "#171B22" if dark else "#E5EBF0"
        sep = "#2A3040" if dark else "#C8D4DC"
        bdr = "#2A3040" if dark else "#C0CCDA"
        self.setStyleSheet(
            f"_ActivityBar {{ background: {bg}; border-right: 1px solid {bdr}; }}"
        )
        self._sep.setStyleSheet(f"background: {sep}; border: none;")
        self._apply_btn_qss(dark)
        self._refresh_logo()

    def _apply_btn_qss(self, dark: bool) -> None:
        if dark:
            # Default: #6A7A8A gives 3.9:1 on the #171B22 sidebar bg — readable
            # but still subdued relative to hover (#8B9BAA, 8.5:1) and checked (#C8D4DF).
            qss = (
                "QPushButton {"
                " border: none; border-left: 3px solid transparent;"
                " border-radius: 0; background: transparent;"
                " color: #6A7A8A; font-size: 20px; padding: 2px; }"
                "QPushButton:hover:enabled {"
                " background: rgba(139,155,170,31); color: #8B9BAA; }"
                "QPushButton:checked {"
                " border-left: 3px solid #8B9BAA;"
                " background: rgba(139,155,170,46); color: #C8D4DF; }"
                "QPushButton:disabled { color: #2E3A48; }"
            )
        else:
            # Default: #5A6A7A gives 4.7:1 on the #E5EBF0 sidebar bg — accessible.
            qss = (
                "QPushButton {"
                " border: none; border-left: 3px solid transparent;"
                " border-radius: 0; background: transparent;"
                " color: #5A6A7A; font-size: 20px; padding: 2px; }"
                "QPushButton:hover:enabled {"
                " background: rgba(78,102,120,26); color: #4E6678; }"
                "QPushButton:checked {"
                " border-left: 3px solid #4E6678;"
                " background: rgba(78,102,120,36); color: #1A2733; }"
                "QPushButton:disabled { color: #B0BECB; }"
            )
        for btn in self._step_btns:
            btn.setStyleSheet(qss)

    def _refresh_logo(self) -> None:
        pix = _tinted_logo_pix(26)
        if not pix.isNull():
            self._logo_lbl.setPixmap(pix)
        else:
            self._logo_lbl.setText("⬡")
            self._logo_lbl.setStyleSheet(
                "color:#8B9BAA; font-size:16px; background:transparent;"
            )


class _PatientInfoStrip(QWidget):
    """Compact 34 px patient-context bar — top of the right work panel.

    Displays: icon · patient name (bold) · ID (muted) · modality badge · ✏

    Signals
    -------
    edit_requested — user clicked the pencil / edit button
    """

    edit_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(34)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(10, 0, 6, 0)
        hl.setSpacing(6)

        self._ico = QLabel("👤")
        self._ico.setFixedWidth(18)
        hl.addWidget(self._ico)

        self._lbl_name = QLabel("Sin paciente cargado")
        self._lbl_name.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        hl.addWidget(self._lbl_name)

        self._lbl_id = QLabel()
        hl.addWidget(self._lbl_id)

        self._lbl_mod = QLabel()
        self._lbl_mod.setVisible(False)
        hl.addWidget(self._lbl_mod)

        hl.addStretch()

        self._btn_edit = QPushButton("✏")
        self._btn_edit.setFixedSize(24, 24)
        self._btn_edit.setFlat(True)
        self._btn_edit.setToolTip("Editar datos del caso")
        self._btn_edit.setCursor(Qt.PointingHandCursor)
        self._btn_edit.clicked.connect(self.edit_requested)
        hl.addWidget(self._btn_edit)

        self.apply_theme()

    def update_info(
        self,
        name: str = "",
        pid: str = "",
        modality: str = "",
    ) -> None:
        """Refresh displayed patient fields."""
        self._lbl_name.setText(name or "Sin paciente cargado")
        self._lbl_id.setText(f"· {pid}" if pid else "")
        self._lbl_mod.setText(modality or "")
        self._lbl_mod.setVisible(bool(modality))

    def apply_theme(self) -> None:
        dark      = _is_dark_mw()
        bg        = "#1A1E28" if dark else "#D8E2EB"
        text      = "#B8C8D8" if dark else "#2A3A4A"
        muted     = "#5A6A7A" if dark else "#7A8A9A"
        badge_bg  = "#253040" if dark else "#B8CAD8"
        badge_fg  = "#8B9BAA" if dark else "#2A3A50"
        btn_c     = "#4A5A68" if dark else "#7A8A9A"
        btn_hover = "rgba(139,155,170,38)" if dark else "rgba(78,102,120,31)"

        self.setStyleSheet(f"background: {bg};")
        self._ico.setStyleSheet(
            f"color: {muted}; background: transparent; font-size: 13px;"
        )
        self._lbl_name.setStyleSheet(
            f"color: {text}; background: transparent;"
            f" font-size: 11px; font-weight: 600;"
        )
        self._lbl_id.setStyleSheet(
            f"color: {muted}; background: transparent; font-size: 10px;"
        )
        self._lbl_mod.setStyleSheet(
            f"color: {badge_fg}; background: {badge_bg};"
            f" border-radius: 8px; padding: 1px 6px; font-size: 9px;"
        )
        self._btn_edit.setStyleSheet(
            f"QPushButton {{ color: {btn_c}; border: none; border-radius: 4px;"
            f" background: transparent; }}"
            f"QPushButton:hover {{ background: {btn_hover}; }}"
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Main window                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #


class MainWindow(QMainWindow):
    #: Emitted by _close_case() BEFORE self.close() so that WelcomeWindow can
    #: distinguish a deliberate "go back to cases" from a plain window close.
    case_closed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        # Use the native Win32 title bar (no FramelessWindowHint).
        # The title bar text is kept in sync with the current workflow step via
        # _update_window_title().  The window icon is set from the app logo so
        # it appears both in the title bar and the Windows taskbar.
        # WA_TranslucentBackground is intentionally NOT set here because it
        # breaks VTK render surfaces (blank viewport).
        from PyQt5.QtGui import QIcon
        # Prefer the multi-resolution .ico so Windows can choose the best
        # size for the title bar, taskbar, and Alt-Tab preview.  Fall back to
        # the tinted PNG if the .ico file is missing.
        _ico = QIcon(_ICON_PATH)
        if _ico.isNull():
            _pix = _tinted_logo_pix(32)
            if not _pix.isNull():
                _ico = QIcon(_pix)
        if not _ico.isNull():
            self.setWindowIcon(_ico)
        self._series: DicomSeries | None = None
        self._load_thread: _LoadThread | None = None
        self._last_session_path: str = ""
        self._planning_window: PlanningWindow | None = None
        self._current_study_id: int = 0    # set by preload_dicom(); 0 = no study context
        self._current_session_id: int = 0  # DB PlanningSession.id; 0 = unsaved / not in DB
        self._pending_session_stents: list[dict] = []  # stents to restore when planning window opens
        self._pending_seg_state:     dict        = {}  # seg params to re-apply after DICOM reload
        self._pending_restore_step:  int         = -1  # step to restore after DICOM reload (-1 = none)
        self._pending_restore_mesh               = None  # vtkPolyData to re-apply to planning window after DICOM reload

        # Initialise DB and auth singletons before any panel is built
        DatabaseManager.instance()
        AuthManager.instance()

        # NOTE: ACT_LOGIN is recorded in app.py immediately after the login
        # dialog returns Accepted, so the audit timestamp reflects the actual
        # login moment and not the time the first case was opened.

        self._build_central()
        self._build_patient_dock()
        self._build_workflow_dock()   # single stepper dock (replaces 10 individual docks)
        self._build_menubar()
        self._build_toolbar()
        self._build_statusbar()
        self._apply_theme()

        self.setWindowTitle("PROSPECTIVE — Hybrid Neurovascular Planning Software Platform v0.1")

        # ── Screen-aware sizing ───────────────────────────────────────────── #
        # Scale the window to 95 % of the available screen geometry so the window
        # always fits on any monitor (critical on 1366 × 768 laptops).
        from PyQt5.QtWidgets import QApplication as _QApp
        _screen = _QApp.primaryScreen()
        if _screen is not None:
            _avail = _screen.availableGeometry()
            _w = min(1600, max(900, int(_avail.width()  * 0.95)))
            _h = min(960,  max(600, int(_avail.height() * 0.92)))
            # Minimum size must never exceed the available geometry.
            _min_w = min(900, _avail.width())
            _min_h = min(600, _avail.height())
            self.setMinimumSize(_min_w, _min_h)
            self.resize(_w, _h)
            # Centre on the primary screen
            self.move(
                _avail.x() + (_avail.width()  - _w) // 2,
                _avail.y() + (_avail.height() - _h) // 2,
            )
        else:
            self.setMinimumSize(900, 600)
            self.resize(1280, 800)
        # Without WA_DeleteOnClose, close() only *hides* this window — the C++
        # object is never destroyed, QObject.destroyed is never emitted, and the
        # WelcomeWindow callback (_on_main_window_closed) never fires.
        # Result: WelcomeWindow stays hidden forever and the process becomes a
        # zombie.  Setting this flag makes close() destroy the object immediately,
        # reliably emitting destroyed and triggering the welcome-back flow.
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self._apply_role_restrictions()
        self._update_user_status_label()

    # ------------------------------------------------------------------ #
    # Qt events                                                             #
    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:  # noqa: N802
        """Gracefully stop background threads before the window is destroyed.

        Without explicit cleanup Qt prints "QThread: Destroyed while thread is
        still running" and the behaviour is undefined (potential crash / hang).
        """
        if self._load_thread is not None and self._load_thread.isRunning():
            self._load_thread.quit()
            if not self._load_thread.wait(3000):   # 3 s max
                self._load_thread.terminate()       # last resort
                self._load_thread.wait(1000)
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        """Re-apply step-adaptive splitter proportions once the window has been
        laid out and has real pixel dimensions.  Without this, a setSizes() call
        before show() may be ignored by Qt's layout engine on first paint."""
        super().showEvent(event)
        # Apply DWM immersive-dark-mode to the native title bar so it matches
        # the dark theme.  Runs on every showEvent (show/restore) so the effect
        # survives minimise→restore cycles.  Safe no-op on non-Windows platforms.
        self._apply_dwm_dark_mode()
        # Also apply the glass / acrylic effect to the client area so the
        # background blur remains active after a minimise→restore.
        try:
            from prospective.ui.glass_utils import enable_acrylic
            enable_acrylic(self)
        except Exception:
            pass
        if hasattr(self, "_stepper") and hasattr(self, "_central_splitter"):
            self._on_step_changed(self._stepper.current_index())
        # Resize the patient dock exactly once — resizeDocks() only works after show()
        if not getattr(self, "_docks_arranged", False):
            self._docks_arranged = True
            self._arrange_docks()

    def _apply_dwm_dark_mode(self) -> None:
        """Tell DWM to colour the native title bar to match the active theme.

        DWMWA_USE_IMMERSIVE_DARK_MODE (attribute 20) makes Windows render the
        title bar in dark mode when set to 1.  This call is a no-op on older
        Windows versions and on non-Windows platforms (ctypes import fails
        silently).  Must be called *after* the native window has been created
        (i.e. from showEvent or later), because winId() forces window creation.
        """
        try:
            import ctypes
            from prospective.ui.themes import is_dark
            hwnd = int(self.winId())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            val = ctypes.c_int(1 if is_dark() else 0)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(val), 4
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # UI construction                                                       #
    # ------------------------------------------------------------------ #

    def _build_central(self) -> None:
        self._mpr = MPRViewer()

        # Triplanar dialog — created lazily on first use.
        # Kept as None until the user first clicks the "3 Planos" toolbar button.
        self._triplanar_dlg: CrossPlaneDialog | None = None

        # Wrap the left panel in a QTabWidget so that step 5 (Planificación)
        # can show the embedded 3D planning view without opening a second window.
        #
        # • Tab 0 — "MPR (4 vistas)"    : always present, active on steps 0-3 and 5
        # • Tab 1 — "Planificación 3D"  : added lazily on first visit to step 5
        #
        self._left_tabs = QTabWidget()
        self._left_tabs.setDocumentMode(True)
        self._left_tabs.setTabBarAutoHide(True)
        self._left_tabs.tabBar().setExpanding(False)   # natural tab width; no elision
        self._left_tabs.addTab(self._mpr, f"{_I.MPR_VIEW}  MPR (4 vistas)")
        # Tab 1 is inserted lazily by _show_planning_tab() on first step-5 visit.

        # H-splitter: MPR tabs (left) │ right panel (patient strip + stepper).
        # The right side is assembled by _build_workflow_dock() after all panels
        # are created and the step definitions are known.
        self._central_splitter = QSplitter(Qt.Horizontal)
        self._central_splitter.setChildrenCollapsible(False)
        self._central_splitter.addWidget(self._left_tabs)

        # ── Level-3 outer container: [_ActivityBar 56px] [splitter…] ─── #
        # The activity bar is created in _build_workflow_dock() (needs step
        # defs); we store the layout so that method can insertWidget(0, bar).
        self._central_container = QWidget()
        self._central_container.setObjectName("centralContainer")
        ch = QHBoxLayout(self._central_container)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.setSpacing(0)
        ch.addWidget(self._central_splitter, 1)   # splitter at index 0 for now
        self._central_h_layout = ch

        self.setCentralWidget(self._central_container)

    def _build_patient_dock(self) -> None:
        """Create patient-info widgets for the Level-3 layout.

        No QDockWidget is created.  The label references (self._lbl_name, etc.)
        are kept intact so the rest of the codebase can read / write them
        without change.  Visual display is handled by ``self._patient_strip``
        which sits above the workflow panel.  The series tree is embedded
        directly in the Step-1 panel by ``_create_patient_step_widget()``.
        """
        # ── Backing-store labels (referenced throughout the codebase) ──── #
        self._lbl_name  = QLabel("—")
        self._lbl_name.setWordWrap(True)
        self._lbl_pid   = QLabel("—")
        self._lbl_pid.setWordWrap(True)
        self._lbl_date  = QLabel("—")
        self._lbl_mod   = QLabel("—")

        # ── Edit-case button ──────────────────────────────────────────── #
        self._btn_edit_case = QPushButton("✏  Datos del caso")
        self._btn_edit_case.setToolTip(
            "Completar o editar los datos clínicos de este caso"
        )
        self._btn_edit_case.clicked.connect(self._edit_current_case)

        # ── Series tree (moved to Step-1 panel in Level-3 layout) ─────── #
        self._series_tree = QTreeWidget()
        self._series_tree.setHeaderLabels(["Descripción", "Slices"])
        self._series_tree.setColumnWidth(0, 150)
        self._series_tree.setAlternatingRowColors(True)

        # ── Compact patient-info strip (top of right work panel) ──────── #
        self._patient_strip = _PatientInfoStrip()
        self._patient_strip.edit_requested.connect(self._edit_current_case)

        # Placeholder so _arrange_docks() is a no-op (no dock to size).
        self._dock_patient = None

    # ------------------------------------------------------------------ #
    # Workflow stepper dock (replaces all individual tool docks)           #
    # ------------------------------------------------------------------ #

    def _build_workflow_dock(self) -> None:
        """Create all panels, wire their signals, embed in WorkflowStepper."""

        # ── helper: compact tab container ─────────────────────────────── #
        def _tab(pairs: list) -> QWidget:
            from PyQt5.QtWidgets import QTabWidget
            tw = QTabWidget()
            tw.setTabPosition(QTabWidget.North)
            # Natural tab width: each tab is as wide as its text + padding.
            # When all tabs exceed the bar width, Qt shows scroll arrows instead
            # of force-squishing text (which would cause "Asistente de cl..." elision).
            tw.tabBar().setExpanding(False)
            tw.tabBar().setUsesScrollButtons(True)
            for label, widget in pairs:
                tw.addTab(widget, label)
            return tw

        # ── Step 1 panel: DICOM loading + windowing ───────────────────── #
        step1 = self._create_patient_step_widget()

        # ── Step 2: Segmentation ──────────────────────────────────────── #
        self._seg_panel = SegmentationPanel()
        self._seg_panel.seg_ready.connect(self._on_seg_ready)
        self._seg_panel.preview_ready.connect(self._on_seg_preview_ready)
        self._seg_panel.mesh_visibility_changed.connect(self._mpr.set_mesh_visible)
        self._seg_panel.volume_visibility_changed.connect(self._mpr.set_volume_visible)
        self._seg_panel.export_stl_requested.connect(self._export_stl)
        self._seg_panel.export_obj_requested.connect(self._export_obj)
        self._seg_panel.request_mpr_seed.connect(self._on_request_mpr_seed)
        # request_auto_threshold signal intentionally NOT connected:
        # the ⚡ Auto-umbral button is hidden (HU cursor tracking was removed
        # because per-pixel sigMouseMoved caused severe visual glitches).

        # ── Step 3: Aneurysm detection ────────────────────────────────── #
        self._aneurysm_panel = AneurysmPanel()
        self._aneurysm_panel.candidate_highlighted.connect(self._on_aneurysm_highlighted)
        self._aneurysm_panel.candidate_isolated.connect(self._on_aneurysm_isolated)
        self._aneurysm_panel.export_requested.connect(self._export_aneurysm_stl)
        # Crop tool: update _vessel_poly + 3D viewer when user crops/restores the mesh
        self._aneurysm_panel.mesh_cropped.connect(self._on_aneurysm_mesh_cropped)
        # Interactive crop-box widget (vtkBoxWidget2 overlaid on MPR 3D view)
        self._aneurysm_panel.box_widget_requested.connect(self._on_box_crop_widget_requested)
        self._aneurysm_panel.interactive_crop_apply_requested.connect(
            self._on_interactive_crop_apply
        )
        # Candidate visibility toggle (show/hide magenta highlight without
        # deselecting the candidate in the table).
        # Uses a handler so the planning window is also updated if it is open.
        self._aneurysm_panel.candidate_visibility_changed.connect(
            self._on_candidate_visibility_changed
        )

        # ── Step 4: Morphometrics + Longitudinal ──────────────────────── #
        self._morpho_panel = MorphometricsPanel()
        # candidate_confirmed is the ONLY signal that triggers morphometrics and
        # advances the stepper.  Browsing (highlighted / isolated) is purely visual.
        self._aneurysm_panel.candidate_confirmed.connect(self._on_candidate_for_morpho)

        self._longitudinal_panel = LongitudinalPanel()
        self._morpho_panel.analysis_done.connect(
            self._longitudinal_panel.set_current_result
        )
        step4 = _tab([
            (f"{_I.STEP_MORPHO} Análisis morfométrico", self._morpho_panel),
            (f"{_I.CENTERLINE} Seguimiento longitudinal", self._longitudinal_panel),
        ])

        # ── Step 5: Planning ──────────────────────────────────────────── #
        # Treatment-strategy decision (CLIP vs ENDO) — shown first as the
        # high-level clinical decision before device-specific planning.
        self._treatment_panel = TreatmentDecisionPanel()
        self._morpho_panel.analysis_done.connect(
            self._treatment_panel.set_morpho_result
        )

        self._clip_recommender_panel = ClipRecommenderPanel()
        self._morpho_panel.analysis_done.connect(
            self._clip_recommender_panel.set_morpho_result
        )

        self._clip_panel = ClipPanel()
        self._clip_panel.clip_placed.connect(self._on_clip_placed)
        self._clip_panel.clip_removed.connect(self._mpr.remove_clip)
        self._clip_panel.clip_removed.connect(lambda _: self._sync_report_clips())
        self._clip_panel.clip_visibility_changed.connect(self._mpr.set_clip_visible)
        self._clip_panel.clip_transform_changed.connect(self._mpr.update_clip_transform)
        self._clip_panel.trajectory_changed.connect(self._mpr.set_trajectory)
        self._morpho_panel.analysis_done.connect(self._on_morpho_done_for_clips)

        self._clip_recommender_panel.clip_selected.connect(
            lambda spec: self._clip_panel.select_clip_by_name(spec.name)
        )

        self._coil_panel = CoilPanel()
        self._coil_panel.coil_placed.connect(self._on_coil_placed)
        self._coil_panel.coil_removed.connect(self._on_coil_removed)
        self._coil_panel.coil_visibility_changed.connect(self._on_coil_visibility)
        self._morpho_panel.analysis_done.connect(self._on_morpho_done_for_coils)

        # ── Perforator risk panel ─────────────────────────────────────── #
        self._perforator_panel = PerforatorRiskPanel()
        self._morpho_panel.analysis_done.connect(self._on_morpho_done_for_perforators)
        self._perforator_panel.overlay_ready.connect(
            self._on_perforator_overlay_ready
        )
        self._perforator_panel.overlay_cleared.connect(
            self._on_perforator_overlay_cleared
        )
        self._perforator_panel.render_requested.connect(
            self._on_perforator_render_requested
        )

        step5 = _tab([
            (f"{_I.STEP_DETECT} Decisión terapéutica",  self._treatment_panel),
            (f"{_I.BRAIN} Perforantes en riesgo", self._perforator_panel),
            (f"{_I.SEARCH} Asistente de clips",    self._clip_recommender_panel),
            (f"{_I.CLIPS} Planificación clips",    self._clip_panel),
            (f"{_I.COIL} Embolización (Coils)",  self._coil_panel),
        ])

        # ── Step 6: Export ────────────────────────────────────────────── #
        self._report_panel = ReportPanel()
        self._report_panel.screenshot_requested.connect(self._capture_screenshot)
        self._morpho_panel.analysis_done.connect(self._report_panel.set_morphometrics)
        self._treatment_panel.decision_updated.connect(
            self._report_panel.set_treatment_decision
        )
        self._treatment_panel.decision_updated.connect(self._on_decision_updated)
        self._clip_panel.trajectory_changed.connect(
            lambda entry, target: self._report_panel.set_trajectory(
                list(entry) if entry is not None else [],
                list(target) if target is not None else [],
            )
        )

        self._print_panel = PrintPrepPanel()

        step6 = _tab([
            (f"{_I.DOC} Informe PDF", self._report_panel),
            (f"{_I.STEP_EXPORT} Impresión 3D", self._print_panel),
        ])

        # ── Assemble stepper ──────────────────────────────────────────── #
        steps = [
            WorkflowStepDef(
                "paciente", 1, _I.STEP_PATIENT, "Paciente",
                "Carga el estudio DICOM del paciente. Ajusta la ventana HU para "
                "visualizar correctamente los vasos cerebrales (preset: Angio cerebral).",
                step1,
            ),
            WorkflowStepDef(
                "segmentacion", 2, _I.STEP_SEGMENT, "Segmentación",
                "Segmenta la vasculatura cerebral aplicando un umbral de Hounsfield. "
                "Usa 'Sustracción ósea' para eliminar cráneo. El resultado es una malla 3D.",
                self._seg_panel,
            ),
            WorkflowStepDef(
                "deteccion", 3, _I.STEP_DETECT, "Detección",
                "Detecta candidatos a aneurisma por curvatura local. Selecciona o aísla "
                "cada candidato para revisarlo en 3D. Ajusta el umbral de curvatura si "
                "se detectan falsos positivos.",
                self._aneurysm_panel,
            ),
            WorkflowStepDef(
                "morfometria", 4, _I.STEP_MORPHO, "Morfometría",
                "Analiza la geometría del aneurisma seleccionado: volumen, diámetro máximo, "
                "cuello, Aspect Ratio, índices BF/UI/EI/NSI y puntuación PHASES. "
                "El seguimiento longitudinal permite comparar sesiones previas.",
                step4,
            ),
            WorkflowStepDef(
                "planificacion", 5, _I.STEP_PLAN, "Planificación",
                "Decisión terapéutica CLIP vs ENDOVASCULAR basada en morfometría y contexto "
                "clínico. El asistente sugiere los mejores clips según la morfología del "
                "cuello. Planifica clips, stents y coils. La vista 3D interactiva se abre "
                "automáticamente en la pestaña izquierda al entrar a este paso.",
                step5,
            ),
            WorkflowStepDef(
                "exportar", 6, _I.STEP_EXPORT, "Exportar",
                "Genera el informe PDF quirúrgico completo con morfometría, clips y "
                "trayectoria. Exporta como DICOM SR o STL para impresión 3D.",
                step6,
            ),
        ]

        self._stepper = WorkflowStepper(steps)

        # Step 1 always available (already set ACTIVE by WorkflowStepper.__init__,
        # but unlock is harmless here and makes intent explicit)
        self._stepper.unlock(0)

        # ── Level-3: hide stepper's own chrome (replaced by _ActivityBar) #
        self._stepper._header.setVisible(False)
        self._stepper._sidebar.setVisible(False)

        # ── Activity bar (56 px left rail) ────────────────────────────── #
        self._activity_bar = _ActivityBar(steps, parent=self._central_container)
        # Insert at index 0 so it sits to the left of the H-splitter.
        self._central_h_layout.insertWidget(0, self._activity_bar)
        self._activity_bar.set_active(0)   # initial state: step 0 active

        # ── Right panel: patient strip (top) + step panels (below) ───── #
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        rv = QVBoxLayout(right_panel)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        rv.addWidget(self._patient_strip)   # compact 34 px patient context bar
        rv.addWidget(self._stepper, 1)      # workflow panels fill remaining height

        right_panel.setMinimumWidth(340)
        self._central_splitter.addWidget(right_panel)
        # [960, 640] → MPR ~704 px, right panel ~469 px at 1600 px window width.
        self._central_splitter.setSizes([960, 640])

        # ── Wire activity bar ↔ stepper ───────────────────────────────── #
        self._stepper.step_changed.connect(self._on_step_changed)
        self._stepper.step_changed.connect(lambda _: self._update_window_title())
        # Keep the activity bar in sync whenever the stepper navigates.
        self._stepper.step_changed.connect(self._activity_bar.set_active)
        self._stepper.step_changed.connect(
            lambda _: self._activity_bar.sync_from_stepper(self._stepper)
        )
        self._activity_bar.step_requested.connect(self._stepper.go_to)

    def _create_patient_step_widget(self) -> QWidget:
        """Builds the Step-1 panel: DICOM load + HU windowing + 3-D presets."""
        from PyQt5.QtWidgets import QTabWidget

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Load DICOM button ─────────────────────────────────────────── #
        btn_load = QPushButton(f"{_I.FOLDER}  Cargar estudio DICOM…")
        btn_load.setMinimumHeight(36)
        btn_load.setStyleSheet(
            "QPushButton { background: #4E6678; color: white; border-radius: 14px; "
            "font-weight: bold; font-size: 11px; }"
            "QPushButton:hover { background: #8B9BAA; }"
        )
        btn_load.clicked.connect(self.open_dicom_directory)
        layout.addWidget(btn_load)

        # ── Status label ──────────────────────────────────────────────── #
        self._lbl_load_status = QLabel("No hay ningún estudio cargado.")
        self._lbl_load_status.setWordWrap(True)
        _mc = "#9B9B9B" if _is_dark_mw() else "#6B6B6B"
        self._lbl_load_status.setStyleSheet(f"color: {_mc}; font-size: 10px;")
        layout.addWidget(self._lbl_load_status)

        # ── Non-3D projection warning (hidden until a 2D series is loaded) #
        self._lbl_projection_warn = QLabel()
        self._lbl_projection_warn.setWordWrap(True)
        self._lbl_projection_warn.setStyleSheet(
            "color: #C07000; background: #FFF8E1; border: 1px solid #E0A000;"
            "border-radius: 4px; padding: 4px 6px; font-size: 10px;"
        )
        self._lbl_projection_warn.setVisible(False)
        layout.addWidget(self._lbl_projection_warn)

        # ── HU windowing ──────────────────────────────────────────────── #
        wl_grp = QGroupBox("Ventana / Nivel (HU)")
        wl_form = QFormLayout(wl_grp)
        wl_form.setLabelAlignment(Qt.AlignRight)

        self._preset_combo = QComboBox()
        for name in WINDOW_PRESETS:
            self._preset_combo.addItem(name)
        idx = self._preset_combo.findText(DEFAULT_PRESET)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        wl_form.addRow("Preset:", self._preset_combo)

        self._wc_spin = QDoubleSpinBox()
        self._wc_spin.setRange(-2000, 4000)
        self._wc_spin.setValue(WINDOW_PRESETS[DEFAULT_PRESET][0])
        self._wc_spin.setSuffix(" HU")
        self._wc_spin.setDecimals(0)
        self._wc_spin.valueChanged.connect(self._on_wl_changed)
        wl_form.addRow("Centro:", self._wc_spin)

        self._ww_spin = QDoubleSpinBox()
        self._ww_spin.setRange(1, 4000)
        self._ww_spin.setValue(WINDOW_PRESETS[DEFAULT_PRESET][1])
        self._ww_spin.setSuffix(" HU")
        self._ww_spin.setDecimals(0)
        self._ww_spin.valueChanged.connect(self._on_wl_changed)
        wl_form.addRow("Ancho:", self._ww_spin)

        layout.addWidget(wl_grp)

        # ── 3D rendering preset ───────────────────────────────────────── #
        vtk3d_grp = QGroupBox("Renderizado volumétrico 3D")
        vtk3d_form = QFormLayout(vtk3d_grp)
        vtk3d_form.setLabelAlignment(Qt.AlignRight)

        self._3d_mode_combo = QComboBox()
        for mode in ("VR (Ray Casting)", "MIP", "MinIP"):
            self._3d_mode_combo.addItem(mode)
        self._3d_mode_combo.currentTextChanged.connect(
            lambda name: self._mpr.set_3d_blend_mode(name)
        )
        vtk3d_form.addRow("Modo:", self._3d_mode_combo)

        from prospective.rendering.transfer_functions import PRESETS_3D
        self._3d_preset_combo = QComboBox()
        for name in PRESETS_3D:
            self._3d_preset_combo.addItem(name)
        self._3d_preset_combo.currentTextChanged.connect(
            lambda name: self._mpr.set_3d_preset(name)
        )
        vtk3d_form.addRow("Preset TF:", self._3d_preset_combo)
        layout.addWidget(vtk3d_grp)

        # ── Volume info (populated by _on_loaded) ─────────────────────── #
        vol_grp = QGroupBox("Información del volumen")
        vol_form = QFormLayout(vol_grp)
        vol_form.setLabelAlignment(Qt.AlignRight)
        self._lbl_dims = QLabel("—")
        self._lbl_spacing = QLabel("—")
        self._lbl_series_desc = QLabel("—")
        self._lbl_series_desc.setWordWrap(True)
        vol_form.addRow("Dimensiones:", self._lbl_dims)
        vol_form.addRow("Espaciado:", self._lbl_spacing)
        vol_form.addRow("Serie:", self._lbl_series_desc)
        layout.addWidget(vol_grp)

        # ── Series DICOM tree (moved from patient dock in Level-3) ────── #
        # self._series_tree is created in _build_patient_dock() which runs
        # before this method, so the widget already exists here.
        series_grp = QGroupBox("Series DICOM")
        series_vl = QVBoxLayout(series_grp)
        series_vl.setContentsMargins(4, 4, 4, 4)
        series_vl.addWidget(self._series_tree)
        layout.addWidget(series_grp)

        layout.addStretch()
        return root

    # ------------------------------------------------------------------ #
    # Step-adaptive layout                                                 #
    # ------------------------------------------------------------------ #

    def _schedule_planning_render(self, delay_ms: int = 80) -> None:
        """Deferred VTK render for the embedded planning window.

        Called after a session restore that navigates back to the planning
        step.  The short delay gives Qt time to fully expose the QTabWidget
        page before VTK tries to render, preventing the blank-viewport bug
        that occurs when Render() fires on a not-yet-painted surface.
        """
        def _do_render() -> None:
            pw = self._planning_window
            if pw is None:
                return
            if not getattr(pw, "_vtk_initialized", False):
                return
            if not getattr(pw, "_mesh_loaded", False):
                return
            try:
                pw._reset_camera()
            except Exception:
                pass   # never crash the clinical workflow

        QTimer.singleShot(delay_ms, _do_render)

    def _on_step_changed(self, idx: int) -> None:
        """Adjust the MPR ↔ workspace split when the active workflow step changes.

        Total virtual width is 1600 units.  The right column (WorkflowStepper)
        needs more room for complex forms; the MPR should be generous when
        inspecting mesh / candidates, minimal when filling in forms.
        """
        # All steps use the same ratio so the right workflow panel stays a
        # consistent width regardless of which step is active.
        # [960, 640] gives ~704px to MPR (3D cell ~352px, bars fit) and
        # ~469px to the stepper — scroll areas inside handle overflow.
        _SPLITS = [
            [960, 640],   # 0  Paciente
            [960, 640],   # 1  Segmentación
            [960, 640],   # 2  Detección
            [960, 640],   # 3  Morfometría
            [960, 640],   # 4  Planificación
            [960, 640],   # 5  Exportar
        ]
        if 0 <= idx < len(_SPLITS):
            self._central_splitter.setSizes(_SPLITS[idx])

        # Step 5 (idx=4, 0-based): show the embedded 3D planning tab.
        # Step 3 (idx=2, detection): also open the 3D planning tab so the
        #   yellow sphere + billboard label created by highlight_candidate()
        #   are actually visible when the user selects a candidate row.
        #   Without this, _planning_window is None at step 3 and
        #   highlight_candidate() is never called → no yellow sphere.
        # Every other step: make sure the MPR tab is active.
        if idx in (2, 4):
            self._show_planning_tab()
        elif hasattr(self, "_left_tabs"):
            self._left_tabs.setCurrentIndex(0)

    def _show_planning_tab(self) -> None:
        """Lazily create and show the embedded 3D planning view in the left tab.

        On the **first** call:
          • Creates PlanningWindow with Qt.Widget flag so it embeds as a plain
            widget (no OS window frame, toolbar stays visible inside the tab).
          • Removes the 1100 × 720 minimum-size constraint the standalone window
            sets so the tab can resize freely with the splitter.
          • Wires stent-panel signals and drains any session-loaded pending stents.
          • Populates the vessel mesh if segmentation has already been run.
          • Adds the window as tab 2 (after MPR at 0 and triplanar at 1).

        On subsequent calls it simply switches to the planning tab.
        """
        if self._planning_window is None:
            # embedded=True sets Qt.Widget BEFORE _build_vtk() runs so the VTK
            # render window binds to the correct native handle from the start.
            # This prevents the transient standalone OS window at (0, 28) that
            # caused QWindowsWindow::setGeometry warnings and a black 3-D viewport.
            self._planning_window = PlanningWindow(parent=self, embedded=True)

            # ── Stent signals ──────────────────────────────────────────── #
            sp = self._planning_window._stent_panel
            sp.stent_placed.connect(lambda *_: self._sync_report_stents())
            sp.stent_removed.connect(lambda *_: self._sync_report_stents())

            # ── Drain pending stents from a session loaded before now ─── #
            pending = getattr(self, "_pending_session_stents", [])
            if pending:
                sp.restore_session_state({"stents": pending})
                self._pending_session_stents = []

            # ── Pre-populate vessel mesh if already segmented ──────────── #
            # Prefer self._vessel_poly (explicitly set by _restore_vessel_mesh /
            # _on_seg_ready) over reading back from the VTK mapper because
            # vtkPolyDataMapper.GetInput() can return a vtkDataSet base-type that
            # silently fails the GetNumberOfPoints() guard on some VTK builds.
            vessel_poly = getattr(self, "_vessel_poly", None)
            if vessel_poly is None:
                # Fallback: read from the 3-D mapper (e.g. loaded from a .vtp
                # companion before _vessel_poly was stored on this instance).
                vtk3d = self._mpr._vtk3d
                vessel_poly = vtk3d._mesh_mapper.GetInput()
            if vessel_poly is not None and vessel_poly.GetNumberOfPoints() > 0:
                self._planning_window.set_vessel_mesh(vessel_poly)

            # ── Insert as next tab (tab 2 if triplanar is at 1) ────────── #
            self._left_tabs.addTab(self._planning_window, f"{_I.STEP_PLAN}  Planificación 3D")

        self._left_tabs.setCurrentIndex(self._left_tabs.indexOf(self._planning_window))

    def _on_triplanar_toggled(self, checked: bool) -> None:
        """Toolbar '3 Planos' toggled — show/hide the floating triplanar dialog."""
        if checked:
            # Lazy creation on first use
            if self._triplanar_dlg is None:
                self._triplanar_dlg = CrossPlaneDialog(parent=self)
                # Uncheck the action when the user closes the dialog via ✕
                self._triplanar_dlg.hidden.connect(self._on_triplanar_hidden)
                # If a volume is already loaded, populate the dialog immediately
                if self._series is not None:
                    self._triplanar_dlg.set_volume(
                        self._series.volume, self._series.spacing
                    )
                    self._triplanar_dlg.set_window(
                        self._wc_spin.value(), self._ww_spin.value()
                    )
            self._triplanar_dlg.show()
            self._triplanar_dlg.raise_()
        else:
            if self._triplanar_dlg is not None:
                self._triplanar_dlg.hide()

    def _on_triplanar_hidden(self) -> None:
        """Called when the dialog's ✕ button is pressed — uncheck toolbar action."""
        if hasattr(self, "_triplanar_act"):
            self._triplanar_act.blockSignals(True)
            self._triplanar_act.setChecked(False)
            self._triplanar_act.blockSignals(False)

    def _arrange_docks(self) -> None:
        """Level-3 layout: no QDockWidgets remain; splitter sizes are set in
        _build_workflow_dock().  This method is kept as a no-op so the
        showEvent call site does not need to change."""

    def _build_menubar(self) -> None:
        mb = self.menuBar()

        # ── Archivo ──────────────────────────────────────────────────── #
        file_m = mb.addMenu("&Archivo")

        act = QAction("Abrir directorio DICOM…", self)
        act.setShortcut(QKeySequence.Open)
        act.triggered.connect(self.open_dicom_directory)
        file_m.addAction(act)

        act2 = QAction("Abrir archivos DICOM…", self)
        act2.triggered.connect(self.open_dicom_files)
        file_m.addAction(act2)

        file_m.addSeparator()

        patients_act = QAction("Gestión de pacientes…", self)
        patients_act.setShortcut("Ctrl+P")
        patients_act.triggered.connect(self._open_patient_manager)
        file_m.addAction(patients_act)

        self._act_users = QAction("Gestión de usuarios…", self)
        self._act_users.triggered.connect(self._open_user_manager)
        file_m.addAction(self._act_users)

        file_m.addSeparator()

        back_act = QAction("← Volver al panel de casos", self)
        back_act.setShortcut("Ctrl+W")
        back_act.setToolTip("Cerrar este caso y volver al panel de casos recientes")
        back_act.triggered.connect(self._close_case)
        file_m.addAction(back_act)

        file_m.addSeparator()

        logout_act = QAction("Cerrar sesión de usuario", self)
        logout_act.triggered.connect(self._logout)
        file_m.addAction(logout_act)

        file_m.addSeparator()

        save_act = QAction("Guardar sesión…", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_session)
        file_m.addAction(save_act)

        load_act = QAction("Cargar sesión…", self)
        load_act.setShortcut("Ctrl+L")
        load_act.triggered.connect(self._load_session)
        file_m.addAction(load_act)

        file_m.addSeparator()

        quit_act = QAction("Salir", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_m.addAction(quit_act)

        # ── Vista ────────────────────────────────────────────────────── #
        view_m = mb.addMenu("&Vista")

        reset_act = QAction("Restablecer ventana/nivel", self)
        reset_act.setShortcut("R")
        reset_act.triggered.connect(self._reset_wl)
        view_m.addAction(reset_act)

        view_m.addSeparator()

        for preset_name in WINDOW_PRESETS:
            pa = QAction(preset_name, self)
            pa.triggered.connect(lambda _, n=preset_name: self._apply_preset(n))
            view_m.addAction(pa)

        view_m.addSeparator()

        from prospective.ui.themes import is_dark
        _theme_icon = _I.THEME_LIGHT if is_dark() else _I.THEME_DARK
        self._theme_act = QAction(f"Cambiar a tema claro  {_theme_icon}", self)
        self._theme_act.setShortcut("Ctrl+T")
        self._theme_act.triggered.connect(self._toggle_theme)
        view_m.addAction(self._theme_act)

        view_m.addSeparator()

        # Panel shortcuts — navigate workflow stepper steps directly
        panels_m = view_m.addMenu("Pasos del flujo")
        for label, step_idx, shortcut in (
            (f"{_I.STEP_PATIENT} Paciente",      0, "Ctrl+1"),
            (f"{_I.STEP_SEGMENT} Segmentación",  1, "Ctrl+2"),
            (f"{_I.STEP_DETECT}  Detección",     2, "Ctrl+3"),
            (f"{_I.STEP_MORPHO}  Morfometría",   3, "Ctrl+4"),
            (f"{_I.STEP_PLAN}    Planificación", 4, "Ctrl+5"),
            (f"{_I.STEP_EXPORT}  Exportar",      5, "Ctrl+6"),
        ):
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(
                lambda _, i=step_idx: self._stepper.go_to(i)
            )
            panels_m.addAction(act)

        # ── Ayuda ────────────────────────────────────────────────────── #
        help_m = mb.addMenu("Ay&uda")
        about = QAction("Acerca de…", self)
        about.triggered.connect(self._show_about)
        help_m.addAction(about)

        help_m.addSeparator()
        act_audit = help_m.addAction(f"{_I.AUDIT} SkullChain — Auditoría")
        act_audit.setToolTip("Ver el registro criptográfico de acciones clínicas")
        act_audit.triggered.connect(self._show_audit_dialog)

    def _build_toolbar(self) -> None:
        # _DraggableToolBar keeps the double-click-to-maximise behaviour on the
        # toolbar area; the native Win32 title bar now handles dragging and the
        # min/max/close buttons.
        tb = _DraggableToolBar("Principal", self)
        self.addToolBar(tb)
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # ── PROSPECTIVE logo ──────────────────────────────────────────── #
        self._logo_lbl = QLabel()
        self._logo_lbl.setStyleSheet("background:transparent; margin: 0 10px 0 6px;")
        # 34 px height — fills ~94 % of the 36 px toolbar interior.
        # scaledToHeight preserves the logo's natural aspect ratio, so a wide
        # logo renders at full width instead of being crushed into a tiny square.
        pix = _tinted_logo_pix(34)
        if not pix.isNull():
            self._logo_lbl.setPixmap(pix)
        else:
            # Fallback: text logo (only shown if logo.png is missing)
            self._logo_lbl.setText("PROSPECTIVE")
            self._logo_lbl.setStyleSheet(
                "font-size:13px; font-weight:bold; "
                "color:#8B9BAA; background:transparent;"
            )
        tb.addWidget(self._logo_lbl)

        # ── Back-to-cases button ──────────────────────────────────────── #
        back_tb_act = QAction("⌂  Casos", self)
        back_tb_act.setToolTip("Volver al panel de casos recientes (Ctrl+W)")
        back_tb_act.triggered.connect(self._close_case)
        tb.addAction(back_tb_act)

        tb.addSeparator()

        open_act = QAction("Abrir DICOM", self)
        open_act.setToolTip("Abrir directorio DICOM (Ctrl+O)")
        open_act.triggered.connect(self.open_dicom_directory)
        tb.addAction(open_act)

        patients_tb_act = QAction("Pacientes", self)
        patients_tb_act.setToolTip("Gestión de pacientes y sesiones (Ctrl+P)")
        patients_tb_act.triggered.connect(self._open_patient_manager)
        tb.addAction(patients_tb_act)

        tb.addSeparator()

        # ── Triplanar view toggle ──────────────────────────────────────── #
        self._triplanar_act = QAction(f"{_I.MPR_VIEW}  3 Planos", self)
        self._triplanar_act.setToolTip(
            "Ver las tres vistas ortogonales (Axial/Coronal/Sagital) en paralelo\n"
            "con sliders individuales de posición X, Y, Z"
        )
        self._triplanar_act.setCheckable(True)
        self._triplanar_act.toggled.connect(self._on_triplanar_toggled)
        tb.addAction(self._triplanar_act)

        # NOTE: "Planificación 3D" and HU window presets (Cerebro / Hemorragia /
        # CTA / Hueso) have been intentionally removed from the toolbar.
        # • Planificación 3D is reached via the workflow stepper (Paso 5).
        # • Window presets are available in the Paso 1 panel combo and Vista menu.

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        self._theme_tb_act = QAction(f"{_I.THEME_LIGHT} Tema claro", self)
        self._theme_tb_act.setToolTip("Cambiar entre tema oscuro y claro (Ctrl+T)")
        self._theme_tb_act.triggered.connect(self._toggle_theme)
        tb.addAction(self._theme_tb_act)

        version_lbl = QLabel("v0.1.0-F01")
        tb.addWidget(version_lbl)

        # Min / max / close are provided by the native Win32 title bar.

    def _build_statusbar(self) -> None:
        self._statusbar: QStatusBar = self.statusBar()

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(180)
        self._progress.setVisible(False)
        self._statusbar.addPermanentWidget(self._progress)

        self._user_lbl = QLabel()
        _mc = "#9B9B9B" if _is_dark_mw() else "#6B6B6B"
        self._user_lbl.setStyleSheet(f"color:{_mc}; font-size:10px; padding-right:8px;")
        self._statusbar.addPermanentWidget(self._user_lbl)

        self._statusbar.showMessage(
            "Listo — abra un directorio DICOM para comenzar (Ctrl+O)"
        )

        # Resizing is handled by the native Win32 window frame (all edges).

    # ------------------------------------------------------------------ #
    # Actions                                                              #
    # ------------------------------------------------------------------ #

    def preload_dicom(self, path: str, study_id: int = 0) -> None:
        """Start loading *path* automatically (called from CaseDashboard).

        *study_id* is stored so that once the DICOM loads successfully the
        study record in the DB is updated with the confirmed path.
        """
        self._current_study_id   = study_id
        self._current_session_id = 0   # fresh case — no active session yet
        # Reflect the case number in the title bar immediately, even before
        # the DICOM finishes loading (patient name is filled in by _on_loaded).
        self._update_window_title()
        if path:
            self._start_load(path)

    def open_dicom_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Seleccionar directorio DICOM", ""
        )
        if path:
            self._start_load(path)

    def open_dicom_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar archivos DICOM", "",
            "Archivos DICOM (*.dcm *.DCM);;Todos los archivos (*)",
        )
        if files:
            from pathlib import Path
            self._start_load(str(Path(files[0]).parent))

    def _start_load(self, path: str) -> None:
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)   # indeterminate spinner
        self._statusbar.showMessage(f"Cargando: {path} …")
        self._last_dicom_path = path   # remember for session save

        self._load_thread = _LoadThread(path, self)
        self._load_thread.finished.connect(self._on_loaded)
        self._load_thread.error.connect(self._on_load_error)
        self._load_thread.start()

    def _on_loaded(self, series: DicomSeries) -> None:
        self._series = series
        self._series._source_path = getattr(self, "_last_dicom_path", "")
        self._progress.setVisible(False)
        self._mpr.load_series(series)
        self._seg_panel.set_volume(series.volume, series.spacing)

        # Push volume and window/level to the triplanar dialog if it is open.
        # If it hasn't been opened yet the data will be supplied at creation time.
        if self._triplanar_dlg is not None:
            self._triplanar_dlg.set_volume(series.volume, series.spacing)
            self._triplanar_dlg.set_window(
                series.metadata.window_center, series.metadata.window_width
            )

        # Clear the embedded planning view so the previous case's mesh is not
        # shown while the user runs segmentation on the new case.
        if self._planning_window is not None:
            self._planning_window.clear_vessel_mesh()
            # When loading as part of a session restore, _apply_session() has
            # already placed the saved mesh everywhere — but clear_vessel_mesh()
            # just erased it from the planning window.  Re-apply it immediately.
            # Prefer _pending_restore_mesh (loaded from the .vtp companion file);
            # fall back to _vessel_poly (set by _restore_vessel_mesh in _apply_session)
            # in case the companion file path was stale/missing but the data is
            # already in memory.
            _mesh_src = self._pending_restore_mesh or getattr(self, "_vessel_poly", None)
            if _mesh_src is not None and _mesh_src.GetNumberOfPoints() > 0:
                self._planning_window.set_vessel_mesh(_mesh_src)
        # Consume the pending mesh regardless of whether a planning window exists;
        # _show_planning_tab() will pick it up via the MPR's mesh mapper instead.
        self._pending_restore_mesh = None

        # Sync spinboxes without re-triggering the signal
        for spin in (self._wc_spin, self._ww_spin):
            spin.blockSignals(True)
        self._wc_spin.setValue(series.metadata.window_center)
        self._ww_spin.setValue(series.metadata.window_width)
        for spin in (self._wc_spin, self._ww_spin):
            spin.blockSignals(False)

        m = series.metadata
        self._seg_panel.set_modality(m.modality, m.window_center, m.window_width)

        # Re-apply saved segmentation parameters when restoring a session.
        # set_modality() above overwrites the threshold with modality defaults
        # (e.g. XA auto-calculates from window/level), so saved values must win.
        if self._pending_seg_state:
            self._seg_panel.restore_session_state(self._pending_seg_state)
            self._pending_seg_state = {}

        self._aneurysm_panel.set_modality(m.modality)   # auto-configure detection params
        # Keep patient_id in sync so SkullChain audit entries are traceable.
        self._current_patient_id: str = m.patient_id or ""
        self._lbl_name.setText(m.patient_name or "—")
        self._lbl_pid.setText(m.patient_id or "—")
        self._lbl_date.setText(
            f"{m.study_date[:4]}-{m.study_date[4:6]}-{m.study_date[6:]}"
            if len(m.study_date) == 8 else m.study_date or "—"
        )
        self._lbl_mod.setText(m.modality or "—")
        # Update the Level-3 patient-info strip above the workflow panel.
        if hasattr(self, "_patient_strip"):
            self._patient_strip.update_info(
                m.patient_name or "",
                m.patient_id   or "",
                m.modality     or "",
            )

        z, y, x = series.volume.shape
        sz, sy, sx = series.spacing
        self._lbl_dims.setText(f"{x} × {y} × {z} voxels")
        self._lbl_spacing.setText(f"{sx:.2f} × {sy:.2f} × {sz:.2f} mm")
        desc = m.series_description or m.study_description or "—"
        self._lbl_series_desc.setText(desc)

        # ── Non-3D projection detection ───────────────────────────────── #
        # A proper 3DRA/CT volume has ≥ 10 slices and roughly isotropic
        # spacing (sz ≤ 4 × in-plane pixel size).  Projection series like
        # 2D XA cine runs or biplane roadmaps have very few slices or
        # extreme z-anisotropy; they load without error but produce
        # meaningless 3D segmentation (disconnected slab stacks).
        # Threshold is 4× (not 3×) to accommodate thick-slice CT volumes
        # such as CAMACHO 606 (spacing 0.54 × 0.54 × 1.75 mm, ratio ≈ 3.24).
        _in_plane = max(sx, sy)
        _is_projection = z < 10 or (sz > 4.0 * _in_plane and _in_plane > 0)
        if hasattr(self, "_lbl_projection_warn"):
            if _is_projection:
                self._lbl_projection_warn.setText(
                    "⚠ Esta serie parece ser una proyección 2D "
                    f"({z} láminas, espaciado Z={sz:.1f} mm). "
                    "La segmentación 3D puede no ser significativa."
                )
                self._lbl_projection_warn.setVisible(True)
            else:
                self._lbl_projection_warn.setVisible(False)

        self._series_tree.clear()
        item = QTreeWidgetItem([desc, str(z)])
        self._series_tree.addTopLevelItem(item)
        self._series_tree.expandAll()

        self._statusbar.showMessage(
            f"{m.patient_name or 'Paciente'} — {z} slices — "
            f"espaciado {sx:.2f}×{sy:.2f}×{sz:.2f} mm — {m.modality}"
        )

        # Stepper: mark step 1 done, unlock step 2
        if hasattr(self, "_stepper"):
            name = m.patient_name or "Paciente"
            self._lbl_load_status.setText(
                f"✔ Cargado: {name} · {z} slices · {m.modality}"
            )
            self._stepper.mark_done_and_advance(0)

            # When loading as part of a session restore, mark_done_and_advance(0)
            # above resets the stepper to step 1 (Segmentation), overwriting the
            # restore_to() call made earlier in _apply_session().  Re-apply the
            # saved step here, after the DICOM load has completed.
            if self._pending_restore_step > 1:
                self._stepper.restore_to(self._pending_restore_step)
                self._pending_restore_step = -1
                # Schedule TWO deferred camera resets:
                # • 200 ms — after Qt's layout engine has run and the VTK
                #   widget has its correct pixel dimensions (ResetCamera needs
                #   the right aspect ratio for the frustum calculation).
                # • 600 ms — safety net in case the first render is overwritten
                #   by a Qt repaint event triggered by the tab becoming visible.
                self._schedule_planning_render(200)
                self._schedule_planning_render(600)

        # Forward series metadata to report panel for DICOM SR header
        study_date_fmt = (
            f"{m.study_date[:4]}-{m.study_date[4:6]}-{m.study_date[6:]}"
            if len(m.study_date) == 8 else m.study_date
        )
        self._report_panel.set_series_meta({
            "patient_name":      m.patient_name,
            "patient_id":        m.patient_id,
            "study_date":        study_date_fmt,
            "study_description": m.study_description or m.series_description,
            "modality":          m.modality,
        })

        # ── Auto-link DICOM path to the active study record ───────────── #
        if self._current_study_id:
            dicom_path = getattr(self, "_last_dicom_path", "")
            if dicom_path:
                try:
                    DatabaseManager.instance().update_study(
                        self._current_study_id, dicom_path=dicom_path
                    )
                    logger.info(
                        "DICOM auto-linked: study_id=%d  path=%r",
                        self._current_study_id, dicom_path,
                    )
                except Exception:
                    logger.warning("Could not auto-link DICOM to study", exc_info=True)

    def _on_load_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        QMessageBox.critical(self, "Error al cargar DICOM", msg)
        self._statusbar.showMessage(f"Error: {msg}")

    def _edit_current_case(self) -> None:
        """Open NuevoCasoDialog for the active study.

        * If a study is already linked (came from the Dashboard): opens in
          *edit* mode with all fields pre-filled.
        * If the DICOM was loaded manually (no study context yet): opens in
          *create* mode so the user can register the case from here, with the
          DICOM path already filled in.
        """
        from prospective.ui.widgets.nuevo_caso_dialog import NuevoCasoDialog
        from PyQt5.QtWidgets import QDialog
        user = AuthManager.instance().current_user
        created_by = user.username if user else ""

        # ── Edit mode: study already linked ──────────────────────────── #
        if self._current_study_id:
            study = DatabaseManager.instance().get_study(self._current_study_id)
            if study is None:
                return
            dlg = NuevoCasoDialog(
                created_by=created_by,
                patient_id=study.patient_id,
                study_id=self._current_study_id,
                parent=self,
            )
            dlg.exec()
            return

        # ── Create mode: DICOM loaded manually, no study yet ─────────── #
        dlg = NuevoCasoDialog(created_by=created_by, parent=self)

        # Pre-fill the DICOM path if one is already loaded
        dicom_path = getattr(self, "_last_dicom_path", "")
        if dicom_path:
            dlg.set_dicom_path(dicom_path)

        if dlg.exec() == QDialog.Accepted:
            # Link the newly-created study to this session
            self._current_study_id = dlg.study_id or 0
            logger.info(
                "Case registered from MainWindow — study_id=%d",
                self._current_study_id,
            )

    # ------------------------------------------------------------------ #
    # Segmentation handlers                                                #
    # ------------------------------------------------------------------ #

    def _on_seg_ready(self, result) -> None:
        # Warn if re-segmenting while devices are already placed.
        # Clips live in _clip_panel (always present); stents live inside
        # the planning window's _stent_panel (uses _placed, not _placed_stents).
        _has_placed_clips = bool(
            hasattr(self, "_clip_panel")
            and self._clip_panel.get_session_state().get("clips", [])
        )
        _has_placed_stents = bool(
            self._planning_window is not None
            and hasattr(self._planning_window, "_stent_panel")
            and getattr(self._planning_window._stent_panel, "_placed", [])
        )
        _has_pending_stents = bool(getattr(self, "_pending_session_stents", []))
        if _has_placed_clips or _has_placed_stents or _has_pending_stents:
            QMessageBox.warning(
                self,
                "Atención — Re-segmentación",
                "Se ha ejecutado una nueva segmentación mientras existe una\n"
                "planificación activa (clips / stents colocados).\n\n"
                "La geometría del vaso ha cambiado: comprueba que los\n"
                "dispositivos sigan correctamente posicionados antes de\n"
                "continuar.",
            )

        self._restore_vessel_mesh(result.poly_data)
        # Stepper: mark step 2 done, unlock step 3
        if hasattr(self, "_stepper"):
            self._stepper.mark_done_and_advance(1)
        self._statusbar.showMessage(
            f"Segmentación: {result.n_vertices:,} verts · "
            f"{result.n_triangles:,} tris · "
            f"umbral {result.threshold_hu:.0f} HU"
        )
        self._auto_save("segmentacion")
        try:
            user = AuthManager.instance().current_user
            SkullChain.instance().append(
                ACT_SEGMENTATION,
                {
                    "n_vertices": result.n_vertices,
                    "n_triangles": result.n_triangles,
                    "threshold_hu": float(result.threshold_hu),
                },
                username=user.username if user else "",
                patient_id=getattr(self, "_current_patient_id", ""),
            )
        except Exception:
            pass  # audit must never block clinical workflow

    def _restore_vessel_mesh(self, poly_data) -> None:
        """Distribute *poly_data* to every panel that needs the vessel mesh.

        Called both from _on_seg_ready (live segmentation) and from
        _apply_session (session restore).  Keeps all downstream widgets in sync
        so no code is duplicated.
        """
        self._vessel_poly = poly_data
        self._mpr.set_mesh(poly_data)
        # Explicitly clear the magenta aneurysm highlight so a stale candidate
        # from a previous detection run is never left visible after re-segmenting.
        self._mpr.set_aneurysm(None)
        self._aneurysm_panel.set_mesh(poly_data)
        self._clip_panel.set_vessel_mesh(poly_data)
        self._morpho_panel.set_mesh(None)   # clear — morpho only runs on candidates
        self._print_panel.set_mesh(poly_data)
        if hasattr(self, "_perforator_panel"):
            self._perforator_panel.set_vessel_mesh(poly_data)
        if self._planning_window is not None:
            self._planning_window.set_vessel_mesh(poly_data)
            self._planning_window.set_aneurysm(None)   # clear in planning window too

    def _on_seg_preview_ready(self, result) -> None:
        """Fast-preview update: refresh MPR mesh only; no downstream panel effects."""
        self._mpr.set_mesh(result.poly_data)
        self._statusbar.showMessage(
            f"Vista previa: {result.n_vertices:,} verts · "
            f"{result.n_triangles:,} tris  (baja resolución)",
            3000,
        )

    def _on_request_auto_threshold(self) -> None:
        """
        DEPRECATED — signal is no longer connected.

        The ⚡ Auto-umbral button is hidden and this slot is never called.
        Per-pixel HU tracking via sigMouseMoved was removed because it caused
        severe visual glitches in all four slice views.  The method is kept
        as a stub to avoid AttributeError if any third-party code still
        references it by name.
        """
        # No-op: HU cursor tracking removed.  Button hidden in segmentation_panel.py.
        return  # pragma: no cover  # noqa: RET501  (explicit for clarity)

    def _on_request_mpr_seed(self) -> None:
        """
        Read the current MPR crosshair voxel and forward it to the
        segmentation panel as a grow-from-seeds seed point.

        The three slice indices give:
          axial index   → z
          coronal index → y
          sagital index → x
        """
        if self._series is None:
            return
        z, y, x = self._mpr.get_current_voxel()
        self._seg_panel.add_grow_seed(z, y, x)
        self._statusbar.showMessage(
            f"Semilla añadida en vóxel Z={z}  Y={y}  X={x}",
            4000,
        )

    def _on_candidate_for_morpho(self, candidate) -> None:
        """Load candidate mesh into morphometrics and run analysis automatically."""
        poly_data = candidate.poly_data if candidate is not None else None
        self._morpho_panel.set_mesh(poly_data)
        if poly_data is not None:
            self._morpho_panel.analyze()
            # Stepper: mark step 3 done, unlock step 4
            if hasattr(self, "_stepper"):
                self._stepper.mark_done_and_advance(2)
            self._auto_save("deteccion")

    def _on_morpho_done_for_clips(self, result) -> None:
        """Forward morphometric results to the clip planning panel and planning window."""
        # Stepper: mark step 4 done, unlock steps 5 & 6
        if hasattr(self, "_stepper"):
            self._stepper.mark_done_and_advance(3)
            self._stepper.unlock(4)
            self._stepper.unlock(5)
        self._auto_save("morfometria")
        self._clip_panel.set_neck_diameter(result.neck_diameter_mm)
        cx, cy, cz = result.centroid
        self._clip_panel.set_aneurysm_centroid(cx, cy, cz)
        if self._planning_window is not None:
            self._planning_window.set_neck_diameter(result.neck_diameter_mm)
            self._planning_window.set_aneurysm_centroid(cx, cy, cz)
            parent_mm = self._morpho_panel._spin_parent_diam.value()
            self._planning_window.set_neck_data_for_sizing(
                result.neck_diameter_mm, parent_mm
            )
            # Feature 5 — pass morpho result + aneurysm mesh for 3D overlay
            aneurysm_poly = self._morpho_panel._pending_mesh
            if aneurysm_poly is not None:
                self._planning_window.set_morpho_result(result, aneurysm_poly)

        # Feature 5 — auto-estimate parent artery diameter from vessel mesh
        if (self._morpho_panel._spin_parent_diam.value() == 0.0
                and hasattr(self, '_vessel_poly') and self._vessel_poly is not None):
            try:
                from prospective.processing.parent_artery import estimate_parent_artery_diameter
                est = estimate_parent_artery_diameter(self._vessel_poly, result)
                if est > 0:
                    self._morpho_panel._spin_parent_diam.setValue(round(est, 2))
            except Exception:
                pass   # non-critical

    def _on_morpho_done_for_coils(self, result) -> None:
        """Forward morphometric results to the coil planning panel."""
        cx, cy, cz = result.centroid
        # Use max dimension as dome diameter approximation
        dome_mm = getattr(result, "max_diameter_mm", result.neck_diameter_mm * 1.5)
        vol_mm3 = getattr(result, "volume_mm3", 0.0)
        self._coil_panel.set_aneurysm_data(dome_mm, vol_mm3, (cx, cy, cz))

    def _on_morpho_done_for_perforators(self, result) -> None:
        """Forward morphometric result + aneurysm mesh to the perforator panel."""
        aneurysm_poly = getattr(self._morpho_panel, "_pending_mesh", None)
        self._perforator_panel.set_morpho_data(result, aneurysm_poly)

    def _on_decision_updated(self, decision) -> None:
        """Audit treatment decision changes via SkullChain."""
        try:
            user = AuthManager.instance().current_user
            SkullChain.instance().append(
                ACT_TREATMENT_DECISION,
                {
                    "recommendation": decision.recommendation,
                    "confidence": decision.confidence,
                    "clip_score": decision.clip_score,
                    "endo_score": decision.endo_score,
                },
                username=user.username if user else "",
                patient_id=getattr(self, "_current_patient_id", ""),
            )
        except Exception:
            pass  # audit must never block clinical workflow

    def _on_perforator_overlay_ready(self, actors: list) -> None:
        """Add perforator risk actors to the planning window if it is open."""
        if self._planning_window is not None:
            self._planning_window.set_perforator_overlay(actors)

    def _on_perforator_overlay_cleared(self) -> None:
        """Remove perforator risk actors from the planning window."""
        if self._planning_window is not None:
            self._planning_window.clear_perforator_overlay()

    def _on_perforator_render_requested(self) -> None:
        """Re-render the planning window after a perforator visibility toggle."""
        if self._planning_window is not None:
            self._planning_window.request_render()

    def _on_coil_placed(self, index: int, name: str, position, poly_data) -> None:
        """Handle coil placement — add actor to planning window and update report."""
        self._statusbar.showMessage(f"Coil depositado #{index}: {name}")
        self._sync_report_coils()
        if self._planning_window is not None:
            # Retrieve spec and sac radius from the coil panel
            placed = next((p for p in self._coil_panel._placed if p.index == index), None)
            spec        = placed.entry.spec if placed else None
            dome_mm     = self._coil_panel._dome_mm or 10.0
            sac_radius  = dome_mm / 2.0
            self._planning_window.add_coil(
                index, name, position, poly_data,
                spec=spec, sac_radius_mm=sac_radius,
            )

    def _on_coil_removed(self, index: int) -> None:
        self._statusbar.showMessage(f"Coil #{index} eliminado.")
        self._sync_report_coils()
        if self._planning_window is not None:
            self._planning_window.remove_coil(index)

    def _on_coil_visibility(self, index: int, visible: bool) -> None:
        if self._planning_window is not None:
            self._planning_window.set_coil_visible(index, visible)

    def _sync_report_coils(self) -> None:
        """Rebuild the report panel's coil list from the current placed coils."""
        from prospective.io.report_generator import CoilEntry as ReportCoilEntry
        entries = []
        for pc in self._coil_panel._placed:
            spec = pc.entry.spec
            entries.append(ReportCoilEntry(
                index       = pc.index,
                name        = pc.entry.name,
                position_mm = tuple(pc.position),
                coil_type   = spec.coil_type.value if spec else "",
                diameter_mm = spec.diameter_mm     if spec else 0.0,
                length_cm   = spec.length_cm       if spec else 0.0,
                manufacturer= spec.manufacturer    if spec else "",
                is_custom   = pc.entry.poly_data is not None,
            ))
        self._report_panel.set_coils(entries)

    def _sync_report_stents(self) -> None:
        """Rebuild the report panel's stent list from the planning window."""
        if self._planning_window is None:
            self._report_panel.set_stents([])
            return
        stent_dicts = []
        for ps in self._planning_window._stent_panel._placed:
            spec = ps.entry.spec
            pos  = list(ps.transform.GetPosition())
            stent_dicts.append({
                "index":        ps.index,
                "name":         ps.entry.name,
                "stent_type":   spec.stent_type.value if spec else "",
                "manufacturer": spec.manufacturer     if spec else "",
                "diameter_mm":  spec.diameter_mm      if spec else 0.0,
                "length_mm":    spec.length_mm        if spec else 0.0,
                "position_mm":  pos,
                "is_custom":    ps.entry.poly_data is not None,
            })
        self._report_panel.set_stents(stent_dicts)

    def _on_clip_placed(self, index: int, name: str, transform, poly_data) -> None:
        placed = self._clip_panel._placed[-1]
        self._mpr.add_clip(index, placed.entry.spec, transform, poly_data)
        blade = (f"  ({placed.entry.spec.blade_length_mm:.0f} mm)"
                 if placed.entry.spec else "")
        self._statusbar.showMessage(f"Clip colocado #{index}: {name}{blade}")
        self._sync_report_clips()

    def _sync_report_clips(self) -> None:
        """Rebuild the report panel's clip list from the current placed clips."""
        from prospective.io.report_generator import ClipEntry as ReportClipEntry
        entries = []
        for pc in self._clip_panel._placed:
            m = vtk.vtkMatrix4x4()
            pc.transform.GetMatrix(m)
            pos = (m.GetElement(0, 3), m.GetElement(1, 3), m.GetElement(2, 3))
            state = self._clip_panel.get_session_state()
            # Find the matching state entry
            clip_state = next((c for c in state["clips"] if c["index"] == pc.index), {})
            ori = tuple(clip_state.get("orientation", [0.0, 0.0, 0.0]))
            entries.append(ReportClipEntry(
                index=pc.index,
                name=pc.entry.name,
                position_mm=pos,
                orientation_deg=ori,
                is_custom=getattr(pc.entry, "_source_path", None) is not None,
            ))
        self._report_panel.set_clips(entries)

    def _on_aneurysm_mesh_cropped(self, poly_data) -> None:
        """User applied or reset a 3D crop in the aneurysm panel.

        Updates the shared vessel mesh reference and the 3D viewer so the
        cropped result is visible immediately.  Does NOT propagate to the
        clip or print panels — those work on the original surgical geometry.
        """
        self._vessel_poly = poly_data
        self._mpr.set_mesh(poly_data)
        # Also update the planning window so the cropped mesh is visible there.
        # At step 3 (Detección) the planning tab is the active view, so without
        # this the user would see the old uncropped mesh after applying the crop.
        if self._planning_window is not None:
            self._planning_window.set_vessel_mesh(poly_data)
        n = poly_data.GetNumberOfPoints() if poly_data is not None else 0
        logger.info("Aneurysm panel crop applied — %d vertices in viewer", n)

    def _on_box_crop_widget_requested(self, enabled: bool) -> None:
        """Show or hide the interactive vtkBoxWidget2 in the MPR 3D view.

        When enabling, we also switch to Tab 0 (MPR) so the user can interact
        with the box directly on the mesh.
        """
        if enabled and hasattr(self, "_left_tabs"):
            # Show the MPR view so the user can drag the box handles
            self._left_tabs.setCurrentIndex(0)
        self._mpr._vtk3d.show_crop_box(enabled)
        if enabled:
            self._statusbar.showMessage(
                "Arrastra los handles del cuadro violeta para definir el recorte. "
                "Luego pulsa 'Aplicar recorte de caja' en el panel de Detección.",
                0,   # stay until next message
            )
        else:
            self._statusbar.clearMessage()

    def _on_interactive_crop_apply(self) -> None:
        """Read the current box widget bounds from VTK and apply them to the mesh."""
        bounds = self._mpr._vtk3d.get_crop_box_bounds()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        logger.info(
            "Interactive crop apply: X[%.1f,%.1f] Y[%.1f,%.1f] Z[%.1f,%.1f]",
            xmin, xmax, ymin, ymax, zmin, zmax,
        )
        self._aneurysm_panel.apply_external_bounds(xmin, xmax, ymin, ymax, zmin, zmax)
        # After the crop, switch back to the planning tab (if open) so the user
        # can immediately see the cropped mesh in the 3D view where they work.
        if self._planning_window is not None and hasattr(self, "_left_tabs"):
            self._left_tabs.setCurrentIndex(self._left_tabs.indexOf(self._planning_window))

    def _on_candidate_visibility_changed(self, visible: bool) -> None:
        """Toggle the magenta candidate highlight in all active 3D viewers.

        Called by AneurysmPanel.candidate_visibility_changed signal.
        Affects both the MPR's embedded VTK widget and the planning window (if open),
        because at step 3 the planning tab is shown while the MPR tab is hidden.
        """
        self._mpr._vtk3d.set_aneurysm_visible(visible)
        if self._planning_window is not None:
            self._planning_window.set_aneurysm_visible(visible)

    def _on_aneurysm_highlighted(self, candidate) -> None:
        """Show selected candidate in magenta in the 3D viewers (browse, no step change).

        Deliberately does NOT force-open the planning tab or advance the stepper.
        The user can browse the full candidate list freely; the step advances only
        when they click "Confirmar candidato → Morfometría" (candidate_confirmed).
        """
        poly_data = candidate.poly_data if candidate is not None else None
        self._mpr.set_aneurysm(poly_data)
        # Update the planning window only if it is already open — never force it.
        if self._planning_window is not None:
            self._planning_window.highlight_candidate(candidate)
        if candidate is not None:
            self._statusbar.showMessage(
                f"Candidato #{candidate.index}  Ø {candidate.diameter_mm:.1f} mm  "
                f"| Score {candidate.score * 100:.0f}%  "
                f"| {candidate.n_points:,} puntos"
            )

    def _on_aneurysm_isolated(self, candidate) -> None:
        """Visual isolation: focus the 3D camera on the candidate (no step change).

        Like highlighted, this is a browse-only action.  The step advances only
        when the user clicks "Confirmar candidato" in the detection panel.
        """
        poly_data = candidate.poly_data if candidate is not None else None
        self._mpr.set_aneurysm(poly_data)
        # Update the planning window only if already open
        if self._planning_window is not None:
            self._planning_window.highlight_candidate(candidate)
        if candidate is not None:
            self._statusbar.showMessage(
                f"Candidato #{candidate.index} — Ø {candidate.diameter_mm:.1f} mm  "
                f"| Vista 3D actualizada"
            )

    def _ensure_planning_window_open(self) -> None:
        """Ensure the 3D planning view is visible (embedded in the left panel tab)."""
        self._show_planning_tab()

    def _open_planning_window(self) -> None:
        """Show the embedded 3D planning view in the left panel tab."""
        self._show_planning_tab()

    def _capture_screenshot(self) -> None:
        """Capture the 3D VTK render as PNG bytes and deliver to report panel."""
        try:
            png_bytes = self._mpr._vtk3d.capture_png()
            self._report_panel.deliver_screenshot(png_bytes)
        except Exception as exc:
            logger.exception("Screenshot capture failed")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error de captura", str(exc))

    def _open_patient_manager(self) -> None:
        """Open the patient / study / session management dialog."""
        dlg = PatientManagerDialog(self, current_session_path=self._last_session_path)
        dlg.session_open_requested.connect(self._load_session_from_path)
        dlg.dicom_open_requested.connect(self._start_load)
        dlg.exec()

    def _open_user_manager(self) -> None:
        """Admin-only: open user management dialog."""
        if not AuthManager.instance().has_role(User.ROLE_ADMIN):
            QMessageBox.warning(
                self, "Acceso denegado",
                "Solo los administradores pueden gestionar usuarios."
            )
            return
        UserManagerDialog(self).exec()

    def _logout(self) -> None:
        reply = QMessageBox.question(
            self, "Cerrar sesión",
            "¿Cerrar la sesión de usuario actual?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        AuthManager.instance().logout()
        dlg = LoginDialog(self)
        dlg.exec()   # blocks; reject() calls sys.exit()
        self._apply_role_restrictions()
        self._update_user_status_label()

    def _apply_role_restrictions(self) -> None:
        """Show/hide or enable/disable features based on the current user's role."""
        auth = AuthManager.instance()
        is_admin = auth.has_role(User.ROLE_ADMIN)
        # User management only for admins
        if hasattr(self, "_act_users"):
            self._act_users.setEnabled(is_admin)

    def _update_user_status_label(self) -> None:
        auth = AuthManager.instance()
        user = auth.current_user
        if user:
            role_label = {
                User.ROLE_ADMIN:    "Administrador",
                User.ROLE_SURGEON:  "Cirujano",
                User.ROLE_RESIDENT: "Residente",
            }.get(user.role, user.role)
            self._user_lbl.setText(
                f"{user.full_name or user.username}  [{role_label}]"
            )
        else:
            self._user_lbl.setText("Sin sesión")

    def _export_aneurysm_stl(self, poly_data) -> None:
        if poly_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar candidato STL", "aneurisma.stl",
            "STL Files (*.stl);;All Files (*)"
        )
        if path:
            MeshExporter.export_stl(poly_data, path)
            self._statusbar.showMessage(f"Candidato exportado: {path}")

    def _export_stl(self) -> None:
        result = self._seg_panel.last_result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar STL", "vasculatura.stl",
            "STL Files (*.stl);;All Files (*)"
        )
        if path:
            MeshExporter.export_stl(result.poly_data, path)
            self._statusbar.showMessage(f"STL exportado: {path}")
            try:
                import hashlib
                import pathlib
                file_hash = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
                user = AuthManager.instance().current_user
                SkullChain.instance().append(
                    ACT_MESH_EXPORTED,
                    {"format": "STL", "filename": pathlib.Path(path).name, "file_hash": file_hash},
                    username=user.username if user else "",
                    patient_id=getattr(self, "_current_patient_id", ""),
                )
            except Exception:
                pass  # audit must never block clinical workflow

    def _export_obj(self) -> None:
        result = self._seg_panel.last_result
        if result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar OBJ", "vasculatura.obj",
            "OBJ Files (*.obj);;All Files (*)"
        )
        if path:
            MeshExporter.export_obj(result.poly_data, path)
            self._statusbar.showMessage(f"OBJ exportado: {path}")

    def _on_preset_changed(self, name: str) -> None:
        if name in WINDOW_PRESETS:
            self._apply_preset(name)

    def _apply_preset(self, name: str) -> None:
        wc, ww = WINDOW_PRESETS[name]
        for spin in (self._wc_spin, self._ww_spin):
            spin.blockSignals(True)
        self._wc_spin.setValue(wc)
        self._ww_spin.setValue(ww)
        for spin in (self._wc_spin, self._ww_spin):
            spin.blockSignals(False)
        self._mpr.set_window(wc, ww)
        if self._triplanar_dlg is not None:
            self._triplanar_dlg.set_window(wc, ww)
        # Update combo without triggering signal
        idx = self._preset_combo.findText(name)
        if idx >= 0:
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentIndex(idx)
            self._preset_combo.blockSignals(False)

    def _on_wl_changed(self) -> None:
        wc, ww = self._wc_spin.value(), self._ww_spin.value()
        self._mpr.set_window(wc, ww)
        if self._triplanar_dlg is not None:
            self._triplanar_dlg.set_window(wc, ww)

    def _reset_wl(self) -> None:
        if self._series:
            m = self._series.metadata
            self._apply_preset_values(m.window_center, m.window_width)

    def _apply_preset_values(self, wc: float, ww: float) -> None:
        for spin in (self._wc_spin, self._ww_spin):
            spin.blockSignals(True)
        self._wc_spin.setValue(wc)
        self._ww_spin.setValue(ww)
        for spin in (self._wc_spin, self._ww_spin):
            spin.blockSignals(False)
        self._mpr.set_window(wc, ww)
        if self._triplanar_dlg is not None:
            self._triplanar_dlg.set_window(wc, ww)

    # ------------------------------------------------------------------ #
    # Session save / load                                                  #
    # ------------------------------------------------------------------ #

    def _collect_session(self) -> SessionData:
        """Build a SessionData snapshot from current UI state."""
        data = SessionData()

        # DICOM path + study link
        if self._series is not None:
            data.dicom_path = getattr(self._series, "_source_path", "")
        data.study_id = self._current_study_id

        # Window / level
        data.wl_center = self._wc_spin.value()
        data.wl_width  = self._ww_spin.value()
        data.wl_preset = self._preset_combo.currentText()

        # 3D rendering
        data.preset_3d  = self._3d_preset_combo.currentText()
        data.blend_mode = self._3d_mode_combo.currentText()
        vtk_widget = self._mpr._vtk3d
        data.hu_min = vtk_widget._hu_min
        data.hu_max = vtk_widget._hu_max

        # Segmentation panel — store the full state dict (all adv_* keys etc.)
        data.seg_state = self._seg_panel.get_session_state()

        # Aneurysm detection panel — full state (all 8 params incl. v5/v7 fields)
        data.det_state = self._aneurysm_panel.get_session_state()

        # Clips + trajectory
        clip_state = self._clip_panel.get_session_state()
        from prospective.io.session import ClipState, CoilState
        for cd in clip_state["clips"]:
            data.clips.append(ClipState(**cd))
        tr = clip_state["trajectory"]
        data.traj_entry   = tr["entry"]
        data.traj_target  = tr["target"]
        data.traj_visible = tr["visible"]

        # Coils
        coil_state = self._coil_panel.get_session_state()
        for cd in coil_state["coils"]:
            data.coils.append(CoilState(
                index       = cd["index"],
                name        = cd["name"],
                is_custom   = cd["is_custom"],
                custom_path = cd["custom_path"],
                position    = cd["position"],
            ))

        # Stents / flow diverters (live inside the 3D planning window)
        from prospective.io.session import StentState
        if self._planning_window is not None:
            stent_state = self._planning_window._stent_panel.get_session_state()
            for sd in stent_state["stents"]:
                data.stents.append(StentState(
                    index       = sd["index"],
                    name        = sd["name"],
                    is_custom   = sd["is_custom"],
                    custom_path = sd["custom_path"],
                    position    = sd["position"],
                    orientation = sd["orientation"],
                ))

        # Perforator risk thresholds
        perf = self._perforator_panel.get_session_state()
        data.perf_r_high = perf["r_high"]
        data.perf_r_med  = perf["r_med"]
        data.perf_r_low  = perf["r_low"]

        # Current workflow step (so the user lands on the same step after reload)
        if hasattr(self, "_stepper"):
            data.current_step = self._stepper.current_index()

        # Morphometrics snapshot (if available)
        if self._morpho_panel._result is not None:
            data.morpho_snapshot = self._morpho_panel._result.to_dict()

        # Clinical state: PHASES score inputs + parent artery diam + treatment decision
        data.clinical_state = {
            **self._morpho_panel.get_session_state(),       # phases + parent_diam
            "treatment": self._treatment_panel.get_session_state(),
        }

        # 3D-print preparation parameters
        data.print_prep_state = self._print_panel.get_session_state()

        # Report / patient data
        rep = self._report_panel.get_session_state()
        data.report_patient_name = rep.get("patient_name", "")
        data.report_patient_id   = rep.get("patient_id", "")
        data.report_patient_dob  = rep.get("patient_dob", "")
        data.report_surgeon      = rep.get("surgeon", "")
        data.report_institution  = rep.get("institution", "")
        data.report_notes        = rep.get("notes", "")
        data.report_treatment    = rep.get("treatment", "")

        return data

    def _auto_save(self, milestone: str = "") -> None:
        """Silent auto-save: write session to an auto-generated path, no dialog.

        Called automatically after key workflow milestones (segmentation done,
        detection done, morphometrics done) so progress is never silently lost.
        A brief status-bar message confirms the save; errors are logged and
        surfaced via status bar but never shown as modal dialogs.
        """
        import datetime
        from pathlib import Path as _Path

        # Choose save directory: study-specific if linked, otherwise generic
        if getattr(self, "_current_study_id", None):
            _dir = (
                _Path.home() / ".prospective" / "cases"
                / str(self._current_study_id)
            )
        else:
            _dir = _Path.home() / ".prospective" / "autosave"
        try:
            _dir.mkdir(parents=True, exist_ok=True)
        except Exception as _e:
            logger.warning("Auto-save: cannot create directory %s — %s", _dir, _e)
            return

        # Re-use the existing session file path when available; otherwise create
        # a timestamped one.  Never overwrite with a *new* name — respect Save As.
        if self._last_session_path and _Path(self._last_session_path).exists():
            _path = self._last_session_path
        else:
            _ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            _tag  = f"_{milestone}" if milestone else ""
            _path = str(_dir / f"autosave{_tag}_{_ts}{SESSION_EXT}")

        try:
            data = self._collect_session()

            # ── Save companion mesh .vtp (mirrors _save_session logic) ────── #
            if getattr(self, "_vessel_poly", None) is not None:
                _mesh_path = str(_Path(_path).with_suffix("")) + "_mesh.vtp"
                try:
                    MeshExporter.save_vtp(self._vessel_poly, _mesh_path)
                    data.seg_mesh_path = _mesh_path
                except Exception as _mesh_err:
                    logger.warning("Auto-save: mesh companion failed — %s", _mesh_err)

            save_session(data, _path)

            # Register in DB if linked to a study
            if self._current_study_id:
                db = DatabaseManager.instance()
                _existing = db.get_planning_session_by_path(_path)
                if _existing:
                    db.update_planning_session(
                        _existing.id,
                        file_path=_path,
                        n_clips=len(data.clips) + len(data.coils) + len(data.stents),
                        notes=data.notes,
                    )
                    if not self._current_session_id:
                        self._current_session_id = _existing.id
                else:
                    _ps = db.add_planning_session(
                        study_id=self._current_study_id,
                        file_path=_path,
                        label=f"autosave_{milestone or 'checkpoint'}",
                        n_clips=len(data.clips) + len(data.coils) + len(data.stents),
                        notes=data.notes,
                    )
                    if not self._current_session_id:
                        self._current_session_id = _ps.id

            self._last_session_path = _path
            _tag_msg = f" ({milestone})" if milestone else ""
            self._statusbar.showMessage(
                f"✔ Guardado automático{_tag_msg}: {_Path(_path).name}", 4000
            )
            logger.info("Auto-save%s: %s", _tag_msg, _path)
        except Exception as _exc:
            logger.warning("Auto-save failed: %s", _exc, exc_info=True)
            self._statusbar.showMessage(f"⚠ Guardado automático fallido: {_exc}", 5000)

    def _save_session(self) -> bool:
        """Save the session interactively.

        Returns True if the session was actually saved to disk (the user
        confirmed the file dialog), False if the user cancelled or an
        error occurred.  Callers that block a navigation action (e.g.
        _close_case) MUST check the return value.
        """
        # ── Build a default path inside the study-specific sessions folder ─ #
        # All sessions for a given study are stored under:
        #   ~/.prospective/cases/{study_id}/
        # This keeps session files organized by project and makes them easy to
        # find from the CaseDashboard without knowing where the user saved them.
        import datetime
        from pathlib import Path as _Path

        if getattr(self, "_current_study_id", None):
            _sessions_dir = (
                _Path.home() / ".prospective" / "cases"
                / str(self._current_study_id)
            )
        else:
            _sessions_dir = _Path.home() / ".prospective" / "sessions"
        _sessions_dir.mkdir(parents=True, exist_ok=True)

        # Re-save to the same file → pre-fill with the existing path so
        # the user can confirm with one Enter key press.
        _last = getattr(self, "_last_session_path", None)
        if _last:
            _default = _last
        else:
            _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            _default = str(_sessions_dir / f"sesion_{_ts}{SESSION_EXT}")

        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar sesión",
            _default,
            f"Sesión PROSPECTIVE (*{SESSION_EXT});;JSON (*.json);;Todos (*)",
        )
        if not path:
            return False   # user cancelled the file dialog
        try:
            data = self._collect_session()

            # ── Save segmented mesh as companion .vtp ─────────────── #
            if getattr(self, "_vessel_poly", None) is not None:
                mesh_path = str(_Path(path).with_suffix("")) + "_mesh.vtp"
                try:
                    MeshExporter.save_vtp(self._vessel_poly, mesh_path)
                    data.seg_mesh_path = mesh_path
                except Exception as _mesh_err:
                    logger.warning("Mesh save failed (session still saved): %s", _mesh_err)

            save_session(data, path)

            # ── Register / update in DB so CaseDashboard can list it ── #
            if self._current_study_id:
                label  = _Path(path).stem          # filename without extension
                n_clips = (len(data.clips) + len(data.coils) + len(data.stents))
                db = DatabaseManager.instance()

                is_same_file = (path == self._last_session_path)
                if self._current_session_id and is_same_file:
                    # Re-save to the same file — update existing record
                    db.update_planning_session(
                        self._current_session_id,
                        file_path=path,
                        n_clips=n_clips,
                        notes=data.notes,
                    )
                else:
                    # New file (Save As) or first save — create a new DB record
                    ps = db.add_planning_session(
                        study_id=self._current_study_id,
                        file_path=path,
                        label=label,
                        n_clips=n_clips,
                        notes=data.notes,
                    )
                    self._current_session_id = ps.id

            self._last_session_path = path
            self._statusbar.showMessage(f"Sesión guardada: {path}")
            return True    # save succeeded
        except Exception as exc:
            QMessageBox.critical(self, "Error al guardar sesión", str(exc))
            return False   # save failed

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar sesión", "",
            f"Sesión PROSPECTIVE (*{SESSION_EXT});;JSON (*.json);;Todos (*)",
        )
        if not path:
            return
        self._load_session_from_path(path)

    def _load_session_from_path(
        self, path: str, *, auto_load_dicom: bool = False
    ) -> None:
        """Load a .prospective file by path (used by dialog and direct call).

        *auto_load_dicom* — when True, the DICOM associated with the session is
        loaded automatically without prompting (used by the Welcome-screen banner
        and CaseDashboard resume flow).  When False (default), a confirmation
        dialog is shown so the user can decline on slow/remote drives.
        """
        try:
            data = load_session(path)
            self._apply_session(data, auto_load_dicom=auto_load_dicom)
            self._last_session_path = path

            # ── Sync with DB so subsequent saves update the correct record ── #
            if data.study_id:
                db = DatabaseManager.instance()
                existing = db.get_planning_session_by_path(path)
                if existing:
                    self._current_session_id = existing.id
                else:
                    # Session file exists on disk but was never registered
                    # (e.g. created before this feature, or imported externally)
                    from pathlib import Path as _Path
                    ps = db.add_planning_session(
                        study_id=data.study_id,
                        file_path=path,
                        label=_Path(path).stem,
                    )
                    self._current_session_id = ps.id

            self._statusbar.showMessage(f"Sesión cargada: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error al cargar sesión", str(exc))

    def _apply_session(
        self, data: SessionData, *, auto_load_dicom: bool = False
    ) -> None:
        """Restore all UI panels from a SessionData object.

        *auto_load_dicom* — when True the DICOM is reloaded silently (no
        modal dialog).  Used by the Welcome-screen / CaseDashboard resume flow.
        """
        # Window / level
        self._apply_preset_values(data.wl_center, data.wl_width)
        idx = self._preset_combo.findText(data.wl_preset)
        if idx >= 0:
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentIndex(idx)
            self._preset_combo.blockSignals(False)

        # 3D rendering
        idx3 = self._3d_preset_combo.findText(data.preset_3d)
        if idx3 >= 0:
            self._3d_preset_combo.setCurrentIndex(idx3)
        idxm = self._3d_mode_combo.findText(data.blend_mode)
        if idxm >= 0:
            self._3d_mode_combo.setCurrentIndex(idxm)
        self._mpr._vtk3d._set_hu_range_silent(int(data.hu_min), int(data.hu_max))

        # Segmentation — pass the full saved state dict directly to the panel
        self._seg_panel.restore_session_state(data.seg_state)

        # Aneurysm detection
        self._aneurysm_panel.restore_session_state(data.det_state)

        # Clips + trajectory
        clip_dict = {
            "clips": [
                {
                    "index":       c.index,
                    "name":        c.name,
                    "is_custom":   c.is_custom,
                    "custom_path": c.custom_path,
                    "position":    c.position,
                    "orientation": c.orientation,
                }
                for c in data.clips
            ],
            "trajectory": {
                "entry":   data.traj_entry,
                "target":  data.traj_target,
                "visible": data.traj_visible,
            },
        }
        self._clip_panel.restore_session_state(clip_dict)

        # Coils
        coil_dict = {
            "coils": [
                {
                    "index":       c.index,
                    "name":        c.name,
                    "is_custom":   c.is_custom,
                    "custom_path": c.custom_path,
                    "position":    c.position,
                }
                for c in data.coils
            ],
        }
        self._coil_panel.restore_session_state(coil_dict)

        # Perforator risk thresholds
        self._perforator_panel.restore_session_state({
            "r_high": data.perf_r_high,
            "r_med":  data.perf_r_med,
            "r_low":  data.perf_r_low,
        })

        # Stents — the 3D planning window may not exist yet when loading a session.
        # We store them as pending; _open_planning_window() applies them on first open.
        self._pending_session_stents = [
            {
                "index":       s.index,
                "name":        s.name,
                "is_custom":   s.is_custom,
                "custom_path": s.custom_path,
                "position":    s.position,
                "orientation": s.orientation,
            }
            for s in data.stents
        ]
        # If the window is already open, apply immediately
        if self._planning_window is not None and self._pending_session_stents:
            self._planning_window._stent_panel.restore_session_state(
                {"stents": self._pending_session_stents}
            )
            self._pending_session_stents = []

        # Morphometrics labels (populated even before mesh reload so the user
        # can review previous results immediately after loading the session)
        if data.morpho_snapshot:
            self._morpho_panel.restore_session_state(
                data.morpho_snapshot,
                data.clinical_state,
            )

        # Treatment decision: restore location + ruptured inputs only
        # (_recompute fires automatically when set_morpho_result() is called later)
        self._treatment_panel.restore_session_state(
            data.clinical_state.get("treatment", {})
        )

        # 3D-print preparation parameters
        self._print_panel.restore_session_state(data.print_prep_state)

        # Report / patient data
        self._report_panel.restore_session_state({
            "patient_name": data.report_patient_name,
            "patient_id":   data.report_patient_id,
            "patient_dob":  data.report_patient_dob,
            "surgeon":      data.report_surgeon,
            "institution":  data.report_institution,
            "notes":        data.report_notes,
            "treatment":    data.report_treatment,
        })

        # ── Restore study link early so downstream code has the ID ────── #
        if data.study_id:
            self._current_study_id = data.study_id
            logger.debug("Session restored with study_id=%d", data.study_id)

        # ── Restore segmented vessel mesh BEFORE step restore ──────────── #
        # _show_planning_tab() (triggered by restore_to()) reads the mesh from
        # the VTK mapper.  The mesh must be in the mapper first so the planning
        # window is populated immediately on creation.
        _restored_poly = None
        if data.seg_mesh_path:
            try:
                poly = MeshExporter.load_vtp(data.seg_mesh_path)
                if poly is not None:
                    _restored_poly = poly
                    self._restore_vessel_mesh(poly)   # puts poly into the mapper
            except Exception as _mesh_err:
                logger.warning("Mesh restore failed: %s", _mesh_err)

        if _restored_poly is None and data.current_step >= 4:
            # Session claims to have reached planning/report but no mesh companion
            # file was found — this happens with legacy autosaves or missing files.
            self._statusbar.showMessage(
                "⚠  Malla vascular no encontrada — ejecute la segmentación de nuevo.",
                6000,
            )

        # ── Restore the workflow step (AFTER mesh is in the mapper) ───── #
        # restore_to() unlocks all preceding steps and marks them DONE before
        # navigating, because go_to() silently fails on LOCKED steps.
        if hasattr(self, "_stepper") and 0 <= data.current_step < 6:
            self._stepper.restore_to(data.current_step)

        # ── Reload the associated DICOM ────────────────────────────────── #
        # Always set pending state so _on_loaded() can re-apply it regardless
        # of whether the reload is automatic or dialog-driven.
        def _do_dicom_reload() -> None:
            self._pending_seg_state = dict(data.seg_state)
            if data.current_step > 1:
                self._pending_restore_step = data.current_step
            if _restored_poly is not None:
                self._pending_restore_mesh = _restored_poly
            self._start_load(data.dicom_path)

        if data.dicom_path and Path(data.dicom_path).exists():
            if auto_load_dicom:
                # Seamless resume from Welcome-screen banner / CaseDashboard:
                # start loading immediately without interrupting the user.
                _do_dicom_reload()
            else:
                reply = QMessageBox.question(
                    self, "Recargar DICOM",
                    f"La sesión hace referencia a:\n{data.dicom_path}\n\n"
                    "¿Desea cargar la serie ahora?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    _do_dicom_reload()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Acerca de PROSPECTIVE",
            "<b>PROSPECTIVE v0.1.0 — F-01</b><br>"
            "Preoperative Planning Platform for Cerebral Aneurysm Clipping<br><br>"
            "<b>Fundación Universitaria Navarra UNINAVARRA</b> · 2026<br>"
            "Python 3.11 · PyQt5 · VTK · PyQtGraph · AWS<br><br>"
            "Documento Confidencial",
        )

    def _show_audit_dialog(self) -> None:
        """Open the SkullChain cryptographic audit trail viewer."""
        from prospective.ui.widgets.audit_dialog import AuditDialog
        dlg = AuditDialog(self)
        dlg.exec()

    # ------------------------------------------------------------------ #
    # Theme management — 2 themes: dark / light (both with glass effects) #
    # ------------------------------------------------------------------ #

    def _apply_theme(self) -> None:
        """Re-apply the active theme and refresh all inline-styled widgets."""
        from prospective.ui.themes import apply_theme, current_theme, is_dark
        apply_theme(current_theme())
        # Sync toolbar / menu labels with the current theme
        if is_dark():
            _lbl = f"Cambiar a tema claro  {_I.THEME_LIGHT}"
            _tb  = f"{_I.THEME_LIGHT} Tema claro"
        else:
            _lbl = f"Cambiar a tema oscuro  {_I.THEME_DARK}"
            _tb  = f"{_I.THEME_DARK} Tema oscuro"
        self._theme_act.setText(_lbl)
        self._theme_tb_act.setText(_tb)
        # Refresh widgets that own inline setStyleSheet calls
        if hasattr(self, "_stepper"):
            self._stepper.apply_theme()
        if hasattr(self, "_aneurysm_panel"):
            self._aneurysm_panel.apply_theme()
        if hasattr(self, "_mpr"):
            self._mpr.apply_theme()
        if self._planning_window is not None:
            self._planning_window.apply_theme()
        # Refresh toolbar logo tint and native title-bar dark mode
        if hasattr(self, "_logo_lbl"):
            pix = _tinted_logo_pix(34)
            if not pix.isNull():
                self._logo_lbl.setPixmap(pix)
        # Update the native Win32 title bar colour (dark / light)
        self._apply_dwm_dark_mode()
        # The .ico icon is already multi-resolution; no tint update needed.
        # Refresh Level-3 layout widgets
        if hasattr(self, "_activity_bar"):
            self._activity_bar.apply_theme()
        if hasattr(self, "_patient_strip"):
            self._patient_strip.apply_theme()
        # Refresh floating dialogs
        if self._triplanar_dlg is not None:
            self._triplanar_dlg.apply_theme()
        # Refresh device panels
        if hasattr(self, "_clip_panel"):
            self._clip_panel.apply_theme()
        if hasattr(self, "_coil_panel"):
            self._coil_panel.apply_theme()

    def _toggle_theme(self) -> None:
        """Cycle dark ↔ light and refresh all inline-styled widgets."""
        from prospective.ui.themes import toggle_theme, is_dark
        toggle_theme()
        if is_dark():
            _lbl = f"Cambiar a tema claro  {_I.THEME_LIGHT}"
            _tb  = f"{_I.THEME_LIGHT} Tema claro"
        else:
            _lbl = f"Cambiar a tema oscuro  {_I.THEME_DARK}"
            _tb  = f"{_I.THEME_DARK} Tema oscuro"
        self._theme_act.setText(_lbl)
        self._theme_tb_act.setText(_tb)
        if hasattr(self, "_stepper"):
            self._stepper.apply_theme()
        if hasattr(self, "_aneurysm_panel"):
            self._aneurysm_panel.apply_theme()
        if hasattr(self, "_mpr"):
            self._mpr.apply_theme()
        if self._planning_window is not None:
            self._planning_window.apply_theme()
        # Refresh toolbar logo tint and native title-bar dark mode
        if hasattr(self, "_logo_lbl"):
            pix = _tinted_logo_pix(34)
            if not pix.isNull():
                self._logo_lbl.setPixmap(pix)
        # Update the native Win32 title bar colour (dark / light)
        self._apply_dwm_dark_mode()
        # The .ico icon is already multi-resolution; no tint update needed.
        # Refresh Level-3 layout widgets
        if hasattr(self, "_activity_bar"):
            self._activity_bar.apply_theme()
        if hasattr(self, "_patient_strip"):
            self._patient_strip.apply_theme()
        # Refresh floating dialogs
        if self._triplanar_dlg is not None:
            self._triplanar_dlg.apply_theme()
        # Refresh device panels
        if hasattr(self, "_clip_panel"):
            self._clip_panel.apply_theme()
        if hasattr(self, "_coil_panel"):
            self._coil_panel.apply_theme()
        # Refresh inline-styled muted labels that depend on the theme colour
        _mc = "#9B9B9B" if is_dark() else "#6B6B6B"
        if hasattr(self, "_lbl_load_status"):
            self._lbl_load_status.setStyleSheet(f"color: {_mc}; font-size: 10px;")
        if hasattr(self, "_user_lbl"):
            self._user_lbl.setStyleSheet(f"color:{_mc}; font-size:10px; padding-right:8px;")

    # ------------------------------------------------------------------ #
    # Window title                                                         #
    # ------------------------------------------------------------------ #

    def _update_window_title(self) -> None:
        """Reflect patient name, case ID and current step in the title bar."""
        _STEP_NAMES = [
            "Paciente", "Segmentación", "Detección",
            "Morfometría", "Planificación", "Exportar",
        ]
        parts = ["PROSPECTIVE"]
        if self._series is not None:
            name = self._series.metadata.patient_name
            if name:
                parts.append(name)
        if self._current_study_id:
            parts.append(f"Caso #{self._current_study_id}")
        if hasattr(self, "_stepper"):
            idx = self._stepper.current_index()
            if 0 <= idx < len(_STEP_NAMES):
                parts.append(f"Paso {idx + 1}: {_STEP_NAMES[idx]}")
        self.setWindowTitle("  —  ".join(parts))

    # ------------------------------------------------------------------ #
    # Close case                                                           #
    # ------------------------------------------------------------------ #

    def _close_case(self) -> None:
        """Close this case and return to the cases panel.

        If there is a loaded DICOM series or an active planning session, asks
        the user for confirmation before closing.  The WelcomeWindow / Dashboard
        are brought back via the ``destroyed`` signal (already wired in
        WelcomeWindow._open_main_for_case).
        """
        from PyQt5.QtWidgets import QMessageBox

        # Build a meaningful prompt only when there is something to lose
        has_data = (
            self._series is not None
            or (self._planning_window is not None)
            or self._current_session_id != 0
        )

        if has_data:
            from PyQt5.QtWidgets import (
                QDialog, QDialogButtonBox, QVBoxLayout as _VBL, QLabel as _Lbl,
            )
            dlg = QDialog(self)
            dlg.setWindowTitle("Cerrar caso")
            dlg.setFixedWidth(400)
            _lay = _VBL(dlg)
            _lay.setSpacing(16)
            _lay.setContentsMargins(20, 18, 20, 14)
            _lbl = _Lbl("¿Qué deseas hacer antes de volver al panel de casos?", dlg)
            _lbl.setWordWrap(True)
            _lay.addWidget(_lbl)
            _btns = QDialogButtonBox(Qt.Horizontal, dlg)
            _btn_save    = _btns.addButton(f"{_I.SAVE}  Guardar y volver",   QDialogButtonBox.AcceptRole)
            _btn_nosave  = _btns.addButton("→  Volver sin guardar",  QDialogButtonBox.DestructiveRole)
            _btn_cancel  = _btns.addButton("Cancelar",               QDialogButtonBox.RejectRole)
            _lay.addWidget(_btns)
            _chosen: list[str] = ["cancel"]

            def _on_clicked(b):
                if b is _btn_save:
                    _chosen[0] = "save"
                elif b is _btn_nosave:
                    _chosen[0] = "nosave"
                else:
                    _chosen[0] = "cancel"
                dlg.accept()

            _btns.clicked.connect(_on_clicked)
            dlg.exec_()

            if _chosen[0] == "cancel":
                return
            if _chosen[0] == "save":
                # _save_session() returns True only if the file was actually written.
                # If the user cancels the file dialog, abort the close action.
                if not self._save_session():
                    return

        # Signal BEFORE close so WelcomeWindow can distinguish navigation
        # from a plain window close and decide whether to show the dashboard.
        self.case_closed.emit()
        self.close()
