"""Workflow stepper — Option-D UX redesign.

Guides the user through the 6-step surgical planning workflow with a
visual step indicator, contextual help text, and sequential navigation.

          ┌─────────────────────────────────────────────────────┐
          │ ①✔ ─── ②✔ ─── ③● ─── ④○ ─── ⑤○ ─── ⑥○           │
          │  Paciente  Seg.  Detect. Morfo.  Plan.  Export      │
          ├─────────────────────────────────────────────────────┤
          │  ℹ  Detección de aneurismas                         │
          │     Selecciona un candidato para ver su geometría   │
          │     en la vista 3D y verificar morfometría.         │
          ├─────────────────────────────────────────────────────┤
          │  [ contenido del panel activo ]                     │
          ├─────────────────────────────────────────────────────┤
          │  [← Anterior]               [Siguiente paso →]      │
          └─────────────────────────────────────────────────────┘

Usage
-----
    steps = [
        WorkflowStepDef("paciente", 1, "📂", "Paciente", "Carga el DICOM...", panel),
        ...
    ]
    stepper = WorkflowStepper(steps)
    stepper.step_changed.connect(on_step_changed)
    # From app logic:
    stepper.mark_done(0)      # step 0 completed
    stepper.unlock(1)         # step 1 now available
    stepper.go_to(1)          # navigate programmatically
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import auto, Enum
from typing import Sequence

from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush, QFontMetrics
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QSpacerItem,
    QStackedWidget, QVBoxLayout, QWidget,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Data types                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

class StepStatus(Enum):
    LOCKED    = auto()   # not yet reachable
    AVAILABLE = auto()   # can jump to
    ACTIVE    = auto()   # currently displayed
    DONE      = auto()   # completed


@dataclass
class WorkflowStepDef:
    """Definition for one step of the workflow.

    Parameters
    ----------
    key         : unique identifier (used programmatically)
    number      : 1-based display number
    icon        : emoji / unicode icon shown inside the bubble
    title       : short name shown below bubble
    description : longer text shown when step is active
    panel       : QWidget shown in the content area when active
    """
    key:         str
    number:      int
    icon:        str
    title:       str
    description: str
    panel:       QWidget


# ──────────────────────────────────────────────────────────────────────────── #
# Step header bar (custom-painted)                                              #
# ──────────────────────────────────────────────────────────────────────────── #

# ── Theme-aware colour helper ─────────────────────────────────────────── #

def _is_dark() -> bool:
    """Return True when the active UI theme has a dark background."""
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True   # safe fallback


# Colours — static (theme-independent)
_C_ACTIVE    = QColor("#8B9BAA")   # brand violet — same for both themes
_C_DONE      = QColor("#1a7f37")   # green       — same for both themes
_C_TEXT_LIGHT = QColor("#f0f6fc")  # white text on coloured bubbles (both)

_SIDEBAR_W   = 52    # right sidebar width (px)
_BUBBLE_R    = 14    # bubble radius px
_BUBBLE_D    = _BUBBLE_R * 2
_SLOT_W_MIN  = 72    # minimum horizontal slot per step (px)
_SLOT_W_PREF = 86    # preferred slot width


def _sidebar_css() -> str:
    """Right icon-navigation sidebar stylesheet."""
    if _is_dark():
        return (
            "QWidget#stepSidebar { background: #1F1F1F; border-left: 1px solid #363636; }"
            "QPushButton { background: transparent; border: none; border-radius: 10px;"
            " color: #9B9B9B; font-size: 18px; padding: 4px; }"
            "QPushButton:hover { background: #363636; color: #EBEBEB; }"
            "QPushButton:checked { background: #182434; color: #8B9BAA; }"
            "QPushButton:disabled { color: #4a4a4a; }"
        )
    return (
        "QWidget#stepSidebar { background: #F7F7F7; border-left: 1px solid #E5E5E5; }"
        "QPushButton { background: transparent; border: none; border-radius: 10px;"
        " color: #6B6B6B; font-size: 18px; padding: 4px; }"
        "QPushButton:hover { background: #EBF0F5; color: #0D0D0D; }"
        "QPushButton:checked { background: #DDE5EC; color: #4E6678; }"
        "QPushButton:disabled { color: #CCCCCC; }"
    )


class _StepHeaderBar(QWidget):
    """Horizontally arranged step bubbles connected by lines."""

    clicked = pyqtSignal(int)   # emits step index

    def __init__(self, steps: Sequence[WorkflowStepDef], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._steps  = list(steps)
        self._status = [StepStatus.LOCKED] * len(steps)
        if steps:
            self._status[0] = StepStatus.AVAILABLE
        self.setMinimumHeight(60)
        self.setCursor(Qt.PointingHandCursor)
        self._dark = _is_dark()   # cached; refreshed by apply_theme()

    def apply_theme(self) -> None:
        """Refresh colours when the UI theme changes."""
        self._dark = _is_dark()
        self.update()

    def _theme_colors(self) -> dict:
        """Return a dict of QColor/str values for the current theme cache."""
        d = self._dark
        return {
            "available_fill":  QColor("#2A2A2A") if d else QColor("#F7F7F7"),
            "locked_fill":     QColor("#1F1F1F") if d else QColor("#EEEEEE"),
            "outline_avail":   QColor("#8B9BAA") if d else QColor("#8B9BAA"),
            "outline_lock":    QColor("#363636") if d else QColor("#E5E5E5"),
            "text_grey":       QColor("#9B9B9B") if d else QColor("#6B6B6B"),
            "line":            QColor("#363636") if d else QColor("#E5E5E5"),
        }

    # Qt size hints so the parent scroll area knows when to add scrollbars
    def sizeHint(self) -> QSize:
        return QSize(len(self._steps) * _SLOT_W_PREF, 62)

    def minimumSizeHint(self) -> QSize:
        return QSize(len(self._steps) * _SLOT_W_MIN, 62)

    # ------------------------------------------------------------------ #
    def set_status(self, idx: int, status: StepStatus) -> None:
        self._status[idx] = status
        self.update()

    def get_status(self, idx: int) -> StepStatus:
        return self._status[idx]

    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802
        n = len(self._steps)
        if n == 0:
            return

        # Theme-aware colours (re-evaluated on each paint so runtime switching works)
        tc = self._theme_colors()
        C_AVAILABLE   = tc["available_fill"]
        C_LOCKED      = tc["locked_fill"]
        C_OUTLINE_AVAIL = tc["outline_avail"]
        C_OUTLINE_LOCK  = tc["outline_lock"]
        C_TEXT_GREY   = tc["text_grey"]
        C_LINE        = tc["line"]

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w  = self.width()
        h  = self.height()
        y_bubble = h // 2 - 8       # bubble centre y
        y_label  = y_bubble + _BUBBLE_R + 4

        # Space each bubble evenly
        xs = [int(w * (i + 0.5) / n) for i in range(n)]

        # ── Connector lines ────────────────────────────────────────────── #
        for i in range(n - 1):
            x1 = xs[i]  + _BUBBLE_R
            x2 = xs[i+1] - _BUBBLE_R
            st_left  = self._status[i]
            st_right = self._status[i+1]
            both_done = (st_left  == StepStatus.DONE and
                         st_right in (StepStatus.DONE, StepStatus.ACTIVE))
            pen = QPen(_C_DONE if both_done else C_LINE, 2)
            painter.setPen(pen)
            painter.drawLine(x1, y_bubble, x2, y_bubble)

        # ── Bubbles ─────────────────────────────────────────────────────── #
        # Use Segoe UI Symbol (monochrome) before the emoji font so that the
        # BMP icon characters from icons.py render in the bubble's fg colour.
        font_icon = QFont()
        font_icon.setFamilies(["Segoe UI Symbol", "Segoe UI", "Arial Unicode MS"])
        font_icon.setPointSize(9)
        font_label = QFont("Inter", 7)
        font_label.setBold(False)
        fm_label   = QFontMetrics(font_label)

        # Half-width of the label slot: never more than half the inter-bubble gap,
        # but at least the bubble radius.
        slot_half = max(_BUBBLE_R, w // (2 * n) - 2) if n > 0 else _BUBBLE_R

        for i, step in enumerate(self._steps):
            cx = xs[i]
            status = self._status[i]

            if status == StepStatus.DONE:
                fill    = _C_DONE
                outline = _C_DONE
                fg      = _C_TEXT_LIGHT
                symbol  = "✓"
            elif status == StepStatus.ACTIVE:
                fill    = _C_ACTIVE
                outline = _C_ACTIVE
                fg      = _C_TEXT_LIGHT
                symbol  = str(step.number)
            elif status == StepStatus.AVAILABLE:
                fill    = C_AVAILABLE
                outline = C_OUTLINE_AVAIL
                fg      = C_OUTLINE_AVAIL
                symbol  = str(step.number)
            else:  # LOCKED
                fill    = C_LOCKED
                outline = C_OUTLINE_LOCK
                fg      = C_TEXT_GREY
                symbol  = str(step.number)

            # Circle
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(outline, 2))
            painter.drawEllipse(cx - _BUBBLE_R, y_bubble - _BUBBLE_R,
                                _BUBBLE_D, _BUBBLE_D)

            # Symbol inside circle
            painter.setFont(font_icon)
            painter.setPen(QPen(fg))
            rect = self.rect()
            rect.setLeft(cx - _BUBBLE_R)
            rect.setRight(cx + _BUBBLE_R)
            rect.setTop(y_bubble - _BUBBLE_R)
            rect.setBottom(y_bubble + _BUBBLE_R)
            painter.drawText(rect, Qt.AlignCenter, symbol)

            # Label below circle — elide if the slot is narrower than the text
            label_color = _C_TEXT_LIGHT if status == StepStatus.ACTIVE else C_TEXT_GREY
            painter.setFont(font_label)
            painter.setPen(QPen(label_color))
            label_w = slot_half * 2
            elided  = fm_label.elidedText(step.title, Qt.ElideRight, label_w)
            lr = self.rect()
            lr.setLeft(cx - slot_half)
            lr.setRight(cx + slot_half)
            lr.setTop(y_label)
            lr.setBottom(h)
            painter.drawText(lr, Qt.AlignHCenter | Qt.AlignTop, elided)

        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        n = len(self._steps)
        if n == 0:
            return
        w  = self.width()
        x  = event.x()
        xs = [int(w * (i + 0.5) / n) for i in range(n)]
        for i, cx in enumerate(xs):
            if abs(x - cx) <= _BUBBLE_R + 6:
                if self._status[i] != StepStatus.LOCKED:
                    self.clicked.emit(i)
                return


# ──────────────────────────────────────────────────────────────────────────── #
# Help banner                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

class _HelpBanner(QWidget):
    """Collapsible contextual help text shown under the step indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._expanded = True
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        # Header row: icon + title + toggle
        row = QHBoxLayout()
        self._ico  = QLabel("ℹ")
        self._title = QLabel()
        self._btn_toggle = QPushButton("▲")
        self._btn_toggle.setFlat(True)
        self._btn_toggle.setFixedSize(20, 20)
        self._btn_toggle.clicked.connect(self._toggle)
        row.addWidget(self._ico)
        row.addWidget(self._title, 1)
        row.addWidget(self._btn_toggle)
        layout.addLayout(row)

        # Description text
        self._desc = QLabel()
        self._desc.setWordWrap(True)
        layout.addWidget(self._desc)

        # Separator line
        self._sep_line = QFrame()
        self._sep_line.setFrameShape(QFrame.HLine)
        layout.addWidget(self._sep_line)

        self.apply_theme()

    def apply_theme(self) -> None:
        """Re-apply theme-dependent colours.  Call after toggling the UI theme."""
        dark = _is_dark()
        bg        = "#1F1F1F" if dark else "#F5F5F5"
        title_c   = "#EBEBEB" if dark else "#0D0D0D"
        muted     = "#9B9B9B" if dark else "#6B6B6B"
        sep_c     = "#363636" if dark else "#E5E5E5"
        ico_c     = "#8B9BAA" if dark else "#8B9BAA"

        self._ico.setStyleSheet(f"color: {ico_c}; font-size: 13px;")
        self._title.setStyleSheet(
            f"color: {title_c}; font-weight: bold; font-size: 11px;"
        )
        self._btn_toggle.setStyleSheet(f"color: {muted}; font-size: 9px;")
        self._desc.setStyleSheet(
            f"color: {muted}; font-size: 10px; padding: 2px 4px;"
        )
        self._sep_line.setStyleSheet(f"color: {sep_c};")
        self.setStyleSheet(
            f"_HelpBanner {{ background: {bg}; border-radius: 14px; }}"
            f"QWidget {{ background: {bg}; }}"
        )

    def set_content(self, title: str, description: str) -> None:
        self._title.setText(title)
        self._desc.setText(description)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._desc.setVisible(self._expanded)
        self._btn_toggle.setText("▲" if self._expanded else "▼")


# ──────────────────────────────────────────────────────────────────────────── #
# Main stepper widget                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

class WorkflowStepper(QWidget):
    """
    Container that wraps a list of :class:`WorkflowStepDef` panels inside a
    guided step-by-step UI.

    Signals
    -------
    step_changed(int)  — emitted when the active step index changes
    """

    step_changed = pyqtSignal(int)

    def __init__(
        self,
        steps: Sequence[WorkflowStepDef],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._steps      = list(steps)
        self._current    = 0
        self._done_steps: set[int] = set()   # tracks which steps have been marked done
        self._build_ui()
        if self._steps:
            self._header.set_status(0, StepStatus.ACTIVE)
            self._go_to_internal(0)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def current_index(self) -> int:
        return self._current

    def step_key(self, idx: int) -> str:
        return self._steps[idx].key

    def mark_done(self, idx: int) -> None:
        """Mark step *idx* as completed (green checkmark)."""
        self._done_steps.add(idx)
        self._header.set_status(idx, StepStatus.DONE)
        self._sync_sidebar()

    def unlock(self, idx: int) -> None:
        """Make step *idx* navigable without making it active."""
        if self._header.get_status(idx) == StepStatus.LOCKED:
            self._header.set_status(idx, StepStatus.AVAILABLE)
            # Refresh nav buttons immediately if this step is adjacent to current
            if idx == self._current + 1:
                self._refresh_nav()
            self._sync_sidebar()

    def mark_done_and_advance(self, idx: int) -> None:
        """Mark step *idx* done and move to *idx+1* if possible."""
        self.mark_done(idx)
        next_idx = idx + 1
        if next_idx < len(self._steps):
            self.unlock(next_idx)
            self.go_to(next_idx)

    def go_to(self, idx: int) -> None:
        """Navigate to step *idx* (must not be LOCKED)."""
        if idx < 0 or idx >= len(self._steps):
            return
        status = self._header.get_status(idx)
        if status == StepStatus.LOCKED:
            return
        self._go_to_internal(idx)

    def restore_to(self, idx: int) -> None:
        """Restore workflow state after loading a saved session.

        Unlike ``go_to``, this method does NOT require prior unlocking.
        It marks every step before *idx* as DONE, unlocks all steps up to
        and including *idx* (plus *idx+1* so the user can advance), and then
        navigates to *idx*.

        This is the correct call site for session restore — ``go_to`` alone
        silently fails when the target step is still LOCKED.
        """
        if idx < 0 or idx >= len(self._steps):
            idx = 0

        # Mark all preceding steps as done and unlock them
        for i in range(idx):
            self._done_steps.add(i)
            self._header.set_status(i, StepStatus.DONE)

        # Unlock the target step so go_to() will accept it
        st = self._header.get_status(idx)
        if st == StepStatus.LOCKED:
            self._header.set_status(idx, StepStatus.AVAILABLE)

        # Also unlock the next step (if any) so the user can advance
        next_idx = idx + 1
        if next_idx < len(self._steps):
            if self._header.get_status(next_idx) == StepStatus.LOCKED:
                self._header.set_status(next_idx, StepStatus.AVAILABLE)

        self._sync_sidebar()
        self._go_to_internal(idx)

    def panel(self, idx: int) -> QWidget:
        return self._steps[idx].panel

    # ------------------------------------------------------------------ #
    # Internal navigation                                                  #
    # ------------------------------------------------------------------ #

    def _go_to_internal(self, idx: int) -> None:
        prev = self._current
        self._current = idx

        # Update header statuses
        for i, step in enumerate(self._steps):
            st = self._header.get_status(i)
            if i == idx:
                self._header.set_status(i, StepStatus.ACTIVE)
            elif st == StepStatus.ACTIVE:
                # Restore to DONE if previously completed, else back to AVAILABLE
                self._header.set_status(
                    i,
                    StepStatus.DONE if i in self._done_steps else StepStatus.AVAILABLE,
                )

        # Update help banner
        step_def = self._steps[idx]
        self._help.set_content(
            f"Paso {step_def.number} — {step_def.icon} {step_def.title}",
            step_def.description,
        )

        # Show correct panel
        self._stack.setCurrentIndex(idx)

        self._refresh_nav()
        self._sync_sidebar()

        if idx != prev:
            self.step_changed.emit(idx)

    def _refresh_nav(self) -> None:
        """Update Prev/Next button enabled state and Next label text."""
        idx = self._current
        self._btn_prev.setEnabled(
            idx > 0 and self._header.get_status(idx - 1) != StepStatus.LOCKED
        )
        next_idx = idx + 1
        has_next = (
            next_idx < len(self._steps)
            and self._header.get_status(next_idx) != StepStatus.LOCKED
        )
        self._btn_next.setEnabled(has_next)
        next_title = self._steps[next_idx].title if has_next else ""
        self._btn_next.setText(
            f"Siguiente: {next_title} →" if has_next else "Siguiente →"
        )

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # ── Outer horizontal layout: [content column | right sidebar] ─── #
        self._sidebar_btns: list[QPushButton] = []   # initialise before any _sync_sidebar call

        outer_h = QHBoxLayout(self)
        outer_h.setContentsMargins(0, 0, 0, 0)
        outer_h.setSpacing(0)

        # ── Left content column ───────────────────────────────────────── #
        content_col = QWidget()
        outer = QVBoxLayout(content_col)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Step header bar ───────────────────────────────────────────── #
        self._header = _StepHeaderBar(self._steps)
        self._header.setFixedHeight(62)
        self._header.clicked.connect(self.go_to)
        outer.addWidget(self._header)

        # ── Help banner ───────────────────────────────────────────────── #
        self._help = _HelpBanner()
        outer.addWidget(self._help)

        # ── Content stack (scrollable) ────────────────────────────────── #
        # Vertical scroll: always enabled (panels can be taller than the dock).
        # Horizontal scroll: shown only when a panel's minimum width exceeds
        # the current dock width — prevents silent content clipping.
        self._stack = QStackedWidget()
        for step in self._steps:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setWidget(step.panel)
            self._stack.addWidget(scroll)
        outer.addWidget(self._stack, 1)

        # ── Navigation bar ─────────────────────────────────────────────── #
        self._nav_bar = QFrame()
        self._nav_bar.setFixedHeight(40)
        nav_layout = QHBoxLayout(self._nav_bar)
        nav_layout.setContentsMargins(8, 4, 8, 4)

        self._btn_prev = QPushButton("← Anterior")
        self._btn_prev.setEnabled(False)
        self._btn_prev.setFlat(True)
        self._btn_prev.clicked.connect(lambda: self.go_to(self._current - 1))

        self._btn_next = QPushButton("Siguiente →")
        self._btn_next.setEnabled(False)
        self._btn_next.setFlat(True)
        self._btn_next.clicked.connect(lambda: self.go_to(self._current + 1))

        nav_layout.addWidget(self._btn_prev)
        nav_layout.addStretch()
        nav_layout.addWidget(self._btn_next)
        outer.addWidget(self._nav_bar)

        # ── Assemble outer HBox ───────────────────────────────────────── #
        outer_h.addWidget(content_col, 1)
        outer_h.addWidget(self._build_sidebar())

        self.apply_theme()

    # ------------------------------------------------------------------ #
    # Right sidebar                                                        #
    # ------------------------------------------------------------------ #

    def _build_sidebar(self) -> QWidget:
        """52 px icon-only right sidebar — one button per workflow step."""
        sidebar = QWidget()
        sidebar.setFixedWidth(_SIDEBAR_W)
        sidebar.setObjectName("stepSidebar")
        self._sidebar = sidebar

        vl = QVBoxLayout(sidebar)
        vl.setContentsMargins(4, 8, 4, 8)
        vl.setSpacing(2)

        for i, step in enumerate(self._steps):
            btn = QPushButton(step.icon)
            btn.setCheckable(True)
            btn.setFixedSize(44, 44)
            btn.setToolTip(f"Paso {step.number}: {step.title}")
            btn.setEnabled(False)   # all locked until unlock() is called
            # Use clicked (not toggled) so navigation is triggered on press,
            # and _sync_sidebar() controls the checked state authoritatively.
            btn.clicked.connect(lambda _=False, idx=i: self.go_to(idx))
            vl.addWidget(btn)
            self._sidebar_btns.append(btn)

        vl.addStretch()
        sidebar.setStyleSheet(_sidebar_css())
        return sidebar

    def _sync_sidebar(self) -> None:
        """Mirror step statuses into the right sidebar button states."""
        _status_suffix = {
            StepStatus.LOCKED:    " — bloqueado",
            StepStatus.AVAILABLE: "",
            StepStatus.ACTIVE:    " — activo",
            StepStatus.DONE:      " ✓ completado",
        }
        for i, btn in enumerate(self._sidebar_btns):
            step   = self._steps[i]
            status = self._header.get_status(i)
            btn.blockSignals(True)
            btn.setEnabled(status != StepStatus.LOCKED)
            btn.setChecked(status == StepStatus.ACTIVE)
            btn.setToolTip(
                f"Paso {step.number}: {step.title}"
                f"{_status_suffix.get(status, '')}"
            )
            btn.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Theme support                                                        #
    # ------------------------------------------------------------------ #

    def apply_theme(self) -> None:
        """
        Re-apply all inline stylesheets to match the current UI theme.
        Call this whenever the application theme is toggled.
        """
        dark      = _is_dark()
        nav_bg    = "#2A2A2A" if dark else "#FFFFFF"
        nav_bdr   = "#363636" if dark else "#E5E5E5"
        btn_color = "#8B9BAA" if dark else "#8B9BAA"

        self._nav_bar.setStyleSheet(
            f"background: {nav_bg}; border-top: 1px solid {nav_bdr};"
        )
        self._btn_prev.setStyleSheet(
            f"color: {btn_color}; font-size: 10px; padding: 2px 8px;"
        )
        self._btn_next.setStyleSheet(
            f"color: {btn_color}; font-size: 10px; padding: 2px 8px;"
        )
        self._help.apply_theme()
        self._header.apply_theme()
        # Sidebar — only exists after _build_sidebar() has run
        if hasattr(self, "_sidebar"):
            self._sidebar.setStyleSheet(_sidebar_css())
