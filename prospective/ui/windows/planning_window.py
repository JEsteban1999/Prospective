"""Dedicated device-planning window — mesh-only 3D view + Clips + Stents.

Trajectory picking
------------------
The user clicks points on the artery surface to define a spline path.
A preview tube of the selected stent's **nominal diameter** is rendered
along the trajectory and colour-coded for viability:

  Green  (clearance ≥ 1 mm)   — stent fits comfortably
  Orange (0 ≤ clearance < 1 mm) — tight but feasible
  Red    (clearance < 0)        — stent wider than vessel; does not fit

Camera rotation is unaffected when pick-mode is OFF.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import vtk
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from prospective.models.clip_library import CLIP_CATALOGUE
from prospective.models.stent_library import STENT_CATALOGUE
from prospective.ui.widgets.centerline_panel import CenterlinePanel
from prospective.ui.widgets.cl_stent_panel import CLStentPanel
from prospective.ui.widgets.clip_panel import ClipPanel
from prospective.ui.widgets.measurement_panel import MeasurementPanel
from prospective.ui.widgets.stent_panel import StentPanel

logger = logging.getLogger(__name__)

_RES       = Path(__file__).resolve().parents[3] / "resources"
_LOGO_PATH = str(_RES / "logo.png")

_SPLINE_SAMPLES  = 500   # arc-length table resolution
_VIAB_SAMPLES    = 120   # points sampled for clearance check


# ──────────────────────────────────────────────────────────────────────────── #
# Theme-aware CSS helpers                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


def _tinted_logo_pix(size: int) -> QPixmap:
    """Return the SkullApp logo tinted for the current theme at *size* px."""
    color = "#ffffff" if _is_dark() else "#4E6678"
    pix = QPixmap(_LOGO_PATH)
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


def _plan_hdr_css() -> str:
    if _is_dark():
        return ("background:#2A2A2A; color:#A8B8C6; font-weight:700;"
                "font-size:11px; padding:6px; letter-spacing:1px;"
                "border-bottom:1px solid #363636;")
    return ("background:#F7F7F7; color:#4E6678; font-weight:700;"
            "font-size:11px; padding:6px; letter-spacing:1px;"
            "border-bottom:1px solid #E5E5E5;")


def _plan_vtk_hdr_css() -> str:
    if _is_dark():
        return ("background:#2A2A2A; color:#A8B8C6; font-weight:700;"
                "font-size:13px; padding:4px; letter-spacing:1px;")
    return ("background:#E8EFF7; color:#4E6678; font-weight:700;"
            "font-size:13px; padding:4px; letter-spacing:1px;")


def _plan_help_css() -> str:
    if _is_dark():
        return ("background:#1F1F1F; color:#9B9B9B; font-size:10px;"
                "padding:5px 8px; border-bottom:1px solid #363636;")
    return ("background:#FFFFFF; color:#6B6B6B; font-size:10px;"
            "padding:5px 8px; border-bottom:1px solid #E5E5E5;")


def _plan_panel_stack_css() -> str:
    if _is_dark():
        return "QStackedWidget { background: #1F1F1F; border-left: 1px solid #363636; }"
    return "QStackedWidget { background: #FFFFFF; border-left: 1px solid #E5E5E5; }"


def _plan_sidebar_css() -> str:
    # font-family: Segoe UI Symbol must come before any emoji font so that BMP
    # characters (and Variation-Selector-15 suffixed glyphs) render monochrome.
    from prospective.ui.icons import FONT as _ICO_FONT
    _ff = f"font-family: {_ICO_FONT};"
    if _is_dark():
        return (f"QWidget#planSidebar {{ background: #1F1F1F; border-right: 1px solid #363636; }}"
                f"QPushButton {{ background: transparent; border: none; border-radius: 10px;"
                f" color: #9B9B9B; font-size: 18px; padding: 4px; {_ff} }}"
                f"QPushButton:hover {{ background: #363636; color: #EBEBEB; }}"
                f"QPushButton:checked {{ background: #1C303F; color: #A8B8C6; }}")
    return (f"QWidget#planSidebar {{ background: #F7F7F7; border-right: 1px solid #E5E5E5; }}"
            f"QPushButton {{ background: transparent; border: none; border-radius: 10px;"
            f" color: #5A5A5A; font-size: 18px; padding: 4px; {_ff} }}"
            f"QPushButton:hover {{ background: #EBF0F5; color: #1A1A1A; }}"
            f"QPushButton:checked {{ background: #DDE5EC; color: #4E6678; }}")


def _plan_muted_btn_css() -> str:
    if _is_dark():
        return ("QPushButton { background: #2A2A2A; border: 1px solid #363636;"
                " border-radius: 14px; color: #EBEBEB; font-size: 10px;"
                " padding: 4px 8px; text-align: left; }"
                "QPushButton:hover { background: #363636; border-color: #A8B8C6; }"
                "QPushButton:checked { background: #1C303F; border-color: #8B9BAA;"
                " color: #A8B8C6; font-weight: bold; }"
                "QPushButton:disabled { color: #5a5a5a; border-color: #363636; }")
    return ("QPushButton { background: #F7F7F7; border: 1px solid #E5E5E5;"
            " border-radius: 14px; color: #0D0D0D; font-size: 10px;"
            " padding: 4px 8px; text-align: left; }"
            "QPushButton:hover { background: #DDE5EC; border-color: #8B9BAA; }"
            "QPushButton:checked { background: #DDE5EC; border-color: #8B9BAA;"
            " color: #4E6678; font-weight: bold; }"
            "QPushButton:disabled { color: #AAAAAA; border-color: #E5E5E5; }")


def _plan_toggle_blue_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#2A2A2A;border:1px solid #363636;"
                "border-radius:9px;color:#EBEBEB;font-weight:bold;padding:4px 8px;}"
                "QPushButton:checked{background:#1C303F;border-color:#8B9BAA;color:#A8B8C6;}"
                "QPushButton:hover{background:#363636;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:9px;color:#0D0D0D;font-weight:bold;padding:4px 8px;}"
            "QPushButton:checked{background:#DDE5EC;border-color:#8B9BAA;color:#4E6678;}"
            "QPushButton:hover{background:#DDE5EC;}")


def _plan_pick_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#2A2A2A;border:1px solid #363636;"
                "border-radius:8px;color:#EBEBEB;padding:4px 8px;}"
                "QPushButton:checked{background:#1a3a1a;border-color:#4caf50;"
                "color:#69f0ae;font-weight:bold;}"
                "QPushButton:hover{background:#363636;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:8px;color:#0D0D0D;padding:4px 8px;}"
            "QPushButton:checked{background:#e6f9e6;border-color:#2da44e;"
            "color:#1a7a35;font-weight:bold;}"
            "QPushButton:hover{background:#DDE5EC;}")


def _plan_primary_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1C303F;border:1px solid #8B9BAA;"
                "border-radius:9px;color:#EBEBEB;font-weight:bold;padding:4px 10px;}"
                "QPushButton:hover{background:#4E6678;border-color:#A8B8C6;}"
                "QPushButton:disabled{background:#2A2A2A;color:#5a5a5a;border-color:#363636;}")
    return ("QPushButton{background:#DDE5EC;border:1px solid #8B9BAA;"
            "border-radius:9px;color:#2E4A5F;font-weight:bold;padding:4px 10px;}"
            "QPushButton:hover{background:#8B9BAA;color:#ffffff;}"
            "QPushButton:disabled{background:#F7F7F7;color:#AAAAAA;border-color:#E5E5E5;}")


def _plan_clip_place_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1a2d1a;border:1px solid #3fb950;"
                "border-radius:9px;color:#EBEBEB;font-weight:bold;padding:4px 10px;}"
                "QPushButton:hover{background:#3fb950;color:#1F1F1F;}"
                "QPushButton:disabled{background:#1F1F1F;color:#5a5a5a;border-color:#363636;}")
    return ("QPushButton{background:#e6f9e6;border:1px solid #2da44e;"
            "border-radius:9px;color:#1a7a35;font-weight:bold;padding:4px 10px;}"
            "QPushButton:hover{background:#2da44e;color:#ffffff;}"
            "QPushButton:disabled{background:#F7F7F7;color:#AAAAAA;border-color:#E5E5E5;}")


def _plan_nudge_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:5px;color:#EBEBEB;padding:1px 5px;font-weight:bold;}"
                "QPushButton:hover{background:#1C303F;border-color:#A8B8C6;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:5px;color:#0D0D0D;padding:1px 5px;font-weight:bold;}"
            "QPushButton:hover{background:#DDE5EC;border-color:#8B9BAA;}")


def _plan_sep_css() -> str:
    return "border-top: 1px solid #363636;" if _is_dark() else "border-top: 1px solid #E5E5E5;"


def _plan_perf_on_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#2d1117;border:1px solid #f85149;"
                "border-radius:9px;color:#f85149;font-weight:bold;padding:4px 8px;}"
                "QPushButton:hover{background:#f85149;color:#ffffff;}")
    return ("QPushButton{background:#ffebe9;border:1px solid #cf222e;"
            "border-radius:9px;color:#cf222e;font-weight:bold;padding:4px 8px;}"
            "QPushButton:hover{background:#cf222e;color:#ffffff;}")


def _plan_perf_off_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:9px;color:#9B9B9B;padding:4px 8px;}"
                "QPushButton:hover{background:#1C2E3E;border-color:#f85149;color:#f85149;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:9px;color:#6B6B6B;padding:4px 8px;}"
            "QPushButton:hover{background:#ffebe9;border-color:#cf222e;color:#cf222e;}")


def _plan_clear_danger_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:9px;color:#9B9B9B;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#2d1117;color:#f85149;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:9px;color:#6B6B6B;font-size:10px;padding:3px 6px;}"
            "QPushButton:hover{background:#ffebe9;color:#cf222e;}")


def _plan_annot_on_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#182434;border:1px solid #A8B8C6;"
                "border-radius:9px;color:#A8B8C6;font-weight:bold;padding:4px 8px;}"
                "QPushButton:hover{background:#A8B8C6;color:#1F1F1F;}")
    return ("QPushButton{background:#EBF0F5;border:1px solid #4E6678;"
            "border-radius:9px;color:#4E6678;font-weight:bold;padding:4px 8px;}"
            "QPushButton:hover{background:#4E6678;color:#ffffff;}")


def _plan_annot_off_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:9px;color:#9B9B9B;padding:4px 8px;}"
                "QPushButton:hover{background:#1C303F;border-color:#A8B8C6;color:#A8B8C6;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:9px;color:#6B6B6B;padding:4px 8px;}"
            "QPushButton:hover{background:#EBF0F5;border-color:#4E6678;color:#4E6678;}")


def _plan_clear_annot_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:9px;color:#9B9B9B;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#182434;color:#A8B8C6;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:9px;color:#6B6B6B;font-size:10px;padding:3px 6px;}"
            "QPushButton:hover{background:#EBF0F5;color:#4E6678;}")


def _plan_bifurc_on_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1e1a0e;border:1px solid #e3b341;"
                "border-radius:9px;color:#e3b341;font-weight:bold;padding:4px 8px;}"
                "QPushButton:hover{background:#e3b341;color:#1F1F1F;}")
    return ("QPushButton{background:#fff8c5;border:1px solid #9a6700;"
            "border-radius:9px;color:#9a6700;font-weight:bold;padding:4px 8px;}"
            "QPushButton:hover{background:#9a6700;color:#ffffff;}")


def _plan_bifurc_off_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:9px;color:#9B9B9B;padding:4px 8px;}"
                "QPushButton:hover{background:#1C2E3E;border-color:#e3b341;color:#e3b341;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:9px;color:#6B6B6B;padding:4px 8px;}"
            "QPushButton:hover{background:#fff8c5;border-color:#9a6700;color:#9a6700;}")


def _plan_clear_bifurc_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:9px;color:#9B9B9B;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#1e1a0e;color:#e3b341;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:9px;color:#6B6B6B;font-size:10px;padding:3px 6px;}"
            "QPushButton:hover{background:#fff8c5;color:#9a6700;}")


def _plan_clear_traj_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{color:#f85149;}"
                "QPushButton:hover{background:#2d1117;border-color:#f85149;}")
    return ("QPushButton{color:#cf222e;}"
            "QPushButton:hover{background:#ffebe9;border-color:#cf222e;}")


def _plan_cut_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1a3a20;border:2px solid #3fb950;"
                "border-radius:10px;color:#3fb950;font-weight:bold;font-size:13px;}"
                "QPushButton:hover{background:#238636;color:#ffffff;}"
                "QPushButton:disabled{background:#1F1F1F;color:#5a5a5a;border-color:#363636;}")
    return ("QPushButton{background:#e6f9e6;border:2px solid #2da44e;"
            "border-radius:10px;color:#2da44e;font-weight:bold;font-size:13px;}"
            "QPushButton:hover{background:#238636;color:#ffffff;}"
            "QPushButton:disabled{background:#F7F7F7;color:#AAAAAA;border-color:#E5E5E5;}")


def _plan_restore_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1F1F1F;border:1px solid #f85149;"
                "border-radius:9px;color:#f85149;font-weight:bold;font-size:11px;}"
                "QPushButton:hover{background:#2d1117;color:#ff7b72;}"
                "QPushButton:disabled{color:#5a5a5a;border-color:#363636;background:#1F1F1F;}")
    return ("QPushButton{background:#ffebe9;border:1px solid #cf222e;"
            "border-radius:9px;color:#cf222e;font-weight:bold;font-size:11px;}"
            "QPushButton:hover{background:#a40e26;color:#ffffff;}"
            "QPushButton:disabled{color:#AAAAAA;border-color:#E5E5E5;background:#F7F7F7;}")


def _plan_plane_cut_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1a3a20;border:2px solid #3fb950;"
                "border-radius:10px;color:#3fb950;font-weight:bold;}"
                "QPushButton:hover{background:#238636;color:#fff;}"
                "QPushButton:disabled{background:#1F1F1F;color:#5a5a5a;border-color:#363636;}")
    return ("QPushButton{background:#e6f9e6;border:2px solid #2da44e;"
            "border-radius:10px;color:#2da44e;font-weight:bold;}"
            "QPushButton:hover{background:#238636;color:#fff;}"
            "QPushButton:disabled{background:#F7F7F7;color:#AAAAAA;border-color:#E5E5E5;}")


def _plan_flip_btn_css() -> str:
    if _is_dark():
        return ("QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:9px;color:#9B9B9B;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#1C303F;color:#A8B8C6;}"
                "QPushButton:disabled{color:#5a5a5a;border-color:#363636;}")
    return ("QPushButton{background:#F7F7F7;border:1px solid #E5E5E5;"
            "border-radius:9px;color:#6B6B6B;font-size:10px;padding:3px 6px;}"
            "QPushButton:hover{background:#DDE5EC;color:#8B9BAA;}"
            "QPushButton:disabled{color:#AAAAAA;border-color:#E5E5E5;}")


def _plan_muted_lbl_css() -> str:
    """Stylesheet for secondary / muted status labels (theme-aware)."""
    return ("color:#9B9B9B; font-size:10px;"
            if _is_dark() else
            "color:#6B6B6B; font-size:10px;")


class PlanningWindow(QMainWindow):
    """
    Standalone surgical-device planning window.

    Public API
    ----------
    set_vessel_mesh(poly_data)     — load vascular mesh
    set_aneurysm(poly_data|None)   — show/hide aneurysm candidate
    set_neck_diameter(mm)          — forward to clip panel for filtering
    set_aneurysm_centroid(x,y,z)   — pre-fill position spinboxes
    """

    def __init__(self, parent: QWidget | None = None, *, embedded: bool = False) -> None:
        super().__init__(parent)
        # When embedded in a QTabWidget the window must be a plain widget, not a
        # standalone OS window.  Set Qt.Widget BEFORE _build_vtk() so the VTK
        # render window gets the correct native handle from the start and the OS
        # never creates a temporary top-level window at position (0, 28) that
        # causes a QWindowsWindow::setGeometry warning and a black 3-D viewport.
        if embedded:
            self.setWindowFlags(Qt.Widget)
            self.setMinimumSize(400, 300)   # relax the standalone 1100×720 constraint
        self.setWindowTitle("PROSPECTIVE — Planificación de Dispositivos")
        if not embedded:
            # Clamp minimum to fit 1366×768 laptops with some breathing room
            self.setMinimumSize(880, 580)

        self._clip_actors:   dict[int, vtk.vtkActor] = {}
        self._stent_actors:  dict[int, vtk.vtkActor] = {}
        self._coil_actors:   dict[int, vtk.vtkActor] = {}
        self._traj_actor:    vtk.vtkActor | None      = None

        # Flag: mesh has been loaded (need camera reset after first show)
        self._mesh_loaded: bool = False

        # Vessel geometry (stored for PCA-based axis estimation + lumen centering)
        self._vessel_poly:         vtk.vtkPolyData | None           = None
        self._kd_locator:          vtk.vtkKdTreePointLocator | None = None
        self._vessel_normals_data: vtk.vtkPolyData | None           = None
        self._obb_tree:            vtk.vtkOBBTree | None            = None

        # Last single-point stent (for nudge controls)
        self._nudge_idx:   int | None        = None   # stent index in _stent_actors
        self._nudge_pos:   np.ndarray | None = None   # current centre (world mm)
        self._nudge_axis:  np.ndarray | None = None   # unit vessel axis (PCA)
        self._nudge_perp1: np.ndarray | None = None   # lateral  (⊥ axis)
        self._nudge_perp2: np.ndarray | None = None   # height   (⊥ axis, ⊥ perp1)

        # Index of the stent being reshaped (None = normal placement mode)
        self._deform_target_idx: int | None = None

        # Perforator marking state
        self._perf_mode:    bool                 = False
        self._perf_pts:     list[np.ndarray]     = []   # world-space centres
        self._perf_actors:  list[vtk.vtkActor]   = []   # red sphere actors
        self._perf_radius:  float                = 4.0  # proximity threshold (mm)

        # 3-D annotation state
        self._annot_mode:          bool       = False
        self._annot_data:          list[dict] = []   # [{"pos":(x,y,z),"text":str}]
        self._annot_label_actors:  list       = []   # vtkBillboardTextActor3D
        self._annot_sphere_actors: list[vtk.vtkActor] = []  # cyan anchor spheres

        # Bifurcation angle measurement state (A-2)
        self._bifurc_mode:    bool                     = False
        self._bifurc_pts:     list[tuple[float, ...]]  = []   # ≤4 picks
        self._bifurc_actors:  list                     = []   # spheres + lines + labels

        # Planning history / undo stack (A-3)
        # Each entry: {"kind": "stent"|"clip", "label": str}
        self._history:    list[dict] = []
        self._max_history: int       = 30

        # Pre/post comparison (A-4)
        self._post_actor:  vtk.vtkActor | None = None

        # Aneurysm candidate marker (sphere + label shown when a candidate is selected)
        self._cand_sphere_actor: vtk.vtkActor | None = None
        self._cand_label_actor:  object | None       = None   # vtkBillboardTextActor3D

        # FD deployment animation (SC-3)
        self._anim_timer:  QTimer | None           = None
        self._anim_actor:  vtk.vtkActor | None     = None
        self._anim_frame:  int                     = 0
        self._anim_frames: int                     = 24   # total frames
        self._anim_transform: vtk.vtkTransform | None = None

        # Deployed centreline stent
        self._cl_stent_actor:    vtk.vtkActor | None    = None

        # Morphometric overlay (Feature 5)
        self._morpho_overlay:    object | None          = None  # MorphoOverlayActors
        self._morpho_result:     object | None          = None  # MorphometricResult
        self._morpho_poly:       vtk.vtkPolyData | None = None  # aneurysm mesh

        # Perforator risk overlay («Perforantes en riesgo»)
        self._perf_overlay_actors: list = []

        # Ruler / caliper state
        self._ruler_mode:        bool                    = False  # waiting for pick A or B
        self._ruler_pt_a:        tuple | None            = None   # first endpoint (mm)
        self._ruler_actors:      dict[int, object]       = {}     # index → RulerActors
        self._ruler_preview:     object | None           = None   # temporary preview actors

        # Centreline pick state
        self._cl_mode:           str | None              = None   # 'source' | 'target' | None
        self._cl_actor:          vtk.vtkActor | None     = None
        self._cl_src_actor:      vtk.vtkActor | None     = None
        self._cl_tgt_actor:      vtk.vtkActor | None     = None
        self._cl_bar_actor:      vtk.vtkActor | None     = None

        # Trajectory picking state
        self._pick_mode:         bool                    = False
        self._pick_points:       list[tuple[float, ...]] = []
        self._pick_diameters:    list[float]             = []   # vessel Ø at each pick point (mm)
        self._pick_pt_actors:    list[vtk.vtkActor]      = []
        self._pick_diam_actors:  list                    = []   # vtkBillboardTextActor3D labels
        self._preview_actor:     vtk.vtkActor | None     = None   # stent-tube preview
        self._viab_locator:      vtk.vtkCellLocator | None = None

        # Box-clip state
        self._box_clip_active:   bool                     = False
        self._box_cut_applied:   bool                     = False   # True after "Cortar" clicked
        self._box_widget:        vtk.vtkBoxWidget2 | None = None
        self._ghost_actor:       vtk.vtkActor | None      = None
        self._mesh_raw_bounds:   tuple[float, ...]        = (0.,) * 6

        # Theme-sensitive widget refs — populated during _build_*() calls
        self._plan_page_hdrs: list  = []   # panel section header QLabels
        self._plan_page_helps: list = []   # help text QLabels
        self._plan_muted_btns: list = []   # settings-page muted buttons
        self._nudge_btns: list      = []   # direction arrow buttons
        self._plan_seps: list       = []   # QFrame separators (HLine)

        self._build_vtk()
        self._build_toolbar()       # ← must run before _build_panels (_act_* used inside)
        self._build_panels()        # creates sub-panels + QStackedWidget pages
        self._build_central_layout()  # assembles sidebar | vtk | panel_stack
        self._connect_signals()

    # ------------------------------------------------------------------ #
    # Qt events                                                            #
    # ------------------------------------------------------------------ #

    def showEvent(self, event) -> None:
        """
        Called every time the window becomes visible.

        On the **first** show we must initialise the VTK interactor here,
        not in __init__, because Windows requires a real HWND/pixel-format
        to create an OpenGL context.  Calling Initialize() before show()
        raises ``wglMakeCurrent failed (code 2004)``.

        Subsequent shows (window hidden → re-shown) only reset the camera.
        """
        super().showEvent(event)
        if not getattr(self, "_vtk_initialized", False):
            self._vtk_initialized = True

            def _init_and_reset() -> None:
                self._iren.Initialize()
                if self._mesh_loaded:
                    self._reset_camera()

            QTimer.singleShot(0, _init_and_reset)
        elif self._mesh_loaded:
            # 150 ms gives Qt's layout engine time to send the resizeEvent to
            # the VTK widget before _reset_camera syncs the render window size.
            QTimer.singleShot(150, self._reset_camera)

    # ------------------------------------------------------------------ #
    # VTK pipeline                                                         #
    # ------------------------------------------------------------------ #

    def _build_vtk(self) -> None:
        central = QWidget()
        vl = QVBoxLayout(central)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        self._vtk_hdr = QLabel("Vista 3D — Malla Vascular")
        self._vtk_hdr.setAlignment(Qt.AlignCenter)
        self._vtk_hdr.setStyleSheet(_plan_vtk_hdr_css())
        vl.addWidget(self._vtk_hdr)

        self._iren = QVTKRenderWindowInteractor()
        self._iren.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(self._iren, stretch=1)

        # Store — setCentralWidget() is called later in _build_central_layout()
        # after the icon sidebar and panel stack are ready.
        self._vtk_container = central

        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.04, 0.06, 0.10)
        self._renderer.SetBackground2(0.01, 0.02, 0.05)
        self._renderer.SetGradientBackground(True)
        self._iren.GetRenderWindow().AddRenderer(self._renderer)

        # Standard trackball style — camera rotation handled by Qt event filter
        self._style = vtk.vtkInteractorStyleTrackballCamera()
        self._iren.SetInteractorStyle(self._style)

        # Qt-level event filter: intercepts left-clicks before VTK sees them
        # when pick mode is active → no AbortFlag, no VTK virtual override needed
        self._iren.installEventFilter(self)

        # Cell picker — only picks from the vessel actor
        self._cell_picker = vtk.vtkCellPicker()
        self._cell_picker.SetTolerance(0.005)

        # Vessel mesh (semi-transparent blue)
        self._mesh_mapper = vtk.vtkPolyDataMapper()
        self._mesh_actor  = vtk.vtkActor()
        self._mesh_actor.SetMapper(self._mesh_mapper)
        p = self._mesh_actor.GetProperty()
        p.SetColor(0.30, 0.60, 0.95)
        p.SetOpacity(0.42)
        p.SetAmbient(0.20)
        p.SetDiffuse(0.70)
        p.SetSpecular(0.35)
        p.SetSpecularPower(20.0)
        self._mesh_actor.VisibilityOff()
        self._renderer.AddActor(self._mesh_actor)

        # Aneurysm candidate (magenta)
        self._aneurysm_mapper = vtk.vtkPolyDataMapper()
        self._aneurysm_actor  = vtk.vtkActor()
        self._aneurysm_actor.SetMapper(self._aneurysm_mapper)
        p2 = self._aneurysm_actor.GetProperty()
        p2.SetColor(0.95, 0.15, 0.85)
        p2.SetOpacity(0.88)
        self._aneurysm_actor.VisibilityOff()
        self._renderer.AddActor(self._aneurysm_actor)

        # Orientation axes
        axes = vtk.vtkAxesActor()
        axes.SetXAxisLabelText("R")
        axes.SetYAxisLabelText("A")
        axes.SetZAxisLabelText("S")
        self._axes_widget = vtk.vtkOrientationMarkerWidget()
        self._axes_widget.SetOrientationMarker(axes)
        self._axes_widget.SetInteractor(self._iren)
        self._axes_widget.SetViewport(0.0, 0.0, 0.14, 0.14)
        self._axes_widget.EnabledOn()
        self._axes_widget.InteractiveOff()

        # NOTE: self._iren.Initialize() is intentionally NOT called here.
        # It is deferred to showEvent() so that Windows has a valid HWND
        # and pixel format before VTK tries to create an OpenGL context.
        # Calling Initialize() before show() causes:
        #   wglMakeCurrent failed in MakeCurrent(), error: unknown error(code 2004)

    # ------------------------------------------------------------------ #
    # Right-side panels                                                    #
    # ------------------------------------------------------------------ #

    # ──────────────────────────────────────────────────────────────────────── #
    # Right-side tool panel (grouped sidebar)                                   #
    # ──────────────────────────────────────────────────────────────────────── #

    def _build_panels(self) -> None:
        """
        Create all sub-panel widgets and assemble them as pages in a
        QStackedWidget.  The stack is hidden until the user presses a
        sidebar icon button; pressing it again (or the same button) hides it.

        Page map
        --------
        0  — empty placeholder (nothing selected)
        1  — ✂  Clips
        2  — 🌀  Flow Diverters
        3  — 🩺  Stent en Línea Central
        4  — 🎯  Trayectoria / Pick mode
        5  — 🫀  Línea Central
        6  — 📏  Mediciones 3D
        7  — ✂  Corte de malla (box + plano)
        8  — ⚙   Vista / Overlays
        """
        # ── Sub-panel widgets ────────────────────────────────────────── #
        self._clip_panel         = ClipPanel()
        self._stent_panel        = StentPanel()
        self._centerline_panel   = CenterlinePanel()
        self._cl_stent_panel     = CLStentPanel()
        self._measurement_panel  = MeasurementPanel()

        # ── Panel stack ──────────────────────────────────────────────── #
        self._panel_stack = QStackedWidget()
        self._panel_stack.setMinimumWidth(240)
        self._panel_stack.setMaximumWidth(380)
        self._panel_stack.setVisible(False)
        self._panel_stack.setStyleSheet(_plan_panel_stack_css())

        # Page 0 — empty placeholder
        self._panel_stack.addWidget(QWidget())                         # 0

        from prospective.ui.icons import I as _I
        # Pages 1-3 — Intervention tools
        self._panel_stack.addWidget(self._wrap_page(
            f"{_I.CLIPS}  CLIPS",
            "Selecciona un clip del catálogo, ajusta posición y ángulo. "
            "Activa 'Trayectoria' en la barra lateral para colocación guiada.",
            self._clip_panel))                                         # 1

        self._panel_stack.addWidget(self._wrap_page(
            f"{_I.FLOW_DIV}  FLOW DIVERTERS",
            "Elige un stent o flow diverter (Pipeline, Surpass, FRED…). "
            "Verde = cabe · Naranja = justo · Rojo = no cabe.",
            self._stent_panel))                                        # 2

        self._panel_stack.addWidget(self._wrap_page(
            f"{_I.STENT_CL}  STENT EN LÍNEA CENTRAL",
            "Calcula la línea central y despliega un stent siguiendo "
            "el eje de curvatura real.",
            self._cl_stent_panel))                                     # 3

        # Page 4 — Trajectory / pick  (content built inline)
        self._panel_stack.addWidget(self._build_trajectory_page())    # 4

        # Pages 5-6 — Analysis tools
        self._panel_stack.addWidget(self._wrap_page(
            f"{_I.CENTERLINE}  LÍNEA CENTRAL",
            "Calcula la línea central entre dos puntos. Perfil de "
            "sección transversal y estenosis mínima.",
            self._centerline_panel))                                   # 5

        self._panel_stack.addWidget(self._wrap_page(
            f"{_I.MEASURE}  MEDICIONES 3D",
            "Calibre 3D. Haz clic en dos puntos de la superficie para "
            "medir la distancia en mm. Exportable como CSV.",
            self._measurement_panel))                                  # 6

        # Page 7 — Box clip + plane cut
        self._panel_stack.addWidget(self._build_cut_page())           # 7

        # Page 8 — Vista / overlays / settings
        self._panel_stack.addWidget(self._build_settings_page())      # 8

        # Sidebar key → stack index mapping
        self._sidebar_page_map: dict[str, int] = {
            "clips":        1,
            "flow_div":     2,
            "stent_cl":     3,
            "trajectory":   4,
            "centerline":   5,
            "measurements": 6,
            "cut":          7,
            "settings":     8,
        }

    # ------------------------------------------------------------------ #
    # Panel page helpers                                                   #
    # ------------------------------------------------------------------ #

    def _wrap_page(self, title: str, help_text: str, panel: QWidget) -> QWidget:
        """Wrap *panel* in a titled page widget suitable for the panel stack."""
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        hdr = QLabel(title)
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet(_plan_hdr_css())
        self._plan_page_hdrs.append(hdr)
        vl.addWidget(hdr)

        if help_text:
            help_lbl = QLabel(help_text)
            help_lbl.setWordWrap(True)
            help_lbl.setStyleSheet(_plan_help_css())
            self._plan_page_helps.append(help_lbl)
            vl.addWidget(help_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        vl.addWidget(scroll, 1)

        return page

    def _build_settings_page(self) -> QWidget:
        """Settings page: Vista / Overlays controls."""
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        hdr = QLabel("⚙  VISTA / OVERLAYS")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet(_plan_hdr_css())
        self._plan_page_hdrs.append(hdr)
        vl.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        root = QWidget()
        cl = QVBoxLayout(root)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(6)

        # FD animation button
        self._btn_deploy_anim = QPushButton("▶  Animar despliegue FD")
        self._btn_deploy_anim.setEnabled(False)
        self._btn_deploy_anim.setToolTip(
            "Simula visualmente el despliegue del último flow diverter colocado."
        )
        self._btn_deploy_anim.setStyleSheet(_plan_muted_btn_css())
        self._btn_deploy_anim.clicked.connect(self._start_deploy_animation)
        self._act_deploy_anim.changed.connect(
            lambda: self._btn_deploy_anim.setEnabled(self._act_deploy_anim.isEnabled())
        )
        self._plan_muted_btns.append(self._btn_deploy_anim)
        cl.addWidget(self._btn_deploy_anim)

        # Morpho overlay toggle
        from prospective.ui.icons import I as _I
        self._btn_morpho_overlay = QPushButton(f"{_I.ANGLE_MEAS}  Overlay morfométrico")
        self._btn_morpho_overlay.setCheckable(True)
        self._btn_morpho_overlay.setEnabled(False)
        self._btn_morpho_overlay.setToolTip(
            "Muestra el plano del cuello, la altura del domo y los diámetros "
            "máximos directamente sobre la escena 3D."
        )
        self._btn_morpho_overlay.setStyleSheet(_plan_muted_btn_css())
        self._btn_morpho_overlay.toggled.connect(self._act_morpho_overlay.setChecked)
        self._act_morpho_overlay.toggled.connect(self._btn_morpho_overlay.setChecked)
        self._act_morpho_overlay.changed.connect(
            lambda: self._btn_morpho_overlay.setEnabled(
                self._act_morpho_overlay.isEnabled()
            )
        )
        self._plan_muted_btns.append(self._btn_morpho_overlay)
        cl.addWidget(self._btn_morpho_overlay)

        # Mesh opacity toggle
        self._btn_opaque = QPushButton("Malla: Opaca / Transparente")
        self._btn_opaque.setCheckable(True)
        self._btn_opaque.setStyleSheet(_plan_muted_btn_css())
        self._btn_opaque.toggled.connect(self._act_opaque.setChecked)
        self._act_opaque.toggled.connect(self._btn_opaque.setChecked)
        self._plan_muted_btns.append(self._btn_opaque)
        cl.addWidget(self._btn_opaque)

        # Post-treatment row
        post_row = QWidget()
        post_rl = QHBoxLayout(post_row)
        post_rl.setContentsMargins(0, 0, 0, 0)
        post_rl.setSpacing(4)

        self._btn_load_post = QPushButton(f"{_I.FOLDER}  Cargar post-tratamiento")
        self._btn_load_post.setStyleSheet(_plan_muted_btn_css())
        self._btn_load_post.setToolTip("Carga malla post-tratamiento (STL/OBJ)")
        self._btn_load_post.clicked.connect(self._load_post_mesh)
        self._plan_muted_btns.append(self._btn_load_post)
        post_rl.addWidget(self._btn_load_post)

        self._btn_hide_post = QPushButton(f"{_I.EYE}  Mostrar")
        self._btn_hide_post.setCheckable(True)
        self._btn_hide_post.setEnabled(False)
        self._btn_hide_post.setToolTip("Mostrar/ocultar malla post-tratamiento")
        self._btn_hide_post.setStyleSheet(_plan_muted_btn_css())
        self._btn_hide_post.toggled.connect(self._act_hide_post.setChecked)
        self._act_hide_post.toggled.connect(self._btn_hide_post.setChecked)
        self._act_hide_post.changed.connect(
            lambda: self._btn_hide_post.setEnabled(self._act_hide_post.isEnabled())
        )
        self._plan_muted_btns.append(self._btn_hide_post)
        post_rl.addWidget(self._btn_hide_post)
        cl.addWidget(post_row)

        cl.addStretch()
        scroll.setWidget(root)
        vl.addWidget(scroll, 1)
        return page

    # ------------------------------------------------------------------ #
    # Top toolbar                                                          #
    # ------------------------------------------------------------------ #

    def _build_toolbar(self) -> None:
        """
        Minimal toolbar — only the most-used actions are visible.
        Secondary view/overlay controls live in the ⚙ Vista page of the
        icon sidebar (built in _build_settings_page).
        """
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence

        tb: QToolBar = self.addToolBar("Acciones")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        tb.setStyleSheet("QToolBar { spacing: 4px; }")

        # ── SkullApp logo ─────────────────────────────────────────────── #
        logo_lbl = QLabel()
        logo_lbl.setStyleSheet("background:transparent; margin: 0 6px 0 8px;")
        pix = _tinted_logo_pix(48)  # 48×21 px — fits the toolbar height
        if not pix.isNull():
            logo_lbl.setPixmap(pix)
        else:
            logo_lbl.setText("💀")
            logo_lbl.setStyleSheet("font-size:16px; background:transparent;")
        tb.addWidget(logo_lbl)

        tb.addSeparator()

        # Camera reset — always needed
        act_cam = tb.addAction("⟳ Cámara")
        act_cam.setToolTip("Resetear la cámara al encuadre original (R)")
        act_cam.triggered.connect(self._reset_camera)

        tb.addSeparator()

        # Undo (Ctrl+Z) — always needed
        self._act_undo = tb.addAction("↶ Deshacer")
        self._act_undo.setToolTip("Elimina el último clip o stent colocado (Ctrl+Z)")
        self._act_undo.setEnabled(False)
        self._act_undo.triggered.connect(self._undo_last_device)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo_last_device)

        # ── Secondary actions — NOT added to the toolbar ─────────────── #
        # Their self._act_* attributes are still created so all handler code
        # works unchanged.  They surface as buttons inside the ⚙ VISTA page
        # of the right sidebar (built in _build_settings_page).

        self._act_deploy_anim = QAction("▶ Animar FD", self)
        self._act_deploy_anim.setToolTip(
            "Simula visualmente el despliegue del último flow diverter colocado."
        )
        self._act_deploy_anim.setEnabled(False)
        self._act_deploy_anim.triggered.connect(self._start_deploy_animation)

        from prospective.ui.icons import I as _I
        self._act_morpho_overlay = QAction(f"{_I.ANGLE_MEAS} Overlay morfométrico", self)
        self._act_morpho_overlay.setCheckable(True)
        self._act_morpho_overlay.setEnabled(False)
        self._act_morpho_overlay.setToolTip(
            "Muestra el plano del cuello, la altura del domo y los diámetros "
            "máximos directamente sobre la escena 3D."
        )
        self._act_morpho_overlay.toggled.connect(self._toggle_morpho_overlay)

        self._act_opaque = QAction("Malla: Opaca", self)
        self._act_opaque.setCheckable(True)
        self._act_opaque.toggled.connect(self._toggle_mesh_opacity)

        self._act_load_post = QAction(f"{_I.FOLDER} Post-tratamiento", self)
        self._act_load_post.setToolTip("Carga malla post-tratamiento (STL/OBJ)")
        self._act_load_post.triggered.connect(self._load_post_mesh)

        self._act_hide_post = QAction(f"{_I.EYE} Post-tratamiento", self)
        self._act_hide_post.setCheckable(True)
        self._act_hide_post.setEnabled(False)
        self._act_hide_post.setToolTip("Mostrar/ocultar malla post-tratamiento")
        self._act_hide_post.toggled.connect(self._toggle_post_mesh_visibility)

        # Spacer + status label at the far right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        _hint_clr = "#9B9B9B" if _is_dark() else "#6B6B6B"
        self._lbl_tb_status = QLabel(
            f"<small style='color:{_hint_clr}'>"
            "Clic izquierdo sobre arteria cuando pick-mode está activo  ·  "
            "Ctrl+Z = deshacer  ·  R = resetear cámara"
            "</small>"
        )
        tb.addWidget(self._lbl_tb_status)

    # ------------------------------------------------------------------ #
    # Trajectory page (sidebar panel, page 4)                             #
    # ------------------------------------------------------------------ #

    def _build_trajectory_page(self) -> QWidget:
        """Build the Trajectory / Pick panel page for the sidebar stack."""
        page = QWidget()
        page_vl = QVBoxLayout(page)
        page_vl.setContentsMargins(0, 0, 0, 0)
        page_vl.setSpacing(0)

        # Page header
        from prospective.ui.icons import I as _I
        hdr = QLabel(f"{_I.TRAJECTORY}  TRAYECTORIA / PICK")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet(_plan_hdr_css())
        self._plan_page_hdrs.append(hdr)
        page_vl.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        root = QWidget()
        vl   = QVBoxLayout(root)
        vl.setContentsMargins(6, 4, 6, 4)
        vl.setSpacing(4)

        # ── Row 1: pick-mode toggle + undo / clear ────────────────────── #
        row = QWidget()
        rl  = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        self._btn_pick = QPushButton("Activar modo pick")
        self._btn_pick.setCheckable(True)
        self._btn_pick.setToolTip(
            "Activa el modo de selección de puntos sobre la arteria.\n"
            "La rotación de cámara se restaura al desactivarlo."
        )
        self._btn_pick.setStyleSheet(_plan_pick_btn_css())
        self._btn_pick.toggled.connect(self._toggle_pick_mode)
        rl.addWidget(self._btn_pick, 1)

        btn_undo = QPushButton("Deshacer")
        btn_undo.setMinimumWidth(90)
        btn_undo.clicked.connect(self._undo_last_point)
        rl.addWidget(btn_undo)

        self._btn_clear_traj = QPushButton("Limpiar")
        self._btn_clear_traj.setMinimumWidth(76)
        self._btn_clear_traj.setStyleSheet(_plan_clear_traj_btn_css())
        self._btn_clear_traj.clicked.connect(self._clear_trajectory)
        rl.addWidget(self._btn_clear_traj)

        vl.addWidget(row)

        # ── Row 2: stent selector + overlap + place ────────────────────── #
        stent_row = QWidget()
        sr = QHBoxLayout(stent_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(6)

        sr.addWidget(QLabel("Stent:"))
        self._traj_stent_combo = QComboBox()
        for spec in STENT_CATALOGUE:
            self._traj_stent_combo.addItem(spec.display_label)
        self._traj_stent_combo.currentIndexChanged.connect(self._on_stent_combo_changed)
        sr.addWidget(self._traj_stent_combo, 1)

        sr.addWidget(QLabel("Sol.:"))
        self._overlap_spin = QDoubleSpinBox()
        self._overlap_spin.setRange(0.0, 5.0)
        self._overlap_spin.setValue(0.5)
        self._overlap_spin.setSingleStep(0.5)
        self._overlap_spin.setDecimals(1)
        self._overlap_spin.setSuffix("mm")
        self._overlap_spin.setMinimumWidth(68)
        self._overlap_spin.setToolTip("Solapamiento entre stents consecutivos (mm)")
        self._overlap_spin.valueChanged.connect(self._on_stent_combo_changed)
        sr.addWidget(self._overlap_spin)

        vl.addWidget(stent_row)

        self._btn_auto_place = QPushButton(f"{_I.STENT} Colocar stents en trayectoria")
        self._btn_auto_place.setEnabled(False)
        self._btn_auto_place.setStyleSheet(_plan_primary_btn_css())
        self._btn_auto_place.clicked.connect(self._place_stents_on_trajectory)
        vl.addWidget(self._btn_auto_place)

        # ── Row 3: clip selector + place ──────────────────────────────── #
        clip_row = QWidget()
        cl = QHBoxLayout(clip_row)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)

        cl.addWidget(QLabel("Clip:"))
        self._traj_clip_combo = QComboBox()
        self._traj_clip_combo.setToolTip(
            f"Clip que se colocará al pulsar '{_I.CLIP_PLACE} Colocar clip en cuello'."
        )
        for spec in CLIP_CATALOGUE:
            self._traj_clip_combo.addItem(
                f"{spec.name}  ({spec.blade_length_mm:.0f} mm)"
            )
        cl.addWidget(self._traj_clip_combo, 1)
        vl.addWidget(clip_row)

        self._btn_clip_place = QPushButton(f"{_I.CLIP_PLACE} Colocar clip en cuello")
        self._btn_clip_place.setEnabled(False)
        self._btn_clip_place.setToolTip(
            "1 punto → clip centrado en ese punto, hoja perpendicular al eje del vaso.\n"
            "2+ puntos → clip en el punto medio, hoja orientada de P1 a P2."
        )
        self._btn_clip_place.setStyleSheet(_plan_clip_place_btn_css())
        self._btn_clip_place.clicked.connect(self._place_clip_on_trajectory)
        vl.addWidget(self._btn_clip_place)

        # ── Status / viability ────────────────────────────────────────── #
        self._pick_status = QLabel(
            "Pulse 'Activar modo pick' y haga clic sobre la arteria para definir puntos."
        )
        self._pick_status.setStyleSheet(_plan_muted_lbl_css())
        vl.addWidget(self._pick_status)

        # ── Nudge controls (shown after single-point placement) ───────── #
        self._nudge_widget = QWidget()
        nv = QVBoxLayout(self._nudge_widget)
        nv.setContentsMargins(0, 2, 0, 0)
        nv.setSpacing(3)

        # ── Row 1: header + step spinbox + viability label ───────────── #
        hdr_row = QWidget()
        hl = QHBoxLayout(hdr_row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        _hdr_clr  = "#A8B8C6" if _is_dark() else "#4E6678"
        _muted_clr = "#9B9B9B" if _is_dark() else "#6B6B6B"
        hl.addWidget(QLabel(f"<b style='color:{_hdr_clr};font-size:10px;'>Mover dispositivo:</b>"))
        hl.addWidget(QLabel(f"<small style='color:{_muted_clr}'>Paso:</small>"))

        self._nudge_step_spin = QDoubleSpinBox()
        self._nudge_step_spin.setRange(0.1, 10.0)
        self._nudge_step_spin.setValue(1.0)
        self._nudge_step_spin.setSingleStep(0.5)
        self._nudge_step_spin.setDecimals(1)
        self._nudge_step_spin.setSuffix(" mm")
        self._nudge_step_spin.setMinimumWidth(68)
        hl.addWidget(self._nudge_step_spin)

        hl.addSpacing(10)
        self._nudge_viab_label = QLabel("—")
        self._nudge_viab_label.setStyleSheet(_plan_muted_lbl_css() + " font-weight:bold;")
        self._nudge_viab_label.setMinimumWidth(260)
        hl.addWidget(self._nudge_viab_label)
        hl.addStretch()
        nv.addWidget(hdr_row)

        # ── Row 2: three axis groups ─────────────────────────────────── #
        axes_row = QWidget()
        al = QHBoxLayout(axes_row)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(4)

        _axes = [
            ("Eje vaso",  "axis",  "◀◀", "◀", "▶", "▶▶"),
            ("Lateral",   "perp1", "◀◀", "◀", "▶", "▶▶"),
            ("Altura",    "perp2", "◀◀", "◀", "▶", "▶▶"),
        ]
        _ax_clr = "#9B9B9B" if _is_dark() else "#6B6B6B"
        for axis_label, dir_key, *btn_labels in _axes:
            al.addWidget(QLabel(
                f"<small style='color:{_ax_clr}'>{axis_label}:</small>"
            ))
            for btn_label in btn_labels:
                mult = {"◀◀": -3, "◀": -1, "▶": +1, "▶▶": +3}[btn_label]
                btn = QPushButton(btn_label)
                btn.setFixedWidth(36)
                btn.setStyleSheet(_plan_nudge_btn_css())
                btn.clicked.connect(
                    lambda _=False, k=dir_key, m=mult: self._nudge_in_dir(k, m)
                )
                self._nudge_btns.append(btn)
                al.addWidget(btn)
            al.addSpacing(8)

        al.addStretch()
        nv.addWidget(axes_row)

        self._nudge_total_offset: float = 0.0
        self._nudge_widget.setVisible(False)
        vl.addWidget(self._nudge_widget)

        # ── Perforator marking ────────────────────────────────────────── #
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(_plan_sep_css())
        self._plan_seps.append(sep)
        vl.addWidget(sep)

        perf_row = QWidget()
        pl = QHBoxLayout(perf_row)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(6)

        _perf_style_on  = _plan_perf_on_css()
        _perf_style_off = _plan_perf_off_css()

        from prospective.ui.icons import I as _I
        self._btn_perf = QPushButton(f"{_I.BRAIN} Marcar perforante")
        self._btn_perf.setCheckable(True)
        self._btn_perf.setToolTip(
            "Activa el modo de marcado de arterias perforantes.\n"
            "Haga clic sobre el vaso para añadir un marcador rojo.\n"
            "Los clips colocados cerca de un perforante generarán una advertencia."
        )
        self._btn_perf.setStyleSheet(_perf_style_off)
        self._btn_perf.toggled.connect(self._toggle_perf_mode)
        pl.addWidget(self._btn_perf)

        self._btn_clear_perf = QPushButton("Limpiar perforantes")
        self._btn_clear_perf.setToolTip("Elimina todos los marcadores de perforantes.")
        self._btn_clear_perf.setStyleSheet(_plan_clear_danger_btn_css())
        self._btn_clear_perf.clicked.connect(self._clear_perforators)
        pl.addWidget(self._btn_clear_perf)

        self._lbl_perf_count = QLabel("0 perforantes marcados")
        self._lbl_perf_count.setStyleSheet(_plan_muted_lbl_css())
        pl.addWidget(self._lbl_perf_count)
        pl.addStretch()

        vl.addWidget(perf_row)

        self._lbl_perf_collision = QLabel("—")
        self._lbl_perf_collision.setStyleSheet(_plan_muted_lbl_css())
        self._lbl_perf_collision.setWordWrap(True)
        vl.addWidget(self._lbl_perf_collision)

        # Store styles for toggling
        self._perf_style_on  = _perf_style_on
        self._perf_style_off = _perf_style_off

        # ── 3-D Annotation mode ───────────────────────────────────────── #
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(_plan_sep_css())
        self._plan_seps.append(sep2)
        vl.addWidget(sep2)

        annot_row = QWidget()
        anl = QHBoxLayout(annot_row)
        anl.setContentsMargins(0, 0, 0, 0)
        anl.setSpacing(6)

        _annot_style_off = _plan_annot_off_css()
        _annot_style_on  = _plan_annot_on_css()
        self._annot_style_on  = _annot_style_on
        self._annot_style_off = _annot_style_off

        from prospective.ui.icons import I as _I
        self._btn_annot = QPushButton(f"{_I.ANNOTATION} Añadir anotación")
        self._btn_annot.setCheckable(True)
        self._btn_annot.setToolTip(
            "Activa el modo de anotación 3D.\n"
            "Haga clic sobre la arteria para anclar una etiqueta de texto."
        )
        self._btn_annot.setStyleSheet(_annot_style_off)
        self._btn_annot.toggled.connect(self._toggle_annot_mode)
        anl.addWidget(self._btn_annot)

        self._btn_del_annot = QPushButton("Deshacer última")
        self._btn_del_annot.setToolTip("Elimina la última anotación añadida.")
        self._btn_del_annot.setStyleSheet(_plan_clear_annot_btn_css())
        self._btn_del_annot.clicked.connect(self._undo_last_annotation)
        anl.addWidget(self._btn_del_annot)

        self._btn_clear_annot = QPushButton("Limpiar anotaciones")
        self._btn_clear_annot.setToolTip("Elimina todas las anotaciones 3D.")
        self._btn_clear_annot.setStyleSheet(_plan_clear_annot_btn_css())
        self._btn_clear_annot.clicked.connect(self._clear_annotations)
        anl.addWidget(self._btn_clear_annot)

        self._lbl_annot_count = QLabel("0 anotaciones")
        self._lbl_annot_count.setStyleSheet(_plan_muted_lbl_css())
        anl.addWidget(self._lbl_annot_count)
        anl.addStretch()
        vl.addWidget(annot_row)

        # ── Bifurcation angle measurement (A-2) ──────────────────────── #
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet(_plan_sep_css())
        self._plan_seps.append(sep3)
        vl.addWidget(sep3)

        bifurc_row = QWidget()
        bl = QHBoxLayout(bifurc_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)

        _bifurc_style_off = _plan_bifurc_off_css()
        _bifurc_style_on  = _plan_bifurc_on_css()
        self._bifurc_style_on  = _bifurc_style_on
        self._bifurc_style_off = _bifurc_style_off

        from prospective.ui.icons import I as _I
        self._btn_bifurc = QPushButton(f"{_I.ANGLE_MEAS} Medir ángulos bifurcación")
        self._btn_bifurc.setCheckable(True)
        self._btn_bifurc.setToolTip(
            "Mide los ángulos de bifurcación vascular (β₁, β₂, θ).\n"
            "Clic 1: punto en arteria madre\n"
            "Clic 2: ápex de bifurcación\n"
            "Clic 3: punto en rama 1\n"
            "Clic 4: punto en rama 2  →  muestra β₁, β₂, θ"
        )
        self._btn_bifurc.setStyleSheet(_bifurc_style_off)
        self._btn_bifurc.toggled.connect(self._toggle_bifurc_mode)
        bl.addWidget(self._btn_bifurc)

        self._btn_clear_bifurc = QPushButton("Limpiar")
        self._btn_clear_bifurc.setStyleSheet(_plan_clear_bifurc_btn_css())
        self._btn_clear_bifurc.clicked.connect(self._clear_bifurc_measurement)
        bl.addWidget(self._btn_clear_bifurc)
        bl.addStretch()
        vl.addWidget(bifurc_row)

        _mc = "#9B9B9B" if _is_dark() else "#6B6B6B"
        self._lbl_bifurc_result = QLabel(
            f"<small style='color:{_mc}'>β₁ — · β₂ — · θ —</small>"
        )
        self._lbl_bifurc_result.setWordWrap(True)
        vl.addWidget(self._lbl_bifurc_result)

        scroll.setWidget(root)
        page_vl.addWidget(scroll, 1)
        return page

    # ------------------------------------------------------------------ #
    # Cut page (sidebar panel, page 7)                                     #
    # ------------------------------------------------------------------ #

    def _build_cut_page(self) -> QWidget:
        """Build the Box Clip + Plane Cut panel page for the sidebar stack."""
        page = QWidget()
        page_vl = QVBoxLayout(page)
        page_vl.setContentsMargins(0, 0, 0, 0)
        page_vl.setSpacing(0)

        hdr = QLabel("✂  CORTE DE MALLA")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet(_plan_hdr_css())
        self._plan_page_hdrs.append(hdr)
        page_vl.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        root = QWidget()
        vl   = QVBoxLayout(root)
        vl.setContentsMargins(8, 6, 8, 6)
        vl.setSpacing(8)

        # ── Enable toggle ─────────────────────────────────────────────── #
        self._btn_box_clip = QPushButton("Activar corte de caja")
        self._btn_box_clip.setCheckable(True)
        self._btn_box_clip.setMinimumHeight(30)
        self._btn_box_clip.setStyleSheet(_plan_toggle_blue_css())
        self._btn_box_clip.toggled.connect(self._toggle_box_clip)
        vl.addWidget(self._btn_box_clip)

        _mc  = "#9B9B9B" if _is_dark() else "#6B6B6B"
        _grn = "#3fb950" if _is_dark() else "#1B7A2E"
        guide = QLabel(
            f"<small style='color:{_mc}'>"
            "① Ajusta el cubo arrastrando sus asas en la vista 3D "
            "o editando los spinboxes.<br>"
            f"② Pulsa <b style='color:{_grn}'>✂ Cortar malla</b> para aplicar el recorte."
            "</small>"
        )
        guide.setWordWrap(True)
        vl.addWidget(guide)

        # ── Bounds spinboxes ──────────────────────────────────────────── #
        bounds_grp = QGroupBox("Límites del cubo (mm)")
        bf = QFormLayout()
        bf.setLabelAlignment(Qt.AlignRight)
        bf.setVerticalSpacing(4)

        self._clip_spins: dict[str, QDoubleSpinBox] = {}
        for axis, label in (("xmin", "X mín"), ("xmax", "X máx"),
                             ("ymin", "Y mín"), ("ymax", "Y máx"),
                             ("zmin", "Z mín"), ("zmax", "Z máx")):
            s = QDoubleSpinBox()
            s.setRange(-1000.0, 1000.0)
            s.setSingleStep(1.0)
            s.setDecimals(1)
            s.setSuffix(" mm")
            s.valueChanged.connect(self._on_clip_spinbox_changed)
            self._clip_spins[axis] = s
            bf.addRow(f"{label}:", s)

        bounds_grp.setLayout(bf)
        vl.addWidget(bounds_grp)

        # ── Sync helpers ──────────────────────────────────────────────── #
        sync_row = QWidget()
        sr = QHBoxLayout(sync_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(6)

        btn_reset_bounds = QPushButton("↺ Límites originales")
        btn_reset_bounds.setToolTip("Vuelve los límites al bounding box original de la malla")
        btn_reset_bounds.clicked.connect(self._reset_box_bounds)
        sr.addWidget(btn_reset_bounds)

        btn_sync = QPushButton("⟳ Desde caja")
        btn_sync.setToolTip("Lee la posición actual del cubo y actualiza los spinboxes")
        btn_sync.clicked.connect(self._sync_spinboxes_from_widget)
        sr.addWidget(btn_sync)

        vl.addWidget(sync_row)

        # ── CUT button  (principal action) ────────────────────────────── #
        self._btn_cut = QPushButton("✂  Cortar malla")
        self._btn_cut.setMinimumHeight(36)
        self._btn_cut.setEnabled(False)
        self._btn_cut.setToolTip(
            "Aplica el recorte definido por el cubo.\n"
            "La malla queda recortada a los límites X/Y/Z indicados."
        )
        self._btn_cut.setStyleSheet(_plan_cut_btn_css())
        self._btn_cut.clicked.connect(self._apply_cut_now)
        vl.addWidget(self._btn_cut)

        # ── Restore button ────────────────────────────────────────────── #
        self._btn_restore_mesh = QPushButton("↺ Restablecer malla completa")
        self._btn_restore_mesh.setMinimumHeight(28)
        self._btn_restore_mesh.setEnabled(False)
        self._btn_restore_mesh.setStyleSheet(_plan_restore_btn_css())
        self._btn_restore_mesh.clicked.connect(self._restore_full_mesh)
        vl.addWidget(self._btn_restore_mesh)

        self._chk_ghost = QCheckBox("Mostrar contorno de referencia")
        self._chk_ghost.setChecked(True)
        self._chk_ghost.setToolTip(
            "Muestra la malla completa en wireframe a baja opacidad "
            "para tener contexto espacial mientras se ajusta el cubo."
        )
        self._chk_ghost.toggled.connect(self._on_ghost_toggled)
        vl.addWidget(self._chk_ghost)

        self._lbl_clip_status = QLabel("Corte inactivo.")
        self._lbl_clip_status.setStyleSheet(_plan_muted_lbl_css())
        self._lbl_clip_status.setWordWrap(True)
        vl.addWidget(self._lbl_clip_status)

        vl.addStretch()

        # ── Free-plane clip (S-1 — 3D Slicer-inspired) ───────────────── #
        sep_fp = QFrame()
        sep_fp.setFrameShape(QFrame.HLine)
        sep_fp.setStyleSheet(_plan_sep_css())
        self._plan_seps.append(sep_fp)
        vl.addWidget(sep_fp)

        _ttl_c = "#A8B8C6" if _is_dark() else "#4E6678"
        _mc    = "#9B9B9B" if _is_dark() else "#6B6B6B"
        vl.addWidget(QLabel(
            f"<b style='color:{_ttl_c}; font-size:11px;'>✂ Plano de corte libre</b>"
        ))

        self._btn_plane_clip = QPushButton("Activar plano de corte")
        self._btn_plane_clip.setCheckable(True)
        self._btn_plane_clip.setMinimumHeight(30)
        self._btn_plane_clip.setStyleSheet(_plan_toggle_blue_css())
        self._btn_plane_clip.toggled.connect(self._toggle_plane_clip)
        vl.addWidget(self._btn_plane_clip)

        vl.addWidget(QLabel(
            f"<small style='color:{_mc}'>"
            "Arrastra el widget de plano en la vista 3D.<br>"
            "La normal (flecha) indica el lado que se conserva."
            "</small>"
        ))

        fp_btn_row = QWidget()
        fpl = QHBoxLayout(fp_btn_row)
        fpl.setContentsMargins(0, 0, 0, 0)
        fpl.setSpacing(6)

        self._btn_plane_cut = QPushButton("✂  Cortar con plano")
        self._btn_plane_cut.setEnabled(False)
        self._btn_plane_cut.setMinimumHeight(32)
        self._btn_plane_cut.setStyleSheet(_plan_plane_cut_btn_css())
        self._btn_plane_cut.clicked.connect(self._apply_plane_cut)
        fpl.addWidget(self._btn_plane_cut)

        self._btn_flip_plane = QPushButton("⇄ Invertir")
        self._btn_flip_plane.setEnabled(False)
        self._btn_flip_plane.setToolTip("Invierte la normal del plano (cambia qué lado se conserva)")
        self._btn_flip_plane.setStyleSheet(_plan_flip_btn_css())
        self._btn_flip_plane.clicked.connect(self._flip_plane_normal)
        fpl.addWidget(self._btn_flip_plane)
        vl.addWidget(fp_btn_row)

        self._lbl_plane_status = QLabel("Plano inactivo.")
        self._lbl_plane_status.setStyleSheet(_plan_muted_lbl_css())
        self._lbl_plane_status.setWordWrap(True)
        vl.addWidget(self._lbl_plane_status)

        scroll.setWidget(root)
        page_vl.addWidget(scroll, 1)

        # Free-plane state (initialised here so set_vessel_mesh can reference them)
        self._plane_clip_active:  bool                           = False
        self._plane_widget:       vtk.vtkImplicitPlaneWidget2 | None = None
        self._plane_cut_applied:  bool                           = False

        return page

    # ------------------------------------------------------------------ #
    # Icon sidebar + central layout                                        #
    # ------------------------------------------------------------------ #

    def _build_icon_sidebar(self) -> QWidget:
        """
        52 px icon-only sidebar — each button is exclusively checkable.
        Pressing a button shows its matching panel page; pressing again hides it.
        """
        sidebar = QWidget()
        sidebar.setFixedWidth(52)
        sidebar.setObjectName("planSidebar")
        sidebar.setStyleSheet(_plan_sidebar_css())
        self._plan_sidebar = sidebar

        vl = QVBoxLayout(sidebar)
        vl.setContentsMargins(4, 8, 4, 8)
        vl.setSpacing(2)

        # (icon, sidebar_key, tooltip)
        # All icons are BMP Unicode + Variation-Selector-15 → monochrome,
        # colour controlled entirely via the CSS color property.
        from prospective.ui.icons import I as _I
        _BUTTONS = [
            (_I.CLIPS,      "clips",        "Clips quirúrgicos"),
            (_I.FLOW_DIV,   "flow_div",     "Flow Diverters / Stents"),
            (_I.STENT_CL,   "stent_cl",     "Stent en Línea Central"),
            (_I.TRAJECTORY, "trajectory",   "Trayectoria / Pick"),
            (_I.CENTERLINE, "centerline",   "Línea Central"),
            (_I.MEASURE,    "measurements", "Mediciones 3D"),
            (_I.CUT,        "cut",          "Corte de malla"),
            (_I.SETTINGS,   "settings",     "Vista / Overlays"),
        ]

        self._sidebar_btns: dict[str, QPushButton] = {}
        for icon, key, tip in _BUTTONS:
            btn = QPushButton(icon)
            btn.setCheckable(True)
            btn.setFixedSize(44, 44)
            btn.setToolTip(tip)
            btn.toggled.connect(
                lambda checked, k=key: self._on_sidebar_btn(k, checked)
            )
            vl.addWidget(btn)
            self._sidebar_btns[key] = btn

        vl.addStretch()
        return sidebar

    def _on_sidebar_btn(self, key: str, checked: bool) -> None:
        """Toggle the panel page for *key* and keep sidebar buttons exclusive."""
        if checked:
            # Uncheck every other button without re-triggering this slot
            for k, btn in self._sidebar_btns.items():
                if k != key:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
            self._panel_stack.setCurrentIndex(self._sidebar_page_map[key])
            self._panel_stack.setVisible(True)
        else:
            # Hide panel only when no button remains checked
            if not any(b.isChecked() for b in self._sidebar_btns.values()):
                self._panel_stack.setVisible(False)
                self._panel_stack.setCurrentIndex(0)

    def _build_central_layout(self) -> None:
        """
        Assemble the final central widget:
            [icon_sidebar (52 px) | vtk_viewport (stretch) | panel_stack (340 px)]

        Must be called after _build_vtk(), _build_toolbar() and _build_panels().
        """
        sidebar = self._build_icon_sidebar()

        container = QWidget()
        hl = QHBoxLayout(container)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        hl.addWidget(sidebar)
        hl.addWidget(self._vtk_container, 1)   # VTK viewport takes all free space
        hl.addWidget(self._panel_stack)

        self.setCentralWidget(container)

    # ------------------------------------------------------------------ #
    # Signal wiring                                                        #
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self._clip_panel.clip_placed.connect(self._on_clip_placed)
        self._clip_panel.clip_removed.connect(self._on_clip_removed)
        self._clip_panel.clip_transform_changed.connect(self._on_clip_transform)
        self._clip_panel.clip_visibility_changed.connect(self._on_clip_visibility)
        self._clip_panel.trajectory_changed.connect(self._on_trajectory)

        self._stent_panel.stent_placed.connect(self._on_stent_placed)
        self._stent_panel.stent_removed.connect(self._on_stent_removed)
        self._stent_panel.stent_transform_changed.connect(self._on_stent_transform)
        self._stent_panel.stent_visibility_changed.connect(self._on_stent_visibility)
        self._stent_panel.stent_deform_requested.connect(self._start_deform_mode)

        self._centerline_panel.pick_source_requested.connect(
            lambda: self._enter_cl_pick('source'))
        self._centerline_panel.pick_target_requested.connect(
            lambda: self._enter_cl_pick('target'))
        self._centerline_panel.centerline_ready.connect(self._on_centerline_ready)
        self._centerline_panel.centerline_cleared.connect(self._on_centerline_cleared)

        self._measurement_panel.ruler_requested.connect(self._enter_ruler_pick)
        self._measurement_panel.ruler_visibility.connect(self._on_ruler_visibility)
        self._measurement_panel.ruler_deleted.connect(self._on_ruler_deleted)

        self._cl_stent_panel.stent_deployed.connect(self._on_cl_stent_deployed)
        self._cl_stent_panel.stent_retracted.connect(self._on_cl_stent_retracted)

    # ------------------------------------------------------------------ #
    # Theme                                                                #
    # ------------------------------------------------------------------ #

    def apply_theme(self) -> None:
        """Re-apply CSS to all theme-sensitive widgets after a palette change."""
        self._vtk_hdr.setStyleSheet(_plan_vtk_hdr_css())
        self._panel_stack.setStyleSheet(_plan_panel_stack_css())
        self._plan_sidebar.setStyleSheet(_plan_sidebar_css())

        for lbl in self._plan_page_hdrs:
            lbl.setStyleSheet(_plan_hdr_css())
        for lbl in self._plan_page_helps:
            lbl.setStyleSheet(_plan_help_css())
        for btn in self._plan_muted_btns:
            btn.setStyleSheet(_plan_muted_btn_css())
        for btn in self._nudge_btns:
            btn.setStyleSheet(_plan_nudge_btn_css())
        for sep in self._plan_seps:
            sep.setStyleSheet(_plan_sep_css())

        self._btn_pick.setStyleSheet(_plan_pick_btn_css())
        self._btn_auto_place.setStyleSheet(_plan_primary_btn_css())
        self._btn_clip_place.setStyleSheet(_plan_clip_place_btn_css())
        self._btn_box_clip.setStyleSheet(_plan_toggle_blue_css())
        self._btn_plane_clip.setStyleSheet(_plan_toggle_blue_css())
        self._btn_cut.setStyleSheet(_plan_cut_btn_css())
        self._btn_restore_mesh.setStyleSheet(_plan_restore_btn_css())
        self._btn_plane_cut.setStyleSheet(_plan_plane_cut_btn_css())
        self._btn_flip_plane.setStyleSheet(_plan_flip_btn_css())
        self._btn_clear_traj.setStyleSheet(_plan_clear_traj_btn_css())
        self._btn_clear_perf.setStyleSheet(_plan_clear_danger_btn_css())
        self._btn_del_annot.setStyleSheet(_plan_clear_annot_btn_css())
        self._btn_clear_annot.setStyleSheet(_plan_clear_annot_btn_css())
        self._btn_clear_bifurc.setStyleSheet(_plan_clear_bifurc_btn_css())

        # Muted / status labels
        self._lbl_perf_count.setStyleSheet(_plan_muted_lbl_css())
        self._lbl_perf_collision.setStyleSheet(_plan_muted_lbl_css())
        self._lbl_annot_count.setStyleSheet(_plan_muted_lbl_css())
        self._lbl_clip_status.setStyleSheet(_plan_muted_lbl_css())
        self._lbl_plane_status.setStyleSheet(_plan_muted_lbl_css())

        # Cascade to sub-panels with their own inline colours
        self._centerline_panel.apply_theme()

        # Toolbar hint label — rebuild HTML with updated colour
        _hint_clr = "#9B9B9B" if _is_dark() else "#6B6B6B"
        self._lbl_tb_status.setText(
            f"<small style='color:{_hint_clr}'>"
            "Clic izquierdo sobre arteria cuando pick-mode está activo  ·  "
            "Ctrl+Z = deshacer  ·  R = resetear cámara"
            "</small>"
        )

        # Toggle buttons: update styles AND refresh stored CSS strings
        self._perf_style_on  = _plan_perf_on_css()
        self._perf_style_off = _plan_perf_off_css()
        self._btn_perf.setStyleSheet(
            self._perf_style_on if self._btn_perf.isChecked() else self._perf_style_off
        )

        self._annot_style_on  = _plan_annot_on_css()
        self._annot_style_off = _plan_annot_off_css()
        self._btn_annot.setStyleSheet(
            self._annot_style_on if self._btn_annot.isChecked() else self._annot_style_off
        )

        self._bifurc_style_on  = _plan_bifurc_on_css()
        self._bifurc_style_off = _plan_bifurc_off_css()
        self._btn_bifurc.setStyleSheet(
            self._bifurc_style_on if self._btn_bifurc.isChecked() else self._bifurc_style_off
        )

        # Cascade to device panel embedded in this window
        if hasattr(self, "_stent_panel"):
            self._stent_panel.apply_theme()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def clear_vessel_mesh(self) -> None:
        """Hide the vessel mesh and mark the window as unpopulated.

        Called by MainWindow._on_loaded() when a new DICOM case is opened so
        the embedded tab does not show stale geometry from the previous case
        until fresh segmentation is performed.
        """
        self._mesh_actor.VisibilityOff()
        self._aneurysm_actor.VisibilityOff()
        self._vessel_poly        = None
        self._kd_locator         = None
        self._vessel_normals_data = None
        self._obb_tree           = None
        self._mesh_loaded        = False
        self._render()

    def set_vessel_mesh(self, poly_data: vtk.vtkPolyData) -> None:
        self._mesh_mapper.SetInputData(poly_data)
        self._mesh_mapper.Update()
        self._mesh_actor.VisibilityOn()

        # Store raw bounds for box-clip reset
        b = poly_data.GetBounds()   # (xmin, xmax, ymin, ymax, zmin, zmax)
        self._mesh_raw_bounds = b
        self._populate_clip_spinboxes(b)
        if self._box_clip_active:
            # Rebuild ghost with new mesh and reapply planes
            self._hide_ghost_mesh()
            self._show_ghost_mesh()
            self._apply_box_clip_planes(list(b))

        # Picker: only pick from the vessel surface
        self._cell_picker.AddPickList(self._mesh_actor)
        self._cell_picker.PickFromListOn()

        # Store raw polydata for PCA-based vessel-axis estimation
        self._vessel_poly = poly_data
        self._kd_locator  = vtk.vtkKdTreePointLocator()
        self._kd_locator.SetDataSet(poly_data)
        self._kd_locator.BuildLocator()

        # Forward to centreline panel
        self._centerline_panel.set_vessel_mesh(poly_data)

        # Precompute cell normals (for lumen-centre estimation)
        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(poly_data)
        nf.ComputeCellNormalsOn()
        nf.ComputePointNormalsOff()
        nf.SplittingOff()
        nf.Update()
        self._vessel_normals_data = nf.GetOutput()

        # OBB tree for ray-casting to the opposite vessel wall
        self._obb_tree = vtk.vtkOBBTree()
        self._obb_tree.SetDataSet(poly_data)
        self._obb_tree.BuildLocator()

        # Build a cell locator for viability checks (kept alive on self)
        self._viab_locator = vtk.vtkCellLocator()
        self._viab_locator.SetDataSet(poly_data)
        self._viab_locator.BuildLocator()

        self._mesh_loaded = True

        # If the window is already visible AND VTK is initialised, reset the
        # camera now (deferred so Qt has processed any pending resize events).
        # If VTK is not yet initialised, showEvent → _init_and_reset handles it.
        # Calling _render() before Initialize() on Windows creates a broken GL
        # context, leaving the viewport permanently blank — so we avoid it here.
        if self.isVisible() and getattr(self, "_vtk_initialized", False):
            QTimer.singleShot(0, self._reset_camera)

        self._clip_panel.set_vessel_mesh(poly_data)
        self._stent_panel.set_vessel_mesh(poly_data)

    def set_aneurysm(self, poly_data: vtk.vtkPolyData | None) -> None:
        if poly_data is None:
            self._aneurysm_actor.VisibilityOff()
        else:
            self._aneurysm_mapper.SetInputData(poly_data)
            self._aneurysm_mapper.Update()
            self._aneurysm_actor.VisibilityOn()
        self._render()

    def set_aneurysm_visible(self, visible: bool) -> None:
        """Show or hide the aneurysm candidate highlight without clearing the mesh data."""
        self._aneurysm_actor.SetVisibility(visible)
        self._render()

    def highlight_candidate(self, candidate) -> None:
        """
        Show a 3D sphere + billboard label at the candidate centroid and
        focus the camera on it.  Pass None to clear the marker.

        Parameters
        ----------
        candidate : AneurysmCandidate | None
        """
        self._clear_candidate_marker()

        if candidate is None:
            self._aneurysm_actor.VisibilityOff()
            self._render()
            return

        # ── 1. Show mesh overlay ──────────────────────────────────────── #
        self.set_aneurysm(candidate.poly_data)

        cx, cy, cz = candidate.centroid
        r = max(candidate.radius_mm, 1.0)

        # ── 2. Pulsing sphere at centroid ─────────────────────────────── #
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(cx, cy, cz)
        sphere.SetRadius(r * 1.25)       # slightly larger than the candidate
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(24)
        sphere.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        mapper.ScalarVisibilityOff()

        self._cand_sphere_actor = vtk.vtkActor()
        self._cand_sphere_actor.SetMapper(mapper)
        prop = self._cand_sphere_actor.GetProperty()
        prop.SetColor(1.0, 0.85, 0.0)   # yellow
        prop.SetOpacity(0.25)
        prop.SetRepresentationToSurface()
        prop.SetAmbient(0.4)
        prop.SetDiffuse(0.6)
        self._renderer.AddActor(self._cand_sphere_actor)

        # ── 3. Billboard text label ────────────────────────────────────── #
        # Sac interpretation from positive Gaussian curvature fraction
        pgf = getattr(candidate, "positive_gauss_frac", 0.5)
        sac_tag = "saco" if pgf > 0.80 else ("mixto" if pgf > 0.55 else "bifurcación")

        label = vtk.vtkBillboardTextActor3D()
        label.SetInput(
            f"#{candidate.index}  Ø {candidate.diameter_mm:.1f} mm\n"
            f"Score: {candidate.score * 100:.0f}%  [{sac_tag}]"
        )
        label.SetPosition(cx, cy, cz)
        label.SetDisplayOffset(12, 8)
        tp = label.GetTextProperty()
        tp.SetFontSize(15)
        tp.SetColor(1.0, 0.95, 0.3)       # yellow-white
        tp.SetBold(True)
        tp.SetShadow(True)
        tp.SetBackgroundColor(0.08, 0.06, 0.02)
        tp.SetBackgroundOpacity(0.75)
        self._renderer.AddActor(label)
        self._cand_label_actor = label

        # ── 4. Focus camera on the candidate ─────────────────────────── #
        bounds = candidate.poly_data.GetBounds()
        # Expand bounds by 50% to include surrounding vessel context
        margin = max(r * 3.0, 10.0)
        focus_bounds = (
            bounds[0] - margin, bounds[1] + margin,
            bounds[2] - margin, bounds[3] + margin,
            bounds[4] - margin, bounds[5] + margin,
        )
        self._renderer.ResetCamera(focus_bounds)
        self._render()

    def _clear_candidate_marker(self) -> None:
        """Remove previous candidate sphere + label from the scene."""
        if self._cand_sphere_actor is not None:
            self._renderer.RemoveActor(self._cand_sphere_actor)
            self._cand_sphere_actor = None
        if self._cand_label_actor is not None:
            self._renderer.RemoveActor(self._cand_label_actor)
            self._cand_label_actor = None

    def set_neck_diameter(self, neck_mm: float) -> None:
        self._clip_panel.set_neck_diameter(neck_mm)

    def set_neck_data_for_sizing(
        self, neck_length_mm: float, parent_artery_mm: float = 0.0
    ) -> None:
        """Forward morphometric neck/artery data to the stent sizing widget."""
        self._stent_panel.set_neck_data(neck_length_mm, parent_artery_mm)

    def set_aneurysm_centroid(self, x: float, y: float, z: float) -> None:
        self._clip_panel.set_aneurysm_centroid(x, y, z)
        self._stent_panel.set_aneurysm_centroid(x, y, z)

    def set_morpho_result(
        self,
        result,
        aneurysm_poly: vtk.vtkPolyData,
    ) -> None:
        """
        Store a morphometric result and enable the overlay toggle button.

        Called by MainWindow after analysis completes.  The overlay is NOT
        shown automatically — the user must toggle it via the toolbar button.
        """
        self._morpho_result = result
        self._morpho_poly   = aneurysm_poly
        # Clear any previous overlay
        self._remove_morpho_overlay()
        self._act_morpho_overlay.setEnabled(True)
        self._act_morpho_overlay.setChecked(False)

    def _toggle_morpho_overlay(self, show: bool) -> None:
        if show:
            if self._morpho_overlay is None:
                self._build_morpho_overlay()
            if self._morpho_overlay is not None:
                self._morpho_overlay.set_visible(True)
        else:
            if self._morpho_overlay is not None:
                self._morpho_overlay.set_visible(False)
        self._render()

    def _build_morpho_overlay(self) -> None:
        if self._morpho_result is None or self._morpho_poly is None:
            return
        try:
            from prospective.rendering.morpho_overlay import build_morpho_overlay
            self._morpho_overlay = build_morpho_overlay(
                self._morpho_result, self._morpho_poly
            )
            for actor in self._morpho_overlay:
                self._renderer.AddActor(actor)
        except Exception:
            logger.exception("Failed to build morpho overlay")
            self._morpho_overlay = None

    def _remove_morpho_overlay(self) -> None:
        if self._morpho_overlay is not None:
            for actor in self._morpho_overlay:
                self._renderer.RemoveActor(actor)
            self._morpho_overlay = None

    # ------------------------------------------------------------------ #
    # Perforator risk overlay — public API                                 #
    # ------------------------------------------------------------------ #

    def set_perforator_overlay(self, actors: list) -> None:
        """
        Replace any existing perforator-risk overlay with *actors*.

        Called by MainWindow when :class:`PerforatorRiskPanel` emits
        ``overlay_ready``.
        """
        self.clear_perforator_overlay()
        for actor in actors:
            self._renderer.AddActor(actor)
        self._perf_overlay_actors = list(actors)
        self._render()

    def clear_perforator_overlay(self) -> None:
        """Remove all perforator risk actors from the renderer."""
        for actor in self._perf_overlay_actors:
            self._renderer.RemoveActor(actor)
        self._perf_overlay_actors.clear()
        self._render()

    def request_render(self) -> None:
        """Public render trigger — used by external panels after actor visibility changes."""
        self._render()

    # ------------------------------------------------------------------ #
    # Toolbar handlers                                                     #
    # ------------------------------------------------------------------ #

    def _reset_camera(self) -> None:
        # Sync the VTK render window to the actual widget size before fitting
        # the camera.  When called from showEvent (or a short-delay timer just
        # after a tab is shown), Qt may not have sent a resizeEvent to the VTK
        # interactor yet.  Without this, ResetCamera() uses stale dimensions,
        # computes the wrong frustum, and the mesh appears outside the viewport
        # → black screen even though the geometry is correctly in the pipeline.
        rw = self._iren.GetRenderWindow()
        w = self._iren.width()
        h = self._iren.height()
        if w > 0 and h > 0:
            rw.SetSize(w, h)
        self._renderer.ResetCamera()
        self._render()

    def _toggle_mesh_opacity(self, opaque: bool) -> None:
        self._mesh_actor.GetProperty().SetOpacity(0.90 if opaque else 0.42)
        self._render()

    # ================================================================== #
    # Box Clip                                                             #
    # ================================================================== #

    def _populate_clip_spinboxes(self, bounds) -> None:
        """Fill spinboxes silently (no clip callback triggered)."""
        keys = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        for key, val in zip(keys, bounds):
            s = self._clip_spins[key]
            s.blockSignals(True)
            s.setValue(float(val))
            s.blockSignals(False)

    def _current_clip_bounds(self) -> list[float]:
        return [self._clip_spins[k].value()
                for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")]

    # ── Toggle ────────────────────────────────────────────────────────── #

    def _toggle_box_clip(self, active: bool) -> None:
        """Show/hide the box widget and enable the Cut button."""
        self._box_clip_active = active
        if active:
            self._btn_box_clip.setText("Desactivar corte de caja")
            self._init_box_widget()
            self._show_ghost_mesh()
            self._btn_cut.setEnabled(True)
            self._lbl_clip_status.setText(
                "Ajusta el cubo y pulsa ✂ Cortar malla."
            )
        else:
            self._btn_box_clip.setText("Activar corte de caja")
            self._destroy_box_widget()
            self._hide_ghost_mesh()
            self._mesh_mapper.RemoveAllClippingPlanes()
            self._btn_cut.setEnabled(False)
            if not self._box_cut_applied:
                self._lbl_clip_status.setText("Corte inactivo.")
        self._render()

    # ── Box widget ────────────────────────────────────────────────────── #

    def _init_box_widget(self) -> None:
        """Create and show the interactive VTK box widget."""
        if self._vessel_poly is None:
            return
        rep = vtk.vtkBoxRepresentation()
        rep.SetPlaceFactor(1.0)
        rep.PlaceWidget(self._current_clip_bounds())
        rep.HandlesOn()

        self._box_widget = vtk.vtkBoxWidget2()
        self._box_widget.SetInteractor(self._iren)
        self._box_widget.SetRepresentation(rep)
        self._box_widget.RotationEnabledOff()
        self._box_widget.ScalingEnabledOn()
        self._box_widget.TranslationEnabledOn()
        self._box_widget.On()
        self._box_widget.AddObserver(
            "InteractionEvent", self._on_box_widget_interaction
        )

    def _destroy_box_widget(self) -> None:
        if self._box_widget is not None:
            self._box_widget.Off()
            self._box_widget = None

    # ── Ghost mesh ────────────────────────────────────────────────────── #

    def _show_ghost_mesh(self) -> None:
        """Add a wireframe ghost actor showing the full vessel extent."""
        if self._vessel_poly is None:
            return
        if self._ghost_actor is not None:
            self._ghost_actor.VisibilityOn()
            return
        ghost_mapper = vtk.vtkPolyDataMapper()
        ghost_mapper.SetInputData(self._vessel_poly)
        ghost_mapper.ScalarVisibilityOff()

        self._ghost_actor = vtk.vtkActor()
        self._ghost_actor.SetMapper(ghost_mapper)
        p = self._ghost_actor.GetProperty()
        p.SetColor(0.30, 0.60, 0.95)       # same blue as main mesh
        p.SetOpacity(0.12)
        p.SetRepresentationToWireframe()
        p.SetLineWidth(0.6)
        self._renderer.AddActor(self._ghost_actor)

    def _hide_ghost_mesh(self) -> None:
        if self._ghost_actor is not None:
            self._renderer.RemoveActor(self._ghost_actor)
            self._ghost_actor = None

    def _on_ghost_toggled(self, checked: bool) -> None:
        if self._ghost_actor is not None:
            self._ghost_actor.SetVisibility(checked)
            self._render()

    # ── Core clip logic ───────────────────────────────────────────────── #

    def _on_box_widget_interaction(self, obj, _event) -> None:
        """Sync spinboxes and clip planes when user drags the box handles."""
        # VTK Python binding: GetBounds() returns a tuple — no output arg
        bounds = list(obj.GetRepresentation().GetBounds())
        self._populate_clip_spinboxes(bounds)
        self._apply_box_clip_planes(bounds)

    def _on_clip_spinbox_changed(self) -> None:
        """Called when user edits a bound spinbox manually."""
        if not self._box_clip_active:
            return
        bounds = self._current_clip_bounds()
        # Enforce min ≤ max
        if bounds[0] > bounds[1]: bounds[1] = bounds[0]
        if bounds[2] > bounds[3]: bounds[3] = bounds[2]
        if bounds[4] > bounds[5]: bounds[5] = bounds[4]
        self._apply_box_clip_planes(bounds)
        if self._box_widget is not None:
            self._box_widget.GetRepresentation().PlaceWidget(bounds)
        self._render()

    def _apply_box_clip_planes(self, bounds: list[float]) -> None:
        """Apply 6 mapper-level clipping planes to clip the rendered mesh.

        Each plane clips the side where ``n · (x − origin) < 0``.  Two
        opposing planes per axis produce a slab, and all six together produce
        a box.  The mesh data is **never modified** — only the GPU rendering
        is clipped in real time.

        Plane layout (normal points INWARD so the interior is the positive
        half-space, which survives clipping):

            -X face  origin=(xmin, 0, 0)  normal=(+1, 0, 0)  → keeps x ≥ xmin
            +X face  origin=(xmax, 0, 0)  normal=(−1, 0, 0)  → keeps x ≤ xmax
            -Y face  origin=(0, ymin, 0)  normal=(0, +1, 0)  → keeps y ≥ ymin
            +Y face  origin=(0, ymax, 0)  normal=(0, −1, 0)  → keeps y ≤ ymax
            -Z face  origin=(0, 0, zmin)  normal=(0, 0, +1)  → keeps z ≥ zmin
            +Z face  origin=(0, 0, zmax)  normal=(0, 0, −1)  → keeps z ≤ zmax
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        plane_defs = [
            ([xmin, 0,    0   ], [+1,  0,  0]),
            ([xmax, 0,    0   ], [-1,  0,  0]),
            ([0,    ymin, 0   ], [ 0, +1,  0]),
            ([0,    ymax, 0   ], [ 0, -1,  0]),
            ([0,    0,    zmin], [ 0,  0, +1]),
            ([0,    0,    zmax], [ 0,  0, -1]),
        ]

        self._mesh_mapper.RemoveAllClippingPlanes()
        for origin, normal in plane_defs:
            plane = vtk.vtkPlane()
            plane.SetOrigin(*origin)
            plane.SetNormal(*normal)
            self._mesh_mapper.AddClippingPlane(plane)

        self._lbl_clip_status.setText(
            f"Corte activo  ·  "
            f"X [{xmin:.1f} → {xmax:.1f}]  "
            f"Y [{ymin:.1f} → {ymax:.1f}]  "
            f"Z [{zmin:.1f} → {zmax:.1f}] mm"
        )
        self._render()

    # ── Cut / Restore ─────────────────────────────────────────────────── #

    def _apply_cut_now(self) -> None:
        """Apply the box cut using 6 chained vtkClipPolyData filters.

        Each filter uses a vtkPlane with an **inward** normal (pointing into
        the box interior) and ``InsideOutOff`` (default):
          - ``InsideOutOff`` keeps where ``f(x) = n·(x-origin) > 0``
          - With inward normal ``[+1,0,0]`` at ``[xmin,0,0]``: f = x-xmin > 0 → x > xmin ✓
        Six filters chained together keep only the region strictly inside the
        box on all six sides.
        """
        if self._vessel_poly is None:
            return

        xmin, xmax, ymin, ymax, zmin, zmax = self._current_clip_bounds()

        # (origin on face, INWARD normal pointing into box interior)
        # InsideOutOff keeps where f = n·(x-origin) > 0, i.e. inside half-space
        plane_configs = [
            ([xmin, 0,    0   ], [+1,  0,  0]),   # -X face, inward → keeps x > xmin
            ([xmax, 0,    0   ], [-1,  0,  0]),   # +X face, inward → keeps x < xmax
            ([0,    ymin, 0   ], [ 0, +1,  0]),   # -Y face, inward → keeps y > ymin
            ([0,    ymax, 0   ], [ 0, -1,  0]),   # +Y face, inward → keeps y < ymax
            ([0,    0,    zmin], [ 0,  0, +1]),   # -Z face, inward → keeps z > zmin
            ([0,    0,    zmax], [ 0,  0, -1]),   # +Z face, inward → keeps z < zmax
        ]

        data = self._vessel_poly
        for origin, normal in plane_configs:
            plane = vtk.vtkPlane()
            plane.SetOrigin(*origin)
            plane.SetNormal(*normal)
            clipper = vtk.vtkClipPolyData()
            clipper.SetInputData(data)
            clipper.SetClipFunction(plane)
            clipper.InsideOutOff()   # keep where f = n·(x-origin) > 0 (positive side)
            clipper.Update()
            data = clipper.GetOutput()

        # Apply clipped geometry to mapper (remove GPU planes — no longer needed)
        self._mesh_mapper.RemoveAllClippingPlanes()
        self._mesh_mapper.SetInputData(data)
        self._mesh_mapper.Update()

        # Hide the interactive box and ghost — the cut is done
        self._destroy_box_widget()
        self._hide_ghost_mesh()
        self._box_clip_active = False
        # Reset toggle button without re-triggering the toggled signal
        self._btn_box_clip.blockSignals(True)
        self._btn_box_clip.setChecked(False)
        self._btn_box_clip.setText("Activar corte de caja")
        self._btn_box_clip.blockSignals(False)
        self._btn_cut.setEnabled(False)

        self._box_cut_applied = True
        self._btn_restore_mesh.setEnabled(True)

        n = data.GetNumberOfPoints()
        self._lbl_clip_status.setText(
            f"✂ Corte aplicado — {n:,} vértices · "
            f"X[{xmin:.1f}→{xmax:.1f}] "
            f"Y[{ymin:.1f}→{ymax:.1f}] "
            f"Z[{zmin:.1f}→{zmax:.1f}] mm"
        )
        self._render()

    def _restore_full_mesh(self) -> None:
        """Revert the mesh actor to the full original polydata."""
        if self._vessel_poly is None:
            return
        self._mesh_mapper.RemoveAllClippingPlanes()
        self._mesh_mapper.SetInputData(self._vessel_poly)
        self._mesh_mapper.Update()
        self._box_cut_applied = False
        self._btn_restore_mesh.setEnabled(False)
        self._lbl_clip_status.setText(
            "Malla completa restaurada. Ajusta el cubo y corta de nuevo."
        )
        self._render()

    # ================================================================== #
    # Free-plane clip (S-1 — 3D Slicer-inspired)                          #
    # ================================================================== #

    def _toggle_plane_clip(self, active: bool) -> None:
        self._plane_clip_active = active
        if active:
            self._btn_plane_clip.setText("Desactivar plano de corte")
            self._init_plane_widget()
            self._btn_plane_cut.setEnabled(True)
            self._btn_flip_plane.setEnabled(True)
            self._lbl_plane_status.setText("Arrastra el plano y pulsa ✂ Cortar.")
        else:
            self._btn_plane_clip.setText("Activar plano de corte")
            self._destroy_plane_widget()
            self._mesh_mapper.RemoveAllClippingPlanes()
            self._btn_plane_cut.setEnabled(False)
            self._btn_flip_plane.setEnabled(False)
            if not self._plane_cut_applied:
                self._lbl_plane_status.setText("Plano inactivo.")
        self._render()

    def _init_plane_widget(self) -> None:
        if self._vessel_poly is None:
            return
        b = list(self._mesh_raw_bounds)
        cx = (b[0] + b[1]) / 2.0
        cy = (b[2] + b[3]) / 2.0
        cz = (b[4] + b[5]) / 2.0

        rep = vtk.vtkImplicitPlaneRepresentation()
        rep.SetPlaceFactor(1.2)
        rep.PlaceWidget(b)
        rep.SetOrigin(cx, cy, cz)
        rep.SetNormal(0.0, 0.0, 1.0)   # start with axial plane
        rep.DrawPlaneOff()              # hide plane polygon, show only widget handles
        rep.OutlineTranslationOff()

        self._plane_widget = vtk.vtkImplicitPlaneWidget2()
        self._plane_widget.SetInteractor(self._iren)
        self._plane_widget.SetRepresentation(rep)
        self._plane_widget.On()
        self._plane_widget.AddObserver(
            "InteractionEvent", self._on_plane_widget_interaction
        )
        self._on_plane_widget_interaction(self._plane_widget, None)

    def _destroy_plane_widget(self) -> None:
        if self._plane_widget is not None:
            self._plane_widget.Off()
            self._plane_widget = None

    def _on_plane_widget_interaction(self, obj, _event) -> None:
        """Live-preview: apply mapper clip plane as the widget is dragged."""
        if self._plane_widget is None:
            return
        rep   = self._plane_widget.GetRepresentation()
        plane = vtk.vtkPlane()
        rep.GetUnderlyingPlane(plane)
        origin = plane.GetOrigin()
        normal = plane.GetNormal()
        self._mesh_mapper.RemoveAllClippingPlanes()
        self._mesh_mapper.AddClippingPlane(plane)
        self._lbl_plane_status.setText(
            f"Plano activo · n=({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
        )
        self._render()

    def _flip_plane_normal(self) -> None:
        """Reverse the plane normal so the opposite half is kept."""
        if self._plane_widget is None:
            return
        rep    = self._plane_widget.GetRepresentation()
        plane  = vtk.vtkPlane()
        rep.GetUnderlyingPlane(plane)
        n      = plane.GetNormal()
        rep.SetNormal(-n[0], -n[1], -n[2])
        self._on_plane_widget_interaction(self._plane_widget, None)

    def _apply_plane_cut(self) -> None:
        """Permanently clip the mesh with the current plane using vtkClipPolyData."""
        if self._vessel_poly is None or self._plane_widget is None:
            return
        rep   = self._plane_widget.GetRepresentation()
        plane = vtk.vtkPlane()
        rep.GetUnderlyingPlane(plane)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(self._vessel_poly)
        clipper.SetClipFunction(plane)
        clipper.InsideOutOff()   # keep where f = n·(x-o) > 0 (same side as normal)
        clipper.Update()
        data = clipper.GetOutput()

        self._mesh_mapper.RemoveAllClippingPlanes()
        self._mesh_mapper.SetInputData(data)
        self._mesh_mapper.Update()

        # Hide widget after cut
        self._destroy_plane_widget()
        self._plane_clip_active = False
        self._btn_plane_clip.blockSignals(True)
        self._btn_plane_clip.setChecked(False)
        self._btn_plane_clip.setText("Activar plano de corte")
        self._btn_plane_clip.blockSignals(False)
        self._btn_plane_cut.setEnabled(False)
        self._btn_flip_plane.setEnabled(False)

        self._plane_cut_applied = True
        n = data.GetNumberOfPoints()
        normal = plane.GetNormal()
        self._lbl_plane_status.setText(
            f"✂ Corte planar aplicado — {n:,} vértices · "
            f"n=({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
        )
        self._render()

    # ── Utility ───────────────────────────────────────────────────────── #

    def _reset_box_bounds(self) -> None:
        """Restore clip bounds to the full mesh bounding box."""
        b = list(self._mesh_raw_bounds)
        self._populate_clip_spinboxes(b)
        if self._box_widget is not None:
            self._box_widget.GetRepresentation().PlaceWidget(b)
        if self._box_clip_active:
            self._apply_box_clip_planes(b)
        self._render()

    def _sync_spinboxes_from_widget(self) -> None:
        """Read current box-widget world bounds → update spinboxes and clip."""
        if self._box_widget is None:
            return
        bounds = list(self._box_widget.GetRepresentation().GetBounds())
        self._populate_clip_spinboxes(bounds)
        if self._box_clip_active:
            self._apply_box_clip_planes(bounds)

    # ================================================================== #
    # Trajectory picking                                                   #
    # ================================================================== #

    def eventFilter(self, obj, event) -> bool:
        """Qt event filter installed on self._iren.

        When pick mode is active, intercepts left-button press events *before*
        VTK processes them.  Returning True consumes the event so VTK never
        sees it → camera never rotates.  The actual pick is deferred via
        QTimer so it runs outside Qt's event dispatch (avoids reentrant Render).
        """
        from PyQt5.QtCore import QEvent
        from PyQt5.QtCore import Qt as _Qt

        if (
            obj is self._iren
            and event.type() == QEvent.MouseButtonPress
            and event.button() == _Qt.LeftButton
        ):
            # Scale logical → physical pixels (fixes HiDPI offset on Windows)
            dpr = self._iren.devicePixelRatioF()
            x   = round(event.x() * dpr)
            y   = round((self._iren.height() - 1 - event.y()) * dpr)

            if self._ruler_mode:
                QTimer.singleShot(0, lambda: self._do_ruler_pick(x, y))
                return True

            if self._cl_mode is not None:
                QTimer.singleShot(0, lambda: self._do_cl_pick(x, y))
                return True

            if self._pick_mode:
                QTimer.singleShot(0, lambda: self._do_pick(x, y))
                return True   # consume → VTK never rotates camera

            if self._perf_mode:
                QTimer.singleShot(0, lambda: self._do_perf_pick(x, y))
                return True

            if self._annot_mode:
                QTimer.singleShot(0, lambda: self._do_annot_pick(x, y))
                return True

            if self._bifurc_mode:
                QTimer.singleShot(0, lambda: self._do_bifurc_pick(x, y))
                return True

        return super().eventFilter(obj, event)

    def _toggle_pick_mode(self, active: bool) -> None:
        self._pick_mode = active

        if active and self._perf_mode:
            self._btn_perf.setChecked(False)
            self._toggle_perf_mode(False)
        if active and self._annot_mode:
            self._btn_annot.setChecked(False)
            self._toggle_annot_mode(False)
        if active and self._bifurc_mode:
            self._btn_bifurc.setChecked(False)
            self._toggle_bifurc_mode(False)

        if active:
            # Auto-reveal the trajectory panel page in the sidebar
            if hasattr(self, "_sidebar_btns"):
                traj_btn = self._sidebar_btns.get("trajectory")
                if traj_btn and not traj_btn.isChecked():
                    traj_btn.setChecked(True)   # triggers _on_sidebar_btn → shows page 4

            self._btn_pick.setText("Pick ACTIVO — clic sobre arteria")
            self._pick_status.setStyleSheet(
                "color:#69f0ae; font-size:10px; font-weight:bold;"
            )
            self._pick_status.setText(
                "MODO PICK ACTIVO  —  Haga clic sobre la superficie de la arteria  "
                "(la rotación de cámara está suspendida mientras pick está activo)"
            )
        else:
            self._btn_pick.setText("Activar modo pick")
            self._pick_status.setStyleSheet(_plan_muted_lbl_css())
            self._update_pick_status()

    def _do_pick(self, x: int, y: int) -> None:
        """Perform the pick outside Qt's event loop so Render() works normally."""
        self._cell_picker.Pick(x, y, 0, self._renderer)
        cell_id = self._cell_picker.GetCellId()
        if cell_id >= 0:
            surface_pt = np.array(self._cell_picker.GetPickPosition())
            center_pt, diameter = self._find_lumen_center(surface_pt, cell_id)
            self._add_pick_point(tuple(center_pt), diameter)

    def _find_lumen_center(
        self, surface_pt: np.ndarray, cell_id: int
    ) -> tuple[np.ndarray, float]:
        """
        Project a surface pick-point to the vessel lumen centre.

        Returns
        -------
        center : ndarray  — lumen centre in world mm
        diameter : float  — vessel diameter at that point (mm); 0 if unknown

        Strategy:
          1. Look up the outward cell normal at the picked cell.
          2. Cast a ray inward through the mesh to find the opposite wall.
          3. Return the midpoint (= centre) and the wall-to-wall distance (= Ø).
          4. Fall back to surface_pt − normal × (stent_r), diameter = 0.
        """
        if (
            self._vessel_normals_data is None
            or self._obb_tree is None
            or cell_id < 0
        ):
            return surface_pt, 0.0

        normals = self._vessel_normals_data.GetCellData().GetNormals()
        if normals is None:
            return surface_pt, 0.0

        normal = np.array(normals.GetTuple3(cell_id))
        norm   = np.linalg.norm(normal)
        if norm < 1e-9:
            return surface_pt, 0.0
        normal /= norm

        # Ray: start just outside the surface, end 30 mm inward.
        # Origin is 0.5 mm outside so the near-wall re-entry is the FIRST hit
        # (distance ≈ 0.5 mm from ray_start).  We skip it and use the first
        # hit that is clearly past the near wall (> 1.5 mm) as the opposite wall.
        # Using the *last* hit was wrong for branching geometry: it could land
        # on a completely different vessel, placing the centre outside the lumen.
        ray_origin_np = surface_pt + normal * 0.5
        ray_start     = ray_origin_np.tolist()
        ray_end       = (surface_pt - normal * 30.0).tolist()

        pts      = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()
        self._obb_tree.IntersectWithLine(ray_start, ray_end, pts, cell_ids)

        n_hits = pts.GetNumberOfPoints()
        if n_hits >= 1:
            # Hits are sorted by distance from ray_start (nearest first).
            # Find the first hit that is past the near-wall re-entry (> 1.5 mm).
            opposite: np.ndarray | None = None
            for i in range(n_hits):
                candidate = np.array(pts.GetPoint(i))
                if float(np.linalg.norm(candidate - ray_origin_np)) > 1.5:
                    opposite = candidate
                    break

            if opposite is not None:
                center   = (surface_pt + opposite) / 2.0
                diameter = float(np.linalg.norm(opposite - surface_pt))
                # Sanity: cerebral vessels are 1–20 mm; reject out-of-range hits
                if 1.0 <= diameter <= 20.0:
                    return center, diameter

        # Fallback: push inward by the selected stent's radius
        spec = STENT_CATALOGUE[self._traj_stent_combo.currentIndex()]
        return surface_pt - normal * (spec.diameter_mm / 2.0), 0.0

    def _add_pick_point(self, pos: tuple, diameter: float = 0.0) -> None:
        self._pick_points.append(tuple(pos))
        self._pick_diameters.append(diameter)

        # Small yellow sphere at the picked location
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.2)
        sphere.SetPhiResolution(12)
        sphere.SetThetaResolution(12)
        sphere.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.85, 0.0)
        actor.GetProperty().SetAmbient(0.6)
        self._pick_pt_actors.append(actor)
        self._renderer.AddActor(actor)

        # Billboard text label showing vessel diameter at this point
        if diameter > 0:
            label = vtk.vtkBillboardTextActor3D()
            label.SetInput(f"Ø {diameter:.1f} mm")
            label.SetPosition(pos[0], pos[1], pos[2])
            label.SetDisplayOffset(8, 8)   # pixels right-up from anchor
            tp = label.GetTextProperty()
            tp.SetFontSize(13)
            tp.SetColor(1.0, 0.85, 0.0)
            tp.SetBold(True)
            tp.SetShadow(True)
            tp.SetShadowOffset(1, -1)
            self._pick_diam_actors.append(label)
            self._renderer.AddActor(label)
        else:
            self._pick_diam_actors.append(None)

        self._rebuild_preview()
        self._update_pick_status()

    def _undo_last_point(self) -> None:
        if not self._pick_points:
            return
        self._pick_points.pop()
        if self._pick_diameters:
            self._pick_diameters.pop()
        actor = self._pick_pt_actors.pop()
        self._renderer.RemoveActor(actor)
        if self._pick_diam_actors:
            lbl = self._pick_diam_actors.pop()
            if lbl is not None:
                self._renderer.RemoveActor(lbl)
        self._rebuild_preview()
        self._update_pick_status()

    def _clear_trajectory(self) -> None:
        for actor in self._pick_pt_actors:
            self._renderer.RemoveActor(actor)
        self._pick_pt_actors.clear()
        self._pick_points.clear()
        self._pick_diameters.clear()
        for lbl in self._pick_diam_actors:
            if lbl is not None:
                self._renderer.RemoveActor(lbl)
        self._pick_diam_actors.clear()
        self._remove_preview()
        self._btn_auto_place.setEnabled(False)
        self._btn_auto_place.setText("Colocar stents en trayectoria")
        self._btn_clip_place.setEnabled(False)
        self._deform_target_idx = None
        self._pick_status.setStyleSheet(_plan_muted_lbl_css())
        self._pick_status.setText(
            "Trayectoria borrada. Active el modo pick y haga clic sobre la arteria."
        )
        # Reset nudge state
        self._nudge_idx          = None
        self._nudge_pos          = None
        self._nudge_axis         = None
        self._nudge_perp1        = None
        self._nudge_perp2        = None
        self._nudge_total_offset = 0.0
        self._nudge_widget.setVisible(False)
        self._render()

    def _on_stent_combo_changed(self) -> None:
        """Called when stent type or overlap changes — rebuild preview."""
        self._rebuild_preview()
        self._update_pick_status()

    # ================================================================== #
    # Stent-tube preview + viability                                       #
    # ================================================================== #

    def _rebuild_preview(self) -> None:
        """
        Replace the trajectory preview with a single tube whose radius equals
        the selected stent's nominal diameter / 2.  Colour depends on vessel
        clearance:  green  = fits · orange = tight · red = too wide.

        Also updates the vessel curvature estimate in StentPanel (SC-1/SC-2).
        """
        self._remove_preview()

        n = len(self._pick_points)
        if n < 1:
            self._render()
            return

        # SC-1: update curvature estimate in sizing panel whenever trajectory changes
        if n >= 2:
            radius_mm = self._estimate_curvature_radius()
            self._stent_panel.set_vessel_curvature(radius_mm)

        spec    = STENT_CATALOGUE[self._traj_stent_combo.currentIndex()]
        stent_r = spec.diameter_mm / 2.0

        # ── 1 point: PCA of local vessel neighbourhood → axis ─────────── #
        if n == 1:
            pt   = np.array(self._pick_points[0])
            axis = self._estimate_vessel_axis_at(pt)
            half = spec.length_mm / 2.0
            p0   = pt - axis * half
            p1   = pt + axis * half

            pts_array   = np.array([p0, p1])
            arc_lengths = np.array([0.0, spec.length_mm])

            line = vtk.vtkLineSource()
            line.SetPoint1(*p0.tolist())
            line.SetPoint2(*p1.tolist())
            line.Update()
            source_port = line.GetOutputPort()

        # ── 2+ points: straight line or spline ────────────────────────── #
        else:
            pts_array, arc_lengths, _ = self._build_arc_length_table()

            if n == 2:
                line = vtk.vtkLineSource()
                line.SetPoint1(*self._pick_points[0])
                line.SetPoint2(*self._pick_points[1])
                line.Update()
                source_port = line.GetOutputPort()
            else:
                spline_pts = vtk.vtkPoints()
                for p in self._pick_points:
                    spline_pts.InsertNextPoint(*p)
                param_spline = vtk.vtkParametricSpline()
                param_spline.SetPoints(spline_pts)
                param_spline.ClosedOff()
                spline_src = vtk.vtkParametricFunctionSource()
                spline_src.SetParametricFunction(param_spline)
                spline_src.SetUResolution(300)
                spline_src.Update()
                source_port = spline_src.GetOutputPort()

        # Stent-sized tube
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(source_port)
        tube.SetRadius(stent_r)
        tube.SetNumberOfSides(32)
        tube.CappingOn()
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.ScalarVisibilityOff()

        self._preview_actor = vtk.vtkActor()
        self._preview_actor.SetMapper(mapper)

        # Viability check → colour
        color, opacity, viab_msg = self._check_viability(
            pts_array, arc_lengths, stent_r
        )
        prop = self._preview_actor.GetProperty()
        prop.SetColor(*color)
        prop.SetOpacity(opacity)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.65)
        prop.SetSpecular(0.50)
        prop.SetSpecularPower(40.0)

        self._renderer.AddActor(self._preview_actor)
        self._render()

        # Store viability message for pick status
        self._viab_msg = viab_msg

    def _remove_preview(self) -> None:
        if self._preview_actor is not None:
            self._renderer.RemoveActor(self._preview_actor)
            self._preview_actor = None
        self._viab_msg = ""

    def _check_viability(
        self,
        pts_array: np.ndarray,
        arc_lengths: np.ndarray,
        stent_radius_mm: float,
    ) -> tuple[tuple, float, str]:
        """
        Sample the trajectory and compute the signed clearance at each point.

        Sign convention
        ---------------
        For each sample point P we find the closest cell on the vessel wall
        (point Q, outward normal N).  The sign of dot(P−Q, N) tells us
        whether P is inside (negative dot → clearance = dist − r) or
        outside (positive dot → clearance = −(dist + r)).
        This prevents false "CABE" when the stent centre is outside the vessel.

        Returns (rgb_tuple, opacity, message).
        """
        if self._viab_locator is None or len(pts_array) == 0:
            return (0.35, 0.65, 0.95), 0.45, ""

        # Cell normals for inside/outside detection
        cell_normals = None
        if self._vessel_normals_data is not None:
            cell_normals = self._vessel_normals_data.GetCellData().GetNormals()

        total = float(arc_lengths[-1])
        step  = total / _VIAB_SAMPLES
        min_clearance = float("inf")

        closest_pt = [0.0, 0.0, 0.0]
        cell_id    = vtk.reference(0)
        sub_id     = vtk.reference(0)
        dist2      = vtk.reference(0.0)

        for i in range(_VIAB_SAMPLES + 1):
            arc_s = i * step
            pt, _ = self._get_point_and_tangent(arc_s, pts_array, arc_lengths)
            self._viab_locator.FindClosestPoint(
                pt.tolist(), closest_pt, cell_id, sub_id, dist2
            )
            dist = math.sqrt(float(dist2))

            # ── Inside/outside test via outward-normal dot product ────── #
            inside = True   # optimistic default when normals unavailable
            cid = int(cell_id)
            if cell_normals is not None and cid >= 0:
                normal   = np.array(cell_normals.GetTuple3(cid))
                vec_to_pt = pt - np.array(closest_pt)
                # dot < 0  →  pt is on the inward side  →  INSIDE
                inside = float(np.dot(vec_to_pt, normal)) < 0

            if inside:
                clearance = dist - stent_radius_mm
            else:
                # Outside the vessel: report large negative clearance so the
                # stent always shows as "NO CABE" regardless of wall distance
                clearance = -(dist + stent_radius_mm)

            if clearance < min_clearance:
                min_clearance = clearance

        if min_clearance >= 1.0:
            color   = (0.18, 0.78, 0.35)
            opacity = 0.60
            icon    = "✓"
            verdict = f"CABE · holgura mínima {min_clearance:.1f} mm"
            style   = "color:#56d364;"
        elif min_clearance >= 0.0:
            color   = (0.95, 0.60, 0.10)
            opacity = 0.65
            icon    = "⚠"
            verdict = f"AJUSTADO · holgura mínima {min_clearance:.1f} mm"
            style   = "color:#e3b341;"
        else:
            color   = (0.88, 0.16, 0.16)
            opacity = 0.70
            icon    = "✗"
            if min_clearance < -stent_radius_mm:
                # Centre is outside → different wording
                verdict = (
                    f"NO CABE · trayectoria fuera del vaso "
                    f"({abs(min_clearance + stent_radius_mm):.1f} mm fuera)"
                )
            else:
                verdict = (
                    f"NO CABE · el stent (Ø{stent_radius_mm*2:.1f} mm) "
                    f"excede el lumen en {abs(min_clearance):.1f} mm"
                )
            style = "color:#f85149;"

        self._pick_status.setStyleSheet(f"font-size:10px; font-weight:bold; {style}")
        return color, opacity, f"{icon}  {verdict}"

    def _update_pick_status(self) -> None:
        n = len(self._pick_points)

        # Clip button: enabled with any pick point
        self._btn_clip_place.setEnabled(n >= 1)

        if n == 0:
            self._btn_auto_place.setEnabled(False)
            self._pick_status.setStyleSheet(_plan_muted_lbl_css())
            if not self._pick_mode:
                self._pick_status.setText(
                    "Sin puntos — active el modo pick y haga clic sobre la arteria."
                )
            return

        if n == 1:
            # Single-point mode: PCA axis used — allow immediate placement
            self._btn_auto_place.setEnabled(True)
            spec = STENT_CATALOGUE[self._traj_stent_combo.currentIndex()]
            viab = getattr(self, "_viab_msg", "")
            diam_str = ""
            if self._pick_diameters and self._pick_diameters[0] > 0:
                diam_str = f"  ·  Ø vaso {self._pick_diameters[0]:.1f} mm"
            msg  = (
                f"1 punto  ·  eje estimado por PCA{diam_str}  ·  "
                f"Ø {spec.diameter_mm:.1f} mm × {spec.length_mm:.0f} mm"
            )
            self._pick_status.setText(f"{msg}   {viab}" if viab else msg)
            return

        _, arc_lengths, total = self._build_arc_length_table()
        spec     = STENT_CATALOGUE[self._traj_stent_combo.currentIndex()]
        overlap  = self._overlap_spin.value()
        step     = max(0.1, spec.length_mm - overlap)
        n_stents = (
            int((total - spec.length_mm / 2.0) / step) + 1
            if total >= spec.length_mm else 0
        )

        # Always allow placement with 2+ points — short trajectories get 1 stent at midpoint
        self._btn_auto_place.setEnabled(True)

        viab = getattr(self, "_viab_msg", "")
        count_str = str(n_stents) if n_stents > 0 else "1  ⚡ trayectoria corta"

        # Vessel diameter summary along picked points
        valid_diams = [d for d in self._pick_diameters if d > 0]
        if valid_diams:
            dmin = min(valid_diams)
            dmax = max(valid_diams)
            if abs(dmax - dmin) < 0.2:
                diam_str = f"  ·  Ø vaso {dmin:.1f} mm"
            else:
                diam_str = f"  ·  Ø vaso {dmin:.1f}–{dmax:.1f} mm"
        else:
            diam_str = ""

        base = (
            f"{n} puntos  ·  trayectoria {total:.1f} mm{diam_str}  ·  "
            f"Ø stent {spec.diameter_mm:.1f} mm  ·  "
            f"stents a colocar: {count_str}"
        )
        self._pick_status.setText(f"{base}   {viab}" if viab else base)

    # ================================================================== #
    # Automatic stent placement along trajectory                           #
    # ================================================================== #

    def _place_stents_on_trajectory(self) -> None:
        n    = len(self._pick_points)
        spec = STENT_CATALOGUE[self._traj_stent_combo.currentIndex()]

        # Grab (and immediately clear) deform target so re-entry is safe
        deform_idx              = self._deform_target_idx
        self._deform_target_idx = None
        self._btn_auto_place.setText("Colocar stents en trayectoria")

        # ── Single-point mode: PCA axis, one stent ────────────────────── #
        if n == 1:
            pt      = np.array(self._pick_points[0])
            tangent = self._estimate_vessel_axis_at(pt)
            transform = self._transform_z_to_tangent(pt, tangent)

            if deform_idx is not None:
                # Reshape existing stent: swap actor, reuse the same list entry
                old_actor = self._stent_actors.pop(deform_idx, None)
                if old_actor:
                    self._renderer.RemoveActor(old_actor)
                new_actor = self._make_stent_actor(spec.name, None)
                new_actor.SetUserTransform(transform)
                self._stent_actors[deform_idx] = new_actor
                self._renderer.AddActor(new_actor)
                self._stent_panel.update_placed_transform(deform_idx, transform)
                idx = deform_idx
                logger.info(
                    "Reshaped stent #%d: %s at (%.1f, %.1f, %.1f)",
                    deform_idx, spec.name, *pt.tolist(),
                )
            else:
                idx = self._stent_panel.place_stent_programmatic(spec.name, transform)
                logger.info(
                    "Single-point placement: %s at (%.1f, %.1f, %.1f) "
                    "axis=(%.2f, %.2f, %.2f)",
                    spec.name, *pt.tolist(), *tangent.tolist(),
                )

            # Remove viability preview tube — the helix actor takes its place
            self._remove_preview()

            # Compute orthonormal frame for 3-D nudge
            ax = tangent.copy()
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(ax, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            p1 = np.cross(ax, ref);  p1 /= np.linalg.norm(p1)
            p2 = np.cross(ax, p1);   p2 /= np.linalg.norm(p2)

            # Store for nudge controls
            self._nudge_idx          = idx
            self._nudge_pos          = pt.copy()
            self._nudge_axis         = ax
            self._nudge_perp1        = p1
            self._nudge_perp2        = p2
            self._nudge_total_offset = 0.0
            self._nudge_widget.setVisible(True)
            self._recheck_viability_for_nudge()

            viab = getattr(self, "_viab_msg", "")
            action = "Redefinido" if deform_idx is not None else "Colocado"
            self._pick_status.setText(
                f"{action} 1 × {spec.name} por eje PCA"
                + (f"   {viab}" if viab else "")
            )
            return

        # ── Trajectory mode: 2+ points ────────────────────────────────── #
        if n < 2:
            return

        pts_array, arc_lengths, total = self._build_arc_length_table()

        # ── Deform mode: reshape existing stent as a curved tube ─────── #
        if deform_idx is not None:
            old_actor = self._stent_actors.pop(deform_idx, None)
            if old_actor:
                self._renderer.RemoveActor(old_actor)
            curved_actor = self._make_curved_tube_actor(
                spec, self._pick_points, pts_array, arc_lengths
            )
            self._stent_actors[deform_idx] = curved_actor
            self._renderer.AddActor(curved_actor)
            # Store identity transform (tube already in world coords)
            identity = vtk.vtkTransform()
            identity.Identity()
            self._stent_panel.update_placed_transform(deform_idx, identity)
            self._remove_preview()
            self._nudge_widget.setVisible(False)
            logger.info(
                "Reshaped stent #%d as curved tube along %.1f mm trajectory",
                deform_idx, total,
            )
            viab = getattr(self, "_viab_msg", "")
            self._pick_status.setText(
                f"Stent #{deform_idx} redefinido · trayectoria {total:.1f} mm"
                + (f"   {viab}" if viab else "")
            )
            self._render()
            return

        # ── Normal multi-point placement ──────────────────────────────── #
        overlap = self._overlap_spin.value()
        step    = max(0.1, spec.length_mm - overlap)

        if total < spec.length_mm:
            # Trajectory shorter than one stent — place 1 stent at midpoint
            mid_arc        = total / 2.0
            pt, tangent    = self._get_point_and_tangent(mid_arc, pts_array, arc_lengths)
            transform      = self._transform_z_to_tangent(pt, tangent)
            self._stent_panel.place_stent_programmatic(spec.name, transform)
            self._remove_preview()
            logger.info(
                "Short-trajectory placement: 1 × %s at midpoint (%.1f mm < %.0f mm stent)",
                spec.name, total, spec.length_mm,
            )
            viab = getattr(self, "_viab_msg", "")
            self._pick_status.setText(
                f"Colocado 1 × {spec.name}  ·  trayectoria {total:.1f} mm "
                f"(más corta que el stent {spec.length_mm:.0f} mm)"
                + (f"   {viab}" if viab else "")
            )
            return

        arc_positions: list[float] = []
        pos = spec.length_mm / 2.0
        while pos <= total + 1e-3:
            arc_positions.append(pos)
            pos += step

        for arc_s in arc_positions:
            pt, tangent = self._get_point_and_tangent(arc_s, pts_array, arc_lengths)
            transform   = self._transform_z_to_tangent(pt, tangent)
            self._stent_panel.place_stent_programmatic(spec.name, transform)

        # Remove trajectory preview — placed helix actors take its place
        self._remove_preview()

        logger.info(
            "Auto-placed %d stents (%s) along %.1f mm trajectory",
            len(arc_positions), spec.name, total,
        )
        viab = getattr(self, "_viab_msg", "")
        self._pick_status.setText(
            f"Colocados {len(arc_positions)} × {spec.name}  ·  "
            f"trayectoria {total:.1f} mm  ·  solapamiento {overlap:.1f} mm"
            + (f"   {viab}" if viab else "")
        )

    # ------------------------------------------------------------------ #
    # Arc-length geometry helpers                                          #
    # ------------------------------------------------------------------ #

    def _nudge_in_dir(self, dir_key: str, multiplier: int) -> None:
        """Move the placed stent in one of three vessel-local directions.

        *dir_key* is ``'axis'``, ``'perp1'``, or ``'perp2'``.
        *multiplier* is ±1 (small step) or ±3 (large step).
        The actual displacement = multiplier × step-spinbox value.
        """
        if self._nudge_idx is None or self._nudge_pos is None:
            return

        dirs = {
            "axis":  self._nudge_axis,
            "perp1": self._nudge_perp1,
            "perp2": self._nudge_perp2,
        }
        direction = dirs.get(dir_key)
        if direction is None:
            return

        step  = self._nudge_step_spin.value()
        delta = multiplier * step
        self._nudge_pos          = self._nudge_pos + direction * delta
        self._nudge_total_offset += delta if dir_key == "axis" else 0.0

        transform = self._transform_z_to_tangent(self._nudge_pos, self._nudge_axis)

        actor = self._stent_actors.get(self._nudge_idx)
        if actor:
            actor.SetUserTransform(transform)

        self._stent_panel.update_placed_transform(self._nudge_idx, transform)
        self._recheck_viability_for_nudge()
        self._render()

    def _recheck_viability_for_nudge(self) -> None:
        """Re-run viability check for the current nudge position and update labels."""
        if self._nudge_pos is None or self._nudge_axis is None:
            return

        spec    = STENT_CATALOGUE[self._traj_stent_combo.currentIndex()]
        half    = spec.length_mm / 2.0
        stent_r = spec.diameter_mm / 2.0
        p0      = self._nudge_pos - self._nudge_axis * half
        p1      = self._nudge_pos + self._nudge_axis * half

        pts_array   = np.array([p0, p1])
        arc_lengths = np.array([0.0, spec.length_mm])

        color, opacity, viab_msg = self._check_viability(
            pts_array, arc_lengths, stent_r
        )

        # Update nudge viability label
        if "NO CABE" in viab_msg or "✗" in viab_msg:
            vstyle = "color:#f85149; font-size:10px; font-weight:bold;"
        elif "AJUST" in viab_msg or "⚠" in viab_msg:
            vstyle = "color:#e3b341; font-size:10px; font-weight:bold;"
        elif "CABE" in viab_msg or "✓" in viab_msg:
            vstyle = "color:#56d364; font-size:10px; font-weight:bold;"
        else:
            vstyle = _plan_muted_lbl_css()
        self._nudge_viab_label.setStyleSheet(vstyle)
        self._nudge_viab_label.setText(viab_msg if viab_msg else "Sin malla cargada")

        # Tint the placed stent actor for real-time visual viability feedback
        actor = self._stent_actors.get(self._nudge_idx)
        if actor:
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetOpacity(min(1.0, opacity + 0.15))

        # Also tint the preview tube if it still exists
        if self._preview_actor is not None:
            self._preview_actor.GetProperty().SetColor(*color)
            self._preview_actor.GetProperty().SetOpacity(opacity)

    def _estimate_vessel_axis_at(self, point: np.ndarray) -> np.ndarray:
        """
        Estimate the local vessel-axis direction at *point* via PCA on nearby
        mesh vertices.  The first principal component of a tubular neighbourhood
        aligns with the tube's long axis.

        Search radius = max(stent_length × 0.6, 8 mm).  If too few vertices are
        found within that radius the search falls back to the 40 nearest points.

        Returns a **unit** vector, or (0, 0, 1) as a safe fallback.
        """
        if self._kd_locator is None or self._vessel_poly is None:
            return np.array([0.0, 0.0, 1.0])

        spec   = STENT_CATALOGUE[self._traj_stent_combo.currentIndex()]
        radius = max(spec.length_mm * 0.6, 8.0)

        id_list = vtk.vtkIdList()
        self._kd_locator.FindPointsWithinRadius(radius, point.tolist(), id_list)

        if id_list.GetNumberOfIds() < 6:
            # Too sparse — fall back to N nearest neighbours
            self._kd_locator.FindClosestNPoints(40, point.tolist(), id_list)

        n = id_list.GetNumberOfIds()
        if n < 3:
            return np.array([0.0, 0.0, 1.0])

        pts      = np.array([self._vessel_poly.GetPoint(id_list.GetId(i))
                             for i in range(n)], dtype=float)
        centered = pts - pts.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            axis  = Vt[0]
            norm  = np.linalg.norm(axis)
            return axis / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        except Exception:
            return np.array([0.0, 0.0, 1.0])

    def _estimate_curvature_radius(self) -> float:
        """Estimate mean radius of curvature (mm) of the trajectory spline.

        Uses the discrete Menger curvature on consecutive triplets of sampled
        spline points and returns the *median* radius (robust to outliers at
        inflection points).  Returns 0.0 when there are fewer than 3 points
        (straight or unknown).
        """
        pts_array, _, total = self._build_arc_length_table()
        n = len(pts_array)
        if n < 3 or total < 1.0:
            return 0.0

        radii: list[float] = []
        # Sample every ~2 mm along the spline for speed
        step = max(1, n // 60)
        for i in range(1, n - 1, step):
            a = pts_array[i - 1]
            b = pts_array[i]
            c = pts_array[i + 1] if i + 1 < n else pts_array[i]
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ac = np.linalg.norm(c - a)
            # Area of triangle via cross product
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            if area < 1e-9 or ab < 1e-9 or bc < 1e-9 or ac < 1e-9:
                continue
            # Menger curvature κ = 4·Area / (|ab|·|bc|·|ac|)
            kappa = 4.0 * area / (ab * bc * ac)
            if kappa > 1e-9:
                radii.append(1.0 / kappa)

        if not radii:
            return 0.0
        return float(np.median(radii))

    def _build_arc_length_table(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n = len(self._pick_points)
        if n < 2:
            return np.zeros((0, 3)), np.array([0.0]), 0.0

        if n == 2:
            pts   = np.array(self._pick_points)
            total = float(np.linalg.norm(pts[1] - pts[0]))
            return pts, np.array([0.0, total]), total

        spline_pts = vtk.vtkPoints()
        for p in self._pick_points:
            spline_pts.InsertNextPoint(*p)

        param_spline = vtk.vtkParametricSpline()
        param_spline.SetPoints(spline_pts)
        param_spline.ClosedOff()

        spline_src = vtk.vtkParametricFunctionSource()
        spline_src.SetParametricFunction(param_spline)
        spline_src.SetUResolution(_SPLINE_SAMPLES)
        spline_src.Update()

        spline_poly = spline_src.GetOutput()
        n_pts       = spline_poly.GetNumberOfPoints()
        pts_array   = np.array([spline_poly.GetPoint(i) for i in range(n_pts)])

        diffs       = np.diff(pts_array, axis=0)
        seg_len     = np.linalg.norm(diffs, axis=1)
        arc_lengths = np.concatenate([[0.0], np.cumsum(seg_len)])
        return pts_array, arc_lengths, float(arc_lengths[-1])

    @staticmethod
    def _get_point_and_tangent(
        arc_s: float,
        pts_array: np.ndarray,
        arc_lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        arc_s = float(np.clip(arc_s, arc_lengths[0], arc_lengths[-1]))
        idx   = int(np.searchsorted(arc_lengths, arc_s))
        idx   = min(max(idx, 1), len(pts_array) - 1)

        s0, s1 = arc_lengths[idx - 1], arc_lengths[idx]
        t      = (arc_s - s0) / (s1 - s0) if s1 > s0 else 0.0
        pt     = pts_array[idx - 1] + t * (pts_array[idx] - pts_array[idx - 1])

        tangent = pts_array[idx] - pts_array[idx - 1]
        norm    = np.linalg.norm(tangent)
        tangent = tangent / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        return pt, tangent

    @staticmethod
    def _transform_z_to_tangent(
        pos: np.ndarray,
        tangent: np.ndarray,
    ) -> vtk.vtkTransform:
        """vtkTransform: translate to *pos*, rotate local Z to *tangent*."""
        z  = np.array([0.0, 0.0, 1.0])
        t  = np.array(tangent, dtype=float)
        n  = np.linalg.norm(t)
        t /= n if n > 1e-9 else 1.0

        cross      = np.cross(z, t)
        cross_norm = np.linalg.norm(cross)
        dot        = float(np.dot(z, t))

        transform = vtk.vtkTransform()
        transform.Identity()
        transform.Translate(float(pos[0]), float(pos[1]), float(pos[2]))

        if cross_norm > 1e-6:
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            axis  = cross / cross_norm
            transform.RotateWXYZ(angle, float(axis[0]), float(axis[1]), float(axis[2]))
        elif dot < 0.0:
            transform.RotateWXYZ(180.0, 1.0, 0.0, 0.0)

        return transform

    # ================================================================== #
    # Clip auto-placement from trajectory picks                           #
    # ================================================================== #

    def _place_clip_on_trajectory(self) -> None:
        """Auto-place the selected clip using the current pick points.

        1 point  → clip centred at that point; blade (local +X) perpendicular
                   to vessel axis; jaws (local ±Y) close along vessel tangent.
        2+ points → clip at arc midpoint; blade direction = vector from first
                    to last pick point (user defines span of the neck).
        """
        if not self._pick_points:
            return

        spec = CLIP_CATALOGUE[self._traj_clip_combo.currentIndex()]
        n    = len(self._pick_points)

        if n == 1:
            pt      = np.array(self._pick_points[0])
            tangent = self._estimate_vessel_axis_at(pt)
            # Blade perp to vessel tangent; jaws close along tangent
            up       = np.array([0.0, 0.0, 1.0])
            blade_x  = np.cross(tangent, up)
            if np.linalg.norm(blade_x) < 1e-9:
                up      = np.array([1.0, 0.0, 0.0])
                blade_x = np.cross(tangent, up)
            transform = self._transform_clip_at_point(pt, blade_x, tangent)
        else:
            # Blade direction = first→last pick point (user-defined neck span)
            p0 = np.array(self._pick_points[0])
            pN = np.array(self._pick_points[-1])
            blade_dir = pN - p0
            b_norm    = np.linalg.norm(blade_dir)
            blade_dir = blade_dir / b_norm if b_norm > 1e-9 else np.array([1.0, 0.0, 0.0])

            # Position: arc midpoint (more accurate on curved trajectories)
            pts_array, arc_lengths, total = self._build_arc_length_table()
            mid_pt, tangent = self._get_point_and_tangent(
                total / 2.0, pts_array, arc_lengths
            )
            transform = self._transform_clip_at_point(mid_pt, blade_dir, tangent)

        self._clip_panel.place_clip_programmatic(spec.name, transform)
        self._clear_trajectory()
        self._render()

    @staticmethod
    def _transform_clip_at_point(
        pos: np.ndarray,
        blade_dir: np.ndarray,
        up_hint: np.ndarray,
    ) -> vtk.vtkTransform:
        """Build a vtkTransform that places a clip at *pos* with:

          local +X → blade_dir  (hoja del clip apunta en esta dirección)
          local +Y → derived from up_hint  (dirección de cierre de mordazas)
          local +Z → cross(X, Y)  (profundidad de la hoja)

        Clip local frame (clip_actor.py convention):
          +X → spring/blade axis  (blade length extends along +X)
          +Y → jaw span axis      (jaws open ±Y)
          +Z → blade depth axis
        """
        x = np.array(blade_dir, dtype=float)
        x_n = np.linalg.norm(x)
        x = x / x_n if x_n > 1e-9 else np.array([1.0, 0.0, 0.0])

        # Build Y from up_hint, orthogonalised against X
        y = np.array(up_hint, dtype=float)
        y -= np.dot(y, x) * x
        y_n = np.linalg.norm(y)
        if y_n < 1e-9:
            # up_hint parallel to blade — pick any perpendicular
            candidates = [np.array([0., 0., 1.]), np.array([0., 1., 0.])]
            for c in candidates:
                y = c - np.dot(c, x) * x
                y_n = np.linalg.norm(y)
                if y_n > 1e-9:
                    break
        y = y / y_n

        z = np.cross(x, y)

        mat = vtk.vtkMatrix4x4()
        mat.Identity()
        for i in range(3):
            mat.SetElement(i, 0, float(x[i]))
            mat.SetElement(i, 1, float(y[i]))
            mat.SetElement(i, 2, float(z[i]))
        mat.SetElement(0, 3, float(pos[0]))
        mat.SetElement(1, 3, float(pos[1]))
        mat.SetElement(2, 3, float(pos[2]))

        transform = vtk.vtkTransform()
        transform.SetMatrix(mat)
        return transform

    # ================================================================== #
    # Pre/post surgical comparison (A-4 — Ankyras-inspired)               #
    # ================================================================== #

    def _load_post_mesh(self) -> None:
        """Open a STL/OBJ file as the post-treatment mesh overlay."""
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar malla post-tratamiento",
            "", "Mesh files (*.stl *.obj *.vtk);;All files (*)"
        )
        if not path:
            return
        # Remove previous post mesh if any
        if self._post_actor is not None:
            self._renderer.RemoveActor(self._post_actor)
            self._post_actor = None

        # Read depending on extension
        ext = path.lower().rsplit(".", 1)[-1]
        if ext == "stl":
            reader = vtk.vtkSTLReader()
        elif ext == "obj":
            reader = vtk.vtkOBJReader()
        else:
            reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()

        poly = reader.GetOutput()
        if poly.GetNumberOfPoints() == 0:
            QMessageBox.warning(self, "Malla post-tratamiento",
                                "El archivo no contiene geometría válida.")
            return

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()

        self._post_actor = vtk.vtkActor()
        self._post_actor.SetMapper(mapper)
        p = self._post_actor.GetProperty()
        p.SetColor(0.20, 0.90, 0.50)   # green — post-treatment
        p.SetOpacity(0.35)
        p.SetAmbient(0.25)
        p.SetDiffuse(0.65)
        self._renderer.AddActor(self._post_actor)

        self._act_hide_post.setEnabled(True)
        self._act_hide_post.setChecked(False)
        self._act_hide_post.setText("Ocultar post")
        self._render()

        n = poly.GetNumberOfPoints()
        import os
        QMessageBox.information(
            self, "Malla post-tratamiento cargada",
            f"{os.path.basename(path)}\n{n:,} vértices\n\n"
            "Se muestra en verde semi-transparente superpuesta a la malla pre-quirúrgica (azul)."
        )

    def _toggle_post_mesh_visibility(self, hidden: bool) -> None:
        if self._post_actor is not None:
            self._post_actor.SetVisibility(not hidden)
            self._act_hide_post.setText("Mostrar post" if hidden else "Ocultar post")
            self._render()

    # ================================================================== #
    # Planning history / undo (A-3 — Ankyras-inspired)                    #
    # ================================================================== #

    def _push_history(self, kind: str, label: str) -> None:
        """Record a placed-device action for undo."""
        self._history.append({"kind": kind, "label": label})
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._act_undo.setEnabled(True)
        self._act_undo.setText(f"↶ Deshacer  ({label})")

    def _undo_last_device(self) -> None:
        """Remove the most recently placed clip or stent."""
        if not self._history:
            return
        entry = self._history.pop()
        if entry["kind"] == "stent":
            self._stent_panel.remove_last_placed()
        else:
            self._clip_panel.remove_last_placed()

        if self._history:
            prev = self._history[-1]
            self._act_undo.setText(f"↶ Deshacer  ({prev['label']})")
            self._act_undo.setEnabled(True)
        else:
            self._act_undo.setText("↶ Deshacer")
            self._act_undo.setEnabled(False)

    # ================================================================== #
    # SC-3 — Virtual FD deployment animation                              #
    # ================================================================== #

    def _start_deploy_animation(self) -> None:
        """Animate the last placed flow-diverter expanding from compressed to full diameter."""
        placed = self._stent_panel._placed  # noqa: SLF001  (intentional internal access)
        if not placed:
            return

        last = placed[-1]
        actor = self._stent_actors.get(last.index)
        if actor is None:
            return

        # Stop any running animation first
        if self._anim_timer is not None:
            self._anim_timer.stop()
            self._anim_timer = None
        if self._anim_actor is not None:
            self._renderer.RemoveActor(self._anim_actor)
            self._anim_actor = None

        # Build an animation actor: deep-copy the existing mapper input
        mapper_in = actor.GetMapper().GetInput()
        if mapper_in is None:
            return

        pd_copy = vtk.vtkPolyData()
        pd_copy.DeepCopy(mapper_in)

        anim_mapper = vtk.vtkPolyDataMapper()
        anim_mapper.SetInputData(pd_copy)
        anim_mapper.ScalarVisibilityOff()

        self._anim_actor = vtk.vtkActor()
        self._anim_actor.SetMapper(anim_mapper)
        prop = self._anim_actor.GetProperty()
        prop.SetColor(1.0, 0.75, 0.20)   # gold — animation actor
        prop.SetOpacity(0.90)
        prop.SetSpecular(0.80)
        prop.SetSpecularPower(60.0)
        prop.SetRepresentationToSurface()

        # Copy base transform from placed stent (position + orientation)
        base_tf = last.transform
        self._anim_transform = vtk.vtkTransform()
        self._anim_transform.PostMultiply()
        # start collapsed: XY scaled to 0.05 (5 % of nominal diameter)
        self._anim_transform.Scale(0.05, 0.05, 1.0)
        # apply position / orientation from base transform
        self._anim_transform.Concatenate(base_tf.GetMatrix())
        self._anim_actor.SetUserTransform(self._anim_transform)

        # Hide the static actor during animation
        actor.SetVisibility(False)
        self._renderer.AddActor(self._anim_actor)

        self._anim_frame = 0
        self._act_deploy_anim.setEnabled(False)
        self._render()

        # Start timer — one step every 40 ms → 24 frames ≈ ~1 s total
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(40)
        self._anim_timer.timeout.connect(self._animation_step)
        self._anim_timer.start()

    def _animation_step(self) -> None:
        """Called ~25 fps by QTimer; advances the deployment animation one frame."""
        self._anim_frame += 1
        t = min(self._anim_frame / self._anim_frames, 1.0)
        # Ease-in-out: smooth start and end
        t_ease = t * t * (3.0 - 2.0 * t)
        # Scale XY from 0.05 → 1.0 (radial expansion), Z stays 1.0
        xy_scale = 0.05 + 0.95 * t_ease

        placed = self._stent_panel._placed  # noqa: SLF001
        if not placed:
            self._anim_timer.stop()
            return
        last = placed[-1]
        base_tf = last.transform

        tf = vtk.vtkTransform()
        tf.PostMultiply()
        tf.Scale(xy_scale, xy_scale, 1.0)
        tf.Concatenate(base_tf.GetMatrix())
        self._anim_actor.SetUserTransform(tf)
        self._render()

        if self._anim_frame >= self._anim_frames:
            # Animation complete — restore static actor, remove anim actor
            self._anim_timer.stop()
            self._anim_timer = None

            actor = self._stent_actors.get(last.index)
            if actor is not None:
                actor.SetVisibility(True)

            self._renderer.RemoveActor(self._anim_actor)
            self._anim_actor = None
            self._anim_transform = None

            self._act_deploy_anim.setEnabled(True)
            self._render()

    # ================================================================== #
    # Clip handlers                                                        #
    # ================================================================== #

    def _on_clip_placed(self, index, spec_name, transform, poly_data) -> None:
        actor = self._make_clip_actor(spec_name, poly_data)
        actor.SetUserTransform(transform)
        self._clip_actors[index] = actor
        self._renderer.AddActor(actor)
        self._check_perforator_collisions()
        self._push_history("clip", spec_name)
        self._render()

    def _on_clip_removed(self, index) -> None:
        actor = self._clip_actors.pop(index, None)
        if actor:
            self._renderer.RemoveActor(actor)
            self._check_perforator_collisions()
            self._render()

    def _on_clip_transform(self, index, transform) -> None:
        actor = self._clip_actors.get(index)
        if actor:
            actor.SetUserTransform(transform)
            self._check_perforator_collisions()
            self._render()

    def _on_clip_visibility(self, index, visible) -> None:
        actor = self._clip_actors.get(index)
        if actor:
            actor.SetVisibility(visible)
            self._render()

    def _on_trajectory(self, entry, target) -> None:
        if self._traj_actor is not None:
            self._renderer.RemoveActor(self._traj_actor)
            self._traj_actor = None

        if entry is None or target is None:
            self._render()
            return

        dx, dy, dz = target[0]-entry[0], target[1]-entry[1], target[2]-entry[2]
        if math.sqrt(dx*dx + dy*dy + dz*dz) < 1e-3:
            return

        line = vtk.vtkLineSource()
        line.SetPoint1(*entry)
        line.SetPoint2(*target)
        line.Update()

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(3.0)
        tube.SetNumberOfSides(16)
        tube.CappingOn()
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.ScalarVisibilityOff()

        self._traj_actor = vtk.vtkActor()
        self._traj_actor.SetMapper(mapper)
        self._traj_actor.GetProperty().SetColor(0.20, 0.85, 0.95)
        self._traj_actor.GetProperty().SetOpacity(0.30)
        self._renderer.AddActor(self._traj_actor)
        self._render()

    # ================================================================== #
    # Stent handlers                                                       #
    # ================================================================== #

    # ================================================================== #
    # Perforator marking (A-04-03)                                        #
    # ================================================================== #

    def _toggle_perf_mode(self, active: bool) -> None:
        self._perf_mode = active
        if active:
            # Suspend other exclusive modes
            if self._pick_mode:
                self._btn_pick.setChecked(False)
            if self._annot_mode:
                self._btn_annot.setChecked(False)
                self._toggle_annot_mode(False)
            if self._bifurc_mode:
                self._btn_bifurc.setChecked(False)
                self._toggle_bifurc_mode(False)
            from prospective.ui.icons import I as _I
            self._btn_perf.setStyleSheet(self._perf_style_on)
            self._btn_perf.setText(f"{_I.BRAIN} Marcando perforantes…")
        else:
            from prospective.ui.icons import I as _I
            self._btn_perf.setStyleSheet(self._perf_style_off)
            self._btn_perf.setText(f"{_I.BRAIN} Marcar perforante")

    def _do_perf_pick(self, x: int, y: int) -> None:
        """Pick a point on the vessel surface and add a perforator marker."""
        self._cell_picker.Pick(x, y, 0, self._renderer)
        if self._cell_picker.GetCellId() < 0:
            return
        pt = np.array(self._cell_picker.GetPickPosition())
        self._add_perforator(pt)

    def _add_perforator(self, pt: np.ndarray) -> None:
        """Add a red sphere marker at *pt* and trigger a proximity check."""
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(pt.tolist())
        sphere.SetRadius(self._perf_radius * 0.35)   # visual radius < proximity radius
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)
        sphere.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.95, 0.15, 0.15)
        actor.GetProperty().SetOpacity(0.85)
        actor.GetProperty().SetAmbient(0.4)

        self._renderer.AddActor(actor)
        self._perf_pts.append(pt.copy())
        self._perf_actors.append(actor)

        self._lbl_perf_count.setText(
            f"{len(self._perf_pts)} perforante(s) marcado(s)"
        )
        self._check_perforator_collisions()
        self._render()

    def _clear_perforators(self) -> None:
        for actor in self._perf_actors:
            self._renderer.RemoveActor(actor)
        self._perf_actors.clear()
        self._perf_pts.clear()
        self._lbl_perf_count.setText("0 perforantes marcados")
        self._lbl_perf_collision.setText("—")
        self._lbl_perf_collision.setStyleSheet(_plan_muted_lbl_css())
        self._render()

    # ================================================================== #
    # 3-D Annotations (A-1 — Ankyras-inspired)                            #
    # ================================================================== #

    def _toggle_annot_mode(self, active: bool) -> None:
        self._annot_mode = active
        if active:
            if self._pick_mode:
                self._btn_pick.setChecked(False)
            if self._perf_mode:
                self._btn_perf.setChecked(False)
                self._toggle_perf_mode(False)
            from prospective.ui.icons import I as _I
            self._btn_annot.setStyleSheet(self._annot_style_on)
            self._btn_annot.setText(f"{_I.ANNOTATION} Anotando… (clic para anclar)")
        else:
            from prospective.ui.icons import I as _I
            self._btn_annot.setStyleSheet(self._annot_style_off)
            self._btn_annot.setText(f"{_I.ANNOTATION} Añadir anotación")

    def _do_annot_pick(self, x: int, y: int) -> None:
        """Pick a surface point and prompt for annotation text."""
        self._cell_picker.Pick(x, y, 0, self._renderer)
        if self._cell_picker.GetCellId() < 0:
            return
        pos = tuple(self._cell_picker.GetPickPosition())

        from PyQt5.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(
            self,
            "Nueva anotación 3D",
            "Texto de la etiqueta:",
        )
        if not ok or not text.strip():
            return
        self._add_annotation(pos, text.strip())

    def _add_annotation(self, pos: tuple, text: str) -> None:
        """Place a billboard text label + small anchor sphere at *pos*."""
        # Anchor sphere (cyan, 1 mm radius)
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.0)
        sphere.SetPhiResolution(10)
        sphere.SetThetaResolution(10)
        sphere.Update()
        sm = vtk.vtkPolyDataMapper()
        sm.SetInputConnection(sphere.GetOutputPort())
        sm.ScalarVisibilityOff()
        sa = vtk.vtkActor()
        sa.SetMapper(sm)
        sa.GetProperty().SetColor(0.55, 0.88, 1.0)
        sa.GetProperty().SetAmbient(0.7)
        sa.GetProperty().SetOpacity(0.90)
        self._renderer.AddActor(sa)
        self._annot_sphere_actors.append(sa)

        # Billboard text label (always faces camera)
        label = vtk.vtkBillboardTextActor3D()
        label.SetInput(text)
        label.SetPosition(*pos)
        label.SetDisplayOffset(10, 6)      # pixels right + up of anchor
        tp = label.GetTextProperty()
        tp.SetFontSize(14)
        tp.SetColor(0.85, 0.75, 1.0)       # soft purple
        tp.SetBold(True)
        tp.SetShadow(True)
        tp.SetShadowOffset(1, -1)
        tp.SetBackgroundColor(0.05, 0.05, 0.12)
        tp.SetBackgroundOpacity(0.65)
        self._renderer.AddActor(label)
        self._annot_label_actors.append(label)

        self._annot_data.append({"pos": pos, "text": text})
        self._lbl_annot_count.setText(f"{len(self._annot_data)} anotación(es)")
        self._render()

    def _undo_last_annotation(self) -> None:
        """Remove the most recently added annotation."""
        if not self._annot_data:
            return
        self._annot_data.pop()
        for actor_list in (self._annot_label_actors, self._annot_sphere_actors):
            actor = actor_list.pop()
            self._renderer.RemoveActor(actor)
        self._lbl_annot_count.setText(
            f"{len(self._annot_data)} anotación(es)"
            if self._annot_data else "0 anotaciones"
        )
        self._render()

    def _clear_annotations(self) -> None:
        """Remove all 3-D annotation labels and spheres."""
        for actor in self._annot_label_actors + self._annot_sphere_actors:
            self._renderer.RemoveActor(actor)
        self._annot_label_actors.clear()
        self._annot_sphere_actors.clear()
        self._annot_data.clear()
        self._lbl_annot_count.setText("0 anotaciones")
        self._render()

    # ================================================================== #
    # Bifurcation angle measurement (A-2 — Ankyras-inspired)              #
    # ================================================================== #

    _BIFURC_LABELS   = ["Arteria madre", "Ápex", "Rama 1", "Rama 2"]
    _BIFURC_COLORS   = [
        (0.35, 0.65, 1.00),   # blue   — parent
        (1.00, 0.85, 0.00),   # yellow — apex
        (0.25, 0.85, 0.35),   # green  — branch 1
        (1.00, 0.55, 0.10),   # orange — branch 2
    ]

    def _toggle_bifurc_mode(self, active: bool) -> None:
        self._bifurc_mode = active
        if active:
            if self._pick_mode:
                self._btn_pick.setChecked(False)
            if self._perf_mode:
                self._btn_perf.setChecked(False)
                self._toggle_perf_mode(False)
            if self._annot_mode:
                self._btn_annot.setChecked(False)
                self._toggle_annot_mode(False)
            from prospective.ui.icons import I as _I
            self._btn_bifurc.setStyleSheet(self._bifurc_style_on)
            self._btn_bifurc.setText(f"{_I.ANGLE_MEAS} Midiendo… (0/4 puntos)")
        else:
            from prospective.ui.icons import I as _I
            self._btn_bifurc.setStyleSheet(self._bifurc_style_off)
            self._btn_bifurc.setText(f"{_I.ANGLE_MEAS} Medir ángulos bifurcación")

    def _do_bifurc_pick(self, x: int, y: int) -> None:
        """Add one bifurcation measurement point (max 4)."""
        if len(self._bifurc_pts) >= 4:
            return
        self._cell_picker.Pick(x, y, 0, self._renderer)
        if self._cell_picker.GetCellId() < 0:
            return
        pos = tuple(self._cell_picker.GetPickPosition())
        idx = len(self._bifurc_pts)
        self._bifurc_pts.append(pos)

        # Coloured sphere
        sph = vtk.vtkSphereSource()
        sph.SetCenter(*pos)
        sph.SetRadius(1.3)
        sph.SetPhiResolution(12)
        sph.SetThetaResolution(12)
        sph.Update()
        sm = vtk.vtkPolyDataMapper()
        sm.SetInputConnection(sph.GetOutputPort())
        sm.ScalarVisibilityOff()
        sa = vtk.vtkActor()
        sa.SetMapper(sm)
        sa.GetProperty().SetColor(*self._BIFURC_COLORS[idx])
        sa.GetProperty().SetAmbient(0.6)
        self._renderer.AddActor(sa)
        self._bifurc_actors.append(sa)

        # Step label
        lbl = vtk.vtkBillboardTextActor3D()
        lbl.SetInput(self._BIFURC_LABELS[idx])
        lbl.SetPosition(*pos)
        lbl.SetDisplayOffset(8, 8)
        tp = lbl.GetTextProperty()
        tp.SetFontSize(12)
        r, g, b = self._BIFURC_COLORS[idx]
        tp.SetColor(r, g, b)
        tp.SetBold(True)
        tp.SetShadow(True)
        tp.SetBackgroundColor(0.05, 0.05, 0.05)
        tp.SetBackgroundOpacity(0.60)
        self._renderer.AddActor(lbl)
        self._bifurc_actors.append(lbl)

        n = len(self._bifurc_pts)
        from prospective.ui.icons import I as _I
        self._btn_bifurc.setText(f"{_I.ANGLE_MEAS} Midiendo… ({n}/4 puntos)")

        if n == 4:
            self._compute_bifurc_angles()
        self._render()

    def _compute_bifurc_angles(self) -> None:
        """Compute β₁, β₂, θ and draw the measurement lines."""
        p_parent, p_apex, p_b1, p_b2 = [np.array(p) for p in self._bifurc_pts]

        v_parent = p_parent - p_apex   # apex → parent direction
        v_b1     = p_b1     - p_apex   # apex → branch 1
        v_b2     = p_b2     - p_apex   # apex → branch 2

        def _angle(a: np.ndarray, b: np.ndarray) -> float:
            cos_a = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            return math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))

        beta1 = _angle(v_parent, v_b1)   # parent ↔ branch 1
        beta2 = _angle(v_parent, v_b2)   # parent ↔ branch 2
        theta = _angle(v_b1,     v_b2)   # branch 1 ↔ branch 2 (bifurcation angle)

        # Draw 3 measurement lines (apex → each neighbour)
        line_defs = [
            (p_apex, p_parent, self._BIFURC_COLORS[0]),
            (p_apex, p_b1,     self._BIFURC_COLORS[2]),
            (p_apex, p_b2,     self._BIFURC_COLORS[3]),
        ]
        for pt_a, pt_b, col in line_defs:
            pts = vtk.vtkPoints()
            pts.InsertNextPoint(*pt_a)
            pts.InsertNextPoint(*pt_b)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, 1)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(line)
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetLines(cells)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*col)
            actor.GetProperty().SetLineWidth(2.0)
            self._renderer.AddActor(actor)
            self._bifurc_actors.append(actor)

        # Angle labels at midpoints of each arm
        mid_b1 = (p_apex + p_b1) / 2
        mid_b2 = (p_apex + p_b2) / 2
        for mid, angle_val, text in [
            (mid_b1, beta1, f"β₁ = {beta1:.1f}°"),
            (mid_b2, beta2, f"β₂ = {beta2:.1f}°"),
        ]:
            lbl = vtk.vtkBillboardTextActor3D()
            lbl.SetInput(text)
            lbl.SetPosition(*mid.tolist())
            lbl.SetDisplayOffset(4, 4)
            tp = lbl.GetTextProperty()
            tp.SetFontSize(13)
            tp.SetColor(1.0, 0.92, 0.55)
            tp.SetBold(True)
            tp.SetShadow(True)
            tp.SetBackgroundColor(0.05, 0.05, 0.05)
            tp.SetBackgroundOpacity(0.65)
            self._renderer.AddActor(lbl)
            self._bifurc_actors.append(lbl)

        # θ label near the apex
        lbl_theta = vtk.vtkBillboardTextActor3D()
        lbl_theta.SetInput(f"θ = {theta:.1f}°")
        lbl_theta.SetPosition(*p_apex.tolist())
        lbl_theta.SetDisplayOffset(-40, 16)
        tp = lbl_theta.GetTextProperty()
        tp.SetFontSize(13)
        tp.SetColor(1.0, 0.75, 0.25)
        tp.SetBold(True)
        tp.SetShadow(True)
        tp.SetBackgroundColor(0.05, 0.05, 0.05)
        tp.SetBackgroundOpacity(0.65)
        self._renderer.AddActor(lbl_theta)
        self._bifurc_actors.append(lbl_theta)

        # Update UI result label
        self._lbl_bifurc_result.setText(
            f"<small>"
            f"<b style='color:#3fb950'>β₁ = {beta1:.1f}°</b>  "
            f"(madre↔rama 1) &nbsp;|&nbsp; "
            f"<b style='color:#e3b341'>β₂ = {beta2:.1f}°</b>  "
            f"(madre↔rama 2) &nbsp;|&nbsp; "
            f"<b style='color:#ff8c00'>θ = {theta:.1f}°</b>  "
            f"(bifurcación)"
            f"</small>"
        )
        # Deactivate mode after measurement
        self._btn_bifurc.blockSignals(True)
        self._btn_bifurc.setChecked(False)
        self._btn_bifurc.blockSignals(False)
        self._bifurc_mode = False
        self._btn_bifurc.setStyleSheet(self._bifurc_style_off)
        from prospective.ui.icons import I as _I
        self._btn_bifurc.setText(f"{_I.ANGLE_MEAS} Medir ángulos bifurcación")

    def _clear_bifurc_measurement(self) -> None:
        for actor in self._bifurc_actors:
            self._renderer.RemoveActor(actor)
        self._bifurc_actors.clear()
        self._bifurc_pts.clear()
        _mc = "#9B9B9B" if _is_dark() else "#6B6B6B"
        self._lbl_bifurc_result.setText(
            f"<small style='color:{_mc}'>β₁ — · β₂ — · θ —</small>"
        )
        if self._bifurc_mode:
            from prospective.ui.icons import I as _I
            self._btn_bifurc.setText(f"{_I.ANGLE_MEAS} Midiendo… (0/4 puntos)")
        self._render()

    def _check_perforator_collisions(self) -> None:
        """
        Check all placed clips against marked perforator positions.

        A clip is considered "at risk" when its world-space position
        (transform origin) is within self._perf_radius mm of any perforator.
        Result is shown in self._lbl_perf_collision.
        """
        if not self._perf_pts:
            self._lbl_perf_collision.setText("—")
            self._lbl_perf_collision.setStyleSheet(_plan_muted_lbl_css())
            return

        if not self._clip_actors:
            self._lbl_perf_collision.setText(
                "Sin clips colocados — coloque clips para verificar."
            )
            self._lbl_perf_collision.setStyleSheet(_plan_muted_lbl_css())
            return

        at_risk: list[str] = []
        for clip_idx, actor in self._clip_actors.items():
            t = actor.GetUserTransform()
            clip_pos = np.array(t.GetPosition()) if t else np.array(actor.GetCenter())
            for i, perf_pt in enumerate(self._perf_pts):
                dist = float(np.linalg.norm(clip_pos - perf_pt))
                if dist <= self._perf_radius:
                    at_risk.append(
                        f"Clip #{clip_idx} a {dist:.1f} mm del perforante #{i+1}"
                    )

        if at_risk:
            self._lbl_perf_collision.setStyleSheet(
                "color:#f85149; font-size:10px; font-weight:bold;"
            )
            self._lbl_perf_collision.setText(
                "⚠ RIESGO PERFORANTE: " + "  ·  ".join(at_risk)
            )
        else:
            self._lbl_perf_collision.setStyleSheet(
                "color:#56d364; font-size:10px; font-weight:bold;"
            )
            self._lbl_perf_collision.setText(
                f"✓ Sin clips a menos de {self._perf_radius:.0f} mm de perforantes"
            )

    # ================================================================== #
    # Stent deform / reshape                                              #
    # ================================================================== #

    def _start_deform_mode(self, index: int) -> None:
        """
        Enter trajectory-redefinition mode for an already-placed stent.

        Clears the current trajectory, activates pick mode, and flags
        *index* as the target so that the next "Colocar" action updates
        the existing stent actor instead of creating a new one.
        """
        # Clear picks (resets _deform_target_idx via _clear_trajectory) …
        self._clear_trajectory()
        # … then set the target (must come after _clear_trajectory reset)
        self._deform_target_idx = index

        # Activate pick mode if not already on
        if not self._pick_mode:
            self._btn_pick.setChecked(True)   # triggers _toggle_pick_mode(True)

        self._btn_auto_place.setText("Aplicar nueva forma")
        self._pick_status.setStyleSheet(
            "color:#e3b341; font-size:10px; font-weight:bold;"
        )
        self._pick_status.setText(
            f"✏  Redefinir stent #{index} — haga clic sobre la arteria "
            "para añadir puntos de control y pulse 'Aplicar nueva forma'."
        )
        self._render()

    @staticmethod
    def _make_curved_tube_actor(
        spec,
        pick_points: list[tuple],
        pts_array:   np.ndarray,
        arc_lengths: np.ndarray,
    ) -> vtk.vtkActor:
        """
        Build a smooth tube actor that follows the trajectory spline in world
        coordinates.  Used for stents reshaped by the 'Redefinir forma' tool.
        """
        stent_r = spec.diameter_mm / 2.0
        n = len(pick_points)

        if n == 2:
            line = vtk.vtkLineSource()
            line.SetPoint1(*pick_points[0])
            line.SetPoint2(*pick_points[1])
            line.Update()
            source_port = line.GetOutputPort()
        else:
            spline_pts = vtk.vtkPoints()
            for p in pick_points:
                spline_pts.InsertNextPoint(*p)
            param_spline = vtk.vtkParametricSpline()
            param_spline.SetPoints(spline_pts)
            param_spline.ClosedOff()
            spline_src = vtk.vtkParametricFunctionSource()
            spline_src.SetParametricFunction(param_spline)
            spline_src.SetUResolution(300)
            spline_src.Update()
            source_port = spline_src.GetOutputPort()

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(source_port)
        tube.SetRadius(stent_r)
        tube.SetNumberOfSides(32)
        tube.CappingOn()
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(0.60, 0.78, 0.95)
        prop.SetOpacity(0.88)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.60)
        prop.SetSpecular(0.80)
        prop.SetSpecularPower(60.0)
        return actor

    def _on_stent_placed(self, index, spec_name, transform, poly_data) -> None:
        actor = self._make_stent_actor(spec_name, poly_data)
        actor.SetUserTransform(transform)
        self._stent_actors[index] = actor
        self._renderer.AddActor(actor)
        self._push_history("stent", spec_name)
        # Enable deploy animation button when at least one stent is present
        self._act_deploy_anim.setEnabled(True)
        self._render()

    def _on_stent_removed(self, index) -> None:
        actor = self._stent_actors.pop(index, None)
        if actor:
            self._renderer.RemoveActor(actor)
            self._render()

    def _on_stent_transform(self, index, transform) -> None:
        actor = self._stent_actors.get(index)
        if actor:
            actor.SetUserTransform(transform)
            self._render()

    def _on_stent_visibility(self, index, visible) -> None:
        actor = self._stent_actors.get(index)
        if actor:
            actor.SetVisibility(visible)
            self._render()

    # ================================================================== #
    # Centreline handlers                                                  #
    # ================================================================== #

    def _enter_cl_pick(self, mode: str) -> None:
        """Activate centreline pick mode ('source' or 'target')."""
        # Suspend all other exclusive pick modes
        if self._pick_mode:
            self._btn_pick.setChecked(False)
            self._toggle_pick_mode(False)
        if self._perf_mode:
            self._btn_perf.setChecked(False)
            self._toggle_perf_mode(False)
        self._cl_mode = mode
        tip = "origen (verde)" if mode == "source" else "destino (rojo)"
        self.statusBar().showMessage(
            f"LÍNEA CENTRAL — haga clic sobre el vaso para marcar {tip}"
        )

    def _do_cl_pick(self, x: int, y: int) -> None:
        """Pick the source or target point for centreline extraction."""
        self._cell_picker.Pick(x, y, 0, self._renderer)
        if self._cell_picker.GetCellId() < 0:
            return

        pt = tuple(self._cell_picker.GetPickPosition())

        if self._cl_mode == "source":
            # Remove old source marker
            if self._cl_src_actor:
                self._renderer.RemoveActor(self._cl_src_actor)
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*pt)
            sphere.SetRadius(1.0)
            sphere.SetPhiResolution(16); sphere.SetThetaResolution(16)
            sphere.Update()
            m = vtk.vtkPolyDataMapper(); m.SetInputConnection(sphere.GetOutputPort())
            self._cl_src_actor = vtk.vtkActor(); self._cl_src_actor.SetMapper(m)
            self._cl_src_actor.GetProperty().SetColor(0.10, 0.85, 0.20)
            self._renderer.AddActor(self._cl_src_actor)
            self._centerline_panel.set_source_point(pt)

        elif self._cl_mode == "target":
            if self._cl_tgt_actor:
                self._renderer.RemoveActor(self._cl_tgt_actor)
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*pt)
            sphere.SetRadius(1.0)
            sphere.SetPhiResolution(16); sphere.SetThetaResolution(16)
            sphere.Update()
            m = vtk.vtkPolyDataMapper(); m.SetInputConnection(sphere.GetOutputPort())
            self._cl_tgt_actor = vtk.vtkActor(); self._cl_tgt_actor.SetMapper(m)
            self._cl_tgt_actor.GetProperty().SetColor(0.90, 0.15, 0.10)
            self._renderer.AddActor(self._cl_tgt_actor)
            self._centerline_panel.set_target_point(pt)

        # Exit pick mode after one click
        self._cl_mode = None
        self.statusBar().clearMessage()
        self._render()

    def _on_centerline_ready(self, result) -> None:
        """Display centreline tube actor coloured by radius."""
        from prospective.rendering.centerline_actor import (
            build_centerline_actor, build_scalar_bar,
        )
        # Remove previous centreline actor
        self._on_centerline_cleared()

        self._cl_actor = build_centerline_actor(result)
        self._renderer.AddActor(self._cl_actor)

        # Feed result to the CL stent panel
        self._cl_stent_panel.set_centerline(result)

        # Scalar bar (colour legend)
        self._cl_bar_actor = build_scalar_bar(result)
        self._renderer.AddActor(self._cl_bar_actor)

        self._render()

    def _on_centerline_cleared(self) -> None:
        for attr in ("_cl_actor", "_cl_src_actor", "_cl_tgt_actor", "_cl_bar_actor"):
            a = getattr(self, attr, None)
            if a is not None:
                self._renderer.RemoveActor(a)
                setattr(self, attr, None)
        self._cl_stent_panel.clear_centerline()
        self._render()

    # ================================================================== #
    # Centreline stent deployment                                          #
    # ================================================================== #

    def _on_cl_stent_deployed(self, result) -> None:
        """Add deployed stent actor to the 3-D scene."""
        self._on_cl_stent_retracted()   # remove previous if any

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(result.stent_poly_data)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(0.75, 0.76, 0.78)   # platinum
        prop.SetOpacity(0.90)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.65)
        prop.SetSpecular(0.55)
        prop.SetSpecularPower(40)

        self._cl_stent_actor = actor
        self._renderer.AddActor(actor)
        self.statusBar().showMessage(
            f"Stent desplegado: {result.summary()}", 6000
        )
        self._render()

    def _on_cl_stent_retracted(self) -> None:
        if self._cl_stent_actor is not None:
            self._renderer.RemoveActor(self._cl_stent_actor)
            self._cl_stent_actor = None
            self._render()

    # ================================================================== #
    # Ruler / caliper                                                       #
    # ================================================================== #

    def _enter_ruler_pick(self) -> None:
        """Activate ruler-pick mode: next two mesh clicks = endpoints A & B."""
        # Suspend other exclusive pick modes
        if self._pick_mode:
            self._btn_pick.setChecked(False)
            self._toggle_pick_mode(False)
        if self._perf_mode:
            self._btn_perf.setChecked(False)
            self._toggle_perf_mode(False)
        if self._cl_mode is not None:
            self._cl_mode = None

        self._ruler_mode  = True
        self._ruler_pt_a  = None
        self._measurement_panel.set_picking(True)
        self.statusBar().showMessage(
            "MEDICIÓN — haz clic en el punto A sobre la malla"
        )

    def _do_ruler_pick(self, x: int, y: int) -> None:
        """Process one ruler click (point A then point B)."""
        self._cell_picker.Pick(x, y, 0, self._renderer)
        if self._cell_picker.GetCellId() < 0:
            return

        pt = tuple(self._cell_picker.GetPickPosition())

        if self._ruler_pt_a is None:
            # First click → store A, show a sphere marker
            self._ruler_pt_a = pt
            self.statusBar().showMessage(
                "MEDICIÓN — haz clic en el punto B para completar la medición"
            )
            self._render()
            return

        # Second click → complete the ruler
        pt_a = self._ruler_pt_a
        pt_b = pt
        self._ruler_mode = False
        self._ruler_pt_a = None

        import math
        dist = math.sqrt(sum((b - a) ** 2 for a, b in zip(pt_a, pt_b)))

        from prospective.rendering.ruler_actor import build_ruler
        idx = self._measurement_panel.add_measurement(pt_a, pt_b, dist)
        actors = build_ruler(pt_a, pt_b)
        self._ruler_actors[idx] = actors
        for a in actors:
            self._renderer.AddActor(a)

        self.statusBar().clearMessage()
        self._render()

    def _on_ruler_visibility(self, index: int, visible: bool) -> None:
        actors = self._ruler_actors.get(index)
        if actors:
            actors.set_visible(visible)
            self._render()

    def _on_ruler_deleted(self, index: int) -> None:
        actors = self._ruler_actors.pop(index, None)
        if actors:
            for a in actors:
                self._renderer.RemoveActor(a)
            self._render()

    # ================================================================== #
    # Coil handlers                                                        #
    # ================================================================== #

    def add_coil(
        self,
        index: int,
        name: str,
        position: tuple,
        poly_data,
        spec=None,
        sac_radius_mm: float = 5.0,
    ) -> None:
        """Place a coil actor at *position* inside the sac."""
        from prospective.rendering.coil_actor import build_coil_actor, build_coil_in_sac

        if poly_data is not None:
            # Custom STL coil — use provided geometry
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.90, 0.89, 0.88)  # platinum
        elif spec is not None:
            # Catalogue coil — generate constrained-walk geometry in sac
            actor = build_coil_in_sac(spec, position, sac_radius_mm, coil_index=index)
        else:
            # Fallback: small sphere marker at position
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*position)
            sphere.SetRadius(1.5)
            sphere.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.90, 0.89, 0.88)

        self._coil_actors[index] = actor
        self._renderer.AddActor(actor)
        self._render()

    def remove_coil(self, index: int) -> None:
        actor = self._coil_actors.pop(index, None)
        if actor:
            self._renderer.RemoveActor(actor)
            self._render()

    def set_coil_visible(self, index: int, visible: bool) -> None:
        actor = self._coil_actors.get(index)
        if actor:
            actor.SetVisibility(visible)
            self._render()

    # ================================================================== #
    # Actor factories                                                      #
    # ================================================================== #

    @staticmethod
    def _make_clip_actor(spec_name, poly_data) -> vtk.vtkActor:
        if poly_data is not None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(poly_data)
            normals.ComputePointNormalsOn()
            normals.SplittingOff()
            normals.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.75, 0.80, 0.90)
            actor.GetProperty().SetSpecular(0.80)
            actor.GetProperty().SetSpecularPower(60.0)
        else:
            from prospective.models.clip_library import CLIP_CATALOGUE
            from prospective.rendering.clip_actor import build_clip_actor
            spec  = next((s for s in CLIP_CATALOGUE if s.name == spec_name), None)
            actor = build_clip_actor(spec) if spec else vtk.vtkActor()
        return actor

    @staticmethod
    def _make_stent_actor(spec_name, poly_data) -> vtk.vtkActor:
        if poly_data is not None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(poly_data)
            normals.ComputePointNormalsOn()
            normals.SplittingOff()
            normals.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.65, 0.80, 0.95)
            actor.GetProperty().SetSpecular(0.90)
            actor.GetProperty().SetSpecularPower(80.0)
        else:
            from prospective.rendering.stent_actor import build_stent_actor
            spec  = next((s for s in STENT_CATALOGUE if s.name == spec_name), None)
            actor = build_stent_actor(spec) if spec else vtk.vtkActor()
        return actor

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _render(self) -> None:
        self._iren.GetRenderWindow().Render()

    def closeEvent(self, event) -> None:
        # Hide instead of destroy.
        #
        # Calling GetRenderWindow().Finalize() / TerminateApp() here leaves
        # the VTK interactor in an invalid state.  When _open_planning_window()
        # in MainWindow reuses the same Python instance (self._planning_window
        # is still not None), any subsequent VTK call on the dead render-window
        # crashes the whole application.
        #
        # Solution: intercept the close event, hide the window and tell Qt NOT
        # to destroy it (event.ignore()).  The VTK pipeline stays valid and a
        # simple show() / raise_() on the same instance works correctly next
        # time.  The window is fully cleaned up when the main application exits.
        self.hide()
        event.ignore()
