"""3D volume rendering widget — VTK embedded in PyQt5.

A-02-01: QVTKRenderWindowInteractor integration
A-02-02: MIP / VR modes with configurable transfer functions + HU range clipping
"""
from __future__ import annotations

import logging

import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QMessageBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from prospective.dicom.preprocessor import DicomPreprocessor
from prospective.rendering.transfer_functions import PRESETS_3D, apply_preset

logger = logging.getLogger(__name__)


def _is_dark() -> bool:
    """Return True when the application is using a dark-background theme."""
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


class _BoneSubtractionWorker(QThread):
    """Background thread that computes bone subtraction on the raw volume."""

    finished = pyqtSignal(object)  # emits np.ndarray

    def __init__(self, volume: np.ndarray, threshold_hu: float = 300.0) -> None:
        super().__init__()
        self._volume = volume
        self._threshold = threshold_hu

    def run(self) -> None:
        logger.info("Bone subtraction started (threshold=%.0f HU)", self._threshold)
        result = DicomPreprocessor.subtract_bone(self._volume, self._threshold)
        logger.info("Bone subtraction complete")
        self.finished.emit(result)

_BLEND_MODES = {
    "VR (Ray Casting)": 0,
    "MIP":              1,
    "MinIP":            2,
}

# Default HU clip range shown on startup
_DEFAULT_HU_MIN = -100
_DEFAULT_HU_MAX = 1500


class VTKVolumeWidget(QWidget):
    """
    Embeds a VTK render window inside a PyQt5 widget.

    Public API
    ----------
    set_volume(volume, spacing)      — load a numpy volume (z,y,x) + mm spacing
    set_preset(name)                 — switch colour/opacity transfer function
    set_blend_mode(name)             — switch VR / MIP / MinIP
    set_hu_range(min_hu, max_hu)     — clip visible HU range (removes bone, etc.)
    set_mesh(poly_data)              — display a vtkPolyData mesh as surface
    clear_mesh()                     — remove the mesh actor
    set_mesh_visible(visible)        — show/hide mesh without removing it
    set_volume_visible(visible)      — show/hide the volume actor
    reset_camera()                   — frame the volume
    """

    # Mesh colour: semi-transparent red (vessels)
    _MESH_COLOR   = (0.95, 0.25, 0.18)
    _MESH_OPACITY = 0.82

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._volume_loaded = False
        self._current_preset = "CTA"
        self._current_mode = "VR (Ray Casting)"
        self._hu_min: float = _DEFAULT_HU_MIN
        self._hu_max: float = _DEFAULT_HU_MAX
        self._mesh_actor: vtk.vtkActor | None = None
        # clip index → vtkActor
        self._clip_actors: dict[int, vtk.vtkActor] = {}

        # Bone subtraction state
        self._raw_volume: np.ndarray | None = None
        self._raw_spacing: tuple[float, float, float] | None = None
        self._modality: str = ""                          # DICOM modality tag
        self._bone_sub_volume: np.ndarray | None = None   # computed lazily
        self._bone_sub_active: bool = False
        self._bone_sub_worker: _BoneSubtractionWorker | None = None

        # Modalities for which bone subtraction is meaningful (Hounsfield-calibrated CT)
        self._CT_MODALITIES: frozenset[str] = frozenset({"CT", "CTA", "PT"})

        # Interactive crop box (vtkBoxWidget2 — created lazily on first use)
        self._box_widget: vtk.vtkBoxWidget2 | None = None
        # Axis-aligned bounds of the crop box, kept in sync via an observer
        # that fires on EndInteractionEvent.  Reliable even if GetPolyData()
        # returns stale/zeroed coordinates on programmatic reads.
        # Format: (xmin, xmax, ymin, ymax, zmin, zmax) in world mm.
        self._current_box_bounds: tuple | None = None

        self._build_vtk_pipeline()
        self._build_ui()

    # ------------------------------------------------------------------ #
    # VTK pipeline                                                         #
    # ------------------------------------------------------------------ #

    def _build_vtk_pipeline(self) -> None:
        self._iren = QVTKRenderWindowInteractor()
        self._iren.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.05, 0.07, 0.09)
        self._renderer.SetBackground2(0.02, 0.03, 0.06)
        self._renderer.SetGradientBackground(True)
        self._iren.GetRenderWindow().AddRenderer(self._renderer)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self._iren.SetInteractorStyle(style)

        self._mapper = vtk.vtkSmartVolumeMapper()
        self._mapper.SetBlendModeToComposite()
        self._mapper.SetRequestedRenderModeToGPU()

        self._color_tf  = vtk.vtkColorTransferFunction()
        self._opacity_tf = vtk.vtkPiecewiseFunction()

        self._prop = vtk.vtkVolumeProperty()
        self._prop.SetColor(self._color_tf)
        self._prop.SetScalarOpacity(self._opacity_tf)
        self._prop.SetInterpolationTypeToLinear()

        self._volume_actor = vtk.vtkVolume()
        self._volume_actor.SetMapper(self._mapper)
        self._volume_actor.SetProperty(self._prop)
        self._renderer.AddVolume(self._volume_actor)

        # Mesh actor (segmentation result — initially hidden)
        self._mesh_mapper = vtk.vtkPolyDataMapper()
        self._mesh_actor  = vtk.vtkActor()
        self._mesh_actor.SetMapper(self._mesh_mapper)
        self._mesh_actor.GetProperty().SetColor(0.95, 0.72, 0.35)   # warm gold
        self._mesh_actor.GetProperty().SetOpacity(0.75)
        self._mesh_actor.GetProperty().SetAmbient(0.15)
        self._mesh_actor.GetProperty().SetDiffuse(0.85)
        self._mesh_actor.GetProperty().SetSpecular(0.30)
        self._mesh_actor.GetProperty().SetSpecularPower(20.0)
        self._mesh_actor.VisibilityOff()
        self._renderer.AddActor(self._mesh_actor)

        # Approach trajectory actor (semi-transparent cyan tube, initially hidden)
        self._traj_actor: vtk.vtkActor | None = None

        # Aneurysm actor (candidate highlight — magenta, initially hidden)
        self._aneurysm_mapper = vtk.vtkPolyDataMapper()
        self._aneurysm_actor  = vtk.vtkActor()
        self._aneurysm_actor.SetMapper(self._aneurysm_mapper)
        self._aneurysm_actor.GetProperty().SetColor(0.95, 0.15, 0.85)   # magenta
        self._aneurysm_actor.GetProperty().SetOpacity(0.92)
        self._aneurysm_actor.GetProperty().SetAmbient(0.20)
        self._aneurysm_actor.GetProperty().SetDiffuse(0.80)
        self._aneurysm_actor.GetProperty().SetSpecular(0.50)
        self._aneurysm_actor.GetProperty().SetSpecularPower(30.0)
        self._aneurysm_actor.VisibilityOff()
        self._renderer.AddActor(self._aneurysm_actor)

        # Orientation axes (bottom-left corner)
        axes_actor = vtk.vtkAxesActor()
        axes_actor.SetXAxisLabelText("R")
        axes_actor.SetYAxisLabelText("A")
        axes_actor.SetZAxisLabelText("S")
        self._axes = vtk.vtkOrientationMarkerWidget()
        self._axes.SetOrientationMarker(axes_actor)
        self._axes.SetInteractor(self._iren)
        self._axes.SetViewport(0.0, 0.0, 0.15, 0.15)
        self._axes.EnabledOn()
        self._axes.InteractiveOff()

        apply_preset(self._current_preset, self._color_tf, self._opacity_tf, self._prop)
        self._clip_opacity()
        self._iren.Initialize()

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header — kept dark-green regardless of theme (sits above the dark VTK canvas)
        header = QLabel("3D")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(
            "background:#0d2a1a; color:#69f0ae; font-weight:bold; "
            "font-size:11px; padding:3px; letter-spacing:1px;"
        )
        root.addWidget(header)

        # VTK canvas
        root.addWidget(self._iren, stretch=1)

        # ── Mode / preset bar ─────────────────────────────────────────── #
        self._bar1 = QWidget()
        b1 = QHBoxLayout(self._bar1)
        b1.setContentsMargins(4, 2, 4, 2)
        b1.setSpacing(4)

        _exp_pref = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        b1.addWidget(QLabel("Modo:"))
        self._mode_combo = QComboBox()
        self._mode_combo.setMinimumWidth(100)
        self._mode_combo.setMaximumWidth(150)
        self._mode_combo.setSizePolicy(_exp_pref)
        for name in _BLEND_MODES:
            self._mode_combo.addItem(name)
        self._mode_combo.currentTextChanged.connect(self.set_blend_mode)
        b1.addWidget(self._mode_combo)

        b1.addWidget(QLabel("TF:"))
        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(80)
        self._preset_combo.setMaximumWidth(120)
        self._preset_combo.setSizePolicy(_exp_pref)
        for name in PRESETS_3D:
            self._preset_combo.addItem(name)
        self._preset_combo.currentTextChanged.connect(self.set_preset)
        b1.addWidget(self._preset_combo)

        btn_reset = QPushButton("⟳")
        btn_reset.setFixedSize(24, 24)
        btn_reset.setToolTip("Restablecer cámara")
        btn_reset.clicked.connect(self.reset_camera)
        b1.addWidget(btn_reset)

        self._btn_bone_sub = QPushButton("⊖ Hueso")
        self._btn_bone_sub.setCheckable(True)
        self._btn_bone_sub.setMaximumWidth(90)
        self._btn_bone_sub.setSizePolicy(_exp_pref)
        self._btn_bone_sub.setToolTip(
            "Sustracción de hueso (HU > 300 → -1000).\n"
            "Cambia automáticamente al preset Vasos CTA."
        )
        self._btn_bone_sub.clicked.connect(self._toggle_bone_subtraction)
        b1.addWidget(self._btn_bone_sub)

        b1.addStretch()
        self._info_lbl = QLabel("")
        self._info_lbl.setMaximumWidth(140)
        b1.addWidget(self._info_lbl)

        root.addWidget(self._bar1)

        # ── HU density range bar ──────────────────────────────────────── #
        self._bar2 = QWidget()
        b2 = QHBoxLayout(self._bar2)
        b2.setContentsMargins(4, 2, 4, 2)
        b2.setSpacing(4)

        _exp_h = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Min HU — slider + spinbox (no text label; tooltip on slider)
        self._hu_min_slider = QSlider(Qt.Horizontal)
        self._hu_min_slider.setRange(-1000, 2000)
        self._hu_min_slider.setValue(_DEFAULT_HU_MIN)
        self._hu_min_slider.setMinimumWidth(40)
        self._hu_min_slider.setSizePolicy(_exp_h)
        self._hu_min_slider.setToolTip("HU mínimo visible")
        self._hu_min_slider.valueChanged.connect(self._on_hu_min_slider)
        b2.addWidget(self._hu_min_slider)

        self._hu_min_spin = QSpinBox()
        self._hu_min_spin.setRange(-1000, 2000)
        self._hu_min_spin.setValue(_DEFAULT_HU_MIN)
        self._hu_min_spin.setSuffix(" H")
        self._hu_min_spin.setMinimumWidth(78)
        self._hu_min_spin.valueChanged.connect(self._on_hu_min_spin)
        b2.addWidget(self._hu_min_spin)

        # Max HU — slider + spinbox
        self._hu_max_slider = QSlider(Qt.Horizontal)
        self._hu_max_slider.setRange(-1000, 3000)
        self._hu_max_slider.setValue(_DEFAULT_HU_MAX)
        self._hu_max_slider.setMinimumWidth(40)
        self._hu_max_slider.setSizePolicy(_exp_h)
        self._hu_max_slider.setToolTip("HU máximo visible")
        self._hu_max_slider.valueChanged.connect(self._on_hu_max_slider)
        b2.addWidget(self._hu_max_slider)

        self._hu_max_spin = QSpinBox()
        self._hu_max_spin.setRange(-1000, 3000)
        self._hu_max_spin.setValue(_DEFAULT_HU_MAX)
        self._hu_max_spin.setSuffix(" H")
        self._hu_max_spin.setMinimumWidth(78)
        self._hu_max_spin.valueChanged.connect(self._on_hu_max_spin)
        b2.addWidget(self._hu_max_spin)

        # Quick-preset buttons for common views
        b2.addSpacing(3)
        self._hu_preset_btns: list[QPushButton] = []
        for label, lo, hi in (
            ("Vasc",  100,  350),
            ("Cereb", -50,  120),
            ("Crán",  200, 3000),
        ):
            btn = QPushButton(label)
            btn.setFixedWidth(48)
            btn.setToolTip(f"Rango HU: {lo} – {hi}")
            btn.clicked.connect(lambda _, a=lo, b=hi: self._set_hu_range_buttons(a, b))
            b2.addWidget(btn)
            self._hu_preset_btns.append(btn)

        root.addWidget(self._bar2)

        # Apply initial theme-aware styles
        self._apply_bar_styles()

    # ------------------------------------------------------------------ #
    # Theme                                                                #
    # ------------------------------------------------------------------ #

    def _apply_bar_styles(self) -> None:
        """Apply theme-aware styles to the bottom control bars and their buttons."""
        dark = _is_dark()
        if dark:
            bar_bg1          = "#161616"
            bar_bg2          = "#111111"
            bar_border2      = "#2A2A2A"
            muted_clr        = "#9B9B9B"
            info_clr         = "#6B6B6B"
            btn_bg           = "#1F1F1F"
            btn_border       = "#363636"
            btn_hover_bg     = "#1C303F"
            btn_hover_clr    = "#EBEBEB"
            bone_checked_bg  = "#1a3a1a"
            bone_checked_bdr = "#4caf50"
            bone_checked_clr = "#69f0ae"
        else:
            bar_bg1          = "#F7F7F7"
            bar_bg2          = "#EEEEEE"
            bar_border2      = "#E5E5E5"
            muted_clr        = "#6B6B6B"
            info_clr         = "#6B6B6B"
            btn_bg           = "#F7F7F7"
            btn_border       = "#E5E5E5"
            btn_hover_bg     = "#DDE5EC"
            btn_hover_clr    = "#8B9BAA"
            bone_checked_bg  = "#e6f9e6"
            bone_checked_bdr = "#2ea043"
            bone_checked_clr = "#1a7a35"

        self._bar1.setStyleSheet(f"background:{bar_bg1};")
        self._bar2.setStyleSheet(
            f"background:{bar_bg2}; border-top:1px solid {bar_border2};"
        )

        self._info_lbl.setStyleSheet(f"color:{info_clr}; font-size:10px;")

        self._btn_bone_sub.setStyleSheet(
            f"QPushButton{{background:{btn_bg};border:1px solid {btn_border};"
            f"border-radius:5px;color:{muted_clr};font-size:10px;padding:2px 6px;}}"
            f"QPushButton:hover{{background:{btn_hover_bg};color:{btn_hover_clr};}}"
            f"QPushButton:checked{{background:{bone_checked_bg};"
            f"border-color:{bone_checked_bdr};color:{bone_checked_clr};font-weight:bold;}}"
        )

        hu_btn_qss = (
            f"QPushButton{{background:{btn_bg};border:1px solid {btn_border};"
            f"border-radius:5px;color:{muted_clr};font-size:10px;padding:1px 4px;}}"
            f"QPushButton:hover{{background:{btn_hover_bg};color:{btn_hover_clr};}}"
        )
        for btn in self._hu_preset_btns:
            btn.setStyleSheet(hu_btn_qss)

    def apply_theme(self) -> None:
        """Called by the main window when the user toggles the colour theme."""
        self._apply_bar_styles()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_volume(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        modality: str = "",
    ) -> None:
        logger.info(
            "Loading volume into VTK: shape=%s spacing=%s modality=%s",
            volume.shape, spacing, modality,
        )
        self._modality = modality.upper().strip()

        # Reset bone subtraction state for the new series
        self._raw_volume = volume
        self._raw_spacing = spacing
        self._bone_sub_volume = None
        self._bone_sub_active = False
        self._btn_bone_sub.setChecked(False)

        is_ct = self._modality in self._CT_MODALITIES or self._modality == ""
        self._btn_bone_sub.setEnabled(is_ct)
        self._btn_bone_sub.setToolTip(
            "Sustracción de hueso — solo disponible para CT/CTA.\n"
            f"Modalidad actual: {self._modality or 'desconocida'}"
            if not is_ct else
            "Sustracción de hueso (HU > 300 → -1000).\n"
            "Cambia automáticamente al preset Vasos CTA."
        )

        self._load_vtk_volume(volume, spacing)

    def _load_vtk_volume(
        self, volume: np.ndarray, spacing: tuple[float, float, float]
    ) -> None:
        """Push *volume* into the VTK mapper and refresh the render."""
        img_data = self._numpy_to_vtk_image(volume, spacing)
        self._mapper.SetInputData(img_data)
        self._apply_current_tf()
        self._renderer.ResetCamera()
        self._iren.GetRenderWindow().Render()

        z, y, x = volume.shape
        sz, sy, sx = spacing
        self._info_lbl.setText(f"{x}×{y}×{z}  |  {sx:.2f}×{sy:.2f}×{sz:.2f} mm")
        self._volume_loaded = True

    def set_preset(self, name: str) -> None:
        self._current_preset = name
        self._apply_current_tf()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def set_blend_mode(self, name: str) -> None:
        self._current_mode = name
        if name == "MIP":
            self._mapper.SetBlendModeToMaximumIntensity()
        elif name == "MinIP":
            self._mapper.SetBlendModeToMinimumIntensity()
        else:
            self._mapper.SetBlendModeToComposite()
        self._preset_combo.setEnabled(name == "VR (Ray Casting)")
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def set_hu_range(self, min_hu: float, max_hu: float) -> None:
        """Set the visible HU range. Voxels outside [min_hu, max_hu] → transparent."""
        self._hu_min = float(min_hu)
        self._hu_max = float(max_hu)
        self._apply_current_tf()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        self._renderer.ResetCamera()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def set_mesh(self, poly_data: vtk.vtkPolyData) -> None:
        """Load a segmentation mesh and display it as a surface overlay."""
        self._mesh_mapper.SetInputData(poly_data)
        self._mesh_mapper.Update()
        self._mesh_actor.VisibilityOn()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()
        logger.info(
            "Mesh loaded: %d verts / %d tris",
            poly_data.GetNumberOfPoints(),
            poly_data.GetNumberOfPolys(),
        )

    def set_mesh_visible(self, visible: bool) -> None:
        """Show or hide the segmentation mesh actor."""
        self._mesh_actor.SetVisibility(visible)
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def set_volume_visible(self, visible: bool) -> None:
        """Show or hide the volume rendering actor."""
        self._volume_actor.SetVisibility(visible)
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def set_trajectory(
        self,
        entry: tuple[float, float, float] | None,
        target: tuple[float, float, float] | None,
    ) -> None:
        """Show/update/hide the surgical approach corridor tube."""
        # Remove previous actor
        if self._traj_actor is not None:
            self._renderer.RemoveActor(self._traj_actor)
            self._traj_actor = None

        if entry is None or target is None:
            if self._volume_loaded:
                self._iren.GetRenderWindow().Render()
            return

        import math
        dx = target[0] - entry[0]
        dy = target[1] - entry[1]
        dz = target[2] - entry[2]
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 1e-3:
            return

        # Line source
        line = vtk.vtkLineSource()
        line.SetPoint1(*entry)
        line.SetPoint2(*target)
        line.SetResolution(1)
        line.Update()

        # Tube around the line (radius ≈ 5 mm — typical surgical corridor)
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(5.0)
        tube.SetNumberOfSides(24)
        tube.CappingOn()
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.ScalarVisibilityOff()

        self._traj_actor = vtk.vtkActor()
        self._traj_actor.SetMapper(mapper)
        prop = self._traj_actor.GetProperty()
        prop.SetColor(0.20, 0.85, 0.95)   # cyan
        prop.SetOpacity(0.25)
        prop.SetRepresentationToSurface()
        prop.LightingOn()

        # Entry point sphere
        sphere_src = vtk.vtkSphereSource()
        sphere_src.SetCenter(*entry)
        sphere_src.SetRadius(3.0)
        sphere_src.SetPhiResolution(16)
        sphere_src.SetThetaResolution(16)
        sphere_src.Update()

        append = vtk.vtkAppendPolyData()
        append.AddInputConnection(tube.GetOutputPort())
        append.AddInputConnection(sphere_src.GetOutputPort())
        append.Update()

        mapper2 = vtk.vtkPolyDataMapper()
        mapper2.SetInputConnection(append.GetOutputPort())
        mapper2.ScalarVisibilityOff()
        self._traj_actor.SetMapper(mapper2)

        self._renderer.AddActor(self._traj_actor)
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def set_aneurysm(self, poly_data: vtk.vtkPolyData | None) -> None:
        """Display an aneurysm candidate mesh (magenta) over the scene."""
        if poly_data is None:
            self._aneurysm_actor.VisibilityOff()
        else:
            self._aneurysm_mapper.SetInputData(poly_data)
            self._aneurysm_mapper.Update()
            self._aneurysm_actor.VisibilityOn()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def set_aneurysm_visible(self, visible: bool) -> None:
        """Show or hide the aneurysm highlight actor."""
        self._aneurysm_actor.SetVisibility(visible)
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    # ------------------------------------------------------------------ #
    # Clip management                                                      #
    # ------------------------------------------------------------------ #

    def add_clip(
        self,
        index: int,
        spec,
        transform: vtk.vtkTransform,
        custom_poly_data: vtk.vtkPolyData | None = None,
    ) -> None:
        """Add a new clip actor to the scene.

        If *custom_poly_data* is provided it is used directly (imported STL/OBJ).
        Otherwise the actor geometry is built from *spec* via clip_actor.py.
        """
        if custom_poly_data is not None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(custom_poly_data)
            normals.ComputePointNormalsOn()
            normals.SplittingOff()
            normals.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.75, 0.80, 0.90)
            actor.GetProperty().SetAmbient(0.20)
            actor.GetProperty().SetDiffuse(0.70)
            actor.GetProperty().SetSpecular(0.80)
            actor.GetProperty().SetSpecularPower(60.0)
        else:
            from prospective.rendering.clip_actor import build_clip_actor
            actor = build_clip_actor(spec)

        actor.SetUserTransform(transform)
        self._clip_actors[index] = actor
        self._renderer.AddActor(actor)
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def remove_clip(self, index: int) -> None:
        actor = self._clip_actors.pop(index, None)
        if actor is not None:
            self._renderer.RemoveActor(actor)
            if self._volume_loaded:
                self._iren.GetRenderWindow().Render()

    def set_clip_visible(self, index: int, visible: bool) -> None:
        actor = self._clip_actors.get(index)
        if actor is not None:
            actor.SetVisibility(visible)
            if self._volume_loaded:
                self._iren.GetRenderWindow().Render()

    def update_clip_transform(self, index: int, transform: vtk.vtkTransform) -> None:
        actor = self._clip_actors.get(index)
        if actor is not None:
            actor.SetUserTransform(transform)
            if self._volume_loaded:
                self._iren.GetRenderWindow().Render()

    # ------------------------------------------------------------------ #
    # Bone subtraction                                                     #
    # ------------------------------------------------------------------ #

    def _toggle_bone_subtraction(self, checked: bool) -> None:
        """Toggle between the raw volume and the bone-subtracted volume."""
        if self._raw_volume is None:
            self._btn_bone_sub.setChecked(False)
            return

        # Guard: bone subtraction is only valid for Hounsfield-calibrated CT volumes.
        if checked and self._modality and self._modality not in self._CT_MODALITIES:
            self._btn_bone_sub.setChecked(False)
            QMessageBox.warning(
                self,
                "Sustracción de hueso no disponible",
                f"<b>Modalidad: {self._modality}</b><br><br>"
                "La sustracción de hueso por umbral HU solo es válida para volúmenes "
                "<b>CT / CTA</b> con valores Hounsfield calibrados.<br><br>"
                "Para datos <b>XA / DSA</b> utilice el modo <b>MIP</b> con el clip "
                "HU ajustado al rango de contraste vascular (~100–3000 HU).",
            )
            return

        self._bone_sub_active = checked

        if not checked:
            # Restore original volume and reset HU clip to a broad default
            self._set_hu_range_silent(_DEFAULT_HU_MIN, _DEFAULT_HU_MAX)
            self._load_vtk_volume(self._raw_volume, self._raw_spacing)
            return

        if self._bone_sub_volume is not None:
            # Use cached result immediately
            self._apply_bone_sub_volume(self._bone_sub_volume)
            return

        # Compute in background — disable button while working
        self._btn_bone_sub.setEnabled(False)
        self._btn_bone_sub.setText("Procesando…")
        self._bone_sub_worker = _BoneSubtractionWorker(self._raw_volume)
        self._bone_sub_worker.finished.connect(self._on_bone_sub_done)
        self._bone_sub_worker.start()

    def _on_bone_sub_done(self, result: np.ndarray) -> None:
        """Called from the main thread when the background worker finishes."""
        self._bone_sub_volume = result
        self._btn_bone_sub.setEnabled(True)
        self._btn_bone_sub.setText("⊖ Hueso")

        if self._bone_sub_active:
            self._apply_bone_sub_volume(result)

    def _apply_bone_sub_volume(self, volume: np.ndarray) -> None:
        """Load the bone-subtracted volume and switch to the Vasos CTA preset."""
        # 1. Switch preset BEFORE loading so _apply_current_tf uses the right TF
        self._current_preset = "Vasos CTA"
        idx = self._preset_combo.findText("Vasos CTA")
        if idx >= 0:
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentIndex(idx)
            self._preset_combo.blockSignals(False)

        # 2. Reset HU clip sliders to the vessel window (80–400 HU) so
        #    _clip_opacity() does not zero out the vascular opacity range.
        self._set_hu_range_silent(80, 400)

        # 3. Now load the volume — _load_vtk_volume calls _apply_current_tf
        #    which already has the correct preset and HU range.
        self._load_vtk_volume(volume, self._raw_spacing)

    def _set_hu_range_silent(self, lo: int, hi: int) -> None:
        """Update the HU clip range and sync all controls without rendering."""
        for w in (self._hu_min_slider, self._hu_min_spin,
                  self._hu_max_slider, self._hu_max_spin):
            w.blockSignals(True)
        self._hu_min_slider.setValue(lo)
        self._hu_min_spin.setValue(lo)
        self._hu_max_slider.setValue(hi)
        self._hu_max_spin.setValue(hi)
        for w in (self._hu_min_slider, self._hu_min_spin,
                  self._hu_max_slider, self._hu_max_spin):
            w.blockSignals(False)
        self._hu_min = float(lo)
        self._hu_max = float(hi)

    # ------------------------------------------------------------------ #
    # Interactive crop-box widget (vtkBoxWidget2)                          #
    # ------------------------------------------------------------------ #

    def show_crop_box(
        self,
        enabled: bool,
        init_bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> None:
        """Show or hide a draggable 3-D bounding box overlaid on the mesh.

        Parameters
        ----------
        enabled:
            True  → create (if needed) and enable the box widget.
            False → hide the box widget.
        init_bounds:
            (xmin, xmax, ymin, ymax, zmin, zmax) in world coordinates.
            When None the box is initialised to the current mesh bounds.
            Ignored when *enabled* is False.
        """
        if enabled:
            # ── Lazy creation ─────────────────────────────────────────── #
            if self._box_widget is None:
                rep = vtk.vtkBoxRepresentation()
                rep.SetPlaceFactor(1.0)
                # Make handles slightly larger for easier grabbing
                rep.HandlesOn()
                # Style: translucent violet faces, white edges
                rep.GetFaceProperty().SetColor(0.49, 0.24, 0.93)   # #4E6678
                rep.GetFaceProperty().SetOpacity(0.12)
                rep.GetOutlineProperty().SetColor(0.61, 0.48, 0.88)  # #8B9BAA
                rep.GetOutlineProperty().SetLineWidth(1.5)
                rep.GetSelectedFaceProperty().SetColor(0.61, 0.48, 0.88)
                rep.GetSelectedFaceProperty().SetOpacity(0.28)

                self._box_widget = vtk.vtkBoxWidget2()
                self._box_widget.SetInteractor(self._iren)
                self._box_widget.SetRepresentation(rep)
                self._box_widget.RotationEnabledOff()    # keep axis-aligned; rotation
                                                          # would break the AABB crop
                self._box_widget.TranslationEnabledOn()
                self._box_widget.ScalingEnabledOn()
                # Observer: capture reliable bounds every time the user
                # finishes dragging a handle.  VTK guarantees the widget
                # geometry is current at EndInteractionEvent, whereas
                # GetPolyData() called on demand can return stale/zero data.
                self._box_widget.AddObserver(
                    "EndInteractionEvent", self._on_box_end_interaction
                )

            # ── Place box at init_bounds (or mesh bounds) ─────────────── #
            rep = self._box_widget.GetRepresentation()
            if init_bounds is None:
                # Prefer the mesh actor bounds; fall back to volume bounds
                if (self._mesh_actor is not None
                        and self._mesh_actor.GetVisibility()
                        and self._mesh_mapper.GetInput() is not None
                        and self._mesh_mapper.GetInput().GetNumberOfPoints() > 0):
                    init_bounds = self._mesh_actor.GetBounds()
                elif self._volume_loaded:
                    init_bounds = self._volume_actor.GetBounds()
                else:
                    init_bounds = (-50.0, 50.0, -50.0, 50.0, -50.0, 50.0)

            # Set the renderer explicitly BEFORE calling On().
            # Without this, VTK internally calls FindPokedRenderer(0, 0)
            # which returns None when the widget is activated programmatically
            # (i.e., no prior mouse event over the render window), causing
            # On() to silently abort — the box appears but is not interactive.
            self._box_widget.SetCurrentRenderer(self._renderer)
            rep.PlaceWidget(init_bounds)
            self._box_widget.On()

            # Seed the cached bounds from the initial placement so that
            # get_crop_box_bounds() returns valid data even before the user
            # moves a handle for the first time.
            self._current_box_bounds = tuple(float(v) for v in init_bounds)

            # Always render immediately so the box appears without waiting
            # for the next user interaction.
            self._iren.GetRenderWindow().Render()
        else:
            if self._box_widget is not None:
                self._box_widget.Off()
            if self._volume_loaded:
                self._iren.GetRenderWindow().Render()

    def get_crop_box_bounds(
        self,
    ) -> tuple[float, float, float, float, float, float]:
        """Return the current axis-aligned bounding box of the crop widget.

        Priority:
        1. ``_current_box_bounds`` — updated by the EndInteractionEvent observer
           every time the user finishes moving a handle; also seeded on placement.
           This is the most reliable source.
        2. Fallback: read the representation's corner points on-demand (may be
           stale before the first EndInteractionEvent fires).
        3. Fallback: the mesh actor's own bounds (box not active / not moved).

        Returns (xmin, xmax, ymin, ymax, zmin, zmax) in world coordinates.
        """
        if self._box_widget is not None and self._box_widget.GetEnabled():
            # Primary: observer-cached value (guaranteed current)
            if self._current_box_bounds is not None:
                return self._current_box_bounds  # type: ignore[return-value]

            # Secondary: read the rep's corner geometry on demand
            rep = self._box_widget.GetRepresentation()
            poly = vtk.vtkPolyData()
            rep.GetPolyData(poly)           # 15-point poly: 8 corners + handles
            pts = poly.GetPoints()
            n = pts.GetNumberOfPoints()
            if n > 0:
                from vtkmodules.util.numpy_support import vtk_to_numpy
                coords = vtk_to_numpy(pts.GetData()).reshape(n, 3)
                xmin, ymin, zmin = coords.min(axis=0)
                xmax, ymax, zmax = coords.max(axis=0)
                return (float(xmin), float(xmax),
                        float(ymin), float(ymax),
                        float(zmin), float(zmax))

        # Fallback to mesh actor bounds
        if (self._mesh_actor is not None
                and self._mesh_mapper.GetInput() is not None
                and self._mesh_mapper.GetInput().GetNumberOfPoints() > 0):
            b = self._mesh_actor.GetBounds()
            return (b[0], b[1], b[2], b[3], b[4], b[5])
        return (-100.0, 100.0, -100.0, 100.0, -100.0, 100.0)

    # ── EndInteractionEvent observer ──────────────────────────────────── #

    def _on_box_end_interaction(self, caller: object, event: str) -> None:
        """Called by VTK when the user releases the mouse after dragging a
        box handle.  At this point the representation geometry is guaranteed
        to be current, so we read and cache the axis-aligned bounds.

        vtkBoxRepresentation.GetPolyData() returns up to 15 points:
          0-7  → the 8 box corners  (the only ones that define the bounds)
          8-13 → 6 face-center handles (inside the box, don't affect min/max)
          14   → body-center handle   (also inside the box)
        We use only the first 8 corner points to avoid any numerical noise
        from the inner handle positions.
        """
        if self._box_widget is None:
            return
        rep = self._box_widget.GetRepresentation()
        poly = vtk.vtkPolyData()
        rep.GetPolyData(poly)
        pts = poly.GetPoints()
        if pts is None:
            logger.warning("Crop box: GetPolyData returned no points — bounds not updated")
            return
        n = pts.GetNumberOfPoints()
        if n < 8:
            logger.warning(
                "Crop box: GetPolyData returned only %d points (expected ≥8) — "
                "bounds not updated", n
            )
            return

        from vtkmodules.util.numpy_support import vtk_to_numpy
        coords = vtk_to_numpy(pts.GetData()).reshape(n, 3)

        # Use only the 8 actual corners (indices 0-7)
        corners = coords[:8]

        # Sanity check: if all corners are near zero, VTK hasn't synced yet.
        if np.abs(corners).max() < 1e-6:
            logger.warning(
                "Crop box: corner coords are all near-zero — keeping cached bounds"
            )
            return

        xmin, ymin, zmin = corners.min(axis=0)
        xmax, ymax, zmax = corners.max(axis=0)

        # Guard: degenerate box (collapsed to a plane or point)
        if (xmax - xmin) < 1e-3 or (ymax - ymin) < 1e-3 or (zmax - zmin) < 1e-3:
            logger.warning(
                "Crop box: degenerate box detected "
                "(x=%.3f y=%.3f z=%.3f span) — bounds not updated",
                xmax - xmin, ymax - ymin, zmax - zmin,
            )
            return

        self._current_box_bounds = (
            float(xmin), float(xmax),
            float(ymin), float(ymax),
            float(zmin), float(zmax),
        )
        logger.debug(
            "Crop box updated: x=[%.1f,%.1f] y=[%.1f,%.1f] z=[%.1f,%.1f]",
            *self._current_box_bounds,
        )

    # ------------------------------------------------------------------ #
    # HU slider / spinbox handlers                                         #
    # ------------------------------------------------------------------ #

    def _on_hu_min_slider(self, value: int) -> None:
        self._hu_min_spin.blockSignals(True)
        self._hu_min_spin.setValue(value)
        self._hu_min_spin.blockSignals(False)
        self._hu_min = float(value)
        self._apply_current_tf()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def _on_hu_min_spin(self, value: int) -> None:
        self._hu_min_slider.blockSignals(True)
        self._hu_min_slider.setValue(value)
        self._hu_min_slider.blockSignals(False)
        self._hu_min = float(value)
        self._apply_current_tf()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def _on_hu_max_slider(self, value: int) -> None:
        self._hu_max_spin.blockSignals(True)
        self._hu_max_spin.setValue(value)
        self._hu_max_spin.blockSignals(False)
        self._hu_max = float(value)
        self._apply_current_tf()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def _on_hu_max_spin(self, value: int) -> None:
        self._hu_max_slider.blockSignals(True)
        self._hu_max_slider.setValue(value)
        self._hu_max_slider.blockSignals(False)
        self._hu_max = float(value)
        self._apply_current_tf()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    def _set_hu_range_buttons(self, lo: int, hi: int) -> None:
        """Quick-preset buttons — sync both sliders + spinboxes + re-render."""
        for w in (self._hu_min_slider, self._hu_min_spin,
                  self._hu_max_slider, self._hu_max_spin):
            w.blockSignals(True)
        self._hu_min_slider.setValue(lo)
        self._hu_min_spin.setValue(lo)
        self._hu_max_slider.setValue(hi)
        self._hu_max_spin.setValue(hi)
        for w in (self._hu_min_slider, self._hu_min_spin,
                  self._hu_max_slider, self._hu_max_spin):
            w.blockSignals(False)
        self._hu_min = float(lo)
        self._hu_max = float(hi)
        self._apply_current_tf()
        if self._volume_loaded:
            self._iren.GetRenderWindow().Render()

    # ------------------------------------------------------------------ #
    # Transfer function helpers                                            #
    # ------------------------------------------------------------------ #

    def _apply_current_tf(self) -> None:
        """Re-apply the current preset and then clip to [_hu_min, _hu_max]."""
        apply_preset(self._current_preset, self._color_tf, self._opacity_tf, self._prop)
        self._clip_opacity()

    def _clip_opacity(self) -> None:
        """
        Zero out opacity outside [_hu_min, _hu_max] by modifying the
        piecewise opacity function in-place.

        Strategy:
          1. Sample the opacity value at both clip boundaries.
          2. Remove all existing points outside the clip range.
          3. Insert hard-stop points just outside the range.
        """
        lo, hi = self._hu_min, self._hu_max
        if lo >= hi:
            return

        # Sample opacity at the boundaries before modifying
        op_lo = float(self._opacity_tf.GetValue(lo))
        op_hi = float(self._opacity_tf.GetValue(hi))

        # Collect and remove points outside [lo, hi]
        to_remove = []
        for i in range(self._opacity_tf.GetSize()):
            node = [0.0, 0.0, 0.0, 0.0]
            self._opacity_tf.GetNodeValue(i, node)
            x = node[0]
            if x < lo or x > hi:
                to_remove.append(x)
        for x in to_remove:
            self._opacity_tf.RemovePoint(x)

        # Insert boundary clamps (sharp step over 0.5 HU)
        self._opacity_tf.AddPoint(lo - 0.5, 0.0)
        self._opacity_tf.AddPoint(lo,        op_lo)
        self._opacity_tf.AddPoint(hi,        op_hi)
        self._opacity_tf.AddPoint(hi + 0.5,  0.0)
        self._opacity_tf.Modified()

    # ------------------------------------------------------------------ #
    # Static helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_to_vtk_image(
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> vtk.vtkImageData:
        from vtkmodules.util import numpy_support as ns

        z, y, x = volume.shape
        sz, sy, sx = spacing

        img = vtk.vtkImageData()
        img.SetDimensions(x, y, z)
        img.SetSpacing(sx, sy, sz)
        img.SetOrigin(0.0, 0.0, 0.0)

        flat = np.ascontiguousarray(volume, dtype=np.float32).ravel()
        vtk_arr = ns.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName("HU")
        img.GetPointData().SetScalars(vtk_arr)
        return img

    # ------------------------------------------------------------------ #
    # Screenshot                                                           #
    # ------------------------------------------------------------------ #

    def capture_png(self) -> bytes:
        """Render the current 3D view to a PNG and return the raw bytes."""
        rw = self._iren.GetRenderWindow()
        rw.Render()

        win_to_img = vtk.vtkWindowToImageFilter()
        win_to_img.SetInput(rw)
        win_to_img.SetScale(1)
        win_to_img.ReadFrontBufferOff()
        win_to_img.Update()

        writer = vtk.vtkPNGWriter()
        writer.WriteToMemoryOn()
        writer.SetInputConnection(win_to_img.GetOutputPort())
        writer.Write()

        data = writer.GetResult()
        return bytes(bytearray(data))
