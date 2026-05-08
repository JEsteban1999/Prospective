"""Intracranial stent / flow-diverter planning panel.

Features
--------
* Stent catalogue (Pipeline PED, Surpass, FRED, Neuroform, Enterprise 2, Leo+).
* Filter by device type (flow diverter / intracranial stent).
* Custom STL/OBJ import.
* Live spinbox repositioning of placed stents.
* Export placed-stent plan as CSV.

Signals
-------
stent_placed(index, spec_name, transform, poly_data|None)
stent_removed(index)
stent_transform_changed(index, transform)
stent_visibility_changed(index, bool)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import vtk
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from prospective.models.stent_library import STENT_CATALOGUE, StentSpec, StentType

logger = logging.getLogger(__name__)


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


# ──────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

class _StentEntry:
    """Unified representation for catalogue and custom stents."""

    def __init__(
        self,
        name: str,
        poly_data: vtk.vtkPolyData | None = None,
        spec: StentSpec | None = None,
    ) -> None:
        self.name      = name
        self.poly_data = poly_data
        self.spec      = spec

    @property
    def list_label(self) -> str:
        if self.spec is not None:
            return f"{self.spec.stent_type.value}  |  {self.spec.display_label}"
        return f"[Custom]  {self.name}"

    @property
    def info_text(self) -> str:
        if self.spec is not None:
            return self.spec.info_text
        if self.poly_data is not None:
            n = self.poly_data.GetNumberOfPoints()
            t = self.poly_data.GetNumberOfPolys()
            return f"Stent personalizado — {n:,} verts · {t:,} tris"
        return "—"


class _PlacedStent:
    def __init__(
        self,
        index: int,
        entry: _StentEntry,
        transform: vtk.vtkTransform,
    ) -> None:
        self.index     = index
        self.entry     = entry
        self.transform = transform
        self.visible   = True

    @property
    def label(self) -> str:
        return f"#{self.index}  {self.entry.name}"


# ──────────────────────────────────────────────────────────────────────────── #
# Panel                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

class StentPanel(QWidget):
    """Intracranial stent planning dock panel."""

    # index, spec_name, vtkTransform, vtkPolyData|None
    stent_placed             = pyqtSignal(int, str, object, object)
    stent_removed            = pyqtSignal(int)
    stent_transform_changed  = pyqtSignal(int, object)
    stent_visibility_changed = pyqtSignal(int, bool)
    # Emitted when user clicks "Redefinir forma" — planning window enters deform mode
    stent_deform_requested   = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._placed: list[_PlacedStent]   = []
        self._next_index                   = 1
        self._custom_stents: list[_StentEntry] = []
        self._entries: list[_StentEntry]   = []
        self._editing_placed_row: int      = -1
        self._vessel_mesh: vtk.vtkPolyData | None = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_vessel_mesh(self, poly_data: vtk.vtkPolyData | None) -> None:
        self._vessel_mesh = poly_data

    def set_vessel_curvature(self, radius_mm: float) -> None:
        """Set vessel curvature radius (mm) estimated from the trajectory spline.

        A radius of 0 means straight/unknown.  Updates the LCF label so the
        user sees the correction factor before clicking "Calcular sizing".
        """
        self._spin_curvature.blockSignals(True)
        self._spin_curvature.setValue(max(0.0, radius_mm))
        self._spin_curvature.blockSignals(False)
        lcf = self._compute_lcf()
        self._lbl_lcf.setText(
            f"LCF = {lcf:.3f}" if lcf != 1.0 else "LCF = 1.000  (sin corrección)"
        )

    def _compute_lcf(self) -> float:
        """Length Change Function: corrects nominal FD length for curved vessels.

        Based on Dazeo et al. 2024 — for a stent of diameter d deployed in a
        vessel with curvature radius R, the nominal catalogue length is shorter
        than the actual arc length by a factor:

            LCF = 1 + d / (4 * R)

        Returns 1.0 when curvature is unknown (R == 0) or vessel is straight.
        """
        radius = self._spin_curvature.value()
        if radius < 1.0:          # unknown or effectively straight
            return 1.0
        # Use the reference diameter (mean of prox + dist if both set)
        prox = self._spin_prox.value()
        dist = self._spin_dist.value()
        if prox > 0 and dist > 0:
            ref_diam = (prox + dist) / 2.0
        elif prox > 0:
            ref_diam = prox
        elif dist > 0:
            ref_diam = dist
        else:
            ref_diam = 3.5          # fallback typical FD diameter
        return 1.0 + ref_diam / (4.0 * radius)

    def set_neck_data(
        self, neck_length_mm: float, parent_artery_mm: float = 0.0
    ) -> None:
        """Auto-populate sizing inputs from a morphometrics result.

        Called from MainWindow when morpho_panel emits analysis_done.
        ``neck_length_mm`` maps to the neck-length spinbox.
        ``parent_artery_mm`` (optional) pre-fills both proximal and distal
        diameter spinboxes when a parent-artery diameter was provided.
        """
        if neck_length_mm > 0:
            self._spin_neck_len.setValue(neck_length_mm)
        if parent_artery_mm > 0:
            for spin in (self._spin_prox, self._spin_dist):
                spin.blockSignals(True)
                spin.setValue(parent_artery_mm)
                spin.blockSignals(False)

    def set_aneurysm_centroid(self, x: float, y: float, z: float) -> None:
        """Pre-fill position spinboxes from the aneurysm centroid."""
        for spin, val in zip((self._tx, self._ty, self._tz), (x, y, z)):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    def place_stent_programmatic(
        self, spec_name: str, transform: vtk.vtkTransform
    ) -> None:
        """Place a stent without UI interaction (called from trajectory auto-placement).

        Emits ``stent_placed`` so the planning window adds the actor to the 3D scene.
        The new entry also appears in the placed-stents list for visibility control.
        """
        spec = next((s for s in STENT_CATALOGUE if s.name == spec_name), None)
        if spec is None:
            logger.warning("place_stent_programmatic: spec '%s' not found", spec_name)
            return
        entry  = _StentEntry(spec.name, None, spec)
        placed = _PlacedStent(self._next_index, entry, transform)
        self._placed.append(placed)
        self._next_index += 1
        self._placed_list.addItem(placed.label)
        self._placed_list.setCurrentRow(self._placed_list.count() - 1)
        self._btn_export.setEnabled(True)
        self.stent_placed.emit(placed.index, spec.name, transform, None)
        return placed.index   # caller can track this for nudge controls

    def update_placed_transform(
        self, index: int, transform: vtk.vtkTransform
    ) -> None:
        """Update the stored transform for an already-placed stent.

        Called by PlanningWindow when nudge controls move the stent.
        Updates internal state and emits ``stent_transform_changed`` so the
        3-D actor follows.
        """
        for ps in self._placed:
            if ps.index == index:
                ps.transform = transform
                self.stent_transform_changed.emit(index, transform)
                return

    def restore_session_state(self, d: dict) -> None:
        """Restore stent placements from a session dict (inverse of get_session_state).

        Each stent is placed programmatically and its saved transform is applied.
        Orientation is stored as [rz, rx, ry] degrees, matching the order produced
        by vtkTransform.GetOrientation() → [ori[2], ori[0], ori[1]] in get_session_state.
        """
        for sd in d.get("stents", []):
            pos = sd.get("position", [0.0, 0.0, 0.0])
            ori = sd.get("orientation", [0.0, 0.0, 0.0])  # [rz, rx, ry]

            t = vtk.vtkTransform()
            t.Identity()
            t.Translate(*pos)
            t.RotateZ(ori[0])   # yaw  — matches _build_transform()
            t.RotateX(ori[1])   # pitch
            t.RotateY(ori[2])   # roll

            idx = self.place_stent_programmatic(sd.get("name", ""), t)
            if idx is None:
                logger.warning(
                    "restore_session_state: stent '%s' not found in catalogue — skipped",
                    sd.get("name"),
                )

    def get_session_state(self) -> dict:
        stents = []
        for ps in self._placed:
            pos = list(ps.transform.GetPosition())
            ori = list(ps.transform.GetOrientation())
            stents.append({
                "index":       ps.index,
                "name":        ps.entry.name,
                "is_custom":   ps.entry.poly_data is not None,
                "custom_path": getattr(ps.entry, "_source_path", None),
                "position":    [round(v, 3) for v in pos],
                "orientation": [round(v, 3) for v in [ori[2], ori[0], ori[1]]],
            })
        return {"stents": stents}

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.NoFrame)
        content = QWidget()
        layout  = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # Guide text
        guide = QLabel(
            "<small style='color:#9B9B9B'>"
            "El stent se orienta a lo largo del eje Z local. "
            "Ajuste posición y rotación para alinearlo con el vaso parental."
            "</small>"
        )
        guide.setWordWrap(True)
        layout.addWidget(guide)

        # ── Device selection ──────────────────────────────────────────── #
        sel_grp = QGroupBox("Selección de dispositivo")
        sf = QVBoxLayout()
        sf.setSpacing(4)

        filter_row = QWidget()
        fr = QHBoxLayout(filter_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.setSpacing(8)

        self._chk_flow_div = QCheckBox("Desviadores de flujo")
        self._chk_flow_div.setChecked(True)
        self._chk_flow_div.toggled.connect(self._refresh_stent_list)
        fr.addWidget(self._chk_flow_div)

        self._chk_ic = QCheckBox("Stents IC")
        self._chk_ic.setChecked(True)
        self._chk_ic.toggled.connect(self._refresh_stent_list)
        fr.addWidget(self._chk_ic)
        sf.addWidget(filter_row)

        self._stent_list = QListWidget()
        self._stent_list.setAlternatingRowColors(True)
        self._stent_list.setMinimumHeight(110)
        self._stent_list.currentRowChanged.connect(self._on_entry_changed)
        sf.addWidget(self._stent_list)

        self._lbl_info = QLabel("—")
        self._lbl_info.setWordWrap(True)
        self._lbl_info.setProperty("role", "muted")
        sf.addWidget(self._lbl_info)

        self._btn_import = QPushButton("Importar stent STL/OBJ…")
        if _is_dark():
            self._btn_import.setStyleSheet(
                "QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:8px;color:#9B9B9B;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#1C303F;color:#EBEBEB;}"
            )
        else:
            self._btn_import.setStyleSheet(
                "QPushButton{background:#F7F7F7;border:1px solid #D0D0D0;"
                "border-radius:8px;color:#6B6B6B;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#DDE5EC;color:#0D0D0D;}"
            )
        self._btn_import.clicked.connect(self._import_custom)
        sf.addWidget(self._btn_import)

        sel_grp.setLayout(sf)
        layout.addWidget(sel_grp)

        # ── Position & Orientation ────────────────────────────────────── #
        xform_grp = QGroupBox("Posición / Orientación  (stent seleccionado)")
        xf = QFormLayout()
        xf.setLabelAlignment(Qt.AlignRight)
        xf.setVerticalSpacing(3)

        self._tx = self._make_pos_spin(); xf.addRow("X:", self._tx)
        self._ty = self._make_pos_spin(); xf.addRow("Y:", self._ty)
        self._tz = self._make_pos_spin(); xf.addRow("Z:", self._tz)
        self._ry = self._make_rot_spin(); xf.addRow("Yaw:", self._ry)
        self._rp = self._make_rot_spin(); xf.addRow("Pitch:", self._rp)
        self._rr = self._make_rot_spin(); xf.addRow("Roll:", self._rr)

        for spin in (self._tx, self._ty, self._tz, self._ry, self._rp, self._rr):
            spin.valueChanged.connect(self._on_spinbox_changed)

        xform_grp.setLayout(xf)
        layout.addWidget(xform_grp)

        # ── Place button ──────────────────────────────────────────────── #
        self._btn_place = QPushButton("Colocar stent en escena")
        self._btn_place.setMinimumHeight(30)
        if _is_dark():
            self._btn_place.setStyleSheet(
                "QPushButton{background:#1C303F;border:1px solid #8B9BAA;"
                "border-radius:9px;color:#EBEBEB;font-weight:bold;}"
                "QPushButton:hover{background:#4E6678;}"
            )
        else:
            self._btn_place.setStyleSheet(
                "QPushButton{background:#DDE5EC;border:1px solid #8B9BAA;"
                "border-radius:9px;color:#2E4A5F;font-weight:bold;}"
                "QPushButton:hover{background:#8B9BAA;color:#ffffff;}"
            )
        self._btn_place.clicked.connect(self._place_stent)
        layout.addWidget(self._btn_place)

        # ── Placed stents ─────────────────────────────────────────────── #
        placed_grp = QGroupBox("Stents en escena")
        pl = QVBoxLayout()
        pl.setSpacing(4)

        self._placed_list = QListWidget()
        self._placed_list.setAlternatingRowColors(True)
        self._placed_list.setMaximumHeight(90)
        self._placed_list.currentRowChanged.connect(self._on_placed_selected)
        pl.addWidget(self._placed_list)

        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(4)

        self._btn_toggle = QPushButton("Mostrar/Ocultar")
        self._btn_toggle.setEnabled(False)
        self._btn_toggle.clicked.connect(self._toggle_selected)
        br.addWidget(self._btn_toggle)

        self._btn_remove = QPushButton("Eliminar")
        self._btn_remove.setEnabled(False)
        self._btn_remove.setStyleSheet(
            "QPushButton{color:#f85149;}"
            "QPushButton:hover{background:rgba(248,81,73,15);border-color:#f85149;}"
        )
        self._btn_remove.clicked.connect(self._remove_selected)
        br.addWidget(self._btn_remove)

        pl.addWidget(btn_row)

        self._btn_deform = QPushButton("✏ Redefinir forma")
        self._btn_deform.setEnabled(False)
        self._btn_deform.setToolTip(
            "Remodela este stent usando la ventana 3D — define nuevos puntos "
            "de control para curvar el dispositivo sobre la arteria."
        )
        if _is_dark():
            self._btn_deform.setStyleSheet(
                "QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:8px;color:#A8B8C6;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#1C303F;border-color:#A8B8C6;}"
                "QPushButton:disabled{color:#5a5a5a;border-color:#363636;}"
            )
        else:
            self._btn_deform.setStyleSheet(
                "QPushButton{background:#F0F4F8;border:1px solid #C8D0D8;"
                "border-radius:8px;color:#4E6678;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#DDE5EC;border-color:#8B9BAA;}"
                "QPushButton:disabled{color:#AAAAAA;border-color:#D0D0D0;}"
            )
        self._btn_deform.clicked.connect(self._request_deform)
        pl.addWidget(self._btn_deform)
        placed_grp.setLayout(pl)
        layout.addWidget(placed_grp)

        # ── Export ────────────────────────────────────────────────────── #
        self._btn_export = QPushButton("Exportar plan CSV")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_plan)
        layout.addWidget(self._btn_export)

        # ── Sizing asistido ───────────────────────────────────────────────── #
        size_grp = QGroupBox("Sizing asistido (desviador de flujo)")
        sg = QVBoxLayout()
        sg.setSpacing(4)

        sf2 = QFormLayout()
        sf2.setLabelAlignment(Qt.AlignRight)
        sf2.setVerticalSpacing(3)

        self._spin_prox = self._make_diam_spin()
        sf2.addRow("Ø proximal:", self._spin_prox)

        self._spin_dist = self._make_diam_spin()
        sf2.addRow("Ø distal:", self._spin_dist)

        self._spin_neck_len = self._make_diam_spin()
        self._spin_neck_len.setToolTip(
            "Longitud del cuello del aneurisma (se auto-rellena desde morfometría)"
        )
        sf2.addRow("Long. cuello:", self._spin_neck_len)

        self._spin_margin = QDoubleSpinBox()
        self._spin_margin.setRange(1.0, 20.0)
        self._spin_margin.setSingleStep(0.5)
        self._spin_margin.setDecimals(1)
        self._spin_margin.setSuffix(" mm")
        self._spin_margin.setValue(5.0)
        sf2.addRow("Margen anclaje:", self._spin_margin)

        # SC-1: vessel curvature (radius, set from trajectory)
        self._spin_curvature = QDoubleSpinBox()
        self._spin_curvature.setRange(0.0, 500.0)
        self._spin_curvature.setDecimals(1)
        self._spin_curvature.setSuffix(" mm")
        self._spin_curvature.setSpecialValueText("—  (recto)")
        self._spin_curvature.setValue(0.0)
        self._spin_curvature.setReadOnly(True)
        self._spin_curvature.setToolTip(
            "Radio de curvatura del vaso padre estimado desde la trayectoria.\n"
            "Se actualiza automáticamente al definir 2+ puntos de pick.\n"
            "Valor pequeño = vaso muy curvado → corrige longitud del FD (LCF)."
        )
        sf2.addRow("Radio curvatura:", self._spin_curvature)

        self._lbl_lcf = QLabel("LCF = 1.00")
        self._lbl_lcf.setProperty("role", "muted")
        self._lbl_lcf.setToolTip(
            "Length Change Function — factor de corrección de longitud para vaso curvo.\n"
            "Longitud corregida = longitud nominal × LCF."
        )
        sf2.addRow("", self._lbl_lcf)

        sg.addLayout(sf2)

        self._btn_compute_sizing = QPushButton("Calcular sizing recomendado")
        self._btn_compute_sizing.setMinimumHeight(26)
        if _is_dark():
            self._btn_compute_sizing.setStyleSheet(
                "QPushButton{background:#1C303F;border:1px solid #8B9BAA;"
                "border-radius:8px;color:#EBEBEB;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#4E6678;}"
            )
        else:
            self._btn_compute_sizing.setStyleSheet(
                "QPushButton{background:#DDE5EC;border:1px solid #8B9BAA;"
                "border-radius:8px;color:#2E4A5F;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#8B9BAA;color:#ffffff;}"
            )
        self._btn_compute_sizing.clicked.connect(self._compute_sizing)
        sg.addWidget(self._btn_compute_sizing)

        self._lbl_sizing_result = QLabel("—")
        self._lbl_sizing_result.setWordWrap(True)
        self._lbl_sizing_result.setTextFormat(Qt.RichText)
        self._lbl_sizing_result.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._lbl_sizing_result.setStyleSheet(
            "font-size:10px; padding:4px; background:#1F1F1F; border-radius:5px;"
            if _is_dark() else
            "font-size:10px; padding:4px; background:#F0F4F8; border-radius:5px;"
            " border:1px solid #D8E0E8;"
        )
        self._lbl_sizing_result.setMinimumHeight(70)
        sg.addWidget(self._lbl_sizing_result)

        self._btn_apply_sizing = QPushButton("Aplicar mejor opción en lista")
        self._btn_apply_sizing.setEnabled(False)
        if _is_dark():
            self._btn_apply_sizing.setStyleSheet(
                "QPushButton{background:#1a3a20;border:1px solid #3fb950;"
                "border-radius:8px;color:#3fb950;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#238636;color:#ffffff;}"
                "QPushButton:disabled{color:#5a5a5a;border-color:#363636;background:#1F1F1F;}"
            )
        else:
            self._btn_apply_sizing.setStyleSheet(
                "QPushButton{background:#e6f4ea;border:1px solid #2da44e;"
                "border-radius:8px;color:#1a7a35;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#2da44e;color:#ffffff;}"
                "QPushButton:disabled{color:#AAAAAA;border-color:#D0D0D0;background:#F0F0F0;}"
            )
        self._btn_apply_sizing.clicked.connect(self._apply_best_sizing)
        sg.addWidget(self._btn_apply_sizing)

        size_grp.setLayout(sg)
        layout.addWidget(size_grp)

        layout.addStretch()
        self._refresh_stent_list()
        self._best_sizing_spec: StentSpec | None = None

    # ------------------------------------------------------------------ #
    # Theme                                                                #
    # ------------------------------------------------------------------ #

    def apply_theme(self) -> None:
        """Re-style theme-sensitive buttons/labels to match the current theme."""
        dark = _is_dark()
        if dark:
            self._btn_import.setStyleSheet(
                "QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:8px;color:#9B9B9B;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#1C303F;color:#EBEBEB;}"
            )
            self._btn_place.setStyleSheet(
                "QPushButton{background:#1C303F;border:1px solid #8B9BAA;"
                "border-radius:9px;color:#EBEBEB;font-weight:bold;}"
                "QPushButton:hover{background:#4E6678;}"
            )
            self._btn_deform.setStyleSheet(
                "QPushButton{background:#1F1F1F;border:1px solid #363636;"
                "border-radius:8px;color:#A8B8C6;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#1C303F;border-color:#A8B8C6;}"
                "QPushButton:disabled{color:#5a5a5a;border-color:#363636;}"
            )
            self._btn_compute_sizing.setStyleSheet(
                "QPushButton{background:#1C303F;border:1px solid #8B9BAA;"
                "border-radius:8px;color:#EBEBEB;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#4E6678;}"
            )
            self._lbl_sizing_result.setStyleSheet(
                "font-size:10px; padding:4px; background:#1F1F1F; border-radius:5px;"
            )
            self._btn_apply_sizing.setStyleSheet(
                "QPushButton{background:#1a3a20;border:1px solid #3fb950;"
                "border-radius:8px;color:#3fb950;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#238636;color:#ffffff;}"
                "QPushButton:disabled{color:#5a5a5a;border-color:#363636;background:#1F1F1F;}"
            )
        else:
            self._btn_import.setStyleSheet(
                "QPushButton{background:#F7F7F7;border:1px solid #D0D0D0;"
                "border-radius:8px;color:#6B6B6B;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#DDE5EC;color:#0D0D0D;}"
            )
            self._btn_place.setStyleSheet(
                "QPushButton{background:#DDE5EC;border:1px solid #8B9BAA;"
                "border-radius:9px;color:#2E4A5F;font-weight:bold;}"
                "QPushButton:hover{background:#8B9BAA;color:#ffffff;}"
            )
            self._btn_deform.setStyleSheet(
                "QPushButton{background:#F0F4F8;border:1px solid #C8D0D8;"
                "border-radius:8px;color:#4E6678;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#DDE5EC;border-color:#8B9BAA;}"
                "QPushButton:disabled{color:#AAAAAA;border-color:#D0D0D0;}"
            )
            self._btn_compute_sizing.setStyleSheet(
                "QPushButton{background:#DDE5EC;border:1px solid #8B9BAA;"
                "border-radius:8px;color:#2E4A5F;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#8B9BAA;color:#ffffff;}"
            )
            self._lbl_sizing_result.setStyleSheet(
                "font-size:10px; padding:4px; background:#F0F4F8; border-radius:5px;"
                " border:1px solid #D8E0E8;"
            )
            self._btn_apply_sizing.setStyleSheet(
                "QPushButton{background:#e6f4ea;border:1px solid #2da44e;"
                "border-radius:8px;color:#1a7a35;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#2da44e;color:#ffffff;}"
                "QPushButton:disabled{color:#AAAAAA;border-color:#D0D0D0;background:#F0F0F0;}"
            )

    # ------------------------------------------------------------------ #
    # Catalogue list                                                       #
    # ------------------------------------------------------------------ #

    def _refresh_stent_list(self) -> None:
        self._stent_list.clear()
        show_fd = self._chk_flow_div.isChecked()
        show_ic = self._chk_ic.isChecked()

        cat = [
            s for s in STENT_CATALOGUE
            if (show_fd and s.stent_type == StentType.FLOW_DIVERTER)
            or (show_ic and s.stent_type == StentType.INTRACRANIAL)
        ]

        self._entries = self._custom_stents + [_StentEntry(s.name, None, s) for s in cat]
        for e in self._entries:
            self._stent_list.addItem(e.list_label)

        if self._entries:
            self._stent_list.setCurrentRow(0)

    def _on_entry_changed(self, row: int) -> None:
        self._lbl_info.setText(
            self._entries[row].info_text if 0 <= row < len(self._entries) else "—"
        )

    # ------------------------------------------------------------------ #
    # Import custom stent                                                  #
    # ------------------------------------------------------------------ #

    def _import_custom(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Importar stent 3D", "",
            "Mallas 3D (*.stl *.obj *.STL *.OBJ);;Todos los archivos (*)",
        )
        if not path:
            return

        suffix = Path(path).suffix.lower()
        if suffix == ".stl":
            reader = vtk.vtkSTLReader()
        elif suffix == ".obj":
            reader = vtk.vtkOBJReader()
        else:
            QMessageBox.warning(self, "Formato no soportado", "Solo se aceptan STL y OBJ.")
            return

        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()
        if poly.GetNumberOfPoints() == 0:
            QMessageBox.critical(self, "Error al importar",
                                 f"El archivo no contiene geometría válida:\n{path}")
            return

        name = Path(path).stem
        entry = _StentEntry(name, poly, spec=None)
        entry._source_path = str(Path(path).resolve())
        self._custom_stents.append(entry)
        self._refresh_stent_list()
        self._stent_list.setCurrentRow(0)
        logger.info("Custom stent imported: %s  (%d pts)", name, poly.GetNumberOfPoints())

    # ------------------------------------------------------------------ #
    # Live spinbox update                                                  #
    # ------------------------------------------------------------------ #

    def _on_spinbox_changed(self) -> None:
        row = self._editing_placed_row
        if row < 0 or row >= len(self._placed):
            return
        t = self._build_transform()
        self._placed[row].transform = t
        self.stent_transform_changed.emit(self._placed[row].index, t)

    def _build_transform(self) -> vtk.vtkTransform:
        t = vtk.vtkTransform()
        t.Identity()
        t.Translate(self._tx.value(), self._ty.value(), self._tz.value())
        t.RotateZ(self._ry.value())
        t.RotateX(self._rp.value())
        t.RotateY(self._rr.value())
        return t

    # ------------------------------------------------------------------ #
    # Placement                                                            #
    # ------------------------------------------------------------------ #

    def _place_stent(self) -> None:
        row = self._stent_list.currentRow()
        if row < 0 or not self._entries:
            QMessageBox.warning(self, "Sin stent", "Seleccione un tipo de stent.")
            return

        entry = self._entries[row]
        t     = self._build_transform()

        placed = _PlacedStent(self._next_index, entry, t)
        self._placed.append(placed)
        self._next_index += 1

        self._placed_list.addItem(placed.label)
        self._placed_list.setCurrentRow(self._placed_list.count() - 1)
        self._btn_export.setEnabled(True)

        self.stent_placed.emit(placed.index, entry.name, t, entry.poly_data)
        logger.info("Stent placed #%d: %s at (%.1f, %.1f, %.1f)",
                    placed.index, entry.name,
                    self._tx.value(), self._ty.value(), self._tz.value())

    # ------------------------------------------------------------------ #
    # Placed list                                                          #
    # ------------------------------------------------------------------ #

    def _on_placed_selected(self, row: int) -> None:
        enabled = 0 <= row < len(self._placed)
        self._btn_toggle.setEnabled(enabled)
        self._btn_remove.setEnabled(enabled)
        self._btn_deform.setEnabled(enabled)

        if not enabled:
            self._editing_placed_row = -1
            return

        self._editing_placed_row = -1
        placed = self._placed[row]
        pos = placed.transform.GetPosition()
        ori = placed.transform.GetOrientation()
        self._set_spinboxes_silent(pos[0], pos[1], pos[2], ori[2], ori[0], ori[1])
        self._editing_placed_row = row

    def _toggle_selected(self) -> None:
        row = self._placed_list.currentRow()
        if 0 <= row < len(self._placed):
            self._placed[row].visible = not self._placed[row].visible
            self.stent_visibility_changed.emit(
                self._placed[row].index, self._placed[row].visible
            )

    def remove_last_placed(self) -> bool:
        """Remove the most recently placed stent (used by undo).

        Returns True if a stent was removed, False if the list was empty.
        """
        if not self._placed:
            return False
        row = len(self._placed) - 1
        self._editing_placed_row = -1
        placed = self._placed.pop(row)
        self._placed_list.takeItem(row)
        self.stent_removed.emit(placed.index)
        if not self._placed:
            self._btn_export.setEnabled(False)
        self._btn_toggle.setEnabled(False)
        self._btn_remove.setEnabled(False)
        self._btn_deform.setEnabled(False)
        return True

    def _remove_selected(self) -> None:
        row = self._placed_list.currentRow()
        if row < 0 or row >= len(self._placed):
            return
        self._editing_placed_row = -1
        placed = self._placed.pop(row)
        self._placed_list.takeItem(row)
        self.stent_removed.emit(placed.index)
        if not self._placed:
            self._btn_export.setEnabled(False)
        self._btn_toggle.setEnabled(False)
        self._btn_remove.setEnabled(False)
        self._btn_deform.setEnabled(False)

    def _request_deform(self) -> None:
        """Emit stent_deform_requested for the currently selected placed stent."""
        row = self._placed_list.currentRow()
        if 0 <= row < len(self._placed):
            self.stent_deform_requested.emit(self._placed[row].index)

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def _export_plan(self) -> None:
        if not self._placed:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar plan de stents", "plan_stents.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "Stent#", "Nombre", "Tipo",
                "Diám. (mm)", "Long. (mm)",
                "X (mm)", "Y (mm)", "Z (mm)",
                "Yaw (°)", "Pitch (°)", "Roll (°)",
            ])
            for ps in self._placed:
                pos  = ps.transform.GetPosition()
                ori  = ps.transform.GetOrientation()
                diam = f"{ps.entry.spec.diameter_mm:.1f}" if ps.entry.spec else "—"
                leng = f"{ps.entry.spec.length_mm:.0f}"  if ps.entry.spec else "—"
                tipo = ps.entry.spec.stent_type.value    if ps.entry.spec else "Personalizado"
                writer.writerow([
                    ps.index, ps.entry.name, tipo, diam, leng,
                    f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}",
                    f"{ori[2]:.1f}", f"{ori[0]:.1f}", f"{ori[1]:.1f}",
                ])

        QMessageBox.information(self, "Plan exportado", f"Guardado en:\n{path}")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _set_spinboxes_silent(self, x, y, z, yaw, pitch, roll) -> None:
        for spin, val in zip(
            (self._tx, self._ty, self._tz, self._ry, self._rp, self._rr),
            (x, y, z, yaw, pitch, roll),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Sizing asistido                                                      #
    # ------------------------------------------------------------------ #

    def _compute_sizing(self) -> None:
        """Evaluate FLOW_DIVERTER catalogue and show colour-coded recommendations."""
        prox   = self._spin_prox.value()
        dist   = self._spin_dist.value()
        neck   = self._spin_neck_len.value()
        margin = self._spin_margin.value()

        if prox <= 0 and dist <= 0:
            self._lbl_sizing_result.setText(
                "<span style='color:#f0883e'>"
                "Ingrese al menos un diámetro de arteria (proximal o distal)."
                "</span>"
            )
            self._best_sizing_spec = None
            self._btn_apply_sizing.setEnabled(False)
            return

        # Reference diameter: mean when both available, else whichever is set
        ref_diam = (prox + dist) / 2.0 if (prox > 0 and dist > 0) else max(prox, dist)

        # SC-2: Length Change Function — correct nominal length for vessel curvature
        lcf = self._compute_lcf()
        required_len = (neck + 2.0 * margin) * lcf if neck > 0 else 0.0
        self._lbl_lcf.setText(
            f"LCF = {lcf:.3f}  →  long. corregida: {required_len:.1f} mm"
            if lcf != 1.0 and neck > 0
            else f"LCF = {lcf:.3f}" if lcf != 1.0
            else "LCF = 1.000  (sin corrección)"
        )

        results: list[tuple[int, StentSpec, str, str]] = []
        for spec in STENT_CATALOGUE:
            if spec.stent_type != StentType.FLOW_DIVERTER:
                continue
            ratio = spec.diameter_mm / ref_diam
            if 0.85 <= ratio <= 1.25:
                diam_status = "optimal"
            elif 0.70 <= ratio < 0.85:
                diam_status = "small"
            elif 1.25 < ratio <= 1.40:
                diam_status = "large"
            else:
                continue  # outside usable range — skip

            if required_len <= 0 or spec.length_mm >= required_len:
                len_status = "ok"
            elif spec.length_mm >= required_len * 0.85:
                len_status = "short"
            else:
                len_status = "too_short"

            # Lower score = better match
            score = (0 if diam_status == "optimal" else 1) * 10 + \
                    (0 if len_status == "ok" else 1 if len_status == "short" else 2) * 3
            results.append((score, spec, diam_status, len_status))

        results.sort(key=lambda x: x[0])

        if not results:
            self._lbl_sizing_result.setText(
                f"<span style='color:#f85149'>"
                f"Sin dispositivos compatibles para Ø {ref_diam:.1f} mm. "
                f"Verifique las dimensiones arteriales."
                f"</span>"
            )
            self._best_sizing_spec = None
            self._btn_apply_sizing.setEnabled(False)
            return

        len_note = f"Long. mín. <b>{required_len:.0f} mm</b>" if required_len > 0 else "Long. cualquiera"
        html = (
            f"<small>Ref. Ø <b>{ref_diam:.1f} mm</b>  |  {len_note}</small><br>"
        )
        for _score, spec, ds, ls in results[:6]:
            if ds == "optimal" and ls == "ok":
                color, tag = "#3fb950", "✓"
            elif ds == "optimal" or ls == "ok":
                color, tag = "#e3b341", "~"
            else:
                color, tag = "#f0883e", "!"

            pct = (spec.diameter_mm / ref_diam - 1.0) * 100.0
            pct_str = f"{'+' if pct >= 0 else ''}{pct:.0f}%"
            len_warn = "" if ls == "ok" else " <i>(corto)</i>" if ls == "short" else " <i>(muy corto)</i>"
            html += (
                f"<span style='color:{color}'>{tag} <b>{spec.name}</b> "
                f"({pct_str}, {spec.length_mm:.0f} mm){len_warn}</span><br>"
            )

        self._lbl_sizing_result.setText(html.rstrip("<br>"))
        self._best_sizing_spec = results[0][1]
        self._btn_apply_sizing.setEnabled(True)

    def _apply_best_sizing(self) -> None:
        """Pre-select the best sizing recommendation in the stent catalogue list."""
        if self._best_sizing_spec is None:
            return
        target_name = self._best_sizing_spec.name
        for i, entry in enumerate(self._entries):
            if entry.spec is not None and entry.spec.name == target_name:
                self._stent_list.setCurrentRow(i)
                return
        # Device not currently visible — guide the user
        QMessageBox.information(
            self,
            "Dispositivo no visible",
            f"'{target_name}' no aparece en la lista con los filtros actuales.\n"
            "Active 'Desviadores de flujo' para verlo.",
        )

    @staticmethod
    def _make_pos_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-500.0, 500.0)
        s.setSingleStep(0.5)
        s.setDecimals(1)
        s.setSuffix(" mm")
        s.setValue(0.0)
        return s

    @staticmethod
    def _make_rot_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-180.0, 180.0)
        s.setSingleStep(1.0)
        s.setDecimals(1)
        s.setSuffix(" °")
        s.setValue(0.0)
        return s

    @staticmethod
    def _make_diam_spin() -> QDoubleSpinBox:
        """Spinbox for artery / neck diameters (0 = not set, shows '—')."""
        s = QDoubleSpinBox()
        s.setRange(0.0, 10.0)
        s.setSingleStep(0.1)
        s.setDecimals(1)
        s.setSuffix(" mm")
        s.setSpecialValueText("—")
        s.setValue(0.0)
        return s
