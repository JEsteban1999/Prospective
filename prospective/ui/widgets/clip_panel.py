"""Surgical clip planning panel — F-04 / MOD-05.

Features
--------
* Clip catalogue (Sugita/Aesculap/Codman) + custom STL/OBJ import.
* Live spinbox repositioning of placed clips.
* Surgical approach trajectory (entry point → aneurysm, 3D corridor).
* Clip–vessel collision detection with per-clip status indicator.

Signals
-------
clip_placed(index, spec_name, transform, poly_data|None)
clip_removed(index)
clip_transform_changed(index, transform)
clip_visibility_changed(index, bool)
trajectory_changed(entry_xyz, target_xyz)   — update 3D corridor actor
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import vtk
from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from prospective.models.clip_library import (
    CLIP_CATALOGUE,
    ClipSpec,
    clips_for_neck,
)

logger = logging.getLogger(__name__)


def _is_dark() -> bool:
    try:
        from prospective.ui.themes import is_dark
        return is_dark()
    except Exception:
        return True


# ──────────────────────────────────────────────────────────────────────────── #
# Internal clip entry (catalogue OR custom)                                     #
# ──────────────────────────────────────────────────────────────────────────── #

class _ClipEntry:
    """Unified representation for catalogue and custom clips."""

    def __init__(
        self,
        name: str,
        poly_data: vtk.vtkPolyData | None = None,
        spec: ClipSpec | None = None,
    ) -> None:
        self.name      = name
        self.poly_data = poly_data   # None → build from spec at placement time
        self.spec      = spec        # None for custom clips

    @property
    def list_label(self) -> str:
        if self.spec is not None:
            return (f"{self.spec.shape.value}  |  {self.name}  "
                    f"({self.spec.blade_length_mm:.0f} mm)")
        return f"[Custom]  {self.name}"

    @property
    def info_text(self) -> str:
        if self.spec is not None:
            return (f"Hoja: {self.spec.blade_length_mm:.0f} mm  |  "
                    f"Ancho: {self.spec.blade_width_mm:.1f} mm\n"
                    f"Fuerza cierre: {self.spec.closing_force_g:.0f} g  |  "
                    f"{self.spec.manufacturer}")
        if self.poly_data is not None:
            n = self.poly_data.GetNumberOfPoints()
            t = self.poly_data.GetNumberOfPolys()
            return f"Clip personalizado — {n:,} verts · {t:,} tris"
        return "—"


class _PlacedClip:
    def __init__(
        self,
        index: int,
        entry: _ClipEntry,
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

class ClipPanel(QWidget):
    """Surgical clip planning dock panel."""

    # index, spec_name, vtkTransform, vtkPolyData|None
    clip_placed             = pyqtSignal(int, str, object, object)
    clip_removed            = pyqtSignal(int)
    clip_transform_changed  = pyqtSignal(int, object)
    clip_visibility_changed = pyqtSignal(int, bool)
    # (entry_xyz tuple, target_xyz tuple)
    trajectory_changed      = pyqtSignal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._placed: list[_PlacedClip] = []
        self._next_index   = 1
        self._neck_mm: float | None     = None
        self._custom_clips: list[_ClipEntry] = []
        self._entries: list[_ClipEntry] = []
        self._editing_placed_row: int = -1
        # Vessel mesh for collision checks (set from outside)
        self._vessel_mesh: vtk.vtkPolyData | None = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_session_state(self) -> dict:
        clips = []
        for pc in self._placed:
            pos = list(pc.transform.GetPosition())
            ori = list(pc.transform.GetOrientation())
            clips.append({
                "index":       pc.index,
                "name":        pc.entry.name,
                "is_custom":   pc.entry.poly_data is not None,
                "custom_path": getattr(pc.entry, "_source_path", None),
                "position":    [round(v, 3) for v in pos],
                "orientation": [round(v, 3) for v in [ori[2], ori[0], ori[1]]],
            })
        return {
            "clips": clips,
            "trajectory": {
                "entry":  [self._ent_x.value(), self._ent_y.value(), self._ent_z.value()],
                "target": [self._tgt_x.value(), self._tgt_y.value(), self._tgt_z.value()],
                "visible": self._btn_show_traj.isChecked(),
            },
        }

    def restore_session_state(self, d: dict) -> None:
        """Restore clip positions and trajectory from a session dict."""
        import vtk as _vtk

        tr = d.get("trajectory", {})
        entry  = tr.get("entry",  [0, 0, 0])
        target = tr.get("target", [0, 0, 0])
        self._set_spinboxes_traj_silent(entry, target)
        if tr.get("visible", False):
            self._btn_show_traj.setChecked(True)
            self.trajectory_changed.emit(tuple(entry), tuple(target))

        for cd in d.get("clips", []):
            pos = cd.get("position", [0, 0, 0])
            ori = cd.get("orientation", [0, 0, 0])   # yaw, pitch, roll

            poly_data = None
            entry_obj = None

            if cd.get("is_custom") and cd.get("custom_path"):
                path = cd["custom_path"]
                suffix = Path(path).suffix.lower()
                try:
                    if suffix == ".stl":
                        reader = _vtk.vtkSTLReader()
                    else:
                        reader = _vtk.vtkOBJReader()
                    reader.SetFileName(path)
                    reader.Update()
                    poly_data = reader.GetOutput()
                    entry_obj = _ClipEntry(cd["name"], poly_data, spec=None)
                    entry_obj._source_path = path
                except Exception as exc:
                    logger.warning("Could not reload custom clip %s: %s", path, exc)

            if entry_obj is None:
                # Find matching spec in catalogue
                from prospective.models.clip_library import CLIP_CATALOGUE
                spec = next((s for s in CLIP_CATALOGUE if s.name == cd["name"]), None)
                entry_obj = _ClipEntry(cd["name"], None, spec)

            t = _vtk.vtkTransform()
            t.Identity()
            t.Translate(*pos)
            t.RotateZ(ori[0])   # yaw
            t.RotateX(ori[1])   # pitch
            t.RotateY(ori[2])   # roll

            placed = _PlacedClip(cd.get("index", self._next_index), entry_obj, t)
            placed.index = cd.get("index", self._next_index)
            self._next_index = max(self._next_index, placed.index + 1)
            self._placed.append(placed)
            self._placed_list.addItem(placed.label)
            self.clip_placed.emit(placed.index, entry_obj.name, t, poly_data)

        if self._placed:
            self._btn_export.setEnabled(True)
            self._btn_check_col.setEnabled(True)

    def _set_spinboxes_traj_silent(self, entry, target) -> None:
        for spin, val in zip(
            (self._ent_x, self._ent_y, self._ent_z,
             self._tgt_x, self._tgt_y, self._tgt_z),
            list(entry) + list(target),
        ):
            spin.blockSignals(True)
            spin.setValue(float(val))
            spin.blockSignals(False)

    def set_neck_diameter(self, neck_mm: float) -> None:
        self._neck_mm = neck_mm
        self._refresh_clip_list()
        self._lbl_neck.setText(f"Cuello: {neck_mm:.2f} mm")

    def select_clip_by_name(self, name: str) -> bool:
        """
        Highlight the clip with *name* in the catalogue list.

        Returns True if found and selected, False otherwise.
        Called by the clip recommender panel.
        """
        for row, entry in enumerate(self._entries):
            if entry.name == name:
                self._clip_list.setCurrentRow(row)
                self._clip_list.scrollToItem(self._clip_list.item(row))
                return True
        # Not in filtered list — disable filter and retry
        if self._chk_filter.isChecked():
            self._chk_filter.setChecked(False)
            return self.select_clip_by_name(name)
        return False

    def set_vessel_mesh(self, poly_data: vtk.vtkPolyData | None) -> None:
        """Provide the vascular mesh for collision checks."""
        self._vessel_mesh = poly_data

    def set_aneurysm_centroid(self, x: float, y: float, z: float) -> None:
        """Pre-fill the trajectory target from morphometrics centroid."""
        for spin, val in zip((self._tgt_x, self._tgt_y, self._tgt_z), (x, y, z)):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._update_trajectory_info()

    def set_placement_point(
        self,
        x: float, y: float, z: float,
        nx: float = 0.0, ny: float = 0.0, nz: float = 1.0,
    ) -> None:
        """Pre-fill spinboxes from a 3D pick point."""
        import math
        yaw   = math.degrees(math.atan2(ny, nx))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, nz))))
        self._set_spinboxes_silent(x, y, z, yaw, pitch, 0.0)

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
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # ── Clip selection ────────────────────────────────────────────── #
        sel_grp = QGroupBox("Selección de clip")
        sf = QVBoxLayout()
        sf.setSpacing(4)

        self._lbl_neck = QLabel("Cuello: — mm")
        self._lbl_neck.setProperty("role", "muted")
        sf.addWidget(self._lbl_neck)

        self._chk_filter = QCheckBox("Filtrar por diámetro de cuello")
        self._chk_filter.setChecked(True)
        self._chk_filter.setToolTip("Muestra solo clips compatibles con el diámetro de cuello medido")
        self._chk_filter.toggled.connect(self._refresh_clip_list)
        sf.addWidget(self._chk_filter)

        self._clip_list = QListWidget()
        self._clip_list.setAlternatingRowColors(True)
        self._clip_list.setMinimumHeight(100)
        self._clip_list.currentRowChanged.connect(self._on_entry_changed)
        sf.addWidget(self._clip_list)

        self._lbl_clip_info = QLabel("—")
        self._lbl_clip_info.setWordWrap(True)
        self._lbl_clip_info.setProperty("role", "muted")
        sf.addWidget(self._lbl_clip_info)

        # Import custom clip button
        self._btn_import = QPushButton("Importar clip STL/OBJ…")
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
        self._btn_import.setToolTip("Carga un modelo 3D personalizado (STL u OBJ) como clip quirúrgico")
        self._btn_import.clicked.connect(self._import_custom)
        sf.addWidget(self._btn_import)

        sel_grp.setLayout(sf)
        layout.addWidget(sel_grp)

        # ── Position & Orientation ────────────────────────────────────── #
        xform_grp = QGroupBox("Posición / Orientación  (clip seleccionado)")
        xf = QFormLayout()
        xf.setLabelAlignment(Qt.AlignRight)
        xf.setVerticalSpacing(3)

        self._tx = self._make_pos_spin(); xf.addRow("X:", self._tx)
        self._ty = self._make_pos_spin(); xf.addRow("Y:", self._ty)
        self._tz = self._make_pos_spin(); xf.addRow("Z:", self._tz)
        self._ry = self._make_rot_spin(); xf.addRow("Yaw:", self._ry)
        self._rp = self._make_rot_spin(); xf.addRow("Pitch:", self._rp)
        self._rr = self._make_rot_spin(); xf.addRow("Roll:", self._rr)

        # Connect ALL spinboxes → live update the selected placed clip
        for spin in (self._tx, self._ty, self._tz, self._ry, self._rp, self._rr):
            spin.valueChanged.connect(self._on_spinbox_changed)

        xform_grp.setLayout(xf)
        layout.addWidget(xform_grp)

        # ── Place button ──────────────────────────────────────────────── #
        self._btn_place = QPushButton("Colocar clip en escena")
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
        self._btn_place.setToolTip("Coloca el clip seleccionado en las coordenadas indicadas")
        self._btn_place.clicked.connect(self._place_clip)
        layout.addWidget(self._btn_place)

        # ── Placed clips ──────────────────────────────────────────────── #
        placed_grp = QGroupBox("Clips en escena")
        pl = QVBoxLayout()
        pl.setSpacing(4)

        self._placed_list = QListWidget()
        self._placed_list.setAlternatingRowColors(True)
        self._placed_list.setMaximumHeight(90)
        self._placed_list.currentRowChanged.connect(self._on_placed_selected)
        self._placed_list.installEventFilter(self)
        pl.addWidget(self._placed_list)

        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(4)

        self._btn_toggle = QPushButton("Mostrar/Ocultar")
        self._btn_toggle.setEnabled(False)
        self._btn_toggle.setToolTip("Muestra u oculta el clip seleccionado en la vista 3D")
        self._btn_toggle.clicked.connect(self._toggle_selected)
        br.addWidget(self._btn_toggle)

        self._btn_remove = QPushButton("Eliminar")
        self._btn_remove.setEnabled(False)
        self._btn_remove.setToolTip("Elimina el clip seleccionado de la escena  [Supr]")
        self._btn_remove.setStyleSheet(
            "QPushButton{color:#f85149;}"
            "QPushButton:hover{background:rgba(248,81,73,15);border-color:#f85149;}"
        )
        self._btn_remove.clicked.connect(self._remove_selected)
        br.addWidget(self._btn_remove)

        pl.addWidget(btn_row)
        placed_grp.setLayout(pl)
        layout.addWidget(placed_grp)

        # ── Export ────────────────────────────────────────────────────── #
        self._btn_export = QPushButton("Exportar plan CSV")
        self._btn_export.setEnabled(False)
        self._btn_export.setToolTip("Exporta el plan quirúrgico con las posiciones de todos los clips a CSV")
        self._btn_export.clicked.connect(self._export_plan)
        layout.addWidget(self._btn_export)

        # ── Approach trajectory ───────────────────────────────────────── #
        traj_grp = QGroupBox("Trayectoria de abordaje")
        tf = QFormLayout()
        tf.setLabelAlignment(Qt.AlignRight)
        tf.setVerticalSpacing(3)

        tf.addRow(QLabel("<small style='color:#6B6B6B'>Punto de entrada:</small>"))
        self._ent_x = self._make_pos_spin(); tf.addRow("X:", self._ent_x)
        self._ent_y = self._make_pos_spin(); tf.addRow("Y:", self._ent_y)
        self._ent_z = self._make_pos_spin(); tf.addRow("Z:", self._ent_z)

        tf.addRow(QLabel("<small style='color:#6B6B6B'>Punto objetivo (aneurisma):</small>"))
        self._tgt_x = self._make_pos_spin(); tf.addRow("X:", self._tgt_x)
        self._tgt_y = self._make_pos_spin(); tf.addRow("Y:", self._tgt_y)
        self._tgt_z = self._make_pos_spin(); tf.addRow("Z:", self._tgt_z)

        for sp in (self._ent_x, self._ent_y, self._ent_z,
                   self._tgt_x, self._tgt_y, self._tgt_z):
            sp.valueChanged.connect(self._update_trajectory_info)

        self._lbl_traj_info = QLabel("—")
        self._lbl_traj_info.setWordWrap(True)
        self._lbl_traj_info.setProperty("role", "muted")
        tf.addRow("Info:", self._lbl_traj_info)

        traj_btn_row = QWidget()
        tbr = QHBoxLayout(traj_btn_row)
        tbr.setContentsMargins(0, 0, 0, 0)
        tbr.setSpacing(4)

        self._btn_show_traj = QPushButton("Mostrar corredor")
        self._btn_show_traj.setCheckable(True)
        self._btn_show_traj.setToolTip("Muestra el corredor de abordaje quirúrgico en la vista 3D")
        self._btn_show_traj.clicked.connect(self._toggle_trajectory)
        tbr.addWidget(self._btn_show_traj)

        btn_use_clip_pos = QPushButton("Usar pos. clip")
        btn_use_clip_pos.setToolTip(
            "Copia la posición del clip seleccionado como punto objetivo."
        )
        btn_use_clip_pos.clicked.connect(self._use_clip_as_target)
        tbr.addWidget(btn_use_clip_pos)

        tf.addRow(traj_btn_row)
        traj_grp.setLayout(tf)
        layout.addWidget(traj_grp)

        # ── Collision detection ───────────────────────────────────────── #
        col_grp = QGroupBox("Detección de colisiones")
        cf = QVBoxLayout()
        cf.setSpacing(4)

        self._lbl_collision = QLabel("Sin clips colocados.")
        self._lbl_collision.setWordWrap(True)
        self._lbl_collision.setProperty("role", "muted")
        cf.addWidget(self._lbl_collision)

        self._btn_check_col = QPushButton("Verificar colisiones")
        self._btn_check_col.setEnabled(False)
        self._btn_check_col.setToolTip("Verifica si algún clip colocado intersecta con el tejido vascular")
        self._btn_check_col.clicked.connect(self._check_all_collisions)
        cf.addWidget(self._btn_check_col)

        col_grp.setLayout(cf)
        layout.addWidget(col_grp)

        layout.addStretch()
        self._refresh_clip_list()

    # ------------------------------------------------------------------ #
    # Theme                                                                #
    # ------------------------------------------------------------------ #

    def apply_theme(self) -> None:
        """Re-style theme-sensitive buttons to match the current light/dark theme."""
        if _is_dark():
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

    # ------------------------------------------------------------------ #
    # Event filter (keyboard Delete on placed list)                        #
    # ------------------------------------------------------------------ #

    def eventFilter(self, obj: object, event: object) -> bool:
        """Delete key on _placed_list removes the selected clip."""
        if obj is self._placed_list and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self._remove_selected()
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ #
    # Catalogue / custom list                                              #
    # ------------------------------------------------------------------ #

    def _refresh_clip_list(self) -> None:
        self._clip_list.clear()

        if self._chk_filter.isChecked() and self._neck_mm is not None:
            cat = clips_for_neck(self._neck_mm) or CLIP_CATALOGUE
        else:
            cat = CLIP_CATALOGUE

        self._entries = [_ClipEntry(s.name, None, s) for s in cat]
        # Custom clips always appear at the top
        self._entries = self._custom_clips + self._entries

        for e in self._entries:
            self._clip_list.addItem(e.list_label)

        if self._entries:
            self._clip_list.setCurrentRow(0)

    def _on_entry_changed(self, row: int) -> None:
        if 0 <= row < len(self._entries):
            self._lbl_clip_info.setText(self._entries[row].info_text)
        else:
            self._lbl_clip_info.setText("—")

    # ------------------------------------------------------------------ #
    # Import custom clip                                                   #
    # ------------------------------------------------------------------ #

    def _import_custom(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Importar clip 3D", "",
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
            QMessageBox.warning(self, "Formato no soportado",
                                "Solo se aceptan archivos STL y OBJ.")
            return

        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()

        if poly.GetNumberOfPoints() == 0:
            QMessageBox.critical(self, "Error al importar",
                                 f"El archivo no contiene geometría válida:\n{path}")
            return

        name = Path(path).stem
        entry = _ClipEntry(name, poly, spec=None)
        entry._source_path = str(Path(path).resolve())
        self._custom_clips.append(entry)
        self._refresh_clip_list()

        # Select the newly imported clip
        self._clip_list.setCurrentRow(0)
        logger.info("Custom clip imported: %s  (%d pts)", name, poly.GetNumberOfPoints())

    # ------------------------------------------------------------------ #
    # Spinbox live-update                                                  #
    # ------------------------------------------------------------------ #

    def _on_spinbox_changed(self) -> None:
        """If a placed clip is selected, update its transform immediately."""
        row = self._editing_placed_row
        if row < 0 or row >= len(self._placed):
            return
        placed = self._placed[row]
        t = self._build_transform()
        placed.transform = t
        self.clip_transform_changed.emit(placed.index, t)

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

    def place_clip_programmatic(
        self,
        spec_name: str,
        transform: vtk.vtkTransform,
    ) -> int:
        """Place a clip programmatically from a 3D pick (trajectory auto-place).

        Mirrors ``place_stent_programmatic`` in StentPanel.
        Returns the assigned clip index, or -1 if spec_name not found.
        """
        spec = next((s for s in CLIP_CATALOGUE if s.name == spec_name), None)
        if spec is None:
            logger.warning("place_clip_programmatic: unknown spec %r", spec_name)
            return -1

        entry  = _ClipEntry(spec_name, None, spec)
        placed = _PlacedClip(self._next_index, entry, transform)
        self._placed.append(placed)
        self._next_index += 1

        self._placed_list.addItem(placed.label)
        self._placed_list.setCurrentRow(self._placed_list.count() - 1)
        self._btn_toggle.setEnabled(True)
        self._btn_remove.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._btn_check_col.setEnabled(True)
        self._lbl_collision.setText("Pulse 'Verificar colisiones' para comprobar.")
        self.clip_placed.emit(placed.index, spec_name, transform, None)
        logger.info(
            "Clip placed programmatically #%d: %s", placed.index, spec_name
        )
        return placed.index

    def _place_clip(self) -> None:
        row = self._clip_list.currentRow()
        if row < 0 or not self._entries:
            QMessageBox.warning(self, "Sin clip", "Seleccione un tipo de clip.")
            return

        entry = self._entries[row]
        t     = self._build_transform()

        placed = _PlacedClip(self._next_index, entry, t)
        self._placed.append(placed)
        self._next_index += 1

        self._placed_list.addItem(placed.label)
        # Select the new clip so spinboxes control it immediately
        new_row = self._placed_list.count() - 1
        self._placed_list.setCurrentRow(new_row)

        self._btn_export.setEnabled(True)
        self._btn_check_col.setEnabled(True)
        self._lbl_collision.setText("Pulse 'Verificar colisiones' para comprobar.")
        self.clip_placed.emit(placed.index, entry.name, t, entry.poly_data)
        logger.info("Clip placed #%d: %s at (%.1f, %.1f, %.1f)",
                    placed.index, entry.name,
                    self._tx.value(), self._ty.value(), self._tz.value())

    # ------------------------------------------------------------------ #
    # Placed list selection                                                #
    # ------------------------------------------------------------------ #

    def _on_placed_selected(self, row: int) -> None:
        enabled = 0 <= row < len(self._placed)
        self._btn_toggle.setEnabled(enabled)
        self._btn_remove.setEnabled(enabled)

        if not enabled:
            self._editing_placed_row = -1
            return

        # Pause live-update while we sync spinboxes from stored transform
        self._editing_placed_row = -1

        placed = self._placed[row]
        pos = placed.transform.GetPosition()
        ori = placed.transform.GetOrientation()   # (rx, ry, rz) Euler
        self._set_spinboxes_silent(
            pos[0], pos[1], pos[2],
            ori[2],   # Z-rotation → yaw
            ori[0],   # X-rotation → pitch
            ori[1],   # Y-rotation → roll
        )
        # Now re-enable so further changes will update this clip
        self._editing_placed_row = row

    def _toggle_selected(self) -> None:
        row = self._placed_list.currentRow()
        if 0 <= row < len(self._placed):
            self._placed[row].visible = not self._placed[row].visible
            self.clip_visibility_changed.emit(
                self._placed[row].index, self._placed[row].visible
            )

    def remove_last_placed(self) -> bool:
        """Remove the most recently placed clip (used by undo).

        Returns True if a clip was removed, False if the list was empty.
        """
        if not self._placed:
            return False
        row = len(self._placed) - 1
        self._editing_placed_row = -1
        placed = self._placed.pop(row)
        self._placed_list.takeItem(row)
        self.clip_removed.emit(placed.index)
        if not self._placed:
            self._btn_export.setEnabled(False)
        self._btn_toggle.setEnabled(False)
        self._btn_remove.setEnabled(False)
        return True

    def _remove_selected(self) -> None:
        row = self._placed_list.currentRow()
        if row < 0 or row >= len(self._placed):
            return
        self._editing_placed_row = -1
        placed = self._placed.pop(row)
        self._placed_list.takeItem(row)
        self.clip_removed.emit(placed.index)
        if not self._placed:
            self._btn_export.setEnabled(False)
        self._btn_toggle.setEnabled(False)
        self._btn_remove.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Trajectory                                                           #
    # ------------------------------------------------------------------ #

    def _update_trajectory_info(self) -> None:
        import math
        ex, ey, ez = self._ent_x.value(), self._ent_y.value(), self._ent_z.value()
        tx, ty, tz = self._tgt_x.value(), self._tgt_y.value(), self._tgt_z.value()
        dx, dy, dz = tx - ex, ty - ey, tz - ez
        depth = math.sqrt(dx*dx + dy*dy + dz*dz)
        if depth < 1e-3:
            self._lbl_traj_info.setText("Entrada = objetivo.")
            return
        # Angle relative to vertical (Z axis) — simplified axial approach angle
        angle_z = math.degrees(math.acos(max(-1.0, min(1.0, abs(dz) / depth))))
        self._lbl_traj_info.setText(
            f"Profundidad: {depth:.1f} mm\n"
            f"Ángulo axial: {angle_z:.1f}°"
        )
        if self._btn_show_traj.isChecked():
            self.trajectory_changed.emit((ex, ey, ez), (tx, ty, tz))

    def _toggle_trajectory(self, checked: bool) -> None:
        if checked:
            ex = self._ent_x.value(), self._ent_y.value(), self._ent_z.value()
            tg = self._tgt_x.value(), self._tgt_y.value(), self._tgt_z.value()
            self.trajectory_changed.emit(ex, tg)
        else:
            self.trajectory_changed.emit(None, None)   # clear

    def _use_clip_as_target(self) -> None:
        row = self._placed_list.currentRow()
        if row < 0 or row >= len(self._placed):
            return
        pos = self._placed[row].transform.GetPosition()
        for spin, val in zip((self._tgt_x, self._tgt_y, self._tgt_z), pos):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._update_trajectory_info()

    # ------------------------------------------------------------------ #
    # Collision detection                                                  #
    # ------------------------------------------------------------------ #

    def _neck_coverage_html(self, pc: "_PlacedClip") -> str:
        """
        Return an HTML snippet showing the clip-to-neck coverage ratio.

        Coverage = blade_length / neck_diameter × 100 %.
        ✓ ≥ 100 % — blade spans the full neck  (green)
        ⚠  70–99 % — partial coverage          (orange)
        ✗   < 70 % — insufficient coverage     (red)
        """
        if not (self._neck_mm and self._neck_mm > 0 and pc.entry.spec):
            return ""   # neck not measured or custom clip without spec

        ratio = pc.entry.spec.blade_length_mm / self._neck_mm
        pct   = min(ratio * 100.0, 100.0)
        extra = max(0.0, (ratio - 1.0) * 100.0)   # excess beyond neck

        if ratio >= 1.0:
            color = "#56d364"
            icon  = "&#10003;"
            note  = f"+{extra:.0f}% extra" if extra > 0 else "exacto"
        elif ratio >= 0.70:
            color = "#e3b341"
            icon  = "&#9888;"
            note  = "cobertura parcial"
        else:
            color = "#f85149"
            icon  = "&#10007;"
            note  = "insuficiente"

        return (
            f"  <span style='color:{color}'>"
            f"{icon} cuello {pct:.0f}% ({note})</span>"
        )

    def _check_all_collisions(self) -> None:
        from prospective.processing.collision import check_collision
        from prospective.rendering.clip_actor import build_clip_actor

        if not self._placed:
            return
        if self._vessel_mesh is None:
            self._lbl_collision.setText(
                "Sin malla vascular disponible.\n"
                "Realice una segmentación primero."
            )
            return

        lines = []
        for pc in self._placed:
            # Get the clip polydata (build from spec or use custom)
            if pc.entry.poly_data is not None:
                clip_poly = pc.entry.poly_data
            elif pc.entry.spec is not None:
                actor = build_clip_actor(pc.entry.spec)
                clip_poly = actor.GetMapper().GetInput()
            else:
                lines.append(f"#{pc.index} {pc.entry.name}: sin geometría")
                continue

            detected, n = check_collision(
                self._vessel_mesh, clip_poly, pc.transform
            )
            coverage = self._neck_coverage_html(pc)

            if detected:
                lines.append(
                    f"<span style='color:#f85149'>&#9888; #{pc.index} "
                    f"{pc.entry.name}: COLISIÓN ({n} contactos){coverage}</span>"
                )
            else:
                lines.append(
                    f"<span style='color:#56d364'>&#10003; #{pc.index} "
                    f"{pc.entry.name}: sin colisión{coverage}</span>"
                )

        self._lbl_collision.setText("<br>".join(lines))
        self._lbl_collision.setTextFormat(Qt.RichText)

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def _export_plan(self) -> None:
        if not self._placed:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar plan quirúrgico", "plan_clips.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "Clip#", "Nombre", "Tipo",
                "Hoja (mm)", "X (mm)", "Y (mm)", "Z (mm)",
                "Yaw (°)", "Pitch (°)", "Roll (°)",
            ])
            for pc in self._placed:
                pos = pc.transform.GetPosition()
                ori = pc.transform.GetOrientation()
                blade = (f"{pc.entry.spec.blade_length_mm:.0f}"
                         if pc.entry.spec else "—")
                tipo  = (pc.entry.spec.shape.value
                         if pc.entry.spec else "Personalizado")
                writer.writerow([
                    pc.index, pc.entry.name, tipo, blade,
                    f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}",
                    f"{ori[2]:.1f}", f"{ori[0]:.1f}", f"{ori[1]:.1f}",
                ])
        logger.info("Clip plan exported: %s", path)
        QMessageBox.information(self, "Plan exportado", f"Guardado en:\n{path}")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _set_spinboxes_silent(
        self, x, y, z, yaw, pitch, roll
    ) -> None:
        for spin, val in zip(
            (self._tx, self._ty, self._tz, self._ry, self._rp, self._rr),
            (x, y, z, yaw, pitch, roll),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

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
