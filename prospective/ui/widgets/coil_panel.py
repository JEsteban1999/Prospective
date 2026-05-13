"""Embolization coil planning panel — MOD-04 / MOD-05.

Features
--------
* Coil catalogue (Target, Ruby, HydroCoil, MicroPlex, Axium) with type filter.
* Filter by aneurysm dome diameter for automatic sizing suggestions.
* Packing-density calculator: shows % of aneurysm volume packed vs. target.
* Placed-coil list with visibility toggle and remove.
* Export placed-coil plan as CSV.
* Custom STL/OBJ import for proprietary coil models.

Signals
-------
coil_placed(index, spec_name, position_xyz, poly_data|None)
coil_removed(index)
coil_visibility_changed(index, bool)
"""
from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import vtk
from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from prospective.models.coil_library import (
    COIL_CATALOGUE,
    CoilSpec,
    CoilType,
    coils_for_aneurysm,
    estimate_coil_count,
)

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

class _CoilEntry:
    """Unified representation for catalogue and custom coils."""

    def __init__(
        self,
        name: str,
        poly_data: vtk.vtkPolyData | None = None,
        spec: CoilSpec | None = None,
    ) -> None:
        self.name      = name
        self.poly_data = poly_data
        self.spec      = spec

    @property
    def list_label(self) -> str:
        if self.spec is not None:
            return (f"[{self.spec.coil_type.value[:3].upper()}]  "
                    f"{self.spec.display_label}")
        return f"[Custom]  {self.name}"

    @property
    def info_text(self) -> str:
        if self.spec is not None:
            return self.spec.info_text
        if self.poly_data is not None:
            n = self.poly_data.GetNumberOfPoints()
            t = self.poly_data.GetNumberOfPolys()
            return f"Coil personalizado — {n:,} verts · {t:,} tris"
        return "—"


class _PlacedCoil:
    def __init__(
        self,
        index: int,
        entry: _CoilEntry,
        position: tuple[float, float, float],
    ) -> None:
        self.index    = index
        self.entry    = entry
        self.position = position
        self.visible  = True

    @property
    def label(self) -> str:
        return f"#{self.index}  {self.entry.name}"


# ──────────────────────────────────────────────────────────────────────────── #
# Panel                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

class CoilPanel(QWidget):
    """Embolization coil planning dock panel."""

    # index, spec_name, position (x,y,z tuple), vtkPolyData|None
    coil_placed             = pyqtSignal(int, str, object, object)
    coil_removed            = pyqtSignal(int)
    coil_visibility_changed = pyqtSignal(int, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._placed: list[_PlacedCoil]  = []
        self._entries: list[_CoilEntry]  = []
        self._custom_coils: list[_CoilEntry] = []
        self._next_index   = 1
        self._dome_mm: float | None      = None
        self._aneurysm_volume_mm3: float | None = None
        self._aneurysm_centroid: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_aneurysm_data(
        self,
        dome_diameter_mm: float,
        volume_mm3: float,
        centroid: tuple[float, float, float],
    ) -> None:
        """Called by MainWindow when morphometrics are ready."""
        self._dome_mm             = dome_diameter_mm
        self._aneurysm_volume_mm3 = volume_mm3
        self._aneurysm_centroid   = centroid
        # Pre-fill position spinboxes with centroid
        for spin, val in zip(
            (self._pos_x, self._pos_y, self._pos_z), centroid
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._lbl_dome.setText(
            f"Domo: {dome_diameter_mm:.1f} mm  |  "
            f"Vol.: {volume_mm3:.1f} mm³"
        )
        self._refresh_coil_list()
        self._update_packing_display()

    def get_session_state(self) -> dict:
        coils = []
        for pc in self._placed:
            coils.append({
                "index":       pc.index,
                "name":        pc.entry.name,
                "is_custom":   pc.entry.poly_data is not None,
                "custom_path": getattr(pc.entry, "_source_path", None),
                "position":    [round(v, 3) for v in pc.position],
            })
        return {"coils": coils}

    def restore_session_state(self, d: dict) -> None:
        for cd in d.get("coils", []):
            pos = tuple(cd.get("position", [0.0, 0.0, 0.0]))

            poly_data  = None
            entry_obj  = None

            if cd.get("is_custom") and cd.get("custom_path"):
                path   = cd["custom_path"]
                suffix = Path(path).suffix.lower()
                try:
                    reader = vtk.vtkSTLReader() if suffix == ".stl" else vtk.vtkOBJReader()
                    reader.SetFileName(path)
                    reader.Update()
                    poly_data = reader.GetOutput()
                    entry_obj = _CoilEntry(cd["name"], poly_data, spec=None)
                    entry_obj._source_path = path
                except Exception as exc:
                    logger.warning("Could not reload custom coil %s: %s", path, exc)

            if entry_obj is None:
                spec = next(
                    (s for s in COIL_CATALOGUE if s.name == cd["name"]), None
                )
                entry_obj = _CoilEntry(cd["name"], None, spec)

            placed = _PlacedCoil(
                cd.get("index", self._next_index), entry_obj, pos  # type: ignore[arg-type]
            )
            self._next_index = max(self._next_index, placed.index + 1)
            self._placed.append(placed)
            self._placed_list.addItem(placed.label)
            self.coil_placed.emit(placed.index, entry_obj.name, pos, poly_data)

        if self._placed:
            self._btn_export.setEnabled(True)

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.NoFrame)
        content = QWidget()
        layout  = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # ── Aneurysm info ─────────────────────────────────────────────── #
        info_grp = QGroupBox("Aneurisma objetivo")
        inf = QFormLayout()
        self._lbl_dome = QLabel("Sin datos morfométricos")
        self._lbl_dome.setProperty("role", "muted")
        self._lbl_dome.setWordWrap(True)
        inf.addRow("Info:", self._lbl_dome)
        info_grp.setLayout(inf)
        layout.addWidget(info_grp)

        # ── Coil selection ────────────────────────────────────────────── #
        sel_grp = QGroupBox("Selección de coil")
        sf = QVBoxLayout()
        sf.setSpacing(4)

        # Filter controls row
        filter_row = QWidget()
        fr = QHBoxLayout(filter_row)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.setSpacing(4)

        self._chk_filter_size = QCheckBox("Filtrar por tamaño")
        self._chk_filter_size.setChecked(True)
        self._chk_filter_size.setToolTip("Muestra solo coils compatibles con el diámetro del domo del aneurisma")
        self._chk_filter_size.toggled.connect(self._refresh_coil_list)
        fr.addWidget(self._chk_filter_size)

        self._cmb_type = QComboBox()
        self._cmb_type.addItem("Todos los tipos", None)
        for ct in CoilType:
            self._cmb_type.addItem(ct.value, ct)
        self._cmb_type.setToolTip("Filtra la lista por tipo de coil embolizador")
        self._cmb_type.currentIndexChanged.connect(self._refresh_coil_list)
        fr.addWidget(self._cmb_type, stretch=1)

        sf.addWidget(filter_row)

        self._coil_list = QListWidget()
        self._coil_list.setAlternatingRowColors(True)
        self._coil_list.setMinimumHeight(110)
        self._coil_list.currentRowChanged.connect(self._on_entry_changed)
        sf.addWidget(self._coil_list)

        self._lbl_coil_info = QLabel("—")
        self._lbl_coil_info.setWordWrap(True)
        self._lbl_coil_info.setProperty("role", "muted")
        sf.addWidget(self._lbl_coil_info)

        self._btn_import = QPushButton("Importar coil STL/OBJ…")
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
        self._btn_import.setToolTip("Carga un modelo 3D personalizado (STL u OBJ) como coil")
        self._btn_import.clicked.connect(self._import_custom)
        sf.addWidget(self._btn_import)

        sel_grp.setLayout(sf)
        layout.addWidget(sel_grp)

        # ── Placement position ────────────────────────────────────────── #
        pos_grp = QGroupBox("Posición de depósito")
        pf = QFormLayout()
        pf.setLabelAlignment(Qt.AlignRight)
        pf.setVerticalSpacing(3)

        self._pos_x = self._make_pos_spin(); pf.addRow("X:", self._pos_x)
        self._pos_y = self._make_pos_spin(); pf.addRow("Y:", self._pos_y)
        self._pos_z = self._make_pos_spin(); pf.addRow("Z:", self._pos_z)

        btn_use_centroid = QPushButton("Centroide aneurisma")
        btn_use_centroid.setToolTip("Mueve la posición al centroide del saco.")
        btn_use_centroid.setStyleSheet("font-size:10px;")
        btn_use_centroid.clicked.connect(self._use_centroid)
        pf.addRow("", btn_use_centroid)

        pos_grp.setLayout(pf)
        layout.addWidget(pos_grp)

        # ── Deploy button ─────────────────────────────────────────────── #
        self._btn_place = QPushButton("⚫ Depositar coil en saco")
        self._btn_place.setMinimumHeight(30)
        if _is_dark():
            self._btn_place.setStyleSheet(
                "QPushButton{background:#1a2a1a;border:1px solid #2ea043;"
                "border-radius:9px;color:#EBEBEB;font-weight:bold;}"
                "QPushButton:hover{background:#196127;}"
            )
        else:
            self._btn_place.setStyleSheet(
                "QPushButton{background:#D6EFD8;border:1px solid #2ea043;"
                "border-radius:9px;color:#1B5E20;font-weight:bold;}"
                "QPushButton:hover{background:#A5D6A7;}"
            )
        self._btn_place.setToolTip("Deposita el coil seleccionado en el saco del aneurisma")
        self._btn_place.clicked.connect(self._place_coil)
        layout.addWidget(self._btn_place)

        # ── Packing density ───────────────────────────────────────────── #
        pack_grp = QGroupBox("Densidad de empaquetado")
        pkf = QFormLayout()
        pkf.setLabelAlignment(Qt.AlignRight)

        self._lbl_packing = QLabel("—")
        self._lbl_packing.setWordWrap(True)
        pkf.addRow("Estado:", self._lbl_packing)

        target_row = QWidget()
        tr = QHBoxLayout(target_row)
        tr.setContentsMargins(0, 0, 0, 0)
        tr.setSpacing(4)
        tr.addWidget(QLabel("Objetivo:"))
        self._spin_target_pct = QSpinBox()
        self._spin_target_pct.setRange(10, 50)
        self._spin_target_pct.setValue(25)
        self._spin_target_pct.setSuffix(" %")
        self._spin_target_pct.setMinimumWidth(62)
        self._spin_target_pct.setToolTip(
            "Densidad de empaquetado objetivo.\n"
            "≥25% asociado a menores tasas de recanalización\n"
            "(Sluzewski et al., AJNR 2004)."
        )
        self._spin_target_pct.valueChanged.connect(self._update_packing_display)
        tr.addWidget(self._spin_target_pct)
        tr.addStretch()
        pkf.addRow("", target_row)

        pack_grp.setLayout(pkf)
        layout.addWidget(pack_grp)

        # ── Placed coils ──────────────────────────────────────────────── #
        placed_grp = QGroupBox("Coils en saco")
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
        self._btn_toggle.setToolTip("Muestra u oculta el coil seleccionado en la vista 3D")
        self._btn_toggle.clicked.connect(self._toggle_selected)
        br.addWidget(self._btn_toggle)

        self._btn_remove = QPushButton("Eliminar")
        self._btn_remove.setEnabled(False)
        self._btn_remove.setToolTip("Elimina el coil seleccionado del saco  [Supr]")
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
        self._btn_export.setToolTip("Exporta el plan de embolización con todos los coils a CSV")
        self._btn_export.clicked.connect(self._export_plan)
        layout.addWidget(self._btn_export)

        layout.addStretch()
        self._refresh_coil_list()

    # ------------------------------------------------------------------ #
    # Event filter (keyboard Delete on placed list)                        #
    # ------------------------------------------------------------------ #

    def eventFilter(self, obj: object, event: object) -> bool:
        """Delete key on _placed_list removes the selected coil."""
        if obj is self._placed_list and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self._remove_selected()
                return True
        return super().eventFilter(obj, event)

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
                "QPushButton{background:#1a2a1a;border:1px solid #2ea043;"
                "border-radius:9px;color:#EBEBEB;font-weight:bold;}"
                "QPushButton:hover{background:#196127;}"
            )
        else:
            self._btn_import.setStyleSheet(
                "QPushButton{background:#F7F7F7;border:1px solid #D0D0D0;"
                "border-radius:8px;color:#6B6B6B;font-size:10px;padding:3px 6px;}"
                "QPushButton:hover{background:#DDE5EC;color:#0D0D0D;}"
            )
            self._btn_place.setStyleSheet(
                "QPushButton{background:#D6EFD8;border:1px solid #2ea043;"
                "border-radius:9px;color:#1B5E20;font-weight:bold;}"
                "QPushButton:hover{background:#A5D6A7;}"
            )

    # ------------------------------------------------------------------ #
    # Catalogue list                                                       #
    # ------------------------------------------------------------------ #

    def _refresh_coil_list(self) -> None:
        self._coil_list.clear()

        selected_type: CoilType | None = self._cmb_type.currentData()

        if self._chk_filter_size.isChecked() and self._dome_mm is not None:
            cat = coils_for_aneurysm(self._dome_mm, selected_type)
            if not cat:
                cat = COIL_CATALOGUE  # fallback if nothing matches
        else:
            cat = [
                c for c in COIL_CATALOGUE
                if selected_type is None or c.coil_type == selected_type
            ]

        self._entries = self._custom_coils + [
            _CoilEntry(s.name, None, s) for s in cat
        ]

        for e in self._entries:
            self._coil_list.addItem(e.list_label)

        if self._entries:
            self._coil_list.setCurrentRow(0)

    def _on_entry_changed(self, row: int) -> None:
        if 0 <= row < len(self._entries):
            e = self._entries[row]
            self._lbl_coil_info.setText(e.info_text)
            # Update packing suggestion
            if e.spec is not None and self._aneurysm_volume_mm3:
                n = estimate_coil_count(
                    self._aneurysm_volume_mm3, e.spec,
                    self._spin_target_pct.value()
                )
                self._lbl_coil_info.setText(
                    e.info_text + f"\n\nEstimado para {self._spin_target_pct.value()}%: "
                    f"~{n} coil(s)"
                )
        else:
            self._lbl_coil_info.setText("—")

    # ------------------------------------------------------------------ #
    # Placement                                                            #
    # ------------------------------------------------------------------ #

    def _place_coil(self) -> None:
        row = self._coil_list.currentRow()
        if row < 0 or not self._entries:
            QMessageBox.warning(self, "Sin coil", "Seleccione un tipo de coil.")
            return

        entry = self._entries[row]
        pos   = (self._pos_x.value(), self._pos_y.value(), self._pos_z.value())

        placed = _PlacedCoil(self._next_index, entry, pos)
        self._placed.append(placed)
        self._next_index += 1

        self._placed_list.addItem(placed.label)
        self._placed_list.setCurrentRow(self._placed_list.count() - 1)
        self._btn_export.setEnabled(True)
        self._update_packing_display()

        self.coil_placed.emit(placed.index, entry.name, pos, entry.poly_data)
        logger.info(
            "Coil placed #%d: %s at (%.1f, %.1f, %.1f)",
            placed.index, entry.name, *pos,
        )

    def _use_centroid(self) -> None:
        for spin, val in zip(
            (self._pos_x, self._pos_y, self._pos_z),
            self._aneurysm_centroid,
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

    # ------------------------------------------------------------------ #
    # Packing density                                                      #
    # ------------------------------------------------------------------ #

    def _update_packing_display(self) -> None:
        if not self._placed or not self._aneurysm_volume_mm3:
            self._lbl_packing.setText("Sin coils depositados.")
            return

        packed_vol = 0.0
        for pc in self._placed:
            if pc.entry.spec is not None:
                packed_vol += pc.entry.spec.wire_volume_mm3

        total_vol = self._aneurysm_volume_mm3
        pct       = (packed_vol / total_vol * 100.0) if total_vol > 0 else 0.0
        target    = self._spin_target_pct.value()

        if pct >= target:
            color = "#56d364"
            icon  = "✓"
            note  = f"Objetivo alcanzado (+{pct - target:.1f}%)"
        elif pct >= target * 0.8:
            color = "#e3b341"
            icon  = "⚠"
            note  = f"Casi alcanzado (faltan {target - pct:.1f}%)"
        else:
            color = "#f85149"
            icon  = "✗"
            note  = f"Insuficiente (faltan {target - pct:.1f}%)"

        n_coils   = len(self._placed)
        _mc = "#9B9B9B" if _is_dark() else "#6B6B6B"
        self._lbl_packing.setText(
            f"<span style='color:{color}'>{icon} {pct:.1f}%  —  {note}</span><br>"
            f"<small style='color:{_mc}'>"
            f"{n_coils} coil(s) · Vol. hilo: {packed_vol:.2f} mm³ / "
            f"{total_vol:.1f} mm³</small>"
        )
        self._lbl_packing.setTextFormat(Qt.RichText)

    # ------------------------------------------------------------------ #
    # Placed list                                                          #
    # ------------------------------------------------------------------ #

    def _on_placed_selected(self, row: int) -> None:
        enabled = 0 <= row < len(self._placed)
        self._btn_toggle.setEnabled(enabled)
        self._btn_remove.setEnabled(enabled)

    def _toggle_selected(self) -> None:
        row = self._placed_list.currentRow()
        if 0 <= row < len(self._placed):
            self._placed[row].visible = not self._placed[row].visible
            self.coil_visibility_changed.emit(
                self._placed[row].index, self._placed[row].visible
            )

    def remove_last_placed(self) -> bool:
        if not self._placed:
            return False
        placed = self._placed.pop()
        self._placed_list.takeItem(self._placed_list.count() - 1)
        self.coil_removed.emit(placed.index)
        if not self._placed:
            self._btn_export.setEnabled(False)
        self._update_packing_display()
        return True

    def _remove_selected(self) -> None:
        row = self._placed_list.currentRow()
        if row < 0 or row >= len(self._placed):
            return
        placed = self._placed.pop(row)
        self._placed_list.takeItem(row)
        self.coil_removed.emit(placed.index)
        if not self._placed:
            self._btn_export.setEnabled(False)
        self._update_packing_display()
        self._btn_toggle.setEnabled(False)
        self._btn_remove.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Import custom                                                        #
    # ------------------------------------------------------------------ #

    def _import_custom(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Importar coil 3D", "",
            "Mallas 3D (*.stl *.obj *.STL *.OBJ);;Todos los archivos (*)",
        )
        if not path:
            return

        suffix = Path(path).suffix.lower()
        if suffix not in (".stl", ".obj"):
            QMessageBox.warning(self, "Formato no soportado",
                                "Solo se aceptan archivos STL y OBJ.")
            return

        reader = vtk.vtkSTLReader() if suffix == ".stl" else vtk.vtkOBJReader()
        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()

        if poly.GetNumberOfPoints() == 0:
            QMessageBox.critical(self, "Error al importar",
                                 f"El archivo no contiene geometría válida:\n{path}")
            return

        name  = Path(path).stem
        entry = _CoilEntry(name, poly, spec=None)
        entry._source_path = str(Path(path).resolve())
        self._custom_coils.append(entry)
        self._refresh_coil_list()
        self._coil_list.setCurrentRow(0)
        logger.info("Custom coil imported: %s (%d pts)", name, poly.GetNumberOfPoints())

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def _export_plan(self) -> None:
        if not self._placed:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar plan de embolización", "plan_coils.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "Coil#", "Nombre", "Tipo", "Diámetro (mm)", "Longitud (cm)",
                "Fabricante", "X (mm)", "Y (mm)", "Z (mm)",
            ])
            for pc in self._placed:
                tipo   = pc.entry.spec.coil_type.value if pc.entry.spec else "Custom"
                diam   = f"{pc.entry.spec.diameter_mm:.0f}" if pc.entry.spec else "—"
                length = f"{pc.entry.spec.length_cm:.0f}"  if pc.entry.spec else "—"
                mfr    = pc.entry.spec.manufacturer         if pc.entry.spec else "—"
                writer.writerow([
                    pc.index, pc.entry.name, tipo, diam, length, mfr,
                    f"{pc.position[0]:.2f}",
                    f"{pc.position[1]:.2f}",
                    f"{pc.position[2]:.2f}",
                ])
        logger.info("Coil plan exported: %s", path)
        QMessageBox.information(self, "Plan exportado", f"Guardado en:\n{path}")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_pos_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-500.0, 500.0)
        s.setSingleStep(0.5)
        s.setDecimals(1)
        s.setSuffix(" mm")
        s.setValue(0.0)
        return s
