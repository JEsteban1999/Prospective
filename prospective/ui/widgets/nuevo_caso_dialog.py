"""NuevoCasoDialog — clinical intake form for a new aneurysm case.

Creates a Patient + Study record in the database and optionally links
per-modality DICOM directories.  Returns the created study_id and
dicom_path (first non-empty modality path) on accept.

This dialog is used both for **new cases** (from WelcomeWindow "CASO NUEVO"
or CaseDashboard "＋ Nuevo Caso") and for **editing existing cases**
(from the ✏ button on each case card in CaseDashboard).

Layout
------
  Left  panel: decorative angiogram image + app branding (glass dark theme)
  Right panel: scrollable clinical form split into FIVE sections:

    1. Datos del paciente
       Apellidos, nombre, fecha de nacimiento, sexo, NHC, institución,
       ocupación.

    2. Antecedentes personales
       Patológicos, toxicológicos, quirúrgicos, alérgicos, farmacológicos
       (free-text fields).

    3. Datos clínicos
       Síntomas positivos, diagnóstico principal, diagnóstico secundario,
       angiógrafo (marca + tipo).

    4. Caracterización aneurismática
       Tipo de aneurisma, región anatómica, lateralidad,
       tratamiento propuesto (checkboxes: Clipaje / Diversor de flujo +
       Clip / Coils + Clips).

    5. Imágenes diagnósticas
       Four independent modality pickers — each a read-only QLineEdit with
       a 📂 browse button:
         • Tomografía (TAC)    → stored in Study.dicom_tac
         • Angiografía         → stored in Study.dicom_angio
         • Resonancia Magnética → stored in Study.dicom_rm
         • Panangiografía       → stored in Study.dicom_pangio
       At least one path must be filled to accept the form.
       The first non-empty path is also stored in Study.dicom_path for
       backward compatibility with MainWindow's DICOM loader.

Database columns written
------------------------
  Patient: surname, given_name, hospital_id, dob, sex, institution,
           ocupacion, antecedentes_patologicos, antecedentes_toxicologicos,
           antecedentes_quirurgicos, antecedentes_alergicos,
           antecedentes_farmacologicos.
  Study:   created_by, study_date, sintomas_positivos, dx_principal,
           dx_secundario, tipo_aneurisma, region_anatomica, lateralidad,
           tratamiento_propuesto, angiographer,
           dicom_tac, dicom_angio, dicom_rm, dicom_pangio, dicom_path.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QColor, QFont, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────── #
# Colour palette — theme-aware                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

_RES      = Path(__file__).resolve().parents[3] / "resources"
_LOGO_PATH = str(_RES / "logo.png")

_SUCCESS  = "#4dce6a"
_WARNING  = "#ffaa33"

_PALETTES = {
    "dark": dict(
        bg="#1F1F1F", card="#2A2A2A", border="#363636",
        txt="#EBEBEB", muted="#9B9B9B", accent="#A8B8C6", hover="#182434",
    ),
    "light": dict(
        bg="#FFFFFF", card="#F7F7F7", border="#E5E5E5",
        txt="#0D0D0D", muted="#6B6B6B", accent="#8B9BAA", hover="#DDE5EC",
    ),
}


def _pal() -> dict:
    try:
        from prospective.ui.themes import current_theme
        return _PALETTES.get(current_theme(), _PALETTES["dark"])
    except Exception:
        return _PALETTES["dark"]


def _form_style() -> str:
    p = _pal()
    return f"""
QGroupBox {{
    color: {p['accent']};
    font-weight: bold;
    font-size: 11px;
    border: 1px solid {p['border']};
    border-radius: 10px;
    margin-top: 10px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
}}
QLabel {{
    color: {p['txt']};
    font-size: 11px;
}}
QLineEdit, QTextEdit, QComboBox, QDateEdit {{
    background: {p['bg']};
    border: 1px solid {p['border']};
    border-radius: 14px;
    color: {p['txt']};
    padding: 4px 6px;
    font-size: 11px;
}}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QDateEdit:focus {{
    border-color: {p['accent']};
}}
QComboBox QAbstractItemView {{
    background-color: {p['bg']};
    color: {p['txt']};
    border: 1px solid {p['border']};
    border-radius: 14px;
    selection-background-color: {p['accent']};
    selection-color: #ffffff;
    outline: none;
}}
QDateEdit::drop-down {{
    border: none;
    width: 18px;
}}
QCalendarWidget {{
    background-color: {p['bg']};
    color: {p['txt']};
    border: 1px solid {p['border']};
    border-radius: 10px;
}}
QCalendarWidget QWidget {{
    background-color: {p['bg']};
    color: {p['txt']};
}}
QCalendarWidget QWidget#qt_calendar_navigationbar {{
    background-color: {p['card']};
    border-bottom: 1px solid {p['border']};
    border-radius: 10px 10px 0 0;
    padding: 2px 4px;
}}
QCalendarWidget QToolButton {{
    background-color: transparent;
    color: {p['txt']};
    border: none;
    border-radius: 5px;
    padding: 3px 6px;
    font-size: 11px;
    font-weight: 600;
}}
QCalendarWidget QToolButton:hover {{
    background-color: {p['hover']};
    color: {p['accent']};
}}
QCalendarWidget QSpinBox {{
    background-color: {p['bg']};
    color: {p['txt']};
    border: 1px solid {p['border']};
    border-radius: 5px;
}}
QCalendarWidget QAbstractItemView:enabled {{
    background-color: {p['bg']};
    color: {p['txt']};
    selection-background-color: {p['accent']};
    selection-color: #ffffff;
    font-size: 11px;
}}
QCalendarWidget QAbstractItemView:disabled {{
    color: {p['muted']};
}}
QCheckBox {{
    color: {p['txt']};
    font-size: 11px;
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {p['border']};
    border-radius: 5px;
    background: {p['bg']};
}}
QCheckBox::indicator:checked {{
    background: {p['accent']};
    border-color: {p['accent']};
}}
QPushButton {{
    background: {p['card']};
    border: 1px solid {p['border']};
    border-radius: 14px;
    color: {p['txt']};
    padding: 5px 12px;
    font-size: 11px;
}}
QPushButton:hover {{
    background: {p['hover']};
    border-color: {p['accent']};
    color: {p['accent']};
}}
"""

# Module-level colour shorthands — refreshed each time the dialog opens via
# _refresh_palette().  Kept as globals so all existing f-string references
# throughout the module continue to work without changes.
_BG = _CARD = _BORDER = _TXT = _MUTED = _ACCENT = ""


def _refresh_palette() -> None:
    """Write current-theme colours into the module-level shorthand globals."""
    global _BG, _CARD, _BORDER, _TXT, _MUTED, _ACCENT
    p = _pal()
    _BG, _CARD, _BORDER = p["bg"], p["card"], p["border"]
    _TXT, _MUTED, _ACCENT = p["txt"], p["muted"], p["accent"]


# Opciones clínicas
_TRATAMIENTOS_CARACT = [
    "Clipaje",
    "Diversor de flujo + Clip",
    "Coils + Clips",
]

_TIPOS_ANEURISMA = [
    "",
    "Sacular",
    "Fusiforme",
    "Disecante",
    "Blíster",
    "Gigante (> 25 mm)",
    "Grande (10-25 mm)",
    "Pequeño (< 10 mm)",
    "Otro",
]

_REGIONES = [
    "",
    "Arteria cerebral media (ACM)",
    "Arteria comunicante anterior (AComA)",
    "Arteria comunicante posterior (AComP)",
    "Arteria carótida interna (ACI)",
    "Arteria basilar",
    "Arteria cerebelosa posteroinferior (PICA)",
    "Arteria cerebral anterior (ACA)",
    "Arteria cerebral posterior (ACP)",
    "Bifurcación carotídea",
    "Otra",
]

_LATERALIDAD = ["", "Derecha", "Izquierda", "Bilateral", "Línea media"]
_ANGIO_TIPO  = ["", "BIPLANO", "MONOPLANO"]
_SEX_OPTS    = [("", "—"), ("M", "Masculino"), ("F", "Femenino"), ("O", "Otro")]


# ──────────────────────────────────────────────────────────────────────────── #
# Helper widgets                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

def _section(title: str) -> QGroupBox:
    grp = QGroupBox(title)
    grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    return grp


def _label(text: str, muted: bool = False) -> QLabel:
    lbl = QLabel(text)
    if muted:
        lbl.setStyleSheet(f"color:{_MUTED}; font-size:10px;")
    return lbl


def _field(placeholder: str = "", max_len: int = 0) -> QLineEdit:
    w = QLineEdit()
    if placeholder:
        w.setPlaceholderText(placeholder)
    if max_len:
        w.setMaxLength(max_len)
    return w


def _textarea(placeholder: str = "", rows: int = 2) -> QTextEdit:
    w = QTextEdit()
    w.setPlaceholderText(placeholder)
    w.setFixedHeight(rows * 22 + 10)
    return w


# ──────────────────────────────────────────────────────────────────────────── #
# Left panel — decorative                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

class _LeftPanel(QWidget):
    """Dark gradient panel with logo and tagline — visual identity."""

    def paintEvent(self, event):  # type: ignore[override]
        from PyQt5.QtGui import QLinearGradient
        p = QPainter(self)
        bg = _pal()["bg"]
        p.fillRect(self.rect(), QColor(bg))
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0.0, QColor(40, 80, 180, 55))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(self.rect(), g)
        p.end()


# ──────────────────────────────────────────────────────────────────────────── #
# Main dialog                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

class NuevoCasoDialog(QDialog):
    """
    Clinical intake form for a new aneurysm case.

    After accept(), use:
        dialog.patient_id  — id of created/existing Patient
        dialog.study_id    — id of created Study
        dialog.dicom_path  — selected DICOM directory (may be empty)
    """

    def __init__(
        self,
        created_by: str = "",
        patient_id: int | None = None,
        study_id: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._created_by = created_by
        # Edit mode when both IDs are supplied
        self._edit_mode       = patient_id is not None and study_id is not None
        self._edit_patient_id = patient_id
        self._edit_study_id   = study_id

        self.patient_id: int | None = None
        self.study_id:   int | None = None
        self.dicom_path: str = ""

        title = "Editar caso clínico" if self._edit_mode else "Nuevo caso clínico"
        self.setWindowTitle(title)
        self.setMinimumSize(820, 560)
        from PyQt5.QtWidgets import QApplication as _QApp
        _scr = _QApp.primaryScreen()
        if _scr is not None:
            _av = _scr.availableGeometry()
            self.resize(min(1100, int(_av.width() * 0.88)), min(720, int(_av.height() * 0.88)))
        else:
            self.resize(1050, 680)
        self.setWindowFlags(
            Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint
        )
        _refresh_palette()   # fill _BG, _CARD, etc. from current theme
        self.setStyleSheet(f"QDialog {{ background:{_BG}; }}" + _form_style())

        self._build_ui()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        try:
            from prospective.ui.glass_utils import enable_acrylic
            enable_acrylic(self)
        except Exception:
            pass

        if self._edit_mode:
            self._btn_ok.setText("Guardar cambios")
            self._prefill()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left decorative panel ─────────────────────────────────────── #
        left = _LeftPanel()
        left.setFixedWidth(220)
        left.setMinimumWidth(160)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(24, 32, 24, 24)
        left_lay.setSpacing(12)

        # SkullApp logo image — tinted to contrast with left panel background
        skull_img = QLabel()
        skull_img.setStyleSheet("background:transparent;")
        _raw = QPixmap(_LOGO_PATH)
        if not _raw.isNull():
            _raw = _raw.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # Tint: white on dark bg, brand-purple on light bg
            _is_dark_mode = _pal()["bg"] != "#FFFFFF"
            _logo_color = "#ffffff" if _is_dark_mode else "#4E6678"
            _tinted = QPixmap(_raw.size())
            _tinted.fill(Qt.transparent)
            _wp = QPainter(_tinted)
            try:
                _wp.drawPixmap(0, 0, _raw)
                _wp.setCompositionMode(QPainter.CompositionMode_SourceIn)
                _wp.fillRect(_tinted.rect(), QColor(_logo_color))
            finally:
                _wp.end()
            skull_img.setPixmap(_tinted)
        else:
            skull_img.setText("💀")
            skull_img.setStyleSheet("font-size:28px; background:transparent;")
        left_lay.addWidget(skull_img)

        skull_row = QHBoxLayout()
        skull_row.setSpacing(6)
        logo_lbl = QLabel("PROSPECTIVE™")
        f = QFont("Inter",14, QFont.Bold)
        logo_lbl.setFont(f)
        logo_lbl.setStyleSheet(f"color:{_ACCENT};")
        skull_row.addWidget(logo_lbl)
        skull_row.addStretch()
        left_lay.addLayout(skull_row)

        skull_lbl = QLabel("SkullApp")
        skull_lbl.setStyleSheet(f"color:{_MUTED}; font-size:11px;")
        left_lay.addWidget(skull_lbl)

        left_lay.addSpacing(20)

        tagline = QLabel(
            "Registro del caso clínico.\n\n"
            "Complete los datos del paciente\n"
            "y la información clínica antes\n"
            "de cargar las imágenes DICOM."
        )
        tagline.setStyleSheet(f"color:{_MUTED}; font-size:11px; line-height:1.5;")
        tagline.setWordWrap(True)
        left_lay.addWidget(tagline)

        left_lay.addStretch()

        req_note = QLabel("* Campo requerido")
        req_note.setStyleSheet(f"color:{_WARNING}; font-size:10px;")
        left_lay.addWidget(req_note)

        root.addWidget(left)

        # ── Divider ───────────────────────────────────────────────────── #
        div = QWidget()
        div.setFixedWidth(1)
        div.setStyleSheet(f"background:{_BORDER};")
        root.addWidget(div)

        # ── Right: scrollable form ────────────────────────────────────── #
        right_outer = QWidget()
        right_outer.setStyleSheet(f"background:{_BG};")
        right_lay = QVBoxLayout(right_outer)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setStyleSheet(f"background:{_CARD}; border-bottom:1px solid {_BORDER};")
        title_bar.setFixedHeight(56)
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(24, 0, 24, 0)
        tb_title = QLabel("CASO NUEVO")
        tb_title.setFont(QFont("Inter",16, QFont.Bold))
        tb_title.setStyleSheet(f"color:{_TXT};")
        tb_lay.addWidget(tb_title)
        tb_lay.addStretch()
        right_lay.addWidget(title_bar)

        # Scrollable form area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet(f"QScrollArea{{background:{_BG};}}")

        form_container = QWidget()
        form_container.setStyleSheet(f"background:{_BG};")
        fcl = QVBoxLayout(form_container)
        fcl.setContentsMargins(24, 16, 24, 16)
        fcl.setSpacing(16)

        self._build_datos_paciente(fcl)
        self._build_antecedentes(fcl)
        self._build_datos_clinicos(fcl)
        self._build_caracterizacion(fcl)
        self._build_imagenes(fcl)

        fcl.addStretch()
        scroll.setWidget(form_container)
        right_lay.addWidget(scroll, stretch=1)

        # Button box
        btn_bar = QWidget()
        btn_bar.setStyleSheet(
            f"background:{_CARD}; border-top:1px solid {_BORDER};"
        )
        btn_bar.setFixedHeight(56)
        bb_lay = QHBoxLayout(btn_bar)
        bb_lay.setContentsMargins(24, 0, 24, 0)
        bb_lay.addStretch()

        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.setStyleSheet(
            f"QPushButton{{background:{_BG};border:1px solid {_BORDER};"
            f"color:{_MUTED};padding:6px 20px;border-radius:8px;}}"
            f"QPushButton:hover{{border-color:{_ACCENT};color:{_ACCENT};}}"
        )
        self._btn_cancel.clicked.connect(self.reject)
        bb_lay.addWidget(self._btn_cancel)

        bb_lay.addSpacing(8)

        self._btn_ok = QPushButton("Crear caso")
        if _pal()["bg"] == "#FFFFFF":   # light theme
            self._btn_ok.setStyleSheet(
                f"QPushButton{{background:#e6f4ea;border:1px solid {_SUCCESS};"
                f"color:#1a7a35;padding:6px 24px;border-radius:8px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{_SUCCESS};color:#ffffff;}}"
            )
        else:                           # dark theme
            self._btn_ok.setStyleSheet(
                f"QPushButton{{background:#1f4a1f;border:1px solid {_SUCCESS};"
                f"color:{_SUCCESS};padding:6px 24px;border-radius:8px;font-weight:bold;}}"
                f"QPushButton:hover{{background:#2a6a2a;}}"
            )
        self._btn_ok.clicked.connect(self._on_accept)
        bb_lay.addWidget(self._btn_ok)

        right_lay.addWidget(btn_bar)
        root.addWidget(right_outer, stretch=1)

    # ── Section 1: Datos del paciente ──────────────────────────────────── #

    def _build_datos_paciente(self, layout: QVBoxLayout) -> None:
        grp = _section("1. Datos del paciente")
        gf = QFormLayout()
        gf.setLabelAlignment(Qt.AlignRight)
        gf.setSpacing(8)
        gf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Nombre / Apellidos
        name_row = QWidget()
        nl = QHBoxLayout(name_row)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(8)
        self._fld_given = _field("Nombres", 120)
        self._fld_surname = _field("Apellidos", 120)
        nl.addWidget(self._fld_given, stretch=1)
        nl.addWidget(self._fld_surname, stretch=1)
        gf.addRow("Nombre del paciente *:", name_row)

        # Cédula / ID
        self._fld_cedula = _field("Número de cédula / historia clínica", 64)
        gf.addRow("Cédula / NHC *:", self._fld_cedula)

        # DOB + Edad (auto-calculated)
        dob_row = QWidget()
        dl = QHBoxLayout(dob_row)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(8)
        self._fld_dob = QDateEdit()
        self._fld_dob.setDisplayFormat("dd/MM/yyyy")
        self._fld_dob.setCalendarPopup(True)
        self._fld_dob.setDate(QDate(1970, 1, 1))
        self._fld_dob.setMaximumWidth(160)
        self._lbl_edad = QLabel("—")
        self._lbl_edad.setStyleSheet(f"color:{_MUTED}; font-size:11px;")
        self._fld_dob.dateChanged.connect(self._update_age)
        dl.addWidget(self._fld_dob)
        dl.addWidget(QLabel("Edad:"))
        dl.addWidget(self._lbl_edad)
        dl.addStretch()
        gf.addRow("Fecha de nacimiento:", dob_row)

        # Sexo
        self._cmb_sex = QComboBox()
        for val, label in _SEX_OPTS:
            self._cmb_sex.addItem(label, val)
        self._cmb_sex.setMaximumWidth(180)
        gf.addRow("Sexo:", self._cmb_sex)

        # Ocupación
        self._fld_ocupacion = _field("Profesión u ocupación", 200)
        gf.addRow("Ocupación:", self._fld_ocupacion)

        # Hospital
        self._fld_hospital = _field("Hospital o institución", 200)
        gf.addRow("Hospital:", self._fld_hospital)

        # Fecha del caso
        self._fld_fecha = QDateEdit()
        self._fld_fecha.setDisplayFormat("dd/MM/yyyy")
        self._fld_fecha.setCalendarPopup(True)
        self._fld_fecha.setDate(QDate.currentDate())
        self._fld_fecha.setMaximumWidth(160)
        gf.addRow("Fecha del caso:", self._fld_fecha)

        grp.setLayout(gf)
        layout.addWidget(grp)

    # ── Section 2: Antecedentes ────────────────────────────────────────── #

    def _build_antecedentes(self, layout: QVBoxLayout) -> None:
        grp = _section("2. Antecedentes personales")
        gl = QVBoxLayout()
        gl.setSpacing(6)

        header = QLabel(
            "Marque la categoría y describa brevemente los antecedentes relevantes."
        )
        header.setStyleSheet(f"color:{_MUTED}; font-size:10px;")
        gl.addWidget(header)

        categories = [
            ("patologicos",   "Patológicos"),
            ("toxicologicos", "Toxicológicos"),
            ("quirurgicos",   "Quirúrgicos"),
            ("alergicos",     "Alérgicos"),
            ("farmacologicos", "Farmacológicos"),
        ]
        self._ant_fields: dict[str, tuple[QCheckBox, QTextEdit]] = {}

        for key, label in categories:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(8)

            chk = QCheckBox(label)
            chk.setMaximumWidth(160)
            ta = _textarea(f"Detalle los antecedentes {label.lower()}…", rows=2)
            ta.setEnabled(False)
            chk.toggled.connect(ta.setEnabled)

            rl.addWidget(chk)
            rl.addWidget(ta, stretch=1)
            gl.addWidget(row)
            self._ant_fields[key] = (chk, ta)

        grp.setLayout(gl)
        layout.addWidget(grp)

    # ── Section 3: Datos clínicos ──────────────────────────────────────── #

    def _build_datos_clinicos(self, layout: QVBoxLayout) -> None:
        grp = _section("3. Datos clínicos")
        gf = QFormLayout()
        gf.setLabelAlignment(Qt.AlignRight)
        gf.setSpacing(8)
        gf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._fld_sintomas = _textarea("Síntomas actuales, cefalea, déficit neurológico…", 3)
        gf.addRow("Síntomas positivos:", self._fld_sintomas)

        self._fld_dx_principal = _field("Diagnóstico principal")
        gf.addRow("Diagnóstico Principal *:", self._fld_dx_principal)

        self._fld_dx_secundario = _field("Diagnóstico secundario (opcional)")
        gf.addRow("Diagnóstico secundario:", self._fld_dx_secundario)

        grp.setLayout(gf)
        layout.addWidget(grp)

    # ── Section 4: Caracterización Aneurismática ───────────────────────── #

    def _build_caracterizacion(self, layout: QVBoxLayout) -> None:
        grp = _section("4. Caracterización Aneurismática")
        gf = QFormLayout()
        gf.setLabelAlignment(Qt.AlignRight)
        gf.setSpacing(8)
        gf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._cmb_tipo = QComboBox()
        for t in _TIPOS_ANEURISMA:
            self._cmb_tipo.addItem(t or "— seleccionar —", t)
        gf.addRow("Tipo de aneurisma:", self._cmb_tipo)

        self._cmb_region = QComboBox()
        for r in _REGIONES:
            self._cmb_region.addItem(r or "— seleccionar —", r)
        gf.addRow("Región anatómica:", self._cmb_region)

        self._cmb_lat = QComboBox()
        for lat in _LATERALIDAD:
            self._cmb_lat.addItem(lat or "— seleccionar —", lat)
        self._cmb_lat.setMaximumWidth(200)
        gf.addRow("Lateralidad:", self._cmb_lat)

        # Tratamiento propuesto — 3 focused checkboxes
        trat_widget = QWidget()
        trat_lay = QVBoxLayout(trat_widget)
        trat_lay.setContentsMargins(0, 2, 0, 2)
        trat_lay.setSpacing(5)
        self._trat_checks: dict[str, QCheckBox] = {}
        for t in _TRATAMIENTOS_CARACT:
            chk = QCheckBox(t)
            trat_lay.addWidget(chk)
            self._trat_checks[t] = chk
        gf.addRow("Tratamiento propuesto:", trat_widget)

        grp.setLayout(gf)
        layout.addWidget(grp)

    # ── Section 5: Imágenes diagnósticas ──────────────────────────────── #

    def _build_imagenes(self, layout: QVBoxLayout) -> None:
        grp = _section("5. Imágenes diagnósticas")
        gl = QVBoxLayout()
        gl.setSpacing(6)

        note = QLabel(
            "Seleccione al menos una modalidad de imagen diagnóstica (carpeta DICOM). "
            "Puede completar las demás después."
        )
        note.setStyleSheet(f"color:{_MUTED}; font-size:10px;")
        note.setWordWrap(True)
        gl.addWidget(note)

        gl.addSpacing(4)

        # Four modality rows — each returns a read-only QLineEdit
        self._fld_dicom_tac    = self._modality_row(gl, "Tomografía (TAC):")
        self._fld_dicom_angio  = self._modality_row(gl, "Angiografía:")
        self._fld_dicom_rm     = self._modality_row(gl, "Resonancia Magnética:")
        self._fld_dicom_pangio = self._modality_row(gl, "Panangiografía:")

        gl.addSpacing(6)

        # Angiographer info (kept from original design)
        angio_row = QWidget()
        al = QHBoxLayout(angio_row)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(8)
        angio_lbl = QLabel("Angiógrafo:")
        angio_lbl.setMinimumWidth(80)
        angio_lbl.setMaximumWidth(180)
        angio_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        al.addWidget(angio_lbl)
        self._fld_angio_marca = _field("Marca / modelo", 100)
        self._cmb_angio_tipo = QComboBox()
        for t in _ANGIO_TIPO:
            self._cmb_angio_tipo.addItem(t or "Tipo", t)
        self._cmb_angio_tipo.setMaximumWidth(140)
        al.addWidget(self._fld_angio_marca, stretch=1)
        al.addWidget(self._cmb_angio_tipo)
        gl.addWidget(angio_row)

        grp.setLayout(gl)
        layout.addWidget(grp)

    def _modality_row(self, parent_layout: QVBoxLayout, label: str) -> QLineEdit:
        """Add a modality label + read-only path field + browse button to *parent_layout*.

        Returns the QLineEdit so the caller can read / set its text.
        """
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.setSpacing(6)

        lbl = QLabel(label)
        lbl.setMinimumWidth(80)
        lbl.setMaximumWidth(180)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_l.addWidget(lbl)

        fld = QLineEdit()
        fld.setReadOnly(True)
        fld.setPlaceholderText("Carpeta DICOM…")
        row_l.addWidget(fld, stretch=1)

        from prospective.ui.icons import I as _I
        btn = QPushButton(_I.FOLDER)
        btn.setFixedWidth(36)
        btn.setToolTip(f"Examinar carpeta {label.rstrip(':')}")
        btn.clicked.connect(lambda _checked, f=fld: self._browse_dicom_into(f))
        row_l.addWidget(btn)

        parent_layout.addWidget(row_w)
        return fld

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _browse_dicom_into(self, field: QLineEdit) -> None:
        """Open a directory picker and write the chosen path into *field*."""
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta DICOM")
        if path:
            field.setText(path)

    # ------------------------------------------------------------------ #
    # Edit-mode pre-fill                                                   #
    # ------------------------------------------------------------------ #

    def _prefill(self) -> None:
        """Load existing Patient + Study data and populate all form widgets."""
        from prospective.db import DatabaseManager
        db = DatabaseManager.instance()

        patient = db.get_patient(self._edit_patient_id)
        study   = db.get_study(self._edit_study_id)
        if patient is None or study is None:
            return

        # ── Datos del paciente ────────────────────────────────────────── #
        self._fld_given.setText(patient.given_name or "")
        self._fld_surname.setText(patient.surname or "")
        self._fld_cedula.setText(patient.hospital_id or "")

        if patient.dob and len(patient.dob) == 10:
            parts = patient.dob.split("-")
            if len(parts) == 3:
                self._fld_dob.setDate(
                    QDate(int(parts[0]), int(parts[1]), int(parts[2]))
                )

        idx = self._cmb_sex.findData(patient.sex or "")
        if idx >= 0:
            self._cmb_sex.setCurrentIndex(idx)

        self._fld_ocupacion.setText(patient.ocupacion or "")
        self._fld_hospital.setText(patient.institution or "")

        if study.study_date and len(study.study_date) == 10:
            parts = study.study_date.split("-")
            if len(parts) == 3:
                self._fld_fecha.setDate(
                    QDate(int(parts[0]), int(parts[1]), int(parts[2]))
                )

        # ── Antecedentes ──────────────────────────────────────────────── #
        _ant_map = {
            "patologicos":   patient.antecedentes_patologicos,
            "toxicologicos": patient.antecedentes_toxicologicos,
            "quirurgicos":   patient.antecedentes_quirurgicos,
            "alergicos":     patient.antecedentes_alergicos,
            "farmacologicos": patient.antecedentes_farmacologicos,
        }
        for key, (chk, ta) in self._ant_fields.items():
            val = _ant_map.get(key, "")
            if val:
                chk.setChecked(True)
                ta.setPlainText(val)

        # ── Datos clínicos ────────────────────────────────────────────── #
        self._fld_sintomas.setPlainText(study.sintomas_positivos or "")
        self._fld_dx_principal.setText(study.dx_principal or "")
        self._fld_dx_secundario.setText(study.dx_secundario or "")

        idx = self._cmb_tipo.findData(study.tipo_aneurisma or "")
        if idx >= 0:
            self._cmb_tipo.setCurrentIndex(idx)

        idx = self._cmb_region.findData(study.region_anatomica or "")
        if idx >= 0:
            self._cmb_region.setCurrentIndex(idx)

        idx = self._cmb_lat.findData(study.lateralidad or "")
        if idx >= 0:
            self._cmb_lat.setCurrentIndex(idx)

        # Tratamientos — stored as "A, B, C"
        trat_set = {
            t.strip()
            for t in (study.tratamiento_propuesto or "").split(",")
            if t.strip()
        }
        for name, chk in self._trat_checks.items():
            chk.setChecked(name in trat_set)

        # ── Imágenes diagnósticas ─────────────────────────────────────── #
        # Prefer per-modality columns; fall back to legacy dicom_path in TAC
        self._fld_dicom_tac.setText(
            getattr(study, "dicom_tac", None) or study.dicom_path or ""
        )
        self._fld_dicom_angio.setText(getattr(study, "dicom_angio", None) or "")
        self._fld_dicom_rm.setText(getattr(study, "dicom_rm", None) or "")
        self._fld_dicom_pangio.setText(getattr(study, "dicom_pangio", None) or "")

        angio = study.angiographer or ""
        if "|" in angio:
            marca, tipo = angio.split("|", 1)
            self._fld_angio_marca.setText(marca.strip())
            idx = self._cmb_angio_tipo.findData(tipo.strip())
            if idx >= 0:
                self._cmb_angio_tipo.setCurrentIndex(idx)
        else:
            self._fld_angio_marca.setText(angio.strip())

    # ------------------------------------------------------------------ #

    def set_dicom_path(self, path: str) -> None:
        """Pre-fill the TAC DICOM path field (called externally, e.g. from MainWindow)."""
        self._fld_dicom_tac.setText(path)

    def _update_age(self, qdate: QDate) -> None:
        today = date.today()
        dob = date(qdate.year(), qdate.month(), qdate.day())
        if dob >= today or qdate == QDate(1970, 1, 1):
            self._lbl_edad.setText("—")
            return
        age = (
            today.year - dob.year
            - ((today.month, today.day) < (dob.month, dob.day))
        )
        self._lbl_edad.setText(f"{age} años")

    def _on_accept(self) -> None:
        # ── Validation ─────────────────────────────────────────────── #
        given   = self._fld_given.text().strip()
        surname = self._fld_surname.text().strip()
        cedula  = self._fld_cedula.text().strip()
        dx      = self._fld_dx_principal.text().strip()

        dicom_tac    = self._fld_dicom_tac.text().strip()
        dicom_angio  = self._fld_dicom_angio.text().strip()
        dicom_rm     = self._fld_dicom_rm.text().strip()
        dicom_pangio = self._fld_dicom_pangio.text().strip()
        filled_paths = [p for p in (dicom_tac, dicom_angio, dicom_rm, dicom_pangio) if p]
        primary_dicom = filled_paths[0] if filled_paths else ""

        errors = []
        if not given and not surname:
            errors.append("El nombre del paciente es obligatorio.")
        if not cedula:
            errors.append("La cédula / NHC es obligatoria.")
        if not dx:
            errors.append("El diagnóstico principal es obligatorio.")
        if not filled_paths:
            errors.append(
                "Seleccione al menos una modalidad de imagen diagnóstica "
                "(Tomografía, Angiografía, Resonancia Magnética o Panangiografía)."
            )

        if errors:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Campos requeridos", "\n".join(errors))
            return

        # ── Shared field extraction ──────────────────────────────────── #
        from prospective.db import DatabaseManager
        db = DatabaseManager.instance()

        dob_qd  = self._fld_dob.date()
        dob_str = (
            f"{dob_qd.year():04d}-{dob_qd.month():02d}-{dob_qd.day():02d}"
            if dob_qd != QDate(1970, 1, 1) else ""
        )
        sex = self._cmb_sex.currentData() or ""

        ant = {}
        for key, (chk, ta) in self._ant_fields.items():
            ant[key] = ta.toPlainText().strip() if chk.isChecked() else ""

        fecha_qd  = self._fld_fecha.date()
        fecha_str = (
            f"{fecha_qd.year():04d}-{fecha_qd.month():02d}-{fecha_qd.day():02d}"
        )
        tratamientos_sel = [t for t, chk in self._trat_checks.items() if chk.isChecked()]
        angio_marca  = self._fld_angio_marca.text().strip()
        angio_tipo   = self._cmb_angio_tipo.currentData() or ""
        angiographer = f"{angio_marca} | {angio_tipo}".strip(" |")

        _patient_kwargs = dict(
            given_name=given,
            surname=surname,
            hospital_id=cedula,
            dob=dob_str,
            sex=sex,
            institution=self._fld_hospital.text().strip(),
            ocupacion=self._fld_ocupacion.text().strip(),
            antecedentes_patologicos=ant.get("patologicos", ""),
            antecedentes_toxicologicos=ant.get("toxicologicos", ""),
            antecedentes_quirurgicos=ant.get("quirurgicos", ""),
            antecedentes_alergicos=ant.get("alergicos", ""),
            antecedentes_farmacologicos=ant.get("farmacologicos", ""),
        )
        _study_kwargs = dict(
            study_date=fecha_str,
            dicom_path=primary_dicom,
            sintomas_positivos=self._fld_sintomas.toPlainText().strip(),
            dx_principal=dx,
            dx_secundario=self._fld_dx_secundario.text().strip(),
            tipo_aneurisma=self._cmb_tipo.currentData() or "",
            tratamiento_propuesto=", ".join(tratamientos_sel),
            region_anatomica=self._cmb_region.currentData() or "",
            lateralidad=self._cmb_lat.currentData() or "",
            angiographer=angiographer,
            dicom_tac=dicom_tac,
            dicom_angio=dicom_angio,
            dicom_rm=dicom_rm,
            dicom_pangio=dicom_pangio,
        )

        # ── Edit mode: update existing records ───────────────────────── #
        if self._edit_mode:
            db.update_patient(self._edit_patient_id, **_patient_kwargs)
            db.update_study(self._edit_study_id, **_study_kwargs)
            self.patient_id = self._edit_patient_id
            self.study_id   = self._edit_study_id
            self.dicom_path = _study_kwargs["dicom_path"]
            logger.info(
                "Caso actualizado — patient_id=%d  study_id=%d",
                self.patient_id, self.study_id,
            )
            self.accept()
            return

        # ── Create mode: insert new records ──────────────────────────── #
        patient = db.add_patient(**_patient_kwargs)
        study   = db.add_study(
            patient_id=patient.id,
            created_by=self._created_by,
            **_study_kwargs,
        )
        self.patient_id = patient.id
        self.study_id   = study.id
        self.dicom_path = study.dicom_path
        logger.info(
            "Nuevo caso creado — patient_id=%d  study_id=%d  by=%s",
            patient.id, study.id, self._created_by,
        )
        self.accept()
