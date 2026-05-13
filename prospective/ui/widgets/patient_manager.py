"""Patient management dialog — A-04-07.

Three-panel layout:
  Left   — patient list with search + CRUD buttons
  Center — studies for the selected patient
  Right  — planning sessions for the selected study

Signals emitted to MainWindow:
  session_open_requested(str)   — user wants to load a .prospective file
  session_link_requested(int)   — user wants to link the current session to study_id
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from prospective.db.database import DatabaseManager
from prospective.db.models import Patient, PlanningSession, Study
from prospective.ui.glass_utils import dialog_qss, enable_acrylic

logger = logging.getLogger(__name__)


def _apply_style(widget) -> None:
    """Apply the shared glass-theme QSS + acrylic blur to a dialog."""
    widget.setStyleSheet(dialog_qss())

# ──────────────────────────────────────────────────────────────────────────── #
# Helper: editable form dialogs                                                 #
# ──────────────────────────────────────────────────────────────────────────── #


class _PatientDialog(QDialog):
    """Add / edit patient dialog."""

    def __init__(self, parent=None, patient: Patient | None = None) -> None:
        super().__init__(parent)
        self._patient = patient
        self.setWindowTitle("Nuevo paciente" if patient is None else "Editar paciente")
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        _apply_style(self)
        self._build()
        if patient is not None:
            self._populate(patient)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._surname    = QLineEdit()
        self._given_name = QLineEdit()
        self._hosp_id    = QLineEdit()
        self._dob        = QLineEdit(); self._dob.setPlaceholderText("AAAA-MM-DD")
        self._sex        = QComboBox()
        self._sex.addItems(["", "M", "F", "O"])
        self._institution = QLineEdit()
        self._notes      = QTextEdit(); self._notes.setMaximumHeight(70)

        form.addRow("Apellidos:", self._surname)
        form.addRow("Nombre:", self._given_name)
        form.addRow("N.º historia:", self._hosp_id)
        form.addRow("F. nacimiento:", self._dob)
        form.addRow("Sexo:", self._sex)
        form.addRow("Institución:", self._institution)
        form.addRow("Notas:", self._notes)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.Ok).setDefault(True)
        layout.addWidget(btns)

        # Tab order through form fields
        self.setTabOrder(self._surname, self._given_name)
        self.setTabOrder(self._given_name, self._hosp_id)
        self.setTabOrder(self._hosp_id, self._dob)
        self.setTabOrder(self._dob, self._sex)
        self.setTabOrder(self._sex, self._institution)
        self.setTabOrder(self._institution, self._notes)
        # Enter key in single-line fields submits the dialog
        for _f in (self._surname, self._given_name, self._hosp_id,
                   self._dob, self._institution):
            _f.returnPressed.connect(self.accept)

    def _populate(self, p: Patient) -> None:
        self._surname.setText(p.surname)
        self._given_name.setText(p.given_name)
        self._hosp_id.setText(p.hospital_id)
        self._dob.setText(p.dob)
        idx = self._sex.findText(p.sex)
        if idx >= 0:
            self._sex.setCurrentIndex(idx)
        self._institution.setText(p.institution)
        self._notes.setPlainText(p.notes)

    def get_data(self) -> dict:
        return {
            "surname":     self._surname.text().strip(),
            "given_name":  self._given_name.text().strip(),
            "hospital_id": self._hosp_id.text().strip(),
            "dob":         self._dob.text().strip(),
            "sex":         self._sex.currentText(),
            "institution": self._institution.text().strip(),
            "notes":       self._notes.toPlainText().strip(),
        }


class _StudyDialog(QDialog):
    """Add / edit study dialog."""

    def __init__(self, parent=None, study: Study | None = None) -> None:
        super().__init__(parent)
        self._study = study
        self.setWindowTitle("Nuevo estudio" if study is None else "Editar estudio")
        self.setMinimumWidth(460)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        _apply_style(self)
        self._build()
        if study is not None:
            self._populate(study)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._dicom_path   = QLineEdit()
        self._dicom_path.setReadOnly(True)
        self._dicom_path.setPlaceholderText("(ninguno seleccionado)")

        btn_browse = QPushButton("Examinar…")
        btn_browse.clicked.connect(self._browse_dicom)

        path_row = QWidget()
        ph = QHBoxLayout(path_row)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.addWidget(self._dicom_path)
        ph.addWidget(btn_browse)

        self._study_date  = QLineEdit(); self._study_date.setPlaceholderText("AAAA-MM-DD")
        self._modality    = QLineEdit(); self._modality.setPlaceholderText("CTA / MR / …")
        self._description = QLineEdit()
        self._accession   = QLineEdit()
        self._notes       = QTextEdit(); self._notes.setMaximumHeight(60)

        form.addRow("Directorio DICOM:", path_row)
        form.addRow("Fecha estudio:",    self._study_date)
        form.addRow("Modalidad:",        self._modality)
        form.addRow("Descripción:",      self._description)
        form.addRow("N.º acceso:",       self._accession)
        form.addRow("Notas:",            self._notes)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.Ok).setDefault(True)
        layout.addWidget(btns)

        # Tab order through form fields
        self.setTabOrder(self._study_date, self._modality)
        self.setTabOrder(self._modality, self._description)
        self.setTabOrder(self._description, self._accession)
        self.setTabOrder(self._accession, self._notes)
        # Enter key in single-line fields submits the dialog
        for _f in (self._study_date, self._modality,
                   self._description, self._accession):
            _f.returnPressed.connect(self.accept)

    def _browse_dicom(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Seleccionar directorio DICOM")
        if path:
            self._dicom_path.setText(path)

    def _populate(self, s: Study) -> None:
        self._dicom_path.setText(s.dicom_path)
        self._study_date.setText(s.study_date)
        self._modality.setText(s.modality)
        self._description.setText(s.description)
        self._accession.setText(s.accession_number)
        self._notes.setPlainText(s.notes)

    def get_data(self) -> dict:
        return {
            "dicom_path":        self._dicom_path.text().strip(),
            "study_date":        self._study_date.text().strip(),
            "modality":          self._modality.text().strip(),
            "description":       self._description.text().strip(),
            "accession_number":  self._accession.text().strip(),
            "notes":             self._notes.toPlainText().strip(),
        }


class _LinkSessionDialog(QDialog):
    """Link the current in-memory session to a study record."""

    def __init__(self, parent=None, session_path: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Vincular sesión al historial")
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        _apply_style(self)
        self._build(session_path)

    def _build(self, session_path: str) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Datos de la sesión a registrar:"))

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._label      = QLineEdit()
        self._label.setPlaceholderText("ej. Plan definitivo con clip Sugita 7mm")
        self._file_path  = QLineEdit(session_path)
        self._file_path.setReadOnly(True)
        self._risk       = QComboBox()
        self._risk.addItems(["", "Bajo", "Moderado", "Alto"])
        self._neck_mm    = QLineEdit("0.0"); self._neck_mm.setPlaceholderText("mm")
        self._n_clips    = QLineEdit("0")
        self._notes      = QTextEdit(); self._notes.setMaximumHeight(60)
        self._is_final   = QCheckBox("Marcar como plan definitivo")

        form.addRow("Etiqueta:", self._label)
        form.addRow("Archivo:", self._file_path)
        form.addRow("Riesgo:", self._risk)
        form.addRow("Diám. cuello (mm):", self._neck_mm)
        form.addRow("N.º clips:", self._n_clips)
        form.addRow("Notas:", self._notes)
        form.addRow("", self._is_final)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.Ok).setDefault(True)
        layout.addWidget(btns)

        # Enter key in single-line fields submits the dialog
        for _f in (self._label, self._neck_mm, self._n_clips):
            _f.returnPressed.connect(self.accept)

    def get_data(self) -> dict:
        try:
            neck = float(self._neck_mm.text())
        except ValueError:
            neck = 0.0
        try:
            n_clips = int(self._n_clips.text())
        except ValueError:
            n_clips = 0
        return {
            "label":            self._label.text().strip(),
            "file_path":        self._file_path.text().strip(),
            "risk_label":       self._risk.currentText(),
            "neck_diameter_mm": neck,
            "n_clips":          n_clips,
            "notes":            self._notes.toPlainText().strip(),
            "is_final":         self._is_final.isChecked(),
        }


# ──────────────────────────────────────────────────────────────────────────── #
# Main dialog                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #


class PatientManagerDialog(QDialog):
    """
    Three-panel patient management dialog.

    Signals
    -------
    session_open_requested(str)   — path to .prospective file to open
    session_link_requested(int)   — study_id to link current session to
    dicom_open_requested(str)     — DICOM directory from a study record
    """

    session_open_requested = pyqtSignal(str)
    session_link_requested = pyqtSignal(int)    # study_id
    dicom_open_requested   = pyqtSignal(str)

    def __init__(self, parent=None, current_session_path: str = "") -> None:
        super().__init__(parent)
        self._db = DatabaseManager.instance()
        self._current_session_path = current_session_path
        self._selected_patient_id: int | None = None
        self._selected_study_id: int | None = None
        self._selected_session_id_inline: int | None = None

        self.setWindowTitle("Gestión de Pacientes — PROSPECTIVE")
        self.setMinimumSize(820, 540)
        from PyQt5.QtWidgets import QApplication as _QApp
        _scr = _QApp.primaryScreen()
        if _scr is not None:
            _av = _scr.availableGeometry()
            self.resize(min(1200, int(_av.width() * 0.90)), min(680, int(_av.height() * 0.88)))
        else:
            self.resize(1100, 660)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        _apply_style(self)

        self._build_ui()
        self._refresh_patients()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        try:
            enable_acrylic(self)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self._build_patient_panel())
        splitter.addWidget(self._build_study_panel())
        splitter.addWidget(self._build_session_panel())

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 4)

        main_layout.addWidget(splitter, stretch=1)

        # Bottom bar
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 4, 0, 0)

        self._lbl_db = QLabel()
        self._lbl_db.setProperty("role", "muted")
        self._lbl_db.setText(f"BD: {self._db.db_path}")
        bl.addWidget(self._lbl_db)
        bl.addStretch()

        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(self.accept)
        bl.addWidget(close_btn)

        main_layout.addWidget(bottom)

    # ── Patient panel ─────────────────────────────────────────────────── #

    def _build_patient_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("<b>Pacientes</b>"))

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Buscar nombre / NHC…")
        self._search_edit.textChanged.connect(self._refresh_patients)
        layout.addWidget(self._search_edit)

        self._patient_table = QTableWidget()
        self._patient_table.setColumnCount(4)
        self._patient_table.setHorizontalHeaderLabels(["NHC", "Apellidos", "Nombre", "F. Nac."])
        self._patient_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._patient_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._patient_table.setAlternatingRowColors(True)
        self._patient_table.verticalHeader().setVisible(False)
        hdr = self._patient_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._patient_table.currentCellChanged.connect(self._on_patient_selected)
        layout.addWidget(self._patient_table, stretch=1)

        # Buttons
        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(4)

        self._btn_add_patient  = QPushButton("+ Nuevo")
        self._btn_edit_patient = QPushButton("Editar")
        self._btn_del_patient  = QPushButton("Eliminar")
        self._btn_edit_patient.setEnabled(False)
        self._btn_del_patient.setEnabled(False)

        self._btn_add_patient.clicked.connect(self._add_patient)
        self._btn_edit_patient.clicked.connect(self._edit_patient)
        self._btn_del_patient.clicked.connect(self._delete_patient)

        br.addWidget(self._btn_add_patient)
        br.addWidget(self._btn_edit_patient)
        br.addWidget(self._btn_del_patient)
        btn_row.setLayout(br)
        layout.addWidget(btn_row)

        return w

    # ── Study panel ───────────────────────────────────────────────────── #

    def _build_study_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("<b>Estudios y sesiones</b>"))

        # Tree: top-level items = studies, children = planning sessions
        self._study_tree = QTreeWidget()
        self._study_tree.setColumnCount(4)
        self._study_tree.setHeaderLabels(["Fecha", "Descripción / Etiqueta", "Tipo / Riesgo", "✓"])
        self._study_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._study_tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._study_tree.setAlternatingRowColors(True)
        self._study_tree.setRootIsDecorated(True)
        self._study_tree.setUniformRowHeights(True)
        self._study_tree.setAnimated(True)
        hdr = self._study_tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._study_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self._study_tree.itemDoubleClicked.connect(self._on_tree_double_clicked)
        layout.addWidget(self._study_tree, stretch=1)

        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(4)

        self._btn_add_study       = QPushButton("+ Nuevo estudio")
        self._btn_edit_study      = QPushButton("Editar")
        self._btn_del_study       = QPushButton("Eliminar")
        from prospective.ui.icons import I as _I
        self._btn_open_session_inline = QPushButton(f"{_I.DOC} Abrir sesión")
        self._btn_open_dicom      = QPushButton(f"{_I.FOLDER} Abrir DICOM")

        self._btn_add_study.setEnabled(False)
        self._btn_edit_study.setEnabled(False)
        self._btn_del_study.setEnabled(False)
        self._btn_open_session_inline.setEnabled(False)
        self._btn_open_dicom.setEnabled(False)

        self._btn_open_session_inline.setToolTip(
            "Restaura la sesión de planificación seleccionada\n"
            "También puedes hacer doble clic sobre una sesión"
        )
        self._btn_open_dicom.setToolTip(
            "Carga el directorio DICOM del estudio seleccionado\n"
            "También puedes hacer doble clic sobre un estudio"
        )

        self._btn_add_study.clicked.connect(self._add_study)
        self._btn_edit_study.clicked.connect(self._edit_study)
        self._btn_del_study.clicked.connect(self._delete_study)
        self._btn_open_session_inline.clicked.connect(self._open_selected_session_inline)
        self._btn_open_dicom.clicked.connect(self._open_study_dicom)

        br.addWidget(self._btn_add_study)
        br.addWidget(self._btn_edit_study)
        br.addWidget(self._btn_del_study)
        br.addStretch()
        br.addWidget(self._btn_open_session_inline)
        br.addWidget(self._btn_open_dicom)
        btn_row.setLayout(br)
        layout.addWidget(btn_row)

        return w

    # ── Session panel ─────────────────────────────────────────────────── #

    def _build_session_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("<b>Sesiones de planificación</b>"))

        self._session_table = QTableWidget()
        self._session_table.setColumnCount(6)
        self._session_table.setHorizontalHeaderLabels(
            ["Fecha", "Etiqueta", "Riesgo", "Cuello (mm)", "Clips", "Definitiva"]
        )
        self._session_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._session_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._session_table.setAlternatingRowColors(True)
        self._session_table.verticalHeader().setVisible(False)
        hdr = self._session_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self._session_table.currentCellChanged.connect(self._on_session_selected)
        layout.addWidget(self._session_table, stretch=1)

        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(4)

        self._btn_open_session   = QPushButton("Abrir sesión")
        self._btn_link_session   = QPushButton("Vincular sesión actual")
        self._btn_del_session    = QPushButton("Eliminar registro")

        self._btn_open_session.setEnabled(False)
        self._btn_del_session.setEnabled(False)

        self._btn_open_session.setToolTip("Carga el archivo .prospective en el visor")
        self._btn_link_session.setToolTip(
            "Registra la sesión actualmente abierta en el estudio seleccionado"
        )
        if not self._current_session_path:
            self._btn_link_session.setEnabled(False)

        self._btn_open_session.clicked.connect(self._open_session)
        self._btn_link_session.clicked.connect(self._link_current_session)
        self._btn_del_session.clicked.connect(self._delete_session)

        br.addWidget(self._btn_open_session)
        br.addWidget(self._btn_link_session)
        br.addStretch()
        br.addWidget(self._btn_del_session)
        btn_row.setLayout(br)
        layout.addWidget(btn_row)

        return w

    # ------------------------------------------------------------------ #
    # Data refresh helpers                                                 #
    # ------------------------------------------------------------------ #

    def _refresh_patients(self) -> None:
        search = self._search_edit.text().strip()
        patients = self._db.get_patients(search)
        self._patient_table.setRowCount(len(patients))
        self._patient_id_map: dict[int, int] = {}  # row → db id

        for row, p in enumerate(patients):
            self._patient_id_map[row] = p.id
            self._patient_table.setItem(row, 0, QTableWidgetItem(p.hospital_id or "—"))
            self._patient_table.setItem(row, 1, QTableWidgetItem(p.surname))
            self._patient_table.setItem(row, 2, QTableWidgetItem(p.given_name))
            self._patient_table.setItem(row, 3, QTableWidgetItem(p.dob or "—"))

        self._refresh_studies()

    def _refresh_studies(self) -> None:
        # Block signals while rebuilding so clear() doesn't fire _on_tree_selection_changed
        self._study_tree.blockSignals(True)
        self._study_tree.clear()
        self._study_tree.blockSignals(False)

        self._selected_study_id = None
        self._selected_session_id_inline = None
        self._btn_edit_study.setEnabled(False)
        self._btn_del_study.setEnabled(False)
        self._btn_open_session_inline.setEnabled(False)
        self._btn_open_dicom.setEnabled(False)
        self._btn_link_session.setEnabled(False)
        self._session_table.setRowCount(0)

        if self._selected_patient_id is None:
            return

        _risk_colors = {
            "Alto":     QColor("#7f1d1d"),
            "Moderado": QColor("#78350f"),
            "Bajo":     QColor("#14532d"),
        }

        studies = self._db.get_studies(self._selected_patient_id)
        for s in studies:
            # ── Top-level item = study ──────────────────────────────────── #
            study_item = QTreeWidgetItem()
            study_item.setData(0, Qt.UserRole, s.id)
            study_item.setText(0, s.study_date or "—")
            study_item.setText(1, s.description or "—")
            study_item.setText(2, s.modality or "—")
            study_item.setText(3, "")
            bold_font = study_item.font(1)
            bold_font.setBold(True)
            study_item.setFont(1, bold_font)
            self._study_tree.addTopLevelItem(study_item)

            # ── Child items = planning sessions for this study ──────────── #
            sessions = self._db.get_planning_sessions(s.id)
            for ps in sessions:
                sess_item = QTreeWidgetItem(study_item)
                sess_item.setData(0, Qt.UserRole, ps.id)
                date_str = (
                    ps.created_at.strftime("%Y-%m-%d %H:%M")
                    if ps.created_at else "—"
                )
                sess_item.setText(0, date_str)
                sess_item.setText(1, f"  ↳ {ps.label or '(sin etiqueta)'}")
                sess_item.setText(2, ps.risk_label or "—")
                if ps.risk_label in _risk_colors:
                    for col in range(4):
                        sess_item.setBackground(col, _risk_colors[ps.risk_label])
                sess_item.setText(3, "✓" if ps.is_final else "")
                if ps.is_final:
                    bold = QFont()
                    bold.setBold(True)
                    sess_item.setFont(3, bold)

            study_item.setExpanded(True)

    def _refresh_sessions(self) -> None:
        self._session_table.setRowCount(0)
        if self._selected_study_id is None:
            return

        sessions = self._db.get_planning_sessions(self._selected_study_id)
        self._session_id_map: dict[int, int] = {}

        _risk_colors = {
            "Alto":     QColor("#7f1d1d"),
            "Moderado": QColor("#78350f"),
            "Bajo":     QColor("#14532d"),
        }

        self._session_table.setRowCount(len(sessions))
        for row, ps in enumerate(sessions):
            self._session_id_map[row] = ps.id
            date_str = ps.created_at.strftime("%Y-%m-%d %H:%M") if ps.created_at else "—"
            self._session_table.setItem(row, 0, QTableWidgetItem(date_str))
            self._session_table.setItem(row, 1, QTableWidgetItem(ps.label or "—"))

            risk_item = QTableWidgetItem(ps.risk_label or "—")
            if ps.risk_label in _risk_colors:
                risk_item.setBackground(_risk_colors[ps.risk_label])
            self._session_table.setItem(row, 2, risk_item)

            self._session_table.setItem(row, 3, QTableWidgetItem(
                f"{ps.neck_diameter_mm:.1f}" if ps.neck_diameter_mm else "—"
            ))
            self._session_table.setItem(row, 4, QTableWidgetItem(str(ps.n_clips)))

            final_item = QTableWidgetItem("✓" if ps.is_final else "")
            final_item.setTextAlignment(Qt.AlignCenter)
            if ps.is_final:
                bold = QFont(); bold.setBold(True)
                final_item.setFont(bold)
            self._session_table.setItem(row, 5, final_item)

    # ------------------------------------------------------------------ #
    # Selection handlers                                                   #
    # ------------------------------------------------------------------ #

    def _on_patient_selected(self, row: int, *_) -> None:
        has = row >= 0
        self._btn_edit_patient.setEnabled(has)
        self._btn_del_patient.setEnabled(has)
        self._btn_add_study.setEnabled(has)

        self._selected_patient_id = self._patient_id_map.get(row) if has else None
        self._refresh_studies()

    def _on_tree_selection_changed(self) -> None:
        """Handle selection changes in the study/session tree."""
        items = self._study_tree.selectedItems()
        if not items:
            self._selected_study_id = None
            self._selected_session_id_inline = None
            self._btn_edit_study.setEnabled(False)
            self._btn_del_study.setEnabled(False)
            self._btn_open_session_inline.setEnabled(False)
            self._btn_open_dicom.setEnabled(False)
            self._btn_link_session.setEnabled(False)
            self._session_table.setRowCount(0)
            return

        item = items[0]
        is_study = item.parent() is None

        if is_study:
            # Study row selected — clear any inline session selection
            study_id = item.data(0, Qt.UserRole)
            self._selected_study_id = study_id
            self._selected_session_id_inline = None
            self._btn_open_session_inline.setEnabled(False)
        else:
            # Session child row selected
            sess_id  = item.data(0, Qt.UserRole)
            study_id = item.parent().data(0, Qt.UserRole)
            self._selected_study_id = study_id
            self._selected_session_id_inline = sess_id
            self._btn_open_session_inline.setEnabled(True)

        self._btn_edit_study.setEnabled(True)
        self._btn_del_study.setEnabled(True)

        # DICOM button — only when the path actually exists on disk
        dicom_ok = False
        if study_id is not None and self._selected_patient_id is not None:
            st = next(
                (s for s in self._db.get_studies(self._selected_patient_id)
                 if s.id == study_id), None
            )
            dicom_ok = bool(st and st.dicom_path and Path(st.dicom_path).exists())
        self._btn_open_dicom.setEnabled(dicom_ok)

        self._btn_link_session.setEnabled(bool(self._current_session_path))
        self._refresh_sessions()

    def _on_tree_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Double-click a study → open its DICOM; double-click a session → open it."""
        if item.parent() is None:
            self._open_study_dicom()
        else:
            self._open_selected_session_inline()

    def _on_session_selected(self, row: int, *_) -> None:
        has = row >= 0
        self._btn_open_session.setEnabled(has)
        self._btn_del_session.setEnabled(has)

    # ------------------------------------------------------------------ #
    # Patient actions                                                      #
    # ------------------------------------------------------------------ #

    def _add_patient(self) -> None:
        dlg = _PatientDialog(self)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            self._db.add_patient(**data)
            self._refresh_patients()

    def _edit_patient(self) -> None:
        if self._selected_patient_id is None:
            return
        p = self._db.get_patient(self._selected_patient_id)
        if p is None:
            return
        dlg = _PatientDialog(self, patient=p)
        if dlg.exec() == QDialog.Accepted:
            self._db.update_patient(self._selected_patient_id, **dlg.get_data())
            self._refresh_patients()

    def _delete_patient(self) -> None:
        if self._selected_patient_id is None:
            return
        p = self._db.get_patient(self._selected_patient_id)
        if p is None:
            return
        reply = QMessageBox.warning(
            self, "Eliminar paciente",
            f"¿Eliminar a <b>{p.full_name}</b> y todos sus estudios y sesiones?<br>"
            "Esta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            self._db.delete_patient(self._selected_patient_id)
            self._selected_patient_id = None
            self._refresh_patients()

    # ------------------------------------------------------------------ #
    # Study actions                                                        #
    # ------------------------------------------------------------------ #

    def _add_study(self) -> None:
        if self._selected_patient_id is None:
            return
        dlg = _StudyDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self._db.add_study(self._selected_patient_id, **dlg.get_data())
            self._refresh_studies()

    def _edit_study(self) -> None:
        if self._selected_study_id is None:
            return
        studies = self._db.get_studies(self._selected_patient_id)
        st = next((s for s in studies if s.id == self._selected_study_id), None)
        if st is None:
            return
        dlg = _StudyDialog(self, study=st)
        if dlg.exec() == QDialog.Accepted:
            self._db.update_study(self._selected_study_id, **dlg.get_data())
            self._refresh_studies()

    def _delete_study(self) -> None:
        if self._selected_study_id is None:
            return
        reply = QMessageBox.warning(
            self, "Eliminar estudio",
            "¿Eliminar este estudio y todas sus sesiones de planificación?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            self._db.delete_study(self._selected_study_id)
            self._selected_study_id = None
            self._refresh_studies()

    def _open_study_dicom(self) -> None:
        if self._selected_study_id is None:
            return
        studies = self._db.get_studies(self._selected_patient_id)
        st = next((s for s in studies if s.id == self._selected_study_id), None)
        if st and st.dicom_path:
            self.dicom_open_requested.emit(st.dicom_path)
            self.accept()

    # ------------------------------------------------------------------ #
    # Session actions                                                      #
    # ------------------------------------------------------------------ #

    def _open_session(self) -> None:
        row = self._session_table.currentRow()
        ps_id = self._session_id_map.get(row)
        if ps_id is None:
            return
        sessions = self._db.get_planning_sessions(self._selected_study_id)
        ps = next((p for p in sessions if p.id == ps_id), None)
        if ps and ps.file_path:
            if not Path(ps.file_path).exists():
                QMessageBox.warning(
                    self, "Archivo no encontrado",
                    f"No se encontró el archivo:\n{ps.file_path}"
                )
                return
            self.session_open_requested.emit(ps.file_path)
            self.accept()

    def _open_selected_session_inline(self) -> None:
        """Open the session selected directly inside the study tree."""
        if self._selected_session_id_inline is None or self._selected_study_id is None:
            return
        sessions = self._db.get_planning_sessions(self._selected_study_id)
        ps = next(
            (p for p in sessions if p.id == self._selected_session_id_inline), None
        )
        if ps is None or not ps.file_path:
            return
        if not Path(ps.file_path).exists():
            QMessageBox.warning(
                self, "Archivo no encontrado",
                f"No se encontró el archivo:\n{ps.file_path}"
            )
            return
        self.session_open_requested.emit(ps.file_path)
        self.accept()

    def _link_current_session(self) -> None:
        if self._selected_study_id is None or not self._current_session_path:
            return
        dlg = _LinkSessionDialog(self, self._current_session_path)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            self._db.add_planning_session(self._selected_study_id, **data)
            self._refresh_sessions()
            QMessageBox.information(
                self, "Sesión vinculada",
                "La sesión ha sido registrada en el historial del estudio."
            )

    def _delete_session(self) -> None:
        row = self._session_table.currentRow()
        ps_id = self._session_id_map.get(row)
        if ps_id is None:
            return
        reply = QMessageBox.warning(
            self, "Eliminar registro",
            "¿Eliminar este registro de sesión?\n"
            "(El archivo .prospective no se borrará del disco.)",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            self._db.delete_planning_session(ps_id)
            self._refresh_sessions()

    # ------------------------------------------------------------------ #
    # External update                                                      #
    # ------------------------------------------------------------------ #

    def set_current_session_path(self, path: str) -> None:
        """Update the path of the currently-open session (called by MainWindow)."""
        self._current_session_path = path
        self._btn_link_session.setEnabled(
            bool(path) and self._selected_study_id is not None
        )
