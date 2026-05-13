"""User management dialog — A-04-08.

Accessible only to administrators. Allows creating, editing, and
deactivating application users with their roles.
"""
from __future__ import annotations

import logging

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from prospective.auth.auth_manager import AuthManager
from prospective.db.models import User
from prospective.ui.glass_utils import dialog_qss, enable_acrylic

logger = logging.getLogger(__name__)

_ROLE_LABELS = {
    User.ROLE_RESIDENT: "Residente",
    User.ROLE_SURGEON:  "Cirujano",
    User.ROLE_ADMIN:    "Administrador",
}
_ROLE_KEYS = list(_ROLE_LABELS.keys())


class _UserFormDialog(QDialog):
    """Create or edit a user."""

    def __init__(self, parent=None, user: User | None = None) -> None:
        super().__init__(parent)
        self._user = user
        self.setWindowTitle("Nuevo usuario" if user is None else "Editar usuario")
        self.setMinimumWidth(380)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(dialog_qss())
        self._build()
        if user is not None:
            self._populate(user)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(8)

        self._username  = QLineEdit()
        self._full_name = QLineEdit()
        self._role_cb   = QComboBox()
        for key, label in _ROLE_LABELS.items():
            self._role_cb.addItem(label, userData=key)

        self._pw1 = QLineEdit(); self._pw1.setEchoMode(QLineEdit.Password)
        self._pw2 = QLineEdit(); self._pw2.setEchoMode(QLineEdit.Password)
        self._pw2.setPlaceholderText("Confirmar contraseña")
        self._active_cb = QCheckBox("Cuenta activa")
        self._active_cb.setChecked(True)

        if self._user is not None:
            self._username.setReadOnly(True)
            pw_hint = QLabel("(dejar vacío para no cambiar)")
            pw_hint.setProperty("role", "muted")
            pw_hint.setStyleSheet("font-size:9px;")
            form.addRow("Usuario:", self._username)
            form.addRow("Nombre completo:", self._full_name)
            form.addRow("Rol:", self._role_cb)
            form.addRow("Nueva contraseña:", self._pw1)
            form.addRow("", pw_hint)
            form.addRow("Confirmar:", self._pw2)
            form.addRow("", self._active_cb)
        else:
            form.addRow("Usuario:", self._username)
            form.addRow("Nombre completo:", self._full_name)
            form.addRow("Rol:", self._role_cb)
            form.addRow("Contraseña:", self._pw1)
            form.addRow("Confirmar:", self._pw2)
            form.addRow("", self._active_cb)

        layout.addLayout(form)

        self._error_lbl = QLabel("")
        self._error_lbl.setStyleSheet("color:#f87171; font-size:10px;")
        self._error_lbl.setWordWrap(True)
        layout.addWidget(self._error_lbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.Ok).setDefault(True)
        layout.addWidget(btns)

        # Tab order and Enter-to-submit for text fields
        self.setTabOrder(self._username, self._full_name)
        self.setTabOrder(self._full_name, self._role_cb)
        self.setTabOrder(self._role_cb, self._pw1)
        self.setTabOrder(self._pw1, self._pw2)
        for _f in (self._username, self._full_name, self._pw1, self._pw2):
            _f.returnPressed.connect(self._validate_and_accept)

    def _populate(self, u: User) -> None:
        self._username.setText(u.username)
        self._full_name.setText(u.full_name)
        idx = _ROLE_KEYS.index(u.role) if u.role in _ROLE_KEYS else 0
        self._role_cb.setCurrentIndex(idx)
        self._active_cb.setChecked(u.is_active)

    def _validate_and_accept(self) -> None:
        pw1 = self._pw1.text()
        pw2 = self._pw2.text()
        is_new = self._user is None

        if is_new:
            if not self._username.text().strip():
                self._error_lbl.setText("El nombre de usuario es obligatorio.")
                return
            if not pw1:
                self._error_lbl.setText("La contraseña es obligatoria.")
                return

        if pw1 or pw2:
            if pw1 != pw2:
                self._error_lbl.setText("Las contraseñas no coinciden.")
                return
            if len(pw1) < 8:
                self._error_lbl.setText("La contraseña debe tener al menos 8 caracteres.")
                return

        self.accept()

    def get_data(self) -> dict:
        return {
            "username":  self._username.text().strip(),
            "full_name": self._full_name.text().strip(),
            "role":      self._role_cb.currentData(),
            "password":  self._pw1.text(),          # may be empty on edit
            "is_active": self._active_cb.isChecked(),
        }


class UserManagerDialog(QDialog):
    """Full user management panel — admin only.

    Tab 1: Active users — create / edit / delete / change password.
    Tab 2: Pending requests — approve or reject self-registered accounts.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._auth = AuthManager.instance()
        self.setWindowTitle("Gestión de usuarios — PROSPECTIVE")
        self.setMinimumSize(760, 460)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(dialog_qss())
        self._build_ui()
        self._refresh()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        try:
            enable_acrylic(self)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._tabs = QTabWidget()
        from prospective.ui.icons import I as _I
        self._tabs.addTab(self._build_users_tab(),   f"{_I.USERS}  Usuarios activos")
        self._tabs.addTab(self._build_pending_tab(), f"{_I.WAIT}  Solicitudes pendientes")
        layout.addWidget(self._tabs, stretch=1)

        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

    # ── Tab 1: active users ──────────────────────────────────────────── #

    def _build_users_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Usuario", "Nombre completo", "Rol", "Estado", "Último acceso"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.currentCellChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table, stretch=1)

        btn_row = QHBoxLayout()
        self._btn_add     = QPushButton("+ Nuevo usuario")
        self._btn_edit    = QPushButton("Editar")
        self._btn_delete  = QPushButton("Eliminar")
        self._btn_pwreset = QPushButton("Cambiar contraseña")
        for b in (self._btn_edit, self._btn_delete, self._btn_pwreset):
            b.setEnabled(False)
        self._btn_add.clicked.connect(self._add_user)
        self._btn_edit.clicked.connect(self._edit_user)
        self._btn_delete.clicked.connect(self._delete_user)
        self._btn_pwreset.clicked.connect(self._reset_password)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_edit)
        btn_row.addWidget(self._btn_pwreset)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_delete)
        layout.addLayout(btn_row)
        return w

    # ── Tab 2: pending requests ──────────────────────────────────────── #

    def _build_pending_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        info = QLabel(
            "Solicitudes de registro enviadas por nuevos usuarios. "
            "Revisa los datos y <b>aprueba</b> o <b>rechaza</b> cada solicitud."
        )
        info.setWordWrap(True)
        info.setProperty("role", "muted")
        layout.addWidget(info)

        self._pending_table = QTableWidget()
        self._pending_table.setColumnCount(7)
        self._pending_table.setHorizontalHeaderLabels([
            "Usuario", "Nombre completo", "Especialidad",
            "Hospital", "Cargo", "ORCID", "Fecha solicitud",
        ])
        self._pending_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._pending_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._pending_table.setAlternatingRowColors(True)
        self._pending_table.verticalHeader().setVisible(False)
        ph = self._pending_table.horizontalHeader()
        ph.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(1, QHeaderView.Stretch)
        ph.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self._pending_table.currentCellChanged.connect(self._on_pending_selection)
        layout.addWidget(self._pending_table, stretch=1)

        pb_row = QHBoxLayout()
        # Approve / reject use semantic colours that override the base dialog_qss().
        # "Ver CV" intentionally inherits its style from dialog_qss().
        self._btn_approve = QPushButton("✔  Aprobar")
        self._btn_approve.setStyleSheet(
            "QPushButton{background:#1a7f37;border:none;border-radius:9px;"
            "color:#ffffff;padding:6px 16px;font-weight:700;}"
            "QPushButton:hover{background:#2ea043;}"
            "QPushButton:disabled{background:rgba(139,155,170,40);color:rgba(139,155,170,130);}"
        )
        self._btn_reject = QPushButton("✕  Rechazar")
        self._btn_reject.setStyleSheet(
            "QPushButton{background:#7a1f1f;border:none;border-radius:9px;"
            "color:#ffffff;padding:6px 16px;}"
            "QPushButton:hover{background:#9c2c2c;}"
            "QPushButton:disabled{background:rgba(139,155,170,40);color:rgba(139,155,170,130);}"
        )
        from prospective.ui.icons import I as _I
        self._btn_view_cv = QPushButton(f"{_I.DOC}  Ver CV")
        for b in (self._btn_approve, self._btn_reject, self._btn_view_cv):
            b.setEnabled(False)
        self._btn_approve.clicked.connect(self._approve_user)
        self._btn_reject.clicked.connect(self._reject_user)
        self._btn_view_cv.clicked.connect(self._view_cv)
        pb_row.addWidget(self._btn_approve)
        pb_row.addWidget(self._btn_reject)
        pb_row.addStretch()
        pb_row.addWidget(self._btn_view_cv)
        layout.addLayout(pb_row)
        return w

    # ------------------------------------------------------------------ #
    # Refresh                                                              #
    # ------------------------------------------------------------------ #

    def _refresh(self) -> None:
        self._refresh_users()
        self._refresh_pending()

    def _refresh_users(self) -> None:
        users = self._auth.get_all_users()
        # Filter out pending/rejected — show only active/inactive
        users = [u for u in users
                 if getattr(u, "status", "active") not in ("pending", "rejected")]
        self._user_id_map: dict[int, int] = {}
        self._table.setRowCount(len(users))

        for row, u in enumerate(users):
            self._user_id_map[row] = u.id
            self._table.setItem(row, 0, QTableWidgetItem(u.username))
            self._table.setItem(row, 1, QTableWidgetItem(u.full_name or "—"))
            self._table.setItem(row, 2, QTableWidgetItem(_ROLE_LABELS.get(u.role, u.role)))

            status_text = "Activo" if u.is_active else "Inactivo"
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(
                QColor("#3fb950") if u.is_active else QColor("#f85149")
            )
            self._table.setItem(row, 3, status_item)

            last = u.last_login.strftime("%Y-%m-%d %H:%M") if u.last_login else "—"
            self._table.setItem(row, 4, QTableWidgetItem(last))

        for b in (self._btn_edit, self._btn_delete, self._btn_pwreset):
            b.setEnabled(False)

    def _refresh_pending(self) -> None:
        pending = self._auth.get_pending_users()
        self._pending_id_map: dict[int, int] = {}
        self._pending_cv_map: dict[int, str] = {}
        self._pending_table.setRowCount(len(pending))

        for row, u in enumerate(pending):
            self._pending_id_map[row] = u.id
            self._pending_cv_map[row] = getattr(u, "cv_path", "") or ""

            self._pending_table.setItem(row, 0, QTableWidgetItem(u.username))
            self._pending_table.setItem(row, 1, QTableWidgetItem(u.full_name or "—"))
            self._pending_table.setItem(row, 2, QTableWidgetItem(
                getattr(u, "specialty", "") or "—"
            ))
            self._pending_table.setItem(row, 3, QTableWidgetItem(
                getattr(u, "hospital", "") or "—"
            ))
            self._pending_table.setItem(row, 4, QTableWidgetItem(
                getattr(u, "position", "") or "—"
            ))
            self._pending_table.setItem(row, 5, QTableWidgetItem(
                getattr(u, "orcid", "") or "—"
            ))
            created = u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "—"
            self._pending_table.setItem(row, 6, QTableWidgetItem(created))

        # Update tab badge
        count = len(pending)
        label = f"⏳  Solicitudes pendientes  ({count})" if count else "⏳  Solicitudes pendientes"
        self._tabs.setTabText(1, label)
        for b in (self._btn_approve, self._btn_reject, self._btn_view_cv):
            b.setEnabled(False)

    # ------------------------------------------------------------------ #
    # Slots — pending tab                                                  #
    # ------------------------------------------------------------------ #

    def _on_pending_selection(self, row: int, *_) -> None:
        has = row >= 0 and row in self._pending_id_map
        self._btn_approve.setEnabled(has)
        self._btn_reject.setEnabled(has)
        cv = self._pending_cv_map.get(row, "")
        self._btn_view_cv.setEnabled(has and bool(cv))

    def _approve_user(self) -> None:
        row = self._pending_table.currentRow()
        uid = self._pending_id_map.get(row)
        if uid is None:
            return
        username = self._pending_table.item(row, 0).text()
        reply = QMessageBox.question(
            self, "Aprobar usuario",
            f"¿Aprobar la solicitud de <b>{username}</b>?<br>"
            "El usuario podrá iniciar sesión inmediatamente.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        ok, err = self._auth.approve_user(uid)
        if ok:
            QMessageBox.information(
                self, "Usuario aprobado",
                f"La cuenta de <b>{username}</b> ha sido activada correctamente."
            )
            self._refresh()
        else:
            QMessageBox.critical(self, "Error", err)

    def _reject_user(self) -> None:
        row = self._pending_table.currentRow()
        uid = self._pending_id_map.get(row)
        if uid is None:
            return
        username = self._pending_table.item(row, 0).text()
        reply = QMessageBox.warning(
            self, "Rechazar solicitud",
            f"¿Rechazar la solicitud de registro de <b>{username}</b>?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        ok, err = self._auth.reject_user(uid)
        if ok:
            self._refresh()
        else:
            QMessageBox.critical(self, "Error", err)

    def _view_cv(self) -> None:
        row = self._pending_table.currentRow()
        cv_path = self._pending_cv_map.get(row, "")
        if not cv_path:
            return
        import os
        import subprocess
        import sys as _sys
        try:
            if _sys.platform == "win32":
                os.startfile(cv_path)
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", cv_path])
            else:
                subprocess.Popen(["xdg-open", cv_path])
        except Exception as exc:
            QMessageBox.warning(self, "No se pudo abrir el CV", str(exc))

    # ------------------------------------------------------------------ #
    # Slots — users tab                                                    #
    # ------------------------------------------------------------------ #

    def _on_selection_changed(self, row: int, *_) -> None:
        has = row >= 0
        self._btn_edit.setEnabled(has)
        self._btn_pwreset.setEnabled(has)
        # Can't delete yourself
        current = AuthManager.instance().current_user
        uid = self._user_id_map.get(row)
        self._btn_delete.setEnabled(has and uid != (current.id if current else None))

    def _add_user(self) -> None:
        dlg = _UserFormDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        data = dlg.get_data()
        user, err = self._auth.create_user(
            username=data["username"],
            password=data["password"],
            full_name=data["full_name"],
            role=data["role"],
            is_active=data["is_active"],
        )
        if user is None:
            QMessageBox.critical(self, "Error", err)
        else:
            self._refresh()

    def _edit_user(self) -> None:
        row = self._table.currentRow()
        uid = self._user_id_map.get(row)
        if uid is None:
            return
        users = self._auth.get_all_users()
        u = next((x for x in users if x.id == uid), None)
        if u is None:
            return
        dlg = _UserFormDialog(self, user=u)
        if dlg.exec() != QDialog.Accepted:
            return
        data = dlg.get_data()
        ok, err = self._auth.update_user(
            uid,
            full_name=data["full_name"],
            role=data["role"],
            is_active=data["is_active"],
        )
        if not ok:
            QMessageBox.critical(self, "Error", err)
            return
        if data["password"]:
            ok2, err2 = self._auth.change_password(uid, data["password"])
            if not ok2:
                QMessageBox.critical(self, "Error al cambiar contraseña", err2)
        self._refresh()

    def _delete_user(self) -> None:
        row = self._table.currentRow()
        uid = self._user_id_map.get(row)
        if uid is None:
            return
        users = self._auth.get_all_users()
        u = next((x for x in users if x.id == uid), None)
        if u is None:
            return
        reply = QMessageBox.warning(
            self, "Eliminar usuario",
            f"¿Eliminar al usuario <b>{u.username}</b>?<br>"
            "Esta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        ok, err = self._auth.delete_user(uid)
        if not ok:
            QMessageBox.critical(self, "Error", err)
        else:
            self._refresh()

    def _reset_password(self) -> None:
        row = self._table.currentRow()
        uid = self._user_id_map.get(row)
        if uid is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Cambiar contraseña")
        dlg.setMinimumWidth(320)
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        pw1 = QLineEdit(); pw1.setEchoMode(QLineEdit.Password)
        pw2 = QLineEdit(); pw2.setEchoMode(QLineEdit.Password)
        form.addRow("Nueva contraseña:", pw1)
        form.addRow("Confirmar:", pw2)
        lay.addLayout(form)
        err_lbl = QLabel(""); err_lbl.setStyleSheet("color:#f87171;")
        lay.addWidget(err_lbl)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setDefault(True)
        lay.addWidget(btns)

        def _accept():
            if pw1.text() != pw2.text():
                err_lbl.setText("Las contraseñas no coinciden.")
                return
            ok, msg = self._auth.change_password(uid, pw1.text())
            if ok:
                dlg.accept()
            else:
                err_lbl.setText(msg)

        btns.accepted.connect(_accept)
        btns.rejected.connect(dlg.reject)
        pw1.returnPressed.connect(_accept)
        pw2.returnPressed.connect(_accept)
        dlg.exec()
