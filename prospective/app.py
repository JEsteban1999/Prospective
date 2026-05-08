"""Application entry point — creates QApplication and orchestrates the launch flow.

Flow
----
  LoginDialog (cinematic full-screen video bg)
    → success → WelcomeWindow (maximised, looping surgical video bg)
        → "CASO NUEVO"       → NuevoCasoDialog → MainWindow (3D planning)
        → "CASOS EXISTENTES" → CaseDashboard (lazy-created)
                                 → case card click → MainWindow
        → "SKULLCLOUD"       → "Próximamente" toast
        → "3D"               → active MainWindow or prompt
        → "AR/VR"            → "Próximamente" toast
        → "ANALITICA"        → "Próximamente" toast
        → "NOSOTROS"         → About dialog
        → "PQRS"             → _PQRSDialog contact form
        → My ▼ → Cerrar sesión → WelcomeWindow.close() → app.quit

Lifecycle notes
---------------
  * ``app.setQuitOnLastWindowClosed(False)`` keeps the event loop alive when
    WelcomeWindow is hidden (while MainWindow is open).
  * ``welcome.destroyed.connect(app.quit)`` is the actual quit trigger —
    fired only when WelcomeWindow is destroyed (logout or OS close).
  * CaseDashboard is created lazily on first "CASOS EXISTENTES" click;
    signals (open_case, open_session) are connected before show() is called
    to avoid the signal-connection race condition.
"""
from __future__ import annotations

import logging
import sys

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

logger = logging.getLogger(__name__)

# ── Resources ─────────────────────────────────────────────────────────────── #
_RES = Path(__file__).resolve().parent.parent / "resources"


def _app_icon() -> QIcon:
    """Return the best available app icon (ICO > PNG fallback)."""
    for name in ("icon.ico", "icon.png", "logo.ico", "logo.png"):
        path = _RES / name
        if path.exists():
            return QIcon(str(path))
    return QIcon()


def run() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setWindowIcon(_app_icon())          # ← título + taskbar en todos los QWidget
    app.setApplicationName("PROSPECTIVE")
    app.setOrganizationName("Fundación Universitaria Navarra UNINAVARRA")
    app.setApplicationVersion("0.1.0")
    # WelcomeWindow.destroyed → app.quit handles shutdown; disable the
    # default "quit when last window closes" so hiding WelcomeWindow while
    # MainWindow is open does not kill the process.
    app.setQuitOnLastWindowClosed(False)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Imports after QApplication exists (some modules need the app to exist)
    from prospective.db import DatabaseManager
    from prospective.auth import AuthManager
    from prospective.ui.widgets.login_dialog import LoginDialog
    from prospective.ui.windows.welcome_window import WelcomeWindow

    # Initialise singletons so LoginDialog can use them
    DatabaseManager.instance()
    AuthManager.instance()

    # ── Login (cinematic full-screen landing with looping video) ──────── #
    login = LoginDialog()
    if login.exec() != login.Accepted:
        return 0   # user closed / cancelled

    # Resolve the logged-in user
    _auth      = AuthManager.instance()
    _user      = _auth.current_user
    _username  = _user.username  if _user else ""
    _full_name = _user.full_name if _user else ""

    logger.info("PROSPECTIVE started — user: %s", _username or "—")

    # ── Audit: record successful login ───────────────────────────────── #
    try:
        from prospective.audit.skull_chain import SkullChain, ACT_LOGIN
        SkullChain.instance().append(
            ACT_LOGIN,
            {"app": "PROSPECTIVE"},
            username=_username,
        )
    except Exception:
        pass  # audit must never block the clinical workflow

    # ── Welcome screen (root window — app quits when it is destroyed) ── #
    welcome = WelcomeWindow(username=_username, full_name=_full_name)
    welcome.destroyed.connect(app.quit)
    welcome.showMaximized()

    return app.exec_()
