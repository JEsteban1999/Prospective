"""AuthManager — login, logout, password hashing, current user.

Uses PBKDF2-HMAC-SHA256 with 260 000 iterations (NIST SP 800-63B compliant).
No external dependencies beyond the Python standard library.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from prospective.db.database import DatabaseManager
from prospective.db.models import User

logger = logging.getLogger(__name__)

_ITERATIONS = 260_000
_HASH_ALG   = "sha256"
_SALT_BYTES = 32


def _hash_password(password: str, salt_hex: str) -> str:
    """Return the PBKDF2 digest as a hex string."""
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac(_HASH_ALG, password.encode("utf-8"), salt, _ITERATIONS)
    return dk.hex()


def _new_salt() -> str:
    return os.urandom(_SALT_BYTES).hex()


class AuthManager:
    """Singleton that manages the currently-logged-in user."""

    _instance: Optional["AuthManager"] = None

    @classmethod
    def instance(cls) -> "AuthManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        self._db = DatabaseManager.instance()
        self._current_user: User | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def current_user(self) -> User | None:
        return self._current_user

    @property
    def is_logged_in(self) -> bool:
        return self._current_user is not None

    def has_role(self, required: str) -> bool:
        """Check if the current user has *required* role or higher."""
        if self._current_user is None:
            return False
        return self._current_user.has_role(required)

    # ------------------------------------------------------------------ #
    # Login / logout                                                       #
    # ------------------------------------------------------------------ #

    def login(self, username: str, password: str) -> tuple[bool, str]:
        """Attempt login. Returns (success, error_message)."""
        with DbSession(self._db._engine) as s:
            stmt = select(User).where(User.username == username.strip())
            user = s.scalars(stmt).first()

            if user is None:
                logger.warning("Login failed: unknown user %r", username)
                return False, "Usuario no encontrado."

            if getattr(user, "status", User.STATUS_ACTIVE) == User.STATUS_PENDING:
                return False, (
                    "Tu cuenta está pendiente de aprobación.\n"
                    "Un administrador debe activarla antes de que puedas iniciar sesión."
                )
            if getattr(user, "status", User.STATUS_ACTIVE) == User.STATUS_REJECTED:
                return False, "Tu solicitud de registro fue rechazada. Contacta al administrador."
            if not user.is_active:
                return False, "Cuenta desactivada. Contacte al administrador."

            expected = _hash_password(password, user.salt)
            if expected != user.password_hash:
                logger.warning("Login failed: wrong password for %r", username)
                return False, "Contraseña incorrecta."

            # Update last_login
            user.last_login = datetime.now()
            s.commit()
            s.refresh(user)
            s.expunge(user)

        self._current_user = user
        logger.info("Login OK: %s (%s)", user.username, user.role)
        return True, ""

    def logout(self) -> None:
        logger.info("Logout: %s", self._current_user.username if self._current_user else "—")
        self._current_user = None

    # ------------------------------------------------------------------ #
    # User management (admin only)                                         #
    # ------------------------------------------------------------------ #

    def create_user(
        self,
        username: str,
        password: str,
        full_name: str = "",
        role: str = User.ROLE_RESIDENT,
        is_active: bool = True,
    ) -> tuple[User | None, str]:
        """Create a new user. Returns (user, error_message)."""
        username = username.strip()
        if not username:
            return None, "El nombre de usuario no puede estar vacío."
        if len(password) < 8:
            return None, "La contraseña debe tener al menos 8 caracteres."
        if role not in (User.ROLE_RESIDENT, User.ROLE_SURGEON, User.ROLE_ADMIN):
            return None, f"Rol desconocido: {role!r}"

        with DbSession(self._db._engine) as s:
            existing = s.scalars(select(User).where(User.username == username)).first()
            if existing is not None:
                return None, f"El usuario {username!r} ya existe."

            salt = _new_salt()
            pw_hash = _hash_password(password, salt)
            user = User(
                username=username,
                full_name=full_name.strip(),
                role=role,
                password_hash=pw_hash,
                salt=salt,
                is_active=is_active,
            )
            s.add(user)
            s.commit()
            s.refresh(user)
            s.expunge(user)
            logger.info("User created: %s (%s)", username, role)
            return user, ""

    def change_password(self, user_id: int, new_password: str) -> tuple[bool, str]:
        if len(new_password) < 8:
            return False, "La contraseña debe tener al menos 8 caracteres."
        with DbSession(self._db._engine) as s:
            user = s.get(User, user_id)
            if user is None:
                return False, "Usuario no encontrado."
            user.salt = _new_salt()
            user.password_hash = _hash_password(new_password, user.salt)
            s.commit()
        return True, ""

    def request_signup(
        self,
        username: str,
        password: str,
        full_name: str = "",
        national_id: str = "",
        professional_id: str = "",
        specialty: str = "",
        university: str = "",
        hospital: str = "",
        position: str = "",
        orcid: str = "",
        cv_path: str = "",
        photo_path: str = "",
    ) -> tuple[User | None, str]:
        """Self-registration: create a pending account that requires admin approval."""
        username = username.strip()
        if not username:
            return None, "El nombre de usuario no puede estar vacío."
        if len(password) < 8:
            return None, "La contraseña debe tener al menos 8 caracteres."

        with DbSession(self._db._engine) as s:
            existing = s.scalars(select(User).where(User.username == username)).first()
            if existing is not None:
                return None, f"El usuario {username!r} ya existe."

            salt = _new_salt()
            user = User(
                username=username,
                full_name=full_name.strip(),
                role=User.ROLE_SURGEON,      # default role; admin can change on approval
                password_hash=_hash_password(password, salt),
                salt=salt,
                is_active=False,             # not active until approved
                status=User.STATUS_PENDING,
                national_id=national_id.strip(),
                professional_id=professional_id.strip(),
                specialty=specialty.strip(),
                university=university.strip(),
                hospital=hospital.strip(),
                position=position.strip(),
                orcid=orcid.strip(),
                cv_path=cv_path,
                photo_path=photo_path,
            )
            s.add(user)
            s.commit()
            s.refresh(user)
            s.expunge(user)
            logger.info("Sign-up request: %s (%s)", username, specialty)
            return user, ""

    def approve_user(self, user_id: int) -> tuple[bool, str]:
        """Approve a pending registration."""
        with DbSession(self._db._engine) as s:
            user = s.get(User, user_id)
            if user is None:
                return False, "Usuario no encontrado."
            user.is_active = True
            user.status = User.STATUS_ACTIVE
            s.commit()
            logger.info("User approved: %s", user.username)
        return True, ""

    def reject_user(self, user_id: int) -> tuple[bool, str]:
        """Reject a pending registration."""
        with DbSession(self._db._engine) as s:
            user = s.get(User, user_id)
            if user is None:
                return False, "Usuario no encontrado."
            user.status = User.STATUS_REJECTED
            s.commit()
            logger.info("User rejected: %s", user.username)
        return True, ""

    def get_pending_users(self) -> list[User]:
        """Return all users with status=pending."""
        with DbSession(self._db._engine) as s:
            rows = s.scalars(
                select(User).where(User.status == User.STATUS_PENDING)
                .order_by(User.created_at)
            ).all()
            for r in rows:
                s.expunge(r)
            return list(rows)

    def update_user(self, user_id: int, **kwargs) -> tuple[bool, str]:
        """Update profile fields or role/is_active. Password via change_password()."""
        _allowed = {
            "full_name", "role", "is_active", "status",
            "national_id", "professional_id", "specialty",
            "university", "hospital", "position", "orcid",
            "cv_path", "photo_path",
        }
        with DbSession(self._db._engine) as s:
            user = s.get(User, user_id)
            if user is None:
                return False, "Usuario no encontrado."
            for k, v in kwargs.items():
                if k in _allowed:
                    setattr(user, k, v)
            s.commit()
        return True, ""

    def delete_user(self, user_id: int) -> tuple[bool, str]:
        """Delete a user. Cannot delete the last admin."""
        with DbSession(self._db._engine) as s:
            user = s.get(User, user_id)
            if user is None:
                return False, "Usuario no encontrado."
            if user.role == User.ROLE_ADMIN:
                admins = s.scalars(
                    select(User).where(User.role == User.ROLE_ADMIN, User.is_active == True)
                ).all()
                if len(admins) <= 1:
                    return False, "No se puede eliminar el único administrador activo."
            s.delete(user)
            s.commit()
        return True, ""

    def get_all_users(self) -> list[User]:
        with DbSession(self._db._engine) as s:
            rows = s.scalars(select(User).order_by(User.username)).all()
            for r in rows:
                s.expunge(r)
            return list(rows)

    # ------------------------------------------------------------------ #
    # First-run detection                                                  #
    # ------------------------------------------------------------------ #

    def has_any_user(self) -> bool:
        with DbSession(self._db._engine) as s:
            return s.scalars(select(User)).first() is not None
