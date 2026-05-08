"""Integration tests — authentication and roles (A-04-09)."""
from __future__ import annotations

import pytest

from prospective.auth.auth_manager import AuthManager
from prospective.db.database import DatabaseManager
from prospective.db.models import User


@pytest.fixture
def auth(tmp_path):
    """Fresh auth + DB per test."""
    DatabaseManager.reset()
    AuthManager.reset()
    DatabaseManager.instance(str(tmp_path / "auth_test.db"))
    mgr = AuthManager.instance()
    yield mgr
    AuthManager.reset()
    DatabaseManager.reset()


class TestFirstRun:

    def test_no_users_on_fresh_db(self, auth):
        assert auth.has_any_user() is False

    def test_create_first_admin(self, auth):
        user, err = auth.create_user("admin", "Password123", role=User.ROLE_ADMIN)
        assert err == ""
        assert user is not None
        assert user.role == User.ROLE_ADMIN
        assert auth.has_any_user() is True


class TestLogin:

    @pytest.fixture(autouse=True)
    def setup_user(self, auth):
        auth.create_user("drsmith", "Secure!456", full_name="Dr. Smith",
                         role=User.ROLE_SURGEON)

    def test_login_success(self, auth):
        ok, msg = auth.login("drsmith", "Secure!456")
        assert ok is True
        assert msg == ""
        assert auth.current_user is not None
        assert auth.current_user.username == "drsmith"

    def test_login_wrong_password(self, auth):
        ok, msg = auth.login("drsmith", "WrongPass!")
        assert ok is False
        assert msg != ""
        assert auth.current_user is None

    def test_login_unknown_user(self, auth):
        ok, msg = auth.login("nobody", "Password1")
        assert ok is False

    def test_logout_clears_user(self, auth):
        auth.login("drsmith", "Secure!456")
        auth.logout()
        assert auth.current_user is None
        assert auth.is_logged_in is False

    def test_last_login_updated(self, auth):
        auth.login("drsmith", "Secure!456")
        user = auth.current_user
        assert user.last_login is not None


class TestRoleHierarchy:

    def test_admin_has_all_roles(self, auth):
        auth.create_user("admin", "Admin123!", role=User.ROLE_ADMIN)
        auth.login("admin", "Admin123!")
        assert auth.has_role(User.ROLE_ADMIN)
        assert auth.has_role(User.ROLE_SURGEON)
        assert auth.has_role(User.ROLE_RESIDENT)

    def test_surgeon_not_admin(self, auth):
        auth.create_user("surg", "Surg1234!", role=User.ROLE_SURGEON)
        auth.login("surg", "Surg1234!")
        assert auth.has_role(User.ROLE_SURGEON)
        assert auth.has_role(User.ROLE_RESIDENT)
        assert not auth.has_role(User.ROLE_ADMIN)

    def test_resident_only_resident(self, auth):
        auth.create_user("res", "Resid123!", role=User.ROLE_RESIDENT)
        auth.login("res", "Resid123!")
        assert auth.has_role(User.ROLE_RESIDENT)
        assert not auth.has_role(User.ROLE_SURGEON)
        assert not auth.has_role(User.ROLE_ADMIN)

    def test_not_logged_in_no_role(self, auth):
        assert auth.has_role(User.ROLE_RESIDENT) is False


class TestUserManagement:

    def test_duplicate_username_rejected(self, auth):
        auth.create_user("user1", "Pass1234!")
        u2, err = auth.create_user("user1", "OtherPass!")
        assert u2 is None
        assert "ya existe" in err

    def test_short_password_rejected(self, auth):
        _, err = auth.create_user("user2", "short")
        assert "8 caracteres" in err

    def test_change_password(self, auth):
        u, _ = auth.create_user("user3", "OldPass12!")
        ok, _ = auth.change_password(u.id, "NewPass99!")
        assert ok is True
        ok2, _ = auth.login("user3", "NewPass99!")
        assert ok2 is True

    def test_old_password_invalid_after_change(self, auth):
        u, _ = auth.create_user("user4", "OldPass12!")
        auth.change_password(u.id, "NewPass99!")
        ok, _ = auth.login("user4", "OldPass12!")
        assert ok is False

    def test_deactivated_user_cannot_login(self, auth):
        u, _ = auth.create_user("user5", "Active123!", is_active=True)
        auth.update_user(u.id, is_active=False)
        ok, msg = auth.login("user5", "Active123!")
        assert ok is False
        assert "desactivada" in msg.lower()

    def test_cannot_delete_last_admin(self, auth):
        admin, _ = auth.create_user("admin", "Admin123!", role=User.ROLE_ADMIN)
        ok, err = auth.delete_user(admin.id)
        assert ok is False
        assert "único administrador" in err

    def test_can_delete_non_last_admin(self, auth):
        a1, _ = auth.create_user("admin1", "Admin123!", role=User.ROLE_ADMIN)
        a2, _ = auth.create_user("admin2", "Admin456!", role=User.ROLE_ADMIN)
        ok, _ = auth.delete_user(a2.id)
        assert ok is True

    def test_get_all_users(self, auth):
        auth.create_user("u1", "Pass1234!", role=User.ROLE_RESIDENT)
        auth.create_user("u2", "Pass5678!", role=User.ROLE_SURGEON)
        users = auth.get_all_users()
        assert len(users) == 2
