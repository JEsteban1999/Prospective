"""SQLAlchemy ORM models — A-04-07 / A-04-08.

Tables:
  Patient         — demographic record (pseudonymised)
  Study           — one DICOM acquisition linked to a patient
  PlanningSession — one .prospective file linked to a study
  User            — application user with role-based access
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Demographics (pseudonymised — no full name stored by default)
    surname: Mapped[str] = mapped_column(String(120), default="")
    given_name: Mapped[str] = mapped_column(String(120), default="")
    hospital_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    dob: Mapped[str] = mapped_column(String(10), default="")   # YYYY-MM-DD
    sex: Mapped[str] = mapped_column(String(1), default="")    # M / F / O
    institution: Mapped[str] = mapped_column(String(200), default="")
    notes: Mapped[str] = mapped_column(Text, default="")

    # ── Clinical / social fields (added for "Nuevo Caso" form) ─────────── #
    ocupacion: Mapped[str] = mapped_column(String(200), default="")
    antecedentes_patologicos:   Mapped[str] = mapped_column(Text, default="")
    antecedentes_toxicologicos: Mapped[str] = mapped_column(Text, default="")
    antecedentes_quirurgicos:   Mapped[str] = mapped_column(Text, default="")
    antecedentes_alergicos:     Mapped[str] = mapped_column(Text, default="")
    antecedentes_farmacologicos: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    studies: Mapped[list[Study]] = relationship(
        "Study", back_populates="patient", cascade="all, delete-orphan"
    )

    @property
    def full_name(self) -> str:
        parts = [p for p in (self.surname, self.given_name) if p]
        return ", ".join(parts) if parts else "—"

    def __repr__(self) -> str:
        return f"<Patient id={self.id} name={self.full_name!r}>"


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )

    dicom_path: Mapped[str] = mapped_column(Text, default="")
    study_date: Mapped[str] = mapped_column(String(10), default="")   # YYYY-MM-DD
    modality: Mapped[str] = mapped_column(String(16), default="")
    description: Mapped[str] = mapped_column(String(256), default="")
    accession_number: Mapped[str] = mapped_column(String(64), default="")
    notes: Mapped[str] = mapped_column(Text, default="")

    # ── Clinical episode fields (added for "Nuevo Caso" form) ──────────── #
    created_by:         Mapped[str] = mapped_column(String(64), default="")  # username
    sintomas_positivos: Mapped[str] = mapped_column(Text, default="")
    dx_principal:       Mapped[str] = mapped_column(String(500), default="")
    dx_secundario:      Mapped[str] = mapped_column(String(500), default="")
    tipo_aneurisma:     Mapped[str] = mapped_column(String(200), default="")
    tratamiento_propuesto: Mapped[str] = mapped_column(Text, default="")
    region_anatomica:   Mapped[str] = mapped_column(String(300), default="")
    lateralidad:        Mapped[str] = mapped_column(String(100), default="")
    angiographer:       Mapped[str] = mapped_column(String(300), default="")  # "Marca | TIPO"

    # Per-modality DICOM paths (populated by "5. Imágenes diagnósticas")
    dicom_tac:    Mapped[str] = mapped_column(Text, default="")  # Tomografía (TAC)
    dicom_angio:  Mapped[str] = mapped_column(Text, default="")  # Angiografía
    dicom_rm:     Mapped[str] = mapped_column(Text, default="")  # Resonancia Magnética
    dicom_pangio: Mapped[str] = mapped_column(Text, default="")  # Panangiografía

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    patient: Mapped[Patient] = relationship("Patient", back_populates="studies")
    sessions: Mapped[list[PlanningSession]] = relationship(
        "PlanningSession", back_populates="study", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Study id={self.id} date={self.study_date!r} mod={self.modality!r}>"


class PlanningSession(Base):
    __tablename__ = "planning_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("studies.id", ondelete="CASCADE"), nullable=False, index=True
    )

    file_path: Mapped[str] = mapped_column(Text, default="")   # path to .prospective file
    label: Mapped[str] = mapped_column(String(200), default="")
    risk_label: Mapped[str] = mapped_column(String(32), default="")   # Alto/Moderado/Bajo
    neck_diameter_mm: Mapped[float] = mapped_column(Integer, default=0)
    n_clips: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")
    is_final: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    study: Mapped[Study] = relationship("Study", back_populates="sessions")

    def __repr__(self) -> str:
        return f"<PlanningSession id={self.id} label={self.label!r}>"


# ──────────────────────────────────────────────────────────────────────────── #
# User / auth (A-04-08)                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

class User(Base):
    """Application user with role-based access control.

    Roles (ascending privilege):
        resident  — view + run analysis; cannot delete patients or manage users
        surgeon   — full clinical workflow; cannot manage users
        admin     — full access including user management

    Status lifecycle (for self-registration flow):
        pending  — registered via Sign-Up, awaiting admin approval
        active   — approved and able to log in
        rejected — registration denied by admin
    """
    __tablename__ = "users"

    ROLE_RESIDENT = "resident"
    ROLE_SURGEON  = "surgeon"
    ROLE_ADMIN    = "admin"
    _ROLE_RANK    = {ROLE_RESIDENT: 0, ROLE_SURGEON: 1, ROLE_ADMIN: 2}

    STATUS_PENDING  = "pending"
    STATUS_ACTIVE   = "active"
    STATUS_REJECTED = "rejected"

    id: Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str]  = mapped_column(String(64), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(200), default="")
    role: Mapped[str]      = mapped_column(String(16), nullable=False, default=ROLE_RESIDENT)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    salt: Mapped[str]          = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool]    = mapped_column(Boolean, default=True)
    status: Mapped[str]        = mapped_column(String(16), default=STATUS_ACTIVE)

    # Professional profile fields (from Sign-Up form)
    national_id:     Mapped[str] = mapped_column(String(64),  default="")   # cédula / ID
    professional_id: Mapped[str] = mapped_column(String(64),  default="")   # ID profesional
    specialty:       Mapped[str] = mapped_column(String(100), default="")   # especialidad
    university:      Mapped[str] = mapped_column(String(200), default="")   # universidad afiliado
    hospital:        Mapped[str] = mapped_column(String(200), default="")   # hospital
    position:        Mapped[str] = mapped_column(String(100), default="")   # cargo
    orcid:           Mapped[str] = mapped_column(String(64),  default="")   # ORCID
    cv_path:         Mapped[str] = mapped_column(Text, default="")          # path to CV file
    photo_path:      Mapped[str] = mapped_column(Text, default="")          # path to profile photo

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def has_role(self, required: str) -> bool:
        """Return True if this user's role is >= *required* in the hierarchy."""
        return self._ROLE_RANK.get(self.role, -1) >= self._ROLE_RANK.get(required, 99)

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} role={self.role!r} status={self.status!r}>"
