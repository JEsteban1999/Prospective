"""DatabaseManager — single access point for the SQLite persistence layer.

Usage
-----
    from prospective.db import DatabaseManager
    db = DatabaseManager.instance()   # opens ~/.prospective/patients.db
    patients = db.get_patients()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as DbSession, joinedload

from prospective.db.models import Base, Patient, PlanningSession, Study

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".prospective"
_DEFAULT_DB_NAME = "patients.db"


class DatabaseManager:
    """Thin CRUD wrapper around SQLAlchemy + SQLite."""

    _instance: Optional["DatabaseManager"] = None

    # ------------------------------------------------------------------ #
    # Singleton                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def instance(cls, db_path: str | None = None) -> "DatabaseManager":
        """Return the singleton, creating it the first time.

        *db_path* is only honoured on the first call; subsequent calls
        return the existing singleton regardless of *db_path*.
        """
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy singleton (for tests)."""
        cls._instance = None

    # ------------------------------------------------------------------ #
    # Init                                                                 #
    # ------------------------------------------------------------------ #

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = str(_DEFAULT_DB_DIR / _DEFAULT_DB_NAME)

        self._db_path = db_path
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._apply_migrations()
        logger.info("Database opened: %s", db_path)

    # ------------------------------------------------------------------ #
    # Schema migrations (additive only — SQLite ALTER TABLE ADD COLUMN)    #
    # ------------------------------------------------------------------ #

    def _apply_migrations(self) -> None:
        """Add any columns that exist in the ORM model but not yet in the DB.

        This handles upgrades from older installations without requiring
        Alembic.  Only ADD COLUMN is supported (safe, non-destructive).
        """
        from sqlalchemy import inspect, text

        inspector = inspect(self._engine)
        with self._engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                existing_cols = {
                    c["name"] for c in inspector.get_columns(table.name)
                }
                for col in table.columns:
                    if col.name in existing_cols:
                        continue
                    # Build a minimal ALTER TABLE statement
                    col_type = col.type.compile(dialect=self._engine.dialect)
                    default_clause = ""
                    if col.default is not None and col.default.is_scalar:
                        val = col.default.arg
                        if isinstance(val, str):
                            default_clause = f" DEFAULT '{val}'"
                        elif isinstance(val, bool):
                            default_clause = f" DEFAULT {1 if val else 0}"
                        elif val is not None:
                            default_clause = f" DEFAULT {val}"
                    nullable_clause = "" if col.nullable else " NOT NULL"
                    sql = (
                        f"ALTER TABLE {table.name} "
                        f"ADD COLUMN {col.name} {col_type}"
                        f"{default_clause}{nullable_clause}"
                    )
                    try:
                        conn.execute(text(sql))
                        logger.info("Migration: added column %s.%s", table.name, col.name)
                    except Exception as exc:
                        logger.debug("Migration skip %s.%s: %s", table.name, col.name, exc)

    @property
    def db_path(self) -> str:
        return self._db_path

    def _session(self) -> DbSession:
        return DbSession(self._engine)

    # ------------------------------------------------------------------ #
    # Patient CRUD                                                         #
    # ------------------------------------------------------------------ #

    def add_patient(
        self,
        surname: str = "",
        given_name: str = "",
        hospital_id: str = "",
        dob: str = "",
        sex: str = "",
        institution: str = "",
        notes: str = "",
        # Clinical / social fields
        ocupacion: str = "",
        antecedentes_patologicos: str = "",
        antecedentes_toxicologicos: str = "",
        antecedentes_quirurgicos: str = "",
        antecedentes_alergicos: str = "",
        antecedentes_farmacologicos: str = "",
    ) -> Patient:
        with self._session() as s:
            p = Patient(
                surname=surname,
                given_name=given_name,
                hospital_id=hospital_id,
                dob=dob,
                sex=sex,
                institution=institution,
                notes=notes,
                ocupacion=ocupacion,
                antecedentes_patologicos=antecedentes_patologicos,
                antecedentes_toxicologicos=antecedentes_toxicologicos,
                antecedentes_quirurgicos=antecedentes_quirurgicos,
                antecedentes_alergicos=antecedentes_alergicos,
                antecedentes_farmacologicos=antecedentes_farmacologicos,
            )
            s.add(p)
            s.commit()
            s.refresh(p)
            # Detach from session so caller can use the object freely
            s.expunge(p)
            return p

    def update_patient(self, patient_id: int, **kwargs) -> bool:
        _allowed = {"surname", "given_name", "hospital_id", "dob", "sex",
                    "institution", "notes", "ocupacion",
                    "antecedentes_patologicos", "antecedentes_toxicologicos",
                    "antecedentes_quirurgicos", "antecedentes_alergicos",
                    "antecedentes_farmacologicos"}
        with self._session() as s:
            p = s.get(Patient, patient_id)
            if p is None:
                return False
            for k, v in kwargs.items():
                if k in _allowed:
                    setattr(p, k, v)
            s.commit()
            return True

    def delete_patient(self, patient_id: int) -> bool:
        with self._session() as s:
            p = s.get(Patient, patient_id)
            if p is None:
                return False
            s.delete(p)
            s.commit()
            return True

    def get_patients(self, search: str = "") -> list[Patient]:
        with self._session() as s:
            stmt = select(Patient).order_by(Patient.surname, Patient.given_name)
            if search:
                like = f"%{search}%"
                from sqlalchemy import or_
                stmt = stmt.where(
                    or_(
                        Patient.surname.ilike(like),
                        Patient.given_name.ilike(like),
                        Patient.hospital_id.ilike(like),
                    )
                )
            rows = s.scalars(stmt).all()
            # Expunge all to detach from session
            for r in rows:
                s.expunge(r)
            return list(rows)

    def get_patient(self, patient_id: int) -> Patient | None:
        with self._session() as s:
            p = s.get(Patient, patient_id)
            if p is not None:
                s.expunge(p)
            return p

    # ------------------------------------------------------------------ #
    # Dashboard query                                                      #
    # ------------------------------------------------------------------ #

    def get_recent_cases(
        self,
        created_by: str = "",
        limit: int = 12,
    ) -> list[dict]:
        """
        Return a list of *limit* most-recently-created studies with
        associated patient data, optionally filtered by *created_by*.

        Each dict contains all Study + Patient scalar fields flattened.
        """
        with self._session() as s:
            from sqlalchemy import join as sa_join
            stmt = (
                select(Study, Patient)
                .join(Patient, Study.patient_id == Patient.id)
                .order_by(Study.created_at.desc())
            )
            if created_by:
                stmt = stmt.where(Study.created_by == created_by)
            stmt = stmt.limit(limit)
            rows = s.execute(stmt).all()
            result = []
            for study, patient in rows:
                d = {
                    # Study fields
                    "study_id":            study.id,
                    "study_date":          study.study_date,
                    "modality":            study.modality,
                    "dicom_path":          study.dicom_path,
                    "dx_principal":        study.dx_principal,
                    "dx_secundario":       study.dx_secundario,
                    "tipo_aneurisma":      study.tipo_aneurisma,
                    "tratamiento_propuesto": study.tratamiento_propuesto,
                    "region_anatomica":    study.region_anatomica,
                    "lateralidad":         study.lateralidad,
                    "angiographer":        study.angiographer,
                    "sintomas_positivos":  study.sintomas_positivos,
                    "created_at":          study.created_at,
                    "created_by":          study.created_by,
                    # Patient fields
                    "patient_id":          patient.id,
                    "patient_name":        patient.full_name,
                    "hospital_id":         patient.hospital_id,
                    "dob":                 patient.dob,
                    "sex":                 patient.sex,
                    "institution":         patient.institution,
                    "ocupacion":           patient.ocupacion,
                }
                result.append(d)
            return result

    # ------------------------------------------------------------------ #
    # Study CRUD                                                           #
    # ------------------------------------------------------------------ #

    def add_study(
        self,
        patient_id: int,
        dicom_path: str = "",
        study_date: str = "",
        modality: str = "",
        description: str = "",
        accession_number: str = "",
        notes: str = "",
        # Clinical episode fields (Nuevo Caso form)
        created_by: str = "",
        sintomas_positivos: str = "",
        dx_principal: str = "",
        dx_secundario: str = "",
        tipo_aneurisma: str = "",
        tratamiento_propuesto: str = "",
        region_anatomica: str = "",
        lateralidad: str = "",
        angiographer: str = "",
        # Per-modality DICOM paths
        dicom_tac: str = "",
        dicom_angio: str = "",
        dicom_rm: str = "",
        dicom_pangio: str = "",
    ) -> Study:
        with self._session() as s:
            st = Study(
                patient_id=patient_id,
                dicom_path=dicom_path,
                study_date=study_date,
                modality=modality,
                description=description,
                accession_number=accession_number,
                notes=notes,
                created_by=created_by,
                sintomas_positivos=sintomas_positivos,
                dx_principal=dx_principal,
                dx_secundario=dx_secundario,
                tipo_aneurisma=tipo_aneurisma,
                tratamiento_propuesto=tratamiento_propuesto,
                region_anatomica=region_anatomica,
                lateralidad=lateralidad,
                angiographer=angiographer,
                dicom_tac=dicom_tac,
                dicom_angio=dicom_angio,
                dicom_rm=dicom_rm,
                dicom_pangio=dicom_pangio,
            )
            s.add(st)
            s.commit()
            s.refresh(st)
            s.expunge(st)
            return st

    def update_study(self, study_id: int, **kwargs) -> bool:
        _allowed = {"dicom_path", "study_date", "modality",
                    "description", "accession_number", "notes",
                    "created_by", "sintomas_positivos", "dx_principal",
                    "dx_secundario", "tipo_aneurisma", "tratamiento_propuesto",
                    "region_anatomica", "lateralidad", "angiographer",
                    "dicom_tac", "dicom_angio", "dicom_rm", "dicom_pangio"}
        with self._session() as s:
            st = s.get(Study, study_id)
            if st is None:
                return False
            for k, v in kwargs.items():
                if k in _allowed:
                    setattr(st, k, v)
            s.commit()
            return True

    def delete_study(self, study_id: int) -> bool:
        with self._session() as s:
            st = s.get(Study, study_id)
            if st is None:
                return False
            s.delete(st)
            s.commit()
            return True

    def get_study(self, study_id: int) -> Study | None:
        """Return a single Study by primary key, or None if not found."""
        with self._session() as s:
            st = s.get(Study, study_id)
            if st is not None:
                s.expunge(st)
            return st

    def get_studies(self, patient_id: int) -> list[Study]:
        with self._session() as s:
            stmt = (
                select(Study)
                .where(Study.patient_id == patient_id)
                .order_by(Study.study_date.desc())
            )
            rows = s.scalars(stmt).all()
            for r in rows:
                s.expunge(r)
            return list(rows)

    # ------------------------------------------------------------------ #
    # PlanningSession CRUD                                                 #
    # ------------------------------------------------------------------ #

    def add_planning_session(
        self,
        study_id: int,
        file_path: str = "",
        label: str = "",
        risk_label: str = "",
        neck_diameter_mm: float = 0.0,
        n_clips: int = 0,
        notes: str = "",
        is_final: bool = False,
    ) -> PlanningSession:
        with self._session() as s:
            ps = PlanningSession(
                study_id=study_id,
                file_path=file_path,
                label=label,
                risk_label=risk_label,
                neck_diameter_mm=neck_diameter_mm,
                n_clips=n_clips,
                notes=notes,
                is_final=is_final,
            )
            s.add(ps)
            s.commit()
            s.refresh(ps)
            s.expunge(ps)
            return ps

    def update_planning_session(self, session_id: int, **kwargs) -> bool:
        _allowed = {"file_path", "label", "risk_label", "neck_diameter_mm",
                    "n_clips", "notes", "is_final"}
        with self._session() as s:
            ps = s.get(PlanningSession, session_id)
            if ps is None:
                return False
            for k, v in kwargs.items():
                if k in _allowed:
                    setattr(ps, k, v)
            s.commit()
            return True

    def delete_planning_session(self, session_id: int) -> bool:
        with self._session() as s:
            ps = s.get(PlanningSession, session_id)
            if ps is None:
                return False
            s.delete(ps)
            s.commit()
            return True

    def get_planning_sessions(self, study_id: int) -> list[PlanningSession]:
        with self._session() as s:
            stmt = (
                select(PlanningSession)
                .where(PlanningSession.study_id == study_id)
                .order_by(PlanningSession.created_at.desc(), PlanningSession.id.desc())
            )
            rows = s.scalars(stmt).all()
            for r in rows:
                s.expunge(r)
            return list(rows)

    def get_latest_planning_session(self) -> "PlanningSession | None":
        """Return the most recently created PlanningSession across all studies.

        The associated Study and Patient are eagerly loaded so callers can
        safely read ``ps.study.description`` and ``ps.study.patient.full_name``
        after the DB session is closed (no DetachedInstanceError).
        Returns None if no sessions exist yet.
        """
        with self._session() as s:
            stmt = (
                select(PlanningSession)
                .options(
                    joinedload(PlanningSession.study).joinedload(Study.patient)
                )
                .order_by(PlanningSession.created_at.desc(), PlanningSession.id.desc())
                .limit(1)
            )
            ps = s.scalar(stmt)
            if ps is not None:
                # Expunge the session AND all eagerly-loaded relatives so they
                # remain accessible after the DB session closes.
                if ps.study is not None:
                    if ps.study.patient is not None:
                        s.expunge(ps.study.patient)
                    s.expunge(ps.study)
                s.expunge(ps)
            return ps

    def get_planning_session_by_path(self, file_path: str) -> "PlanningSession | None":
        """Return the PlanningSession whose file_path matches exactly, or None."""
        with self._session() as s:
            stmt = select(PlanningSession).where(
                PlanningSession.file_path == str(file_path)
            )
            ps = s.scalar(stmt)
            if ps is not None:
                s.expunge(ps)
            return ps
