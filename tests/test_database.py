"""Integration tests — patient / study / session DB (A-04-09)."""
from __future__ import annotations

import pytest

from prospective.db.database import DatabaseManager
from prospective.db.models import Patient, PlanningSession, Study


@pytest.fixture
def db(tmp_path):
    """Isolated in-memory-like DB per test via temp file."""
    DatabaseManager.reset()
    mgr = DatabaseManager.instance(str(tmp_path / "test.db"))
    yield mgr
    DatabaseManager.reset()


class TestPatientCRUD:

    def test_add_and_get(self, db):
        p = db.add_patient(surname="López", given_name="Ana", hospital_id="NHC-1")
        assert p.id is not None
        patients = db.get_patients()
        assert len(patients) == 1
        assert patients[0].surname == "López"

    def test_full_name_property(self, db):
        p = db.add_patient(surname="García", given_name="Carlos")
        assert p.full_name == "García, Carlos"

    def test_full_name_only_surname(self, db):
        p = db.add_patient(surname="Martínez")
        assert p.full_name == "Martínez"

    def test_search_by_name(self, db):
        db.add_patient(surname="Fernández", given_name="Luis")
        db.add_patient(surname="Rodríguez", given_name="María")
        results = db.get_patients(search="fern")
        assert len(results) == 1
        assert results[0].surname == "Fernández"

    def test_search_by_hospital_id(self, db):
        db.add_patient(surname="Pérez", hospital_id="NHC-999")
        results = db.get_patients(search="NHC-999")
        assert len(results) == 1

    def test_update_patient(self, db):
        p = db.add_patient(surname="Original")
        ok = db.update_patient(p.id, notes="Hypertension")
        assert ok is True
        updated = db.get_patient(p.id)
        assert updated.notes == "Hypertension"

    def test_delete_patient_cascades_studies(self, db):
        p  = db.add_patient(surname="ToDelete")
        st = db.add_study(p.id, description="Study 1")
        db.delete_patient(p.id)
        # Studies for this patient should be gone
        assert db.get_studies(p.id) == []

    def test_get_patient_not_found(self, db):
        assert db.get_patient(9999) is None


class TestStudyCRUD:

    def test_add_study(self, db):
        p  = db.add_patient(surname="Test")
        st = db.add_study(p.id, study_date="2026-01-15", modality="CTA")
        assert st.id is not None
        assert st.modality == "CTA"

    def test_get_studies_ordered_desc(self, db):
        p = db.add_patient(surname="Test")
        db.add_study(p.id, study_date="2025-01-01")
        db.add_study(p.id, study_date="2026-06-15")
        studies = db.get_studies(p.id)
        assert studies[0].study_date == "2026-06-15"

    def test_update_study(self, db):
        p  = db.add_patient(surname="Test")
        st = db.add_study(p.id, description="Old")
        db.update_study(st.id, description="New")
        studies = db.get_studies(p.id)
        assert studies[0].description == "New"

    def test_delete_study_cascades_sessions(self, db):
        p  = db.add_patient(surname="Test")
        st = db.add_study(p.id)
        db.add_planning_session(st.id, label="Draft")
        db.delete_study(st.id)
        assert db.get_planning_sessions(st.id) == []


class TestPlanningSessionCRUD:

    def test_add_session(self, db):
        p  = db.add_patient(surname="Test")
        st = db.add_study(p.id)
        ps = db.add_planning_session(
            st.id,
            label="Plan A",
            risk_label="Moderado",
            neck_diameter_mm=4.5,
            n_clips=1,
            is_final=False,
        )
        assert ps.id is not None
        assert ps.risk_label == "Moderado"

    def test_get_sessions_ordered_desc(self, db):
        p  = db.add_patient(surname="Test")
        st = db.add_study(p.id)
        db.add_planning_session(st.id, label="First")
        db.add_planning_session(st.id, label="Second")
        sessions = db.get_planning_sessions(st.id)
        # Most recent first
        assert sessions[0].label == "Second"

    def test_update_session(self, db):
        p  = db.add_patient(surname="Test")
        st = db.add_study(p.id)
        ps = db.add_planning_session(st.id, is_final=False)
        db.update_planning_session(ps.id, is_final=True)
        updated = db.get_planning_sessions(st.id)[0]
        assert updated.is_final is True

    def test_delete_session(self, db):
        p  = db.add_patient(surname="Test")
        st = db.add_study(p.id)
        ps = db.add_planning_session(st.id, label="To delete")
        db.delete_planning_session(ps.id)
        assert db.get_planning_sessions(st.id) == []
