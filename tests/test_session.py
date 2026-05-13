"""Integration tests — session save / load (A-04-09)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from prospective.io.session import (
    SESSION_VERSION,
    ClipState,
    SessionData,
    load_session,
    save_session,
)


class TestSessionData:

    def test_default_values(self):
        d = SessionData()
        assert d.version == SESSION_VERSION
        assert d.wl_center == pytest.approx(170.0)
        assert d.clips == []

    def test_roundtrip_empty(self, tmp_path):
        d = SessionData()
        path = str(tmp_path / "empty.prospective")
        save_session(d, path)
        d2 = load_session(path)
        assert d2.version == SESSION_VERSION
        assert d2.wl_center == pytest.approx(d.wl_center)
        assert d2.clips == []

    def test_roundtrip_with_clips(self, tmp_path):
        d = SessionData()
        d.wl_center = 200.0
        d.wl_width  = 800.0
        d.clips = [
            ClipState(
                index=0,
                name="Sugita 7mm",
                is_custom=False,
                custom_path=None,
                position=[1.0, 2.0, 3.0],
                orientation=[0.0, 15.0, 0.0],
            )
        ]
        path = str(tmp_path / "clips.prospective")
        save_session(d, path)
        d2 = load_session(path)

        assert len(d2.clips) == 1
        c = d2.clips[0]
        assert c.name == "Sugita 7mm"
        assert c.position == [1.0, 2.0, 3.0]
        assert c.orientation == [0.0, 15.0, 0.0]
        assert c.is_custom is False

    def test_roundtrip_report_fields(self, tmp_path):
        d = SessionData()
        d.report_patient_name = "García, Ana"
        d.report_surgeon      = "Dr. Pérez"
        d.report_notes        = "Hipertensión arterial"
        path = str(tmp_path / "report.prospective")
        save_session(d, path)
        d2 = load_session(path)
        assert d2.report_patient_name == "García, Ana"
        assert d2.report_surgeon      == "Dr. Pérez"
        assert d2.report_notes        == "Hipertensión arterial"

    def test_file_is_valid_json(self, tmp_path):
        d = SessionData()
        path = str(tmp_path / "test.prospective")
        save_session(d, path)
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "version" in raw
        assert "window_level" in raw
        assert "clips" in raw
        assert "report" in raw

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(str(tmp_path / "nonexistent.prospective"))

    def test_extension_added_automatically(self, tmp_path):
        d = SessionData()
        path_no_ext = str(tmp_path / "mysession")
        save_session(d, path_no_ext)
        expected = tmp_path / "mysession.prospective"
        assert expected.exists()

    def test_trajectory_roundtrip(self, tmp_path):
        d = SessionData()
        d.traj_entry   = [10.0, 20.0, 30.0]
        d.traj_target  = [1.0, 2.0, 3.0]
        d.traj_visible = True
        path = str(tmp_path / "traj.prospective")
        save_session(d, path)
        d2 = load_session(path)
        assert d2.traj_entry   == [10.0, 20.0, 30.0]
        assert d2.traj_target  == [1.0, 2.0, 3.0]
        assert d2.traj_visible is True

    # ── v1.6 fields ────────────────────────────────────────────────────── #

    def test_clinical_state_roundtrip(self, tmp_path):
        """clinical_state (PHASES + parent_diam + treatment) survives save/load."""
        d = SessionData()
        d.clinical_state = {
            "phases": {
                "pop_idx": 2,
                "htn": True,
                "age": False,
                "sah": True,
                "site_idx": 1,
            },
            "parent_diam": 4.5,
            "treatment": {
                "location": "Basilar",
                "ruptured": False,
            },
        }
        path = str(tmp_path / "clinical.prospective")
        save_session(d, path)
        d2 = load_session(path)

        cs = d2.clinical_state
        assert cs["phases"]["pop_idx"] == 2
        assert cs["phases"]["htn"] is True
        assert cs["phases"]["sah"] is True
        assert cs["phases"]["site_idx"] == 1
        assert cs["parent_diam"] == pytest.approx(4.5)
        assert cs["treatment"]["location"] == "Basilar"
        assert cs["treatment"]["ruptured"] is False

    def test_clinical_state_defaults_to_empty(self, tmp_path):
        """Older sessions without clinical_state key load without error."""
        d = SessionData()
        raw = d.to_dict()
        raw.pop("clinical_state", None)   # simulate old session file
        with tempfile.NamedTemporaryFile(
            suffix=".prospective", delete=False, mode="w", encoding="utf-8"
        ) as f:
            import json as _json
            _json.dump(raw, f)
            fpath = f.name
        d2 = load_session(fpath)
        assert d2.clinical_state == {}

    def test_print_prep_state_roundtrip(self, tmp_path):
        """print_prep_state (7 mesh-prep params) survives save/load."""
        d = SessionData()
        d.print_prep_state = {
            "size":   120.0,
            "smooth": 30,
            "relax":  0.15,
            "fill":   False,
            "hole":   8.0,
            "sub":    True,
            "bed":    "Bambu Lab X1C (256×256×256 mm)",
        }
        path = str(tmp_path / "print.prospective")
        save_session(d, path)
        d2 = load_session(path)

        ps = d2.print_prep_state
        assert ps["size"]   == pytest.approx(120.0)
        assert ps["smooth"] == 30
        assert ps["relax"]  == pytest.approx(0.15)
        assert ps["fill"]   is False
        assert ps["hole"]   == pytest.approx(8.0)
        assert ps["sub"]    is True
        assert ps["bed"]    == "Bambu Lab X1C (256×256×256 mm)"

    def test_print_prep_state_defaults_to_empty(self, tmp_path):
        """Older sessions without print_prep_state key load without error."""
        d = SessionData()
        raw = d.to_dict()
        raw.pop("print_prep_state", None)   # simulate old session file
        with tempfile.NamedTemporaryFile(
            suffix=".prospective", delete=False, mode="w", encoding="utf-8"
        ) as f:
            import json as _json
            _json.dump(raw, f)
            fpath = f.name
        d2 = load_session(fpath)
        assert d2.print_prep_state == {}

    def test_v16_json_keys_present(self, tmp_path):
        """Saved file must contain both v1.6 top-level keys."""
        d = SessionData()
        d.clinical_state   = {"phases": {"pop_idx": 0}}
        d.print_prep_state = {"size": 80.0}
        path = str(tmp_path / "v16.prospective")
        save_session(d, path)
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "clinical_state"   in raw
        assert "print_prep_state" in raw
