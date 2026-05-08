"""Integration tests — DICOM SR export (A-04-09)."""
from __future__ import annotations

import pytest
import pydicom

from prospective.io.dicom_sr import DicomSRGenerator


_META = {
    "patient_name":      "Test^Patient",
    "patient_id":        "ID-001",
    "study_date":        "2026-03-15",
    "study_description": "CTA Cerebral",
    "modality":          "CTA",
}

_MORPHO = {
    "volume_mm3":         300.0,
    "surface_area_mm2":   220.0,
    "eq_sphere_diam_mm":  8.2,
    "max_diameter_mm":    9.8,
    "neck_diameter_mm":   3.9,
    "dome_height_mm":     7.1,
    "dome_to_neck_ratio": 2.5,
    "aspect_ratio":       1.8,
    "compactness":        0.71,
    "risk_label":         "Alto",
}

_CLIPS = [
    {
        "index":          0,
        "name":           "Sugita #10 - 7mm Straight",
        "position_mm":    (10.0, -3.0, 42.0),
        "orientation_deg":(0.0, 10.0, 0.0),
        "is_custom":      False,
    },
    {
        "index":          1,
        "name":           "Custom clip",
        "position_mm":    (11.0, -4.0, 43.0),
        "orientation_deg":(5.0, 5.0, 0.0),
        "is_custom":      True,
    },
]

_TRAJ = {
    "entry":     [5.0, 10.0, 80.0],
    "target":    [10.0, -3.0, 42.0],
    "depth_mm":  40.5,
    "angle_deg": 27.3,
}


@pytest.fixture
def sr_file(tmp_path):
    gen = DicomSRGenerator(_META, _MORPHO, _CLIPS, _TRAJ)
    out = gen.generate(str(tmp_path / "plan.dcm"))
    return pydicom.dcmread(str(out))


class TestDicomSRStructure:

    def test_sop_class_is_comprehensive_sr(self, sr_file):
        assert sr_file.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.33"

    def test_modality_is_sr(self, sr_file):
        assert sr_file.Modality == "SR"

    def test_patient_name(self, sr_file):
        assert "Test" in str(sr_file.PatientName)

    def test_patient_id(self, sr_file):
        assert sr_file.PatientID == "ID-001"

    def test_study_date_dicom_format(self, sr_file):
        # DICOM DA format: YYYYMMDD (no hyphens)
        assert sr_file.StudyDate == "20260315"
        assert "-" not in sr_file.StudyDate

    def test_has_three_top_level_containers(self, sr_file):
        assert len(sr_file.ContentSequence) == 3

    def test_completion_flag(self, sr_file):
        assert sr_file.CompletionFlag == "COMPLETE"


class TestMorphometryContainer:

    def test_container_name(self, sr_file):
        cont = sr_file.ContentSequence[0]
        assert cont.ConceptNameCodeSequence[0].CodeMeaning == "Aneurysm Morphometry"

    def test_has_ten_items(self, sr_file):
        cont = sr_file.ContentSequence[0]
        assert len(cont.ContentSequence) == 10  # 9 num + 1 risk code

    def test_volume_value(self, sr_file):
        cont = sr_file.ContentSequence[0]
        vol = next(
            i for i in cont.ContentSequence
            if i.ConceptNameCodeSequence[0].CodeMeaning == "Volume"
        )
        assert float(vol.MeasuredValueSequence[0].NumericValue) == pytest.approx(300.0)
        assert vol.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == "mm3"

    def test_risk_code_item(self, sr_file):
        cont = sr_file.ContentSequence[0]
        risk = next(
            i for i in cont.ContentSequence
            if i.ConceptNameCodeSequence[0].CodeMeaning == "Rupture Risk Assessment"
        )
        assert risk.ValueType == "CODE"
        assert risk.ConceptCodeSequence[0].CodeMeaning == "High risk"


class TestClipPlanContainer:

    def test_container_name(self, sr_file):
        cont = sr_file.ContentSequence[1]
        assert cont.ConceptNameCodeSequence[0].CodeMeaning == "Surgical Clip Plan"

    def test_two_clips(self, sr_file):
        cont = sr_file.ContentSequence[1]
        assert len(cont.ContentSequence) == 2

    def test_first_clip_name(self, sr_file):
        clip_cont = sr_file.ContentSequence[1].ContentSequence[0]
        model = next(
            i for i in clip_cont.ContentSequence
            if i.ConceptNameCodeSequence[0].CodeMeaning == "Clip Model"
        )
        assert "Sugita" in model.TextValue

    def test_custom_clip_type(self, sr_file):
        clip_cont = sr_file.ContentSequence[1].ContentSequence[1]
        clip_type = next(
            i for i in clip_cont.ContentSequence
            if i.ConceptNameCodeSequence[0].CodeMeaning == "Clip Type"
        )
        assert clip_type.TextValue == "Personalizado"


class TestTrajectoryContainer:

    def test_container_name(self, sr_file):
        cont = sr_file.ContentSequence[2]
        assert cont.ConceptNameCodeSequence[0].CodeMeaning == "Approach Trajectory"

    def test_eight_trajectory_items(self, sr_file):
        cont = sr_file.ContentSequence[2]
        assert len(cont.ContentSequence) == 8

    def test_depth_value(self, sr_file):
        cont = sr_file.ContentSequence[2]
        depth = next(
            i for i in cont.ContentSequence
            if i.ConceptNameCodeSequence[0].CodeMeaning == "Approach Depth"
        )
        assert float(depth.MeasuredValueSequence[0].NumericValue) == pytest.approx(40.5)

    def test_angle_units(self, sr_file):
        cont = sr_file.ContentSequence[2]
        angle = next(
            i for i in cont.ContentSequence
            if i.ConceptNameCodeSequence[0].CodeMeaning == "Approach Angle"
        )
        assert angle.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == "deg"


class TestEdgeCases:

    def test_empty_clips_and_trajectory(self, tmp_path):
        gen = DicomSRGenerator(_META, _MORPHO, [], {})
        out = gen.generate(str(tmp_path / "no_clips.dcm"))
        ds = pydicom.dcmread(str(out))
        # Only morphometry container — clips and trajectory skipped
        assert len(ds.ContentSequence) == 1

    def test_empty_morphometrics(self, tmp_path):
        gen = DicomSRGenerator(_META, {}, _CLIPS, _TRAJ)
        out = gen.generate(str(tmp_path / "no_morpho.dcm"))
        ds = pydicom.dcmread(str(out))
        # Clips + trajectory only
        assert len(ds.ContentSequence) == 2

    def test_extension_added(self, tmp_path):
        gen = DicomSRGenerator(_META, _MORPHO, [], {})
        out = gen.generate(str(tmp_path / "no_extension"))
        assert out.suffix == ".dcm"
