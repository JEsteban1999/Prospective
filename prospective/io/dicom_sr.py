"""DICOM Structured Report (SR) generator — A-04-06.

Produces a Comprehensive SR (SOP Class 1.2.840.10008.5.1.4.1.1.88.33)
following TID 1500 (Measurement Report) with:

  • Patient / study header linked to the source DICOM series
  • Aneurysm morphometric measurements (NUM content items, UCUM units)
  • Clinical risk assessment (CODE content item)
  • Surgical clip plan (one CONTAINER per clip)
  • Approach trajectory (NUM content items)

Coding schemes used:
  • DCM   — DICOM controlled terminology
  • SCT   — SNOMED CT (selected codes only)
  • UCUM  — Unified Code for Units of Measure
  • 99PRP — private scheme for PROSPECTIVE-specific concepts
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

logger = logging.getLogger(__name__)

# ── SOP Class UIDs ────────────────────────────────────────────────────────── #
_SR_COMPREHENSIVE = "1.2.840.10008.5.1.4.1.1.88.33"
_EXPLICIT_VR_LE   = "1.2.840.10008.1.2.1"

# ── Concept Name codes frequently used ───────────────────────────────────── #
_C = {
    # Containers
    "REPORT":           ("113000",    "DCM",   "Imaging Measurement Report"),
    "MORPHOMETRY":      ("99PRP001",  "99PRP", "Aneurysm Morphometry"),
    "PLAN":             ("99PRP002",  "99PRP", "Surgical Clip Plan"),
    "TRAJECTORY":       ("99PRP003",  "99PRP", "Approach Trajectory"),
    "CLIP":             ("99PRP004",  "99PRP", "Surgical Clip"),
    "STENT_PLAN":       ("99PRP005",  "99PRP", "Endovascular Device Plan"),
    "STENT":            ("99PRP006",  "99PRP", "Endovascular Device"),
    "COIL_PLAN":        ("99PRP007",  "99PRP", "Embolization Coil Plan"),
    "COIL":             ("99PRP008",  "99PRP", "Embolization Coil"),
    # Morphometric measurements
    "VOLUME":           ("121216",    "DCM",   "Volume"),
    "AREA":             ("99PRP010",  "99PRP", "Surface Area"),
    "MAX_DIAM":         ("103339001", "SCT",   "Maximum Diameter"),
    "NECK_DIAM":        ("99PRP011",  "99PRP", "Neck Diameter"),
    "DOME_HEIGHT":      ("99PRP012",  "99PRP", "Dome Height"),
    "DNR":              ("99PRP013",  "99PRP", "Dome-to-Neck Ratio"),
    "AR":               ("99PRP014",  "99PRP", "Aspect Ratio"),
    "COMPACTNESS":      ("99PRP015",  "99PRP", "Compactness (Wadell Sphericity)"),
    "EQ_DIAM":          ("99PRP016",  "99PRP", "Equivalent Sphere Diameter"),
    # Risk
    "RISK":             ("99PRP020",  "99PRP", "Rupture Risk Assessment"),
    # Clip fields
    "CLIP_MODEL":       ("99PRP030",  "99PRP", "Clip Model"),
    "POS_X":            ("99PRP031",  "99PRP", "Clip Position X"),
    "POS_Y":            ("99PRP032",  "99PRP", "Clip Position Y"),
    "POS_Z":            ("99PRP033",  "99PRP", "Clip Position Z"),
    "ROT_X":            ("99PRP034",  "99PRP", "Clip Rotation X"),
    "ROT_Y":            ("99PRP035",  "99PRP", "Clip Rotation Y"),
    "ROT_Z":            ("99PRP036",  "99PRP", "Clip Rotation Z"),
    "CLIP_TYPE":        ("99PRP037",  "99PRP", "Clip Type"),
    # Trajectory
    "ENTRY_X":          ("99PRP040",  "99PRP", "Entry Point X"),
    "ENTRY_Y":          ("99PRP041",  "99PRP", "Entry Point Y"),
    "ENTRY_Z":          ("99PRP042",  "99PRP", "Entry Point Z"),
    "TARGET_X":         ("99PRP043",  "99PRP", "Target Point X"),
    "TARGET_Y":         ("99PRP044",  "99PRP", "Target Point Y"),
    "TARGET_Z":         ("99PRP045",  "99PRP", "Target Point Z"),
    "DEPTH":            ("99PRP046",  "99PRP", "Approach Depth"),
    "ANGLE":            ("99PRP047",  "99PRP", "Approach Angle"),
    # Stent / flow-diverter fields
    "STENT_MODEL":      ("99PRP050",  "99PRP", "Device Model"),
    "STENT_TYPE":       ("99PRP051",  "99PRP", "Device Type"),
    "STENT_DIAM":       ("99PRP052",  "99PRP", "Device Nominal Diameter"),
    "STENT_LENGTH":     ("99PRP053",  "99PRP", "Device Length"),
    "STENT_MFR":        ("99PRP054",  "99PRP", "Device Manufacturer"),
    "STENT_POS_X":      ("99PRP055",  "99PRP", "Device Position X"),
    "STENT_POS_Y":      ("99PRP056",  "99PRP", "Device Position Y"),
    "STENT_POS_Z":      ("99PRP057",  "99PRP", "Device Position Z"),
    # Coil fields
    "COIL_MODEL":       ("99PRP060",  "99PRP", "Coil Model"),
    "COIL_TYPE":        ("99PRP061",  "99PRP", "Coil Type"),
    "COIL_DIAM":        ("99PRP062",  "99PRP", "Coil Nominal Diameter"),
    "COIL_LENGTH":      ("99PRP063",  "99PRP", "Coil Length"),
    "COIL_MFR":         ("99PRP064",  "99PRP", "Coil Manufacturer"),
    "COIL_POS_X":       ("99PRP065",  "99PRP", "Coil Position X"),
    "COIL_POS_Y":       ("99PRP066",  "99PRP", "Coil Position Y"),
    "COIL_POS_Z":       ("99PRP067",  "99PRP", "Coil Position Z"),
}

# ── UCUM unit codes ───────────────────────────────────────────────────────── #
_U = {
    "mm":    ("mm",    "UCUM", "mm"),
    "mm2":   ("mm2",   "UCUM", "mm2"),
    "mm3":   ("mm3",   "UCUM", "mm3"),
    "cm":    ("cm",    "UCUM", "cm"),
    "deg":   ("deg",   "UCUM", "deg"),
    "ratio": ("1",     "UCUM", "no units"),
}


# ──────────────────────────────────────────────────────────────────────────── #
# Low-level dataset helpers                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

def _code_ds(code: str, scheme: str, meaning: str) -> Dataset:
    ds = Dataset()
    ds.CodeValue              = code
    ds.CodingSchemeDesignator = scheme
    ds.CodeMeaning            = meaning
    return ds


def _concept_name(key: str) -> Sequence:
    code, scheme, meaning = _C[key]
    return Sequence([_code_ds(code, scheme, meaning)])


def _num_item(concept_key: str, value: float, unit_key: str,
              relationship: str = "CONTAINS") -> Dataset:
    """Build a NUM content item."""
    item = Dataset()
    item.RelationshipType       = relationship
    item.ValueType              = "NUM"
    item.ConceptNameCodeSequence = _concept_name(concept_key)

    mv = Dataset()
    mv.NumericValue             = f"{value:.4f}"
    ucode, uscheme, umeaning    = _U[unit_key]
    mv.MeasurementUnitsCodeSequence = Sequence([_code_ds(ucode, uscheme, umeaning)])
    item.MeasuredValueSequence  = Sequence([mv])
    return item


def _text_item(concept_key: str, text: str,
               relationship: str = "CONTAINS") -> Dataset:
    item = Dataset()
    item.RelationshipType        = relationship
    item.ValueType               = "TEXT"
    item.ConceptNameCodeSequence = _concept_name(concept_key)
    item.TextValue               = str(text)
    return item


def _code_item(concept_key: str, value_code: str, value_scheme: str,
               value_meaning: str, relationship: str = "CONTAINS") -> Dataset:
    item = Dataset()
    item.RelationshipType        = relationship
    item.ValueType               = "CODE"
    item.ConceptNameCodeSequence = _concept_name(concept_key)
    item.ConceptCodeSequence     = Sequence([_code_ds(value_code, value_scheme, value_meaning)])
    return item


def _container(concept_key: str, children: list[Dataset],
               continuity: str = "SEPARATE",
               relationship: str = "CONTAINS") -> Dataset:
    item = Dataset()
    item.RelationshipType        = relationship
    item.ValueType               = "CONTAINER"
    item.ConceptNameCodeSequence = _concept_name(concept_key)
    item.ContinuityOfContent     = continuity
    item.ContentSequence         = Sequence(children)
    return item


# ──────────────────────────────────────────────────────────────────────────── #
# Risk code mapping                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

_RISK_CODES = {
    "Alto":     ("723509005", "SCT", "High risk"),
    "Moderado": ("723510000", "SCT", "Intermediate risk"),
    "Bajo":     ("723511001", "SCT", "Low risk"),
}


# ──────────────────────────────────────────────────────────────────────────── #
# Main generator                                                                #
# ──────────────────────────────────────────────────────────────────────────── #

class DicomSRGenerator:
    """
    Build a DICOM Comprehensive SR from planning data.

    Parameters
    ----------
    series_meta : dict
        Keys: patient_name, patient_id, study_date, study_description,
              study_instance_uid (optional), series_instance_uid (optional)
    morphometrics : dict
        Keys as in ReportData.morphometrics + 'risk_label'
    clips : list[dict]
        Each dict: index, name, position_mm (3-tuple), orientation_deg (3-tuple),
        is_custom
    trajectory : dict
        Keys: entry (3-list), target (3-list), depth_mm, angle_deg
    """

    def __init__(
        self,
        series_meta: dict[str, Any],
        morphometrics: dict[str, Any],
        clips: list[dict[str, Any]],
        trajectory: dict[str, Any],
        stents: list[dict[str, Any]] | None = None,
        coils: list[dict[str, Any]] | None = None,
    ) -> None:
        self._meta   = series_meta
        self._morpho = morphometrics
        self._clips  = clips
        self._traj   = trajectory
        self._stents = stents or []
        self._coils  = coils  or []

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def generate(self, output_path: str) -> Path:
        p = Path(output_path)
        if p.suffix.lower() != ".dcm":
            p = p.with_suffix(".dcm")

        ds = self._build_dataset()
        pydicom.dcmwrite(str(p), ds)
        logger.info("DICOM SR written: %s", p)
        return p

    # ------------------------------------------------------------------ #
    # Dataset construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_dataset(self) -> FileDataset:
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S.%f")

        sop_instance_uid = generate_uid()

        # ── File meta ─────────────────────────────────────────────────── #
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID    = _SR_COMPREHENSIVE
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID          = _EXPLICIT_VR_LE

        ds = FileDataset(
            filename_or_obj=None,
            dataset={},
            file_meta=file_meta,
            preamble=b"\x00" * 128,
        )

        # ── Patient module ─────────────────────────────────────────────── #
        ds.PatientName   = self._meta.get("patient_name", "ANONIMO")
        ds.PatientID     = self._meta.get("patient_id", "")
        ds.PatientBirthDate = ""
        ds.PatientSex    = ""

        # ── General Study ──────────────────────────────────────────────── #
        ds.StudyInstanceUID  = (self._meta.get("study_instance_uid")
                                or generate_uid())
        raw_date = self._meta.get("study_date", date_str)
        ds.StudyDate         = raw_date.replace("-", "")   # DICOM DA: YYYYMMDD
        ds.StudyTime         = ""
        ds.ReferringPhysicianName = ""
        ds.StudyID           = "1"
        ds.AccessionNumber   = ""
        ds.StudyDescription  = self._meta.get("study_description", "")

        # ── SR Document Series ─────────────────────────────────────────── #
        ds.Modality              = "SR"
        ds.SeriesInstanceUID     = generate_uid()
        ds.SeriesNumber          = "999"
        ds.SeriesDescription     = "PROSPECTIVE Surgical Plan"

        # ── General Equipment ──────────────────────────────────────────── #
        ds.Manufacturer          = "Fundación Universitaria Navarra UNINAVARRA"
        ds.ManufacturerModelName = "PROSPECTIVE"
        ds.SoftwareVersions      = "0.1.0"

        # ── SR Document General ────────────────────────────────────────── #
        ds.SOPClassUID           = _SR_COMPREHENSIVE
        ds.SOPInstanceUID        = sop_instance_uid
        ds.InstanceNumber        = "1"
        ds.ContentDate           = date_str
        ds.ContentTime           = time_str
        ds.CompletionFlag        = "COMPLETE"
        ds.VerificationFlag      = "UNVERIFIED"

        # ── SR Document Content ────────────────────────────────────────── #
        ds.ValueType               = "CONTAINER"
        ds.ConceptNameCodeSequence = _concept_name("REPORT")
        ds.ContinuityOfContent     = "SEPARATE"

        root_items: list[Dataset] = []

        if self._morpho:
            root_items.append(self._build_morphometry_container())
        if self._clips:
            root_items.append(self._build_clip_plan_container())
        if self._stents:
            root_items.append(self._build_stent_plan_container())
        if self._coils:
            root_items.append(self._build_coil_plan_container())
        if self._traj:
            root_items.append(self._build_trajectory_container())

        ds.ContentSequence = Sequence(root_items)

        return ds

    # ------------------------------------------------------------------ #
    # Content containers                                                   #
    # ------------------------------------------------------------------ #

    def _build_morphometry_container(self) -> Dataset:
        m = self._morpho
        items: list[Dataset] = []

        def _add_num(key, field, unit):
            v = m.get(field)
            if v is not None:
                items.append(_num_item(key, float(v), unit))

        _add_num("VOLUME",      "volume_mm3",         "mm3")
        _add_num("AREA",        "surface_area_mm2",   "mm2")
        _add_num("MAX_DIAM",    "max_diameter_mm",    "mm")
        _add_num("EQ_DIAM",     "eq_sphere_diam_mm",  "mm")
        _add_num("NECK_DIAM",   "neck_diameter_mm",   "mm")
        _add_num("DOME_HEIGHT", "dome_height_mm",     "mm")
        _add_num("DNR",         "dome_to_neck_ratio", "ratio")
        _add_num("AR",          "aspect_ratio",       "ratio")
        _add_num("COMPACTNESS", "compactness",        "ratio")

        # Risk code item
        risk = m.get("risk_label", "")
        if risk in _RISK_CODES:
            code, scheme, meaning = _RISK_CODES[risk]
            items.append(_code_item("RISK", code, scheme, meaning))

        return _container("MORPHOMETRY", items)

    def _build_clip_plan_container(self) -> Dataset:
        clip_items: list[Dataset] = []
        for c in self._clips:
            pos = c.get("position_mm", (0, 0, 0))
            ori = c.get("orientation_deg", (0, 0, 0))
            children = [
                _text_item("CLIP_MODEL", c.get("name", "—")),
                _text_item("CLIP_TYPE", "Personalizado" if c.get("is_custom") else "Catálogo"),
                _num_item("POS_X", float(pos[0]), "mm"),
                _num_item("POS_Y", float(pos[1]), "mm"),
                _num_item("POS_Z", float(pos[2]), "mm"),
                _num_item("ROT_X", float(ori[0]), "deg"),
                _num_item("ROT_Y", float(ori[1]), "deg"),
                _num_item("ROT_Z", float(ori[2]), "deg"),
            ]
            clip_items.append(_container("CLIP", children))

        return _container("PLAN", clip_items)

    def _build_trajectory_container(self) -> Dataset:
        tr = self._traj
        entry  = tr.get("entry",  [0, 0, 0])
        target = tr.get("target", [0, 0, 0])
        items  = [
            _num_item("ENTRY_X",  float(entry[0]),              "mm"),
            _num_item("ENTRY_Y",  float(entry[1]),              "mm"),
            _num_item("ENTRY_Z",  float(entry[2]),              "mm"),
            _num_item("TARGET_X", float(target[0]),             "mm"),
            _num_item("TARGET_Y", float(target[1]),             "mm"),
            _num_item("TARGET_Z", float(target[2]),             "mm"),
            _num_item("DEPTH",    float(tr.get("depth_mm", 0)), "mm"),
            _num_item("ANGLE",    float(tr.get("angle_deg", 0)),"deg"),
        ]
        return _container("TRAJECTORY", items)

    def _build_stent_plan_container(self) -> Dataset:
        """One CONTAINER per stent/flow-diverter in the plan."""
        stent_items: list[Dataset] = []
        for s in self._stents:
            pos = s.get("position_mm", (0, 0, 0))
            children: list[Dataset] = [
                _text_item("STENT_MODEL",  s.get("name", "—")),
                _text_item("STENT_TYPE",   s.get("stent_type", "—")),
                _text_item("STENT_MFR",    s.get("manufacturer", "—")),
            ]
            if s.get("diameter_mm") is not None:
                children.append(_num_item("STENT_DIAM",   float(s["diameter_mm"]), "mm"))
            if s.get("length_mm") is not None:
                children.append(_num_item("STENT_LENGTH", float(s["length_mm"]),   "mm"))
            children += [
                _num_item("STENT_POS_X", float(pos[0]), "mm"),
                _num_item("STENT_POS_Y", float(pos[1]), "mm"),
                _num_item("STENT_POS_Z", float(pos[2]), "mm"),
            ]
            stent_items.append(_container("STENT", children))
        return _container("STENT_PLAN", stent_items)

    def _build_coil_plan_container(self) -> Dataset:
        """One CONTAINER per embolization coil in the plan."""
        coil_items: list[Dataset] = []
        for c in self._coils:
            pos = c.get("position_mm", (0, 0, 0))
            children: list[Dataset] = [
                _text_item("COIL_MODEL", c.get("name", "—")),
                _text_item("COIL_TYPE",  c.get("coil_type", "—")),
                _text_item("COIL_MFR",   c.get("manufacturer", "—")),
            ]
            if c.get("diameter_mm") is not None:
                children.append(_num_item("COIL_DIAM",   float(c["diameter_mm"]), "mm"))
            if c.get("length_cm") is not None:
                children.append(_num_item("COIL_LENGTH", float(c["length_cm"]),   "cm"))
            children += [
                _num_item("COIL_POS_X", float(pos[0]), "mm"),
                _num_item("COIL_POS_Y", float(pos[1]), "mm"),
                _num_item("COIL_POS_Z", float(pos[2]), "mm"),
            ]
            coil_items.append(_container("COIL", children))
        return _container("COIL_PLAN", coil_items)
