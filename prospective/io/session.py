"""Session save / load — serialise and restore application state to JSON.

A .prospective file is a UTF-8 JSON document with the following top-level keys:
  version, created, dicom_path,
  window_level, rendering_3d,
  segmentation, aneurysm_detection,
  clips, trajectory, notes

Custom STL/OBJ clips are re-loaded from their original file path.
If the file is missing on load a warning is issued but loading continues.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSION_VERSION = "1.6"   # bumped: clinical_state (PHASES + treatment decision) and print_prep_state added
SESSION_EXT     = ".prospective"


# ──────────────────────────────────────────────────────────────────────────── #
# Data model                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class ClipState:
    index: int
    name: str
    is_custom: bool
    custom_path: str | None          # absolute path for custom STL/OBJ
    position: list[float]            # [x, y, z]
    orientation: list[float]         # [yaw, pitch, roll] degrees


@dataclass
class CoilState:
    index: int
    name: str
    is_custom: bool
    custom_path: str | None          # absolute path for custom STL/OBJ
    position: list[float]            # [x, y, z] — deposit point inside sac


@dataclass
class StentState:
    index: int
    name: str
    is_custom: bool
    custom_path: str | None          # absolute path for custom STL/OBJ
    position: list[float]            # [x, y, z]
    orientation: list[float]         # [rz, rx, ry] degrees (matches VTK GetOrientation order)


@dataclass
class SessionData:
    # ── File info ──────────────────────────────────────────────────────── #
    version: str = SESSION_VERSION
    created: str = ""
    notes: str   = ""

    # ── Data source ────────────────────────────────────────────────────── #
    dicom_path: str = ""
    study_id:   int = 0    # DB Study.id; 0 = not linked to a study record

    # ── Window / level ─────────────────────────────────────────────────── #
    wl_center: float  = 170.0
    wl_width: float   = 600.0
    wl_preset: str    = "CTA"

    # ── 3D rendering ───────────────────────────────────────────────────── #
    preset_3d: str    = "CTA"
    blend_mode: str   = "VR (Ray Casting)"
    hu_min: float     = -100.0
    hu_max: float     = 1500.0

    # ── Segmentation ───────────────────────────────────────────────────── #
    # Full dict returned by SegmentationPanel.get_session_state() — includes
    # threshold_hu, adv_sigma, adv_smooth, adv_dec, limpieza, suavizado, etc.
    # (Replaces the four scalar fields removed in version 1.2)
    seg_state: dict = field(default_factory=dict)
    # Path to a companion .vtp file that stores the segmented vessel mesh.
    # Empty string means no mesh was saved with this session.
    seg_mesh_path: str = ""

    # ── Aneurysm detection ─────────────────────────────────────────────── #
    # Full dict returned by AneurysmPanel.get_session_state() — mirrors the
    # pattern used by seg_state above.  All 8 panel params are persisted
    # (percentile, gauss_percentile, min_r_mm, max_r_mm, min_pts,
    # min_pos_gauss_frac, min_sphericity, pre_smooth_iters).
    # Backward-compat: old sessions only have 4 keys; the panel's
    # restore_session_state fills the rest with sensible defaults.
    det_state: dict = field(default_factory=dict)

    # ── Placed clips ───────────────────────────────────────────────────── #
    clips: list[ClipState] = field(default_factory=list)

    # ── Placed coils ───────────────────────────────────────────────────── #
    coils: list[CoilState] = field(default_factory=list)

    # ── Placed stents / flow diverters ─────────────────────────────────── #
    stents: list[StentState] = field(default_factory=list)

    # ── Approach trajectory ────────────────────────────────────────────── #
    traj_entry: list[float]  = field(default_factory=lambda: [0.0, 0.0, 0.0])
    traj_target: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    traj_visible: bool       = False

    # ── Morphometrics (read-only snapshot) ────────────────────────────── #
    morpho_snapshot: dict[str, Any] = field(default_factory=dict)

    # ── Perforator risk panel settings ────────────────────────────────── #
    perf_r_high: float = 3.0   # mm — high-risk distance threshold
    perf_r_med:  float = 5.0   # mm — medium-risk distance threshold
    perf_r_low:  float = 8.0   # mm — low-risk distance threshold

    # ── Workflow navigation state ──────────────────────────────────────── #
    current_step: int = 0   # index of the active step in WorkflowStepper

    # ── Patient / report data ──────────────────────────────────────────── #
    report_patient_name: str  = ""
    report_patient_id: str    = ""
    report_patient_dob: str   = ""
    report_surgeon: str       = ""
    report_institution: str   = "Fundación Universitaria Navarra UNINAVARRA"
    report_notes: str         = ""
    # Free-text treatment recommendation written by the surgeon
    report_treatment: str     = ""
    # Full narrative report text (anamnesis, findings, plan…)
    report_text: str          = ""

    # ── Clinical context (PHASES score + treatment decision) ───────────── #
    # Dict with keys:
    #   "phases"      → {pop_idx, htn, age, sah, site_idx}
    #   "parent_diam" → float (parent artery diameter in mm, 0 = unknown)
    #   "treatment"   → {location, ruptured}
    clinical_state: dict = field(default_factory=dict)

    # ── 3D-print preparation parameters ───────────────────────────────── #
    # Dict with keys: size, smooth, relax, fill, hole, sub, bed
    print_prep_state: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        d = {
            "version":   self.version,
            "created":   self.created or datetime.now().isoformat(timespec="seconds"),
            "notes":     self.notes,
            "dicom_path": self.dicom_path,
            "study_id":   self.study_id,
            "window_level": {
                "center": self.wl_center,
                "width":  self.wl_width,
                "preset": self.wl_preset,
            },
            "rendering_3d": {
                "preset":     self.preset_3d,
                "blend_mode": self.blend_mode,
                "hu_min":     self.hu_min,
                "hu_max":     self.hu_max,
            },
            "segmentation":  self.seg_state,
            "seg_mesh_path": self.seg_mesh_path,
            "aneurysm_detection": self.det_state,
            "clips": [
                {
                    "index":       c.index,
                    "name":        c.name,
                    "is_custom":   c.is_custom,
                    "custom_path": c.custom_path,
                    "position":    c.position,
                    "orientation": c.orientation,
                }
                for c in self.clips
            ],
            "coils": [
                {
                    "index":       c.index,
                    "name":        c.name,
                    "is_custom":   c.is_custom,
                    "custom_path": c.custom_path,
                    "position":    c.position,
                }
                for c in self.coils
            ],
            "stents": [
                {
                    "index":       s.index,
                    "name":        s.name,
                    "is_custom":   s.is_custom,
                    "custom_path": s.custom_path,
                    "position":    s.position,
                    "orientation": s.orientation,
                }
                for s in self.stents
            ],
            "trajectory": {
                "entry":   self.traj_entry,
                "target":  self.traj_target,
                "visible": self.traj_visible,
            },
            "morphometrics": self.morpho_snapshot,
            "perforator_risk": {
                "r_high": self.perf_r_high,
                "r_med":  self.perf_r_med,
                "r_low":  self.perf_r_low,
            },
            "current_step": self.current_step,
            "report": {
                "patient_name": self.report_patient_name,
                "patient_id":   self.report_patient_id,
                "patient_dob":  self.report_patient_dob,
                "surgeon":      self.report_surgeon,
                "institution":  self.report_institution,
                "notes":        self.report_notes,
                "treatment":    self.report_treatment,
                "text":         self.report_text,
            },
            "clinical_state":   self.clinical_state,
            "print_prep_state": self.print_prep_state,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionData":
        wl   = d.get("window_level", {})
        r3d  = d.get("rendering_3d", {})
        det  = d.get("aneurysm_detection", {})

        # ── Segmentation — backward-compat: v1.1 stored 4 scalar keys ── #
        _seg_raw = d.get("segmentation", {})
        if _seg_raw and "gaussian_sigma" in _seg_raw and "adv_sigma" not in _seg_raw:
            # Upgrade old format: map scalar names → panel's adv_* keys
            seg = {
                "threshold_hu": _seg_raw.get("threshold_hu", 200.0),
                "adv_sigma":    _seg_raw.get("gaussian_sigma", 0.8),
                "adv_smooth":   int(_seg_raw.get("smooth_iter", 15)),
                "adv_dec":      int(float(_seg_raw.get("decimation", 0.70)) * 100),
            }
        else:
            seg = _seg_raw
        tr   = d.get("trajectory", {})
        perf = d.get("perforator_risk", {})

        clips = []
        for c in d.get("clips", []):
            clips.append(ClipState(
                index       = c.get("index", 0),
                name        = c.get("name", ""),
                is_custom   = c.get("is_custom", False),
                custom_path = c.get("custom_path"),
                position    = c.get("position", [0, 0, 0]),
                orientation = c.get("orientation", [0, 0, 0]),
            ))

        coils = []
        for c in d.get("coils", []):
            coils.append(CoilState(
                index       = c.get("index", 0),
                name        = c.get("name", ""),
                is_custom   = c.get("is_custom", False),
                custom_path = c.get("custom_path"),
                position    = c.get("position", [0, 0, 0]),
            ))

        stents = []
        for s in d.get("stents", []):
            stents.append(StentState(
                index       = s.get("index", 0),
                name        = s.get("name", ""),
                is_custom   = s.get("is_custom", False),
                custom_path = s.get("custom_path"),
                position    = s.get("position", [0, 0, 0]),
                orientation = s.get("orientation", [0, 0, 0]),
            ))

        return cls(
            version           = d.get("version", SESSION_VERSION),
            created           = d.get("created", ""),
            notes             = d.get("notes", ""),
            dicom_path        = d.get("dicom_path", ""),
            study_id          = int(d.get("study_id", 0) or 0),
            wl_center         = wl.get("center", 170.0),
            wl_width          = wl.get("width", 600.0),
            wl_preset         = wl.get("preset", "CTA"),
            preset_3d         = r3d.get("preset", "CTA"),
            blend_mode        = r3d.get("blend_mode", "VR (Ray Casting)"),
            hu_min            = r3d.get("hu_min", -100.0),
            hu_max            = r3d.get("hu_max", 1500.0),
            seg_state         = seg,
            seg_mesh_path     = d.get("seg_mesh_path", ""),
            det_state         = det,
            clips             = clips,
            coils             = coils,
            stents            = stents,
            traj_entry        = tr.get("entry", [0, 0, 0]),
            traj_target       = tr.get("target", [0, 0, 0]),
            traj_visible      = tr.get("visible", False),
            morpho_snapshot   = d.get("morphometrics", {}),
            perf_r_high       = float(perf.get("r_high", 3.0)),
            perf_r_med        = float(perf.get("r_med",  5.0)),
            perf_r_low        = float(perf.get("r_low",  8.0)),
            current_step      = int(d.get("current_step", 0)),
            report_patient_name = d.get("report", {}).get("patient_name", ""),
            report_patient_id   = d.get("report", {}).get("patient_id", ""),
            report_patient_dob  = d.get("report", {}).get("patient_dob", ""),
            report_surgeon      = d.get("report", {}).get("surgeon", ""),
            report_institution  = d.get("report", {}).get("institution", "Fundación Universitaria Navarra UNINAVARRA"),
            report_notes        = d.get("report", {}).get("notes", ""),
            report_treatment    = d.get("report", {}).get("treatment", ""),
            report_text         = d.get("report", {}).get("text", ""),
            clinical_state      = d.get("clinical_state", {}),
            print_prep_state    = d.get("print_prep_state", {}),
        )


# ──────────────────────────────────────────────────────────────────────────── #
# File I/O                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

def save_session(data: SessionData, path: str) -> None:
    """Write *data* as a JSON file at *path* (adds .prospective if missing)."""
    p = Path(path)
    if p.suffix.lower() != SESSION_EXT:
        p = p.with_suffix(SESSION_EXT)
    p.write_text(
        json.dumps(data.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Session saved: %s", p)


def load_session(path: str) -> SessionData:
    """Read a JSON session file and return a SessionData."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    data = SessionData.from_dict(raw)
    logger.info("Session loaded: %s  (created %s)", p, data.created)
    return data
