"""Longitudinal aneurysm comparison — Feature 6.

Stores a time-ordered list of :class:`MorphometricSnapshot` records (one per
imaging session) for a single patient and computes inter-session deltas and
a linear trend for each metric.

Data model
----------
``MorphometricSnapshot``
    A thin, serialisable record that captures the key morphometric values from
    one session together with a date and an optional label (e.g. "Baseline",
    "6-month follow-up").

``LongitudinalSeries``
    Ordered collection of snapshots.  Provides:

    * ``add_snapshot()``     — append a new time-point
    * ``remove_snapshot()``  — delete by index
    * ``deltas()``           — consecutive differences for each metric
    * ``trend()``            — linear regression slope (units / day) per metric
    * ``to_dataframe()``     — returns a dict-of-lists suitable for display
    * ``export_csv()``       — write to a file path
    * ``from_morpho_result()`` — convenience constructor from MorphometricResult
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Metrics tracked in the longitudinal comparison
TRACKED_METRICS: list[str] = [
    "volume_mm3",
    "max_diameter_mm",
    "neck_diameter_mm",
    "dome_height_mm",
    "aspect_ratio",
    "dome_to_neck_ratio",
    "bottleneck_factor",
    "undulation_index",
    "ellipticity_index",
    "size_ratio",
]

# Human-readable labels for the UI / export
METRIC_LABELS: dict[str, str] = {
    "volume_mm3":          "Volumen (mm³)",
    "max_diameter_mm":     "Ø máximo (mm)",
    "neck_diameter_mm":    "Ø cuello (mm)",
    "dome_height_mm":      "Altura domo (mm)",
    "aspect_ratio":        "Aspect Ratio",
    "dome_to_neck_ratio":  "Relación domo/cuello",
    "bottleneck_factor":   "Bottleneck Factor",
    "undulation_index":    "Undulation Index",
    "ellipticity_index":   "Ellipticity Index",
    "size_ratio":          "Size Ratio",
}

# ──────────────────────────────────────────────────────────────────────────── #
# Snapshot dataclass                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class MorphometricSnapshot:
    """One time-point of morphometric measurements."""

    session_date:      date
    label:             str   = ""

    # Core morphometric values (subset of MorphometricResult)
    volume_mm3:        float = 0.0
    max_diameter_mm:   float = 0.0
    neck_diameter_mm:  float = 0.0
    dome_height_mm:    float = 0.0
    aspect_ratio:      float = 0.0
    dome_to_neck_ratio:float = 0.0
    bottleneck_factor: float = 0.0
    undulation_index:  float = 0.0
    ellipticity_index: float = 0.0
    size_ratio:        float = 0.0
    rupture_risk:      str   = "—"

    @classmethod
    def from_morpho_result(
        cls,
        result,
        session_date: date | None = None,
        label: str = "",
    ) -> "MorphometricSnapshot":
        """
        Build a snapshot from a :class:`~prospective.processing.morphometrics.MorphometricResult`.

        Parameters
        ----------
        result      : MorphometricResult
        session_date: date (defaults to today)
        label       : free-text session label
        """
        d = session_date or date.today()
        return cls(
            session_date       = d,
            label              = label,
            volume_mm3         = result.volume_mm3,
            max_diameter_mm    = result.max_diameter_mm,
            neck_diameter_mm   = result.neck_diameter_mm,
            dome_height_mm     = result.dome_height_mm,
            aspect_ratio       = result.aspect_ratio,
            dome_to_neck_ratio = result.dome_to_neck_ratio,
            bottleneck_factor  = result.bottleneck_factor,
            undulation_index   = result.undulation_index,
            ellipticity_index  = result.ellipticity_index,
            size_ratio         = result.size_ratio,
            rupture_risk       = result.rupture_risk_label,
        )

    def get_metric(self, key: str) -> float:
        """Return the value of a tracked metric by key name."""
        return float(getattr(self, key, 0.0))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["session_date"] = self.session_date.isoformat()
        return d


# ──────────────────────────────────────────────────────────────────────────── #
# Series                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class LongitudinalSeries:
    """
    Ordered series of morphometric snapshots for one patient.

    Snapshots are always kept sorted by ``session_date`` (ascending).
    """

    patient_id:  str                     = ""
    snapshots:   list[MorphometricSnapshot] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Mutation API                                                          #
    # ------------------------------------------------------------------ #

    def add_snapshot(self, snap: MorphometricSnapshot) -> None:
        """Insert *snap* in chronological order."""
        self.snapshots.append(snap)
        self.snapshots.sort(key=lambda s: s.session_date)
        logger.info(
            "Longitudinal: added snapshot '%s' (%s) — total %d",
            snap.label, snap.session_date.isoformat(), len(self.snapshots),
        )

    def remove_snapshot(self, index: int) -> None:
        """Remove snapshot at position *index* (0-based)."""
        if 0 <= index < len(self.snapshots):
            removed = self.snapshots.pop(index)
            logger.info("Longitudinal: removed snapshot '%s'", removed.label)

    def clear(self) -> None:
        self.snapshots.clear()

    # ------------------------------------------------------------------ #
    # Query API                                                             #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.snapshots)

    def dates(self) -> list[date]:
        return [s.session_date for s in self.snapshots]

    def days_from_first(self) -> list[float]:
        """Return elapsed days from the first session for each snapshot."""
        if not self.snapshots:
            return []
        t0 = self.snapshots[0].session_date
        return [float((s.session_date - t0).days) for s in self.snapshots]

    def values(self, metric: str) -> list[float]:
        """Return the values of *metric* for all snapshots."""
        return [s.get_metric(metric) for s in self.snapshots]

    def deltas(self, metric: str) -> list[float]:
        """
        Consecutive differences for *metric*: snap[i] − snap[i−1].

        Returns a list of length ``len(snapshots) − 1``.
        """
        vals = self.values(metric)
        return [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    def percent_change(self, metric: str) -> list[float]:
        """
        Percentage change relative to the *first* snapshot for each subsequent
        one: (snap[i] − snap[0]) / |snap[0]| × 100 %.

        Returns a list of length ``len(snapshots) − 1``.
        """
        vals = self.values(metric)
        if not vals or abs(vals[0]) < 1e-9:
            return [0.0] * (len(vals) - 1)
        base = vals[0]
        return [(v - base) / abs(base) * 100.0 for v in vals[1:]]

    def trend(self, metric: str) -> tuple[float, float]:
        """
        Linear regression of *metric* vs elapsed days.

        Returns ``(slope, r_squared)`` where *slope* is in units/day.
        Returns ``(0.0, 0.0)`` if fewer than 2 snapshots exist.
        """
        days = self.days_from_first()
        vals = self.values(metric)
        n    = len(days)
        if n < 2:
            return 0.0, 0.0

        x = np.array(days, dtype=float)
        y = np.array(vals, dtype=float)

        # Degenerate: all x the same
        if x.std() < 1e-9:
            return 0.0, 0.0

        slope, intercept = np.polyfit(x, y, 1)
        y_pred  = slope * x + intercept
        ss_res  = float(np.sum((y - y_pred) ** 2))
        ss_tot  = float(np.sum((y - y.mean()) ** 2))
        r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return float(slope), float(max(0.0, min(1.0, r2)))

    def summary_table(self) -> list[dict[str, Any]]:
        """
        Return a list of row-dicts for display in a QTableWidget.

        Each row: metric label, values per session, last delta, trend slope.
        """
        rows = []
        for key in TRACKED_METRICS:
            vals     = self.values(key)
            delts    = self.deltas(key)
            slope, _ = self.trend(key)
            row: dict[str, Any] = {
                "metric":     METRIC_LABELS.get(key, key),
                "values":     [round(v, 3) for v in vals],
                "last_delta": round(delts[-1], 3) if delts else None,
                "trend_slope": round(slope, 4),
            }
            rows.append(row)
        return rows

    def to_dataframe_dict(self) -> dict[str, list]:
        """Dict-of-lists suitable for pandas / display."""
        result: dict[str, list] = {
            "Fecha":   [s.session_date.isoformat() for s in self.snapshots],
            "Etiqueta":[s.label for s in self.snapshots],
            "Riesgo":  [s.rupture_risk for s in self.snapshots],
        }
        for key, label in METRIC_LABELS.items():
            result[label] = [round(s.get_metric(key), 3) for s in self.snapshots]
        return result

    # ------------------------------------------------------------------ #
    # Import / export                                                       #
    # ------------------------------------------------------------------ #

    def export_csv(self, path: str) -> None:
        """Write the series to a CSV file at *path*."""
        data = self.to_dataframe_dict()
        if not data:
            return
        headers = list(data.keys())
        rows    = list(zip(*data.values()))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for row in rows:
                w.writerow(row)
        logger.info("Longitudinal: exported %d snapshots to %s", len(self.snapshots), path)

    @classmethod
    def from_csv(cls, path: str, patient_id: str = "") -> "LongitudinalSeries":
        """
        Load a series from a CSV file previously written by :meth:`export_csv`.

        Returns an empty series if the file cannot be parsed.
        """
        series = cls(patient_id=patient_id)
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = date.fromisoformat(row.get("Fecha", date.today().isoformat()))
                    snap = MorphometricSnapshot(session_date=d, label=row.get("Etiqueta", ""))
                    inv_labels = {v: k for k, v in METRIC_LABELS.items()}
                    for csv_col, key in inv_labels.items():
                        try:
                            setattr(snap, key, float(row.get(csv_col, 0.0) or 0.0))
                        except (ValueError, TypeError):
                            pass
                    snap.rupture_risk = row.get("Riesgo", "—")
                    series.snapshots.append(snap)
        except Exception as exc:
            logger.warning("Longitudinal: failed to load CSV %s — %s", path, exc)
        return series
