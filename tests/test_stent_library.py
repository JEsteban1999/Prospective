"""Unit tests — stent / flow-diverter library (A-03-07)."""
from __future__ import annotations

import pytest

from prospective.models.stent_library import (
    STENT_CATALOGUE,
    StentSpec,
    StentType,
    stents_for_vessel,
)


# ══════════════════════════════════════════════════════════════════════════════
# StentSpec
# ══════════════════════════════════════════════════════════════════════════════


class TestStentSpec:

    def test_frozen_dataclass_immutable(self):
        spec = STENT_CATALOGUE[0]
        with pytest.raises((AttributeError, TypeError)):
            spec.diameter_mm = 99.9  # type: ignore[misc]

    def test_display_label_contains_diameter_and_length(self):
        spec = StentSpec(
            name="TestStent 3.0×20",
            stent_type=StentType.FLOW_DIVERTER,
            diameter_mm=3.0,
            length_mm=20.0,
            porosity_pct=30,
            wire_diameter_um=32,
            n_wires=48,
            manufacturer="Acme",
            compatible_wire='0.027"',
            catheter_id_fr=2.8,
        )
        label = spec.display_label
        assert "3.0" in label
        assert "20" in label
        assert "TestStent" in label

    def test_info_text_contains_required_fields(self):
        spec = STENT_CATALOGUE[0]
        info = spec.info_text
        assert "Tipo" in info
        assert "Diámetro" in info
        assert "Porosidad" in info
        assert "Catéter" in info

    def test_stent_type_enum_values(self):
        assert StentType.FLOW_DIVERTER.value == "Desviador de flujo"
        assert StentType.INTRACRANIAL.value == "Stent intracraneal"


# ══════════════════════════════════════════════════════════════════════════════
# STENT_CATALOGUE contents
# ══════════════════════════════════════════════════════════════════════════════


class TestStentCatalogue:

    def test_catalogue_has_expected_count(self):
        # 6 Pipeline + 4 Surpass + 4 FRED + 4 Neuroform + 3 Enterprise + 3 Leo+
        # + 6 LVIS Jr + 6 LVIS = 36
        assert len(STENT_CATALOGUE) >= 36

    def test_all_entries_are_stent_spec(self):
        for spec in STENT_CATALOGUE:
            assert isinstance(spec, StentSpec)

    def test_all_diameters_positive(self):
        for spec in STENT_CATALOGUE:
            assert spec.diameter_mm > 0, f"{spec.name}: diameter <= 0"

    def test_all_lengths_positive(self):
        for spec in STENT_CATALOGUE:
            assert spec.length_mm > 0, f"{spec.name}: length <= 0"

    def test_all_porosities_in_range(self):
        for spec in STENT_CATALOGUE:
            assert 0 < spec.porosity_pct <= 100, (
                f"{spec.name}: porosity {spec.porosity_pct} out of range"
            )

    def test_flow_diverters_low_porosity(self):
        """Flow diverters must have porosity < 50 % (design requirement)."""
        fds = [s for s in STENT_CATALOGUE if s.stent_type == StentType.FLOW_DIVERTER]
        assert len(fds) > 0
        for spec in fds:
            assert spec.porosity_pct < 50, (
                f"{spec.name}: flow diverter porosity {spec.porosity_pct} >= 50 %"
            )

    def test_ic_stents_high_porosity(self):
        """Intracranial stents must have porosity > 60 % (open-cell design)."""
        ics = [s for s in STENT_CATALOGUE if s.stent_type == StentType.INTRACRANIAL]
        assert len(ics) > 0
        for spec in ics:
            assert spec.porosity_pct > 60, (
                f"{spec.name}: IC stent porosity {spec.porosity_pct} <= 60 %"
            )

    def test_all_n_wires_even(self):
        """Braided stents have paired wire counts (CW + CCW families)."""
        for spec in STENT_CATALOGUE:
            assert spec.n_wires % 2 == 0, f"{spec.name}: odd n_wires {spec.n_wires}"

    def test_all_catheter_fr_positive(self):
        for spec in STENT_CATALOGUE:
            assert spec.catheter_id_fr > 0, f"{spec.name}: catheter_id_fr <= 0"

    def test_pipeline_manufacturer_is_medtronic(self):
        pipeline = [s for s in STENT_CATALOGUE if s.name.startswith("Pipeline")]
        assert len(pipeline) == 6
        for spec in pipeline:
            assert spec.manufacturer == "Medtronic"

    def test_surpass_manufacturer_is_stryker(self):
        surpass = [s for s in STENT_CATALOGUE if s.name.startswith("Surpass")]
        assert len(surpass) == 4
        for spec in surpass:
            assert spec.manufacturer == "Stryker"

    def test_fred_manufacturer_is_microvention(self):
        fred = [s for s in STENT_CATALOGUE if s.name.startswith("FRED")]
        assert len(fred) == 4
        for spec in fred:
            assert spec.manufacturer == "MicroVention"

    def test_all_names_unique(self):
        names = [s.name for s in STENT_CATALOGUE]
        assert len(names) == len(set(names)), "Duplicate stent names in catalogue"

    def test_wire_diameters_in_realistic_range(self):
        """Wire diameters must be between 10 µm and 200 µm."""
        for spec in STENT_CATALOGUE:
            assert 10 <= spec.wire_diameter_um <= 200, (
                f"{spec.name}: wire_diameter_um {spec.wire_diameter_um} out of range"
            )


# ══════════════════════════════════════════════════════════════════════════════
# stents_for_vessel — filtering
# ══════════════════════════════════════════════════════════════════════════════


class TestStentsForVessel:

    def test_returns_list(self):
        result = stents_for_vessel(3.0)
        assert isinstance(result, list)

    def test_diameter_3mm_returns_at_least_one(self):
        result = stents_for_vessel(3.0)
        assert len(result) > 0

    def test_diameter_3mm_within_tolerance(self):
        """All returned stents must have diameter in [0.85×, 1.25×] range."""
        vessel_d = 3.0
        lo, hi   = vessel_d * 0.85, vessel_d * 1.25
        for spec in stents_for_vessel(vessel_d):
            assert lo <= spec.diameter_mm <= hi, (
                f"{spec.name}: diameter {spec.diameter_mm} outside [{lo:.2f}, {hi:.2f}]"
            )

    def test_very_small_vessel_no_stents(self):
        """No stent in catalogue has diameter < 2.5 mm; tiny vessels yield none."""
        result = stents_for_vessel(0.5)
        assert result == []

    def test_very_large_vessel_no_stents(self):
        """No stent > 5 mm; very large vessels yield no matches."""
        result = stents_for_vessel(15.0)
        assert result == []

    def test_diameter_4mm_returns_multiple(self):
        result = stents_for_vessel(4.0)
        assert len(result) >= 3

    def test_filter_is_subset_of_catalogue(self):
        result = stents_for_vessel(3.5)
        for spec in result:
            assert spec in STENT_CATALOGUE

    def test_different_diameters_give_different_results(self):
        r1 = set(s.name for s in stents_for_vessel(2.5))
        r2 = set(s.name for s in stents_for_vessel(5.0))
        # Not identical (extreme sizes have different candidates)
        assert r1 != r2

    def test_returns_stent_spec_instances(self):
        for spec in stents_for_vessel(3.5):
            assert isinstance(spec, StentSpec)

    @pytest.mark.parametrize("d", [2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    def test_valid_vessel_diameter_always_finds_match(self, d: float):
        """Every nominal catalogue diameter should match itself."""
        result = stents_for_vessel(d)
        assert any(abs(s.diameter_mm - d) < 1e-6 for s in result), (
            f"Diameter {d} mm: expected at least one exact match in results"
        )
