"""Unit tests — embolization coil library (MOD-04 / A-03-07)."""
from __future__ import annotations

import math

import pytest

from prospective.models.coil_library import (
    COIL_CATALOGUE,
    CoilSpec,
    CoilType,
    coils_for_aneurysm,
    estimate_coil_count,
)


# ══════════════════════════════════════════════════════════════════════════════
# CoilSpec dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestCoilSpec:

    def test_frozen_dataclass_immutable(self):
        spec = COIL_CATALOGUE[0]
        with pytest.raises((AttributeError, TypeError)):
            spec.diameter_mm = 99.9  # type: ignore[misc]

    def test_display_label_contains_diameter_and_length(self):
        spec = CoilSpec(
            name="Test 5mm×12cm",
            coil_type=CoilType.FILLING,
            diameter_mm=5,
            length_cm=12,
            shape_3d="Helicoidal",
            wire_diameter_um=254,
            manufacturer="Acme",
            compatible_wire='0.014"',
            catheter_id_fr=1.7,
        )
        label = spec.display_label
        assert "5" in label
        assert "12" in label
        assert "Test" in label

    def test_info_text_contains_required_fields(self):
        spec = COIL_CATALOGUE[0]
        info = spec.info_text
        assert "Tipo" in info
        assert "Diámetro" in info
        assert "Longitud" in info
        assert "Catéter" in info

    def test_wire_volume_positive(self):
        spec = COIL_CATALOGUE[0]
        assert spec.wire_volume_mm3 > 0.0

    def test_wire_volume_formula(self):
        """wire_volume = π·r² · length_mm where r = wire_diameter_um/2 in mm."""
        spec = CoilSpec(
            name="Test",
            coil_type=CoilType.FILLING,
            diameter_mm=5,
            length_cm=10,
            shape_3d="Helicoidal",
            wire_diameter_um=200,   # 0.2 mm diameter → r = 0.1 mm
            manufacturer="X",
            compatible_wire='0.014"',
            catheter_id_fr=1.7,
        )
        r_mm = 0.1
        expected = math.pi * r_mm**2 * 100.0   # 100 mm length
        assert abs(spec.wire_volume_mm3 - expected) < 1e-9

    def test_coil_type_enum_values(self):
        assert CoilType.FRAMING.value   == "Enmarcado"
        assert CoilType.FILLING.value   == "Relleno"
        assert CoilType.FINISHING.value == "Acabado"
        assert CoilType.HYDROCOIL.value == "Hidrocoil"
        assert CoilType.COMPLEX_3D.value == "Complejo 3D"


# ══════════════════════════════════════════════════════════════════════════════
# COIL_CATALOGUE integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestCoilCatalogue:

    def test_catalogue_not_empty(self):
        assert len(COIL_CATALOGUE) >= 20

    def test_all_entries_are_coil_spec(self):
        for c in COIL_CATALOGUE:
            assert isinstance(c, CoilSpec)

    def test_all_names_unique(self):
        names = [c.name for c in COIL_CATALOGUE]
        assert len(names) == len(set(names)), "Duplicate coil names found"

    def test_all_diameters_positive(self):
        for c in COIL_CATALOGUE:
            assert c.diameter_mm > 0, f"{c.name}: diameter_mm must be > 0"

    def test_all_lengths_positive(self):
        for c in COIL_CATALOGUE:
            assert c.length_cm > 0, f"{c.name}: length_cm must be > 0"

    def test_all_wire_diameters_positive(self):
        for c in COIL_CATALOGUE:
            assert c.wire_diameter_um > 0, f"{c.name}: wire_diameter_um must be > 0"

    def test_all_catheter_ids_positive(self):
        for c in COIL_CATALOGUE:
            assert c.catheter_id_fr > 0, f"{c.name}: catheter_id_fr must be > 0"

    def test_manufacturers_present(self):
        manufacturers = {c.manufacturer for c in COIL_CATALOGUE}
        assert "Stryker"       in manufacturers
        assert "Penumbra"      in manufacturers
        assert "MicroVention"  in manufacturers
        assert "Medtronic"     in manufacturers

    def test_all_coil_types_represented(self):
        types_in_cat = {c.coil_type for c in COIL_CATALOGUE}
        assert CoilType.FRAMING   in types_in_cat
        assert CoilType.FILLING   in types_in_cat
        assert CoilType.FINISHING in types_in_cat
        assert CoilType.HYDROCOIL in types_in_cat

    def test_target_coils_present(self):
        names = [c.name for c in COIL_CATALOGUE]
        assert any("Target 360°" in n for n in names)
        assert any("Target Ultra" in n for n in names)
        assert any("Target Nano"  in n for n in names)

    def test_ruby_coils_present(self):
        names = [c.name for c in COIL_CATALOGUE]
        assert any("Ruby" in n for n in names)

    def test_hydrocoil_present(self):
        names = [c.name for c in COIL_CATALOGUE]
        assert any("HydroCoil" in n for n in names)

    def test_axium_present(self):
        names = [c.name for c in COIL_CATALOGUE]
        assert any("Axium" in n for n in names)

    def test_microplex_present(self):
        names = [c.name for c in COIL_CATALOGUE]
        assert any("MicroPlex" in n for n in names)

    def test_compatible_wire_format(self):
        """Wire sizes should be '0.014\"' or '0.010\"' or '0.027\"'."""
        valid = {'0.014"', '0.010"', '0.027"'}
        for c in COIL_CATALOGUE:
            assert c.compatible_wire in valid, (
                f"{c.name}: unexpected wire size {c.compatible_wire!r}"
            )

    def test_framing_coils_larger_diameter(self):
        """Framing coils tend to be ≥ 4 mm."""
        framing = [c for c in COIL_CATALOGUE if c.coil_type == CoilType.FRAMING]
        assert all(c.diameter_mm >= 4 for c in framing), \
            "Some framing coils have diameter < 4 mm"

    def test_finishing_coils_smaller_wire(self):
        """Finishing (nano) coils use thinner wire (≤ 200 µm)."""
        finishing = [c for c in COIL_CATALOGUE if c.coil_type == CoilType.FINISHING]
        assert all(c.wire_diameter_um <= 200 for c in finishing), \
            "Some finishing coils have wire diameter > 200 µm"


# ══════════════════════════════════════════════════════════════════════════════
# coils_for_aneurysm()
# ══════════════════════════════════════════════════════════════════════════════

class TestCoilsForAneurysm:

    def test_returns_list(self):
        result = coils_for_aneurysm(6.0)
        assert isinstance(result, list)

    def test_returns_coil_specs(self):
        result = coils_for_aneurysm(6.0)
        assert all(isinstance(c, CoilSpec) for c in result)

    def test_framing_coils_match_dome_size(self):
        """Framing coils for a 6mm dome should have diameter ≈ 5.4–7.8 mm."""
        result = coils_for_aneurysm(6.0, CoilType.FRAMING)
        for c in result:
            assert 6.0 * 0.9 <= c.diameter_mm <= 6.0 * 1.3

    def test_filling_coils_not_larger_than_dome(self):
        """Filling coils for a 6mm dome should not exceed 6 mm."""
        result = coils_for_aneurysm(6.0, CoilType.FILLING)
        for c in result:
            assert c.diameter_mm <= 6.0

    def test_type_filter_works(self):
        framing  = coils_for_aneurysm(8.0, CoilType.FRAMING)
        filling  = coils_for_aneurysm(8.0, CoilType.FILLING)
        all_coils = coils_for_aneurysm(8.0)
        assert set(framing) | set(filling) <= set(all_coils)
        assert all(c.coil_type == CoilType.FRAMING  for c in framing)
        assert all(c.coil_type == CoilType.FILLING   for c in filling)

    def test_no_type_filter_returns_multiple_types(self):
        result = coils_for_aneurysm(6.0)
        types  = {c.coil_type for c in result}
        assert len(types) >= 2

    def test_very_small_aneurysm(self):
        """2 mm aneurysm — should still return some candidates (finishing coils)."""
        result = coils_for_aneurysm(2.0)
        assert len(result) >= 1

    def test_large_aneurysm(self):
        """12 mm giant aneurysm — should return large framing coils."""
        result = coils_for_aneurysm(12.0, CoilType.FRAMING)
        assert len(result) >= 1
        assert all(c.diameter_mm >= 10 for c in result)


# ══════════════════════════════════════════════════════════════════════════════
# estimate_coil_count()
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateCoilCount:

    def test_returns_positive_integer(self):
        spec = COIL_CATALOGUE[0]
        n = estimate_coil_count(100.0, spec, 25.0)
        assert isinstance(n, int)
        assert n >= 1

    def test_minimum_one_coil(self):
        """Even for very small aneurysms the minimum is 1."""
        spec = next(c for c in COIL_CATALOGUE if c.coil_type == CoilType.FINISHING)
        n = estimate_coil_count(1.0, spec, 25.0)
        assert n >= 1

    def test_larger_volume_requires_more_coils(self):
        spec = COIL_CATALOGUE[0]
        n_small = estimate_coil_count(50.0,  spec, 25.0)
        n_large = estimate_coil_count(500.0, spec, 25.0)
        assert n_large > n_small

    def test_higher_target_requires_more_coils(self):
        spec = COIL_CATALOGUE[0]
        n_25 = estimate_coil_count(200.0, spec, 25.0)
        n_35 = estimate_coil_count(200.0, spec, 35.0)
        assert n_35 >= n_25

    def test_formula_consistency(self):
        """
        Manual calculation:
          target_vol = aneurysm_vol × target_pct / 100
          n = ceil(target_vol / wire_volume_per_coil)
        """
        import math as _math
        spec = CoilSpec(
            name="Calc test",
            coil_type=CoilType.FILLING,
            diameter_mm=5,
            length_cm=10,   # 100 mm
            shape_3d="Helicoidal",
            wire_diameter_um=254,   # r ≈ 0.127 mm
            manufacturer="X",
            compatible_wire='0.014"',
            catheter_id_fr=1.7,
        )
        aneurysm_vol = 500.0
        target_pct   = 25.0
        target_vol   = aneurysm_vol * target_pct / 100.0
        expected_n   = _math.ceil(target_vol / spec.wire_volume_mm3)
        assert estimate_coil_count(aneurysm_vol, spec, target_pct) == expected_n


# ══════════════════════════════════════════════════════════════════════════════
# LVIS stents (added in stent_library.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestLvisStents:

    def test_lvis_jr_in_catalogue(self):
        from prospective.models.stent_library import STENT_CATALOGUE, StentType
        lvis_jr = [s for s in STENT_CATALOGUE if "LVIS Jr" in s.name]
        assert len(lvis_jr) >= 4, "Expected at least 4 LVIS Jr sizes"

    def test_lvis_in_catalogue(self):
        from prospective.models.stent_library import STENT_CATALOGUE
        lvis = [s for s in STENT_CATALOGUE if s.name.startswith("LVIS ") and
                "Jr" not in s.name]
        assert len(lvis) >= 4, "Expected at least 4 LVIS sizes"

    def test_lvis_jr_braided_type(self):
        from prospective.models.stent_library import STENT_CATALOGUE, StentType
        lvis_jr = [s for s in STENT_CATALOGUE if "LVIS Jr" in s.name]
        assert all(s.stent_type == StentType.BRAIDED for s in lvis_jr)

    def test_lvis_jr_wire_size(self):
        from prospective.models.stent_library import STENT_CATALOGUE
        lvis_jr = [s for s in STENT_CATALOGUE if "LVIS Jr" in s.name]
        assert all(s.compatible_wire == '0.017"' for s in lvis_jr)

    def test_lvis_jr_microcatheter(self):
        from prospective.models.stent_library import STENT_CATALOGUE
        lvis_jr = [s for s in STENT_CATALOGUE if "LVIS Jr" in s.name]
        assert all(s.catheter_id_fr == 2.1 for s in lvis_jr)

    def test_lvis_low_porosity(self):
        """LVIS has ~23% porosity (high metal coverage)."""
        from prospective.models.stent_library import STENT_CATALOGUE
        lvis = [s for s in STENT_CATALOGUE if "LVIS" in s.name]
        assert all(s.porosity_pct <= 25 for s in lvis)

    def test_stent_type_braided_value(self):
        from prospective.models.stent_library import StentType
        assert StentType.BRAIDED.value == "Stent trenzado"


# ══════════════════════════════════════════════════════════════════════════════
# Yasargil clips (added in clip_library.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestYasargilClips:

    def test_yasargil_clips_in_catalogue(self):
        from prospective.models.clip_library import CLIP_CATALOGUE
        yasargil = [c for c in CLIP_CATALOGUE if "Yasargil" in c.name]
        assert len(yasargil) >= 10, "Expected at least 10 Yasargil clip sizes"

    def test_yasargil_manufacturer(self):
        from prospective.models.clip_library import CLIP_CATALOGUE
        yasargil = [c for c in CLIP_CATALOGUE if "Yasargil" in c.name]
        assert all(c.manufacturer == "Yasargil/KS" for c in yasargil)

    def test_yasargil_shapes_present(self):
        from prospective.models.clip_library import CLIP_CATALOGUE, ClipShape
        yasargil = [c for c in CLIP_CATALOGUE if "Yasargil" in c.name]
        shapes   = {c.shape for c in yasargil}
        assert ClipShape.STRAIGHT    in shapes
        assert ClipShape.CURVED      in shapes
        assert ClipShape.BAYONET     in shapes
        assert ClipShape.FENESTRATED in shapes

    def test_angled_45_shape_present(self):
        from prospective.models.clip_library import CLIP_CATALOGUE, ClipShape
        angled45 = [c for c in CLIP_CATALOGUE if c.shape == ClipShape.ANGLED_45]
        assert len(angled45) >= 2

    def test_all_manufacturers_present(self):
        from prospective.models.clip_library import CLIP_CATALOGUE
        manufacturers = {c.manufacturer for c in CLIP_CATALOGUE}
        assert "Yasargil/KS" in manufacturers
        assert "Sugita"      in manufacturers
        assert "Aesculap"    in manufacturers
        assert "Codman"      in manufacturers

    def test_clips_for_neck_still_works(self):
        from prospective.models.clip_library import clips_for_neck
        result = clips_for_neck(5.0)
        assert len(result) >= 1
        for c in result:
            assert c.blade_length_mm >= 5.0 + 1.0

    def test_clip_catalogue_count_increased(self):
        """Catalogue should now have many more clips than the original 16."""
        from prospective.models.clip_library import CLIP_CATALOGUE
        assert len(CLIP_CATALOGUE) >= 35
