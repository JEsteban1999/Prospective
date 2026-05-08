"""VTK transfer function presets for intracranial CTA volumes.

Each preset configures:
  - vtkColorTransferFunction  → scalar value → RGB
  - vtkPiecewiseFunction      → scalar value → opacity (0–1)
  - vtkVolumeProperty         → lighting parameters
"""
from __future__ import annotations

import vtk

# ─────────────────────────────────────────────────────────────────────────── #
# Public registry                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

PRESETS_3D: list[str] = ["CTA", "Vasos CTA", "Cerebro", "Hemorragia", "Hueso", "Soft Tissue"]


def apply_preset(
    name: str,
    color_tf: vtk.vtkColorTransferFunction,
    opacity_tf: vtk.vtkPiecewiseFunction,
    prop: vtk.vtkVolumeProperty,
) -> None:
    """Apply a named preset in-place to the three VTK objects."""
    presets = {
        "CTA":          _cta,
        "Vasos CTA":    _vessels_cta,
        "Cerebro":      _brain,
        "Hemorragia":   _hemorrhage,
        "Hueso":        _bone,
        "Soft Tissue":  _soft_tissue,
    }
    fn = presets.get(name, _cta)
    color_tf.RemoveAllPoints()
    opacity_tf.RemoveAllPoints()
    fn(color_tf, opacity_tf, prop)
    color_tf.Modified()
    opacity_tf.Modified()


# ─────────────────────────────────────────────────────────────────────────── #
# Preset implementations                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _cta(
    c: vtk.vtkColorTransferFunction,
    o: vtk.vtkPiecewiseFunction,
    p: vtk.vtkVolumeProperty,
) -> None:
    """CT Angiography — vessels bright red/white, skull yellow-white."""
    # Color map (HU → RGB)
    c.AddRGBPoint(-1000, 0.00, 0.00, 0.00)   # air
    c.AddRGBPoint( -300, 0.00, 0.00, 0.00)   # lung
    c.AddRGBPoint(  -80, 0.45, 0.20, 0.12)   # fat / soft tissue boundary
    c.AddRGBPoint(   40, 0.82, 0.60, 0.52)   # soft tissue (pink)
    c.AddRGBPoint(  160, 0.95, 0.28, 0.22)   # contrast / blood (red)
    c.AddRGBPoint(  350, 1.00, 0.72, 0.35)   # calcification (orange)
    c.AddRGBPoint(  600, 1.00, 0.95, 0.82)   # cortical bone (ivory)
    c.AddRGBPoint( 1500, 1.00, 1.00, 1.00)   # dense bone (white)

    # Opacity map
    o.AddPoint(-1000, 0.00)
    o.AddPoint( -300, 0.00)
    o.AddPoint(  -80, 0.00)
    o.AddPoint(   40, 0.03)   # slight tissue haze
    o.AddPoint(  130, 0.18)
    o.AddPoint(  160, 0.45)   # vessel peak
    o.AddPoint(  350, 0.65)
    o.AddPoint(  600, 0.82)
    o.AddPoint( 1500, 0.95)
    o.AddPoint( 3000, 1.00)

    _set_lighting(p, shade=True, ambient=0.15, diffuse=0.90, specular=0.25, power=12.0)


def _vessels_cta(
    c: vtk.vtkColorTransferFunction,
    o: vtk.vtkPiecewiseFunction,
    p: vtk.vtkVolumeProperty,
) -> None:
    """CTA vascular — isolates contrast-enhanced vessels, suppresses bone.

    Designed to work on both raw and bone-subtracted volumes:
    - On bone-subtracted volumes: vessels render cleanly (bone already -1000 HU).
    - On raw volumes: TF-based bone suppression provides a reasonable result,
      though some cortical bone bleed-through may remain at the 300–400 HU boundary.
    """
    # Color: black/transparent below vessels, bright red at peak, invisible above bone
    c.AddRGBPoint(-1000, 0.00, 0.00, 0.00)   # air / bone-subtracted regions
    c.AddRGBPoint(   80, 0.00, 0.00, 0.00)   # soft tissue boundary — invisible
    c.AddRGBPoint(  110, 0.80, 0.08, 0.08)   # vessel ramp start (dark red)
    c.AddRGBPoint(  180, 1.00, 0.18, 0.08)   # main vessels (bright red)
    c.AddRGBPoint(  280, 1.00, 0.60, 0.30)   # calcified wall / dense contrast (orange)
    c.AddRGBPoint(  360, 0.00, 0.00, 0.00)   # above vessel range: invisible
    c.AddRGBPoint( 3000, 0.00, 0.00, 0.00)

    # Opacity: hard zeros below and above the vessel window, steep ramp in between.
    # Designed for bone-subtracted CTA (all bone voxels are already at -1000 HU).
    # The steep ramp gives crisp vessel boundaries instead of a hazy transition.
    o.AddPoint(-1000, 0.00)
    o.AddPoint(   80, 0.00)   # suppress air and soft tissue completely
    o.AddPoint(  100, 0.00)   # dead zone — keeps background clean
    o.AddPoint(  120, 0.75)   # steep ramp up — small vessels appear sharply
    o.AddPoint(  200, 0.95)   # peak opacity for main vessels with contrast
    o.AddPoint(  300, 0.80)   # calcification / dense contrast shoulder
    o.AddPoint(  360, 0.10)   # fast fade — cuts off at bone threshold
    o.AddPoint(  400, 0.00)   # hard stop — bone completely invisible
    o.AddPoint( 3000, 0.00)

    _set_lighting(p, shade=True, ambient=0.10, diffuse=0.95, specular=0.45, power=20.0)


def _brain(
    c: vtk.vtkColorTransferFunction,
    o: vtk.vtkPiecewiseFunction,
    p: vtk.vtkVolumeProperty,
) -> None:
    """Brain parenchyma — grey/white matter visible, vessels subtle."""
    c.AddRGBPoint(-1000, 0.00, 0.00, 0.00)
    c.AddRGBPoint(    0, 0.00, 0.00, 0.00)
    c.AddRGBPoint(   20, 0.42, 0.38, 0.38)   # grey matter
    c.AddRGBPoint(   40, 0.58, 0.52, 0.50)   # white matter
    c.AddRGBPoint(   80, 0.85, 0.75, 0.65)   # hyperintense tissue
    c.AddRGBPoint(  200, 0.95, 0.50, 0.40)   # calcification / contrast
    c.AddRGBPoint(  500, 1.00, 0.90, 0.75)   # bone
    c.AddRGBPoint( 1000, 1.00, 1.00, 1.00)

    o.AddPoint(-1000, 0.00)
    o.AddPoint(    0, 0.00)
    o.AddPoint(   20, 0.08)
    o.AddPoint(   40, 0.14)
    o.AddPoint(   80, 0.22)
    o.AddPoint(  200, 0.45)
    o.AddPoint(  500, 0.70)
    o.AddPoint( 1000, 0.90)

    _set_lighting(p, shade=True, ambient=0.25, diffuse=0.80, specular=0.15, power=8.0)


def _hemorrhage(
    c: vtk.vtkColorTransferFunction,
    o: vtk.vtkPiecewiseFunction,
    p: vtk.vtkVolumeProperty,
) -> None:
    """Intracranial haemorrhage — hyperdense blood bright, tissue suppressed."""
    c.AddRGBPoint(-1000, 0.00, 0.00, 0.00)
    c.AddRGBPoint(   30, 0.00, 0.00, 0.00)
    c.AddRGBPoint(   50, 0.85, 0.65, 0.55)   # isodense blood
    c.AddRGBPoint(   70, 1.00, 0.35, 0.10)   # acute haemorrhage (bright red)
    c.AddRGBPoint(   90, 1.00, 0.90, 0.50)   # hyperdense clot (yellow-white)
    c.AddRGBPoint(  200, 1.00, 1.00, 0.85)
    c.AddRGBPoint(  500, 1.00, 1.00, 1.00)

    o.AddPoint(-1000, 0.00)
    o.AddPoint(   40, 0.00)
    o.AddPoint(   55, 0.30)
    o.AddPoint(   70, 0.70)
    o.AddPoint(   90, 0.90)
    o.AddPoint(  200, 0.80)
    o.AddPoint(  500, 0.70)
    o.AddPoint( 3000, 0.90)

    _set_lighting(p, shade=True, ambient=0.20, diffuse=0.85, specular=0.20, power=10.0)


def _bone(
    c: vtk.vtkColorTransferFunction,
    o: vtk.vtkPiecewiseFunction,
    p: vtk.vtkVolumeProperty,
) -> None:
    """Skull / bone — soft tissue transparent, cortical bone opaque."""
    c.AddRGBPoint(-1000, 0.00, 0.00, 0.00)
    c.AddRGBPoint(  200, 0.00, 0.00, 0.00)
    c.AddRGBPoint(  400, 0.88, 0.80, 0.60)   # cancellous bone (beige)
    c.AddRGBPoint(  700, 1.00, 0.95, 0.82)   # cortical bone (ivory)
    c.AddRGBPoint( 1500, 1.00, 1.00, 1.00)

    o.AddPoint(-1000, 0.00)
    o.AddPoint(  300, 0.00)
    o.AddPoint(  400, 0.30)
    o.AddPoint(  700, 0.80)
    o.AddPoint( 1500, 0.95)
    o.AddPoint( 3000, 1.00)

    _set_lighting(p, shade=True, ambient=0.15, diffuse=0.95, specular=0.45, power=20.0)


def _soft_tissue(
    c: vtk.vtkColorTransferFunction,
    o: vtk.vtkPiecewiseFunction,
    p: vtk.vtkVolumeProperty,
) -> None:
    """Soft tissue — all parenchyma semi-transparent, no bone."""
    c.AddRGBPoint(-1000, 0.00, 0.00, 0.00)
    c.AddRGBPoint( -100, 0.00, 0.00, 0.00)
    c.AddRGBPoint(    0, 0.38, 0.22, 0.18)
    c.AddRGBPoint(   40, 0.75, 0.55, 0.45)
    c.AddRGBPoint(   80, 0.90, 0.78, 0.68)
    c.AddRGBPoint(  200, 1.00, 0.90, 0.70)
    c.AddRGBPoint(  400, 0.00, 0.00, 0.00)   # bone: transparent

    o.AddPoint(-1000, 0.00)
    o.AddPoint( -100, 0.00)
    o.AddPoint(    0, 0.06)
    o.AddPoint(   40, 0.18)
    o.AddPoint(   80, 0.28)
    o.AddPoint(  200, 0.40)
    o.AddPoint(  350, 0.10)
    o.AddPoint(  500, 0.00)   # suppress bone

    _set_lighting(p, shade=True, ambient=0.25, diffuse=0.85, specular=0.10, power=6.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def _set_lighting(
    p: vtk.vtkVolumeProperty,
    *,
    shade: bool,
    ambient: float,
    diffuse: float,
    specular: float,
    power: float,
) -> None:
    if shade:
        p.ShadeOn()
    else:
        p.ShadeOff()
    p.SetAmbient(ambient)
    p.SetDiffuse(diffuse)
    p.SetSpecular(specular)
    p.SetSpecularPower(power)
    p.SetInterpolationTypeToLinear()
