"""Mesh preparation for 3D printing — Feature 7.

Prepares an aneurysm / vessel segmentation mesh for FFF/SLA 3D printing:

1. Fill small holes (vtkFillHolesFilter)
2. Smooth surface (vtkSmoothPolyDataFilter — Laplacian)
3. Optional subdivision for higher resolution output (vtkLinearSubdivisionFilter)
4. Scale uniformly to a target maximum dimension
5. Compute physical properties via vtkMassProperties
6. Watertightness check (open edges via vtkFeatureEdges)
7. Export to STL

Typical usage::

    from prospective.processing.mesh_prep import prepare_mesh_for_print
    result = prepare_mesh_for_print(poly_data, target_size_mm=80.0)
    result.export_stl("/tmp/aneurysm_print.stl")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import vtk

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────── #
# Print-bed presets (name → (x_mm, y_mm, z_mm))                               #
# ──────────────────────────────────────────────────────────────────────────── #
PRINT_BED_PRESETS: dict[str, tuple[float, float, float]] = {
    "Ender 3 / 3 Pro":      (220.0, 220.0, 250.0),
    "Prusa MK4":            (250.0, 210.0, 220.0),
    "Bambu Lab X1C":        (256.0, 256.0, 256.0),
    "Formlabs Form 3":      (145.0, 145.0, 185.0),
    "Ultimaker S3":         (230.0, 190.0, 200.0),
    "Personalizado":        (0.0,   0.0,   0.0),   # user-defined
}


# ──────────────────────────────────────────────────────────────────────────── #
# Result dataclass                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class PrintPrepResult:
    """Output of :func:`prepare_mesh_for_print`."""

    mesh:               "vtk.vtkPolyData"
    scale_factor:       float
    dimensions_mm:      tuple[float, float, float]   # (dx, dy, dz) after scaling
    volume_cm3:         float                         # solid volume (material estimate)
    surface_area_cm2:   float
    is_watertight:      bool
    open_edge_count:    int
    warnings:           list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def fits_in_bed(self, bed: tuple[float, float, float]) -> bool:
        """
        Return True if the mesh's bounding box fits within *bed* (x,y,z) mm.

        Any bed dimension ≤ 0 is treated as unlimited.
        """
        bx, by, bz = bed
        dx, dy, dz = self.dimensions_mm
        if bx > 0 and dx > bx:
            return False
        if by > 0 and dy > by:
            return False
        if bz > 0 and dz > bz:
            return False
        return True

    def export_stl(self, path: str) -> None:
        """Write the prepared mesh to an ASCII STL file."""
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(path)
        writer.SetInputData(self.mesh)
        writer.Write()
        logger.info("PrintPrep: exported STL → %s  (%.2f cm³)", path, self.volume_cm3)

    def summary(self) -> str:
        dx, dy, dz = self.dimensions_mm
        wt = "hermético" if self.is_watertight else f"NO hermético ({self.open_edge_count} bordes abiertos)"
        lines = [
            f"Dimensiones: {dx:.1f} × {dy:.1f} × {dz:.1f} mm",
            f"Escala aplicada: ×{self.scale_factor:.4f}",
            f"Volumen sólido: {self.volume_cm3:.3f} cm³",
            f"Superficie: {self.surface_area_cm2:.2f} cm²",
            f"Malla: {wt}",
        ]
        if self.warnings:
            lines += ["Advertencias:"] + [f"  • {w}" for w in self.warnings]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────── #
# Main function                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def prepare_mesh_for_print(
    poly_data: "vtk.vtkPolyData",
    target_size_mm: float = 80.0,
    smooth_iterations: int = 20,
    smooth_relaxation: float = 0.1,
    fill_holes: bool = True,
    hole_size: float = 5.0,
    subdivide: bool = False,
    progress_cb: Callable[[int], None] | None = None,
) -> PrintPrepResult:
    """
    Prepare *poly_data* for 3D printing.

    Parameters
    ----------
    poly_data        : input surface mesh (vtkPolyData)
    target_size_mm   : desired maximum dimension after scaling (mm); use 0 to skip scaling
    smooth_iterations: Laplacian smoothing iterations
    smooth_relaxation: relaxation factor (0–1) for smoothing
    fill_holes       : whether to apply vtkFillHolesFilter before smoothing
    hole_size        : maximum hole perimeter to fill (mm)
    subdivide        : apply one level of linear subdivision (doubles triangle count)
    progress_cb      : optional callback receiving int 0-100

    Returns
    -------
    PrintPrepResult
    """
    warnings: list[str] = []

    def _progress(pct: int) -> None:
        if progress_cb:
            progress_cb(pct)

    if poly_data is None or poly_data.GetNumberOfPoints() == 0:
        raise ValueError("poly_data is empty or None")

    mesh = poly_data
    _progress(5)

    # ── 1. Normals (consistent orientation) ──────────────────────────── #
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    mesh = normals.GetOutput()
    _progress(15)

    # ── 2. Fill holes ─────────────────────────────────────────────────── #
    if fill_holes:
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputData(mesh)
        filler.SetHoleSize(hole_size)
        filler.Update()
        mesh = filler.GetOutput()
        # Re-compute normals after hole filling
        n2 = vtk.vtkPolyDataNormals()
        n2.SetInputData(mesh)
        n2.ConsistencyOn()
        n2.SplittingOff()
        n2.Update()
        mesh = n2.GetOutput()
    _progress(30)

    # ── 3. Smooth ─────────────────────────────────────────────────────── #
    if smooth_iterations > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(mesh)
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.SetRelaxationFactor(smooth_relaxation)
        smoother.BoundarySmoothingOff()
        smoother.Update()
        mesh = smoother.GetOutput()
    _progress(50)

    # ── 4. Optional subdivision ───────────────────────────────────────── #
    if subdivide:
        sub = vtk.vtkLinearSubdivisionFilter()
        sub.SetInputData(mesh)
        sub.SetNumberOfSubdivisions(1)
        sub.Update()
        mesh = sub.GetOutput()
    _progress(60)

    # ── 5. Triangulate (ensure all faces are triangles) ───────────────── #
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(mesh)
    tri.Update()
    mesh = tri.GetOutput()
    _progress(65)

    # ── 6. Scale to target size ───────────────────────────────────────── #
    bounds = mesh.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    dx_orig = bounds[1] - bounds[0]
    dy_orig = bounds[3] - bounds[2]
    dz_orig = bounds[5] - bounds[4]
    max_dim = max(dx_orig, dy_orig, dz_orig)

    if target_size_mm > 0 and max_dim > 1e-6:
        scale_factor = target_size_mm / max_dim
    else:
        scale_factor = 1.0

    if scale_factor != 1.0:
        transform = vtk.vtkTransform()
        transform.Scale(scale_factor, scale_factor, scale_factor)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(mesh)
        tf.SetTransform(transform)
        tf.Update()
        mesh = tf.GetOutput()
    _progress(75)

    # ── 7. Final normals pass ─────────────────────────────────────────── #
    fn = vtk.vtkPolyDataNormals()
    fn.SetInputData(mesh)
    fn.ConsistencyOn()
    fn.SplittingOff()
    fn.ComputePointNormalsOn()
    fn.Update()
    mesh = fn.GetOutput()
    _progress(80)

    # ── 8. Mass properties ────────────────────────────────────────────── #
    props = vtk.vtkMassProperties()
    props.SetInputData(mesh)
    props.Update()
    volume_mm3      = props.GetVolume()
    surface_area_mm2 = props.GetSurfaceArea()

    # Guard against degenerate mesh (MassProperties returns 0 for open meshes)
    if volume_mm3 < 0:
        volume_mm3 = abs(volume_mm3)
        warnings.append("Volumen negativo calculado — malla puede tener normales invertidas.")

    volume_cm3      = volume_mm3 / 1000.0
    surface_area_cm2 = surface_area_mm2 / 100.0
    _progress(88)

    # ── 9. Watertightness via open edges ─────────────────────────────── #
    feat = vtk.vtkFeatureEdges()
    feat.SetInputData(mesh)
    feat.BoundaryEdgesOn()
    feat.FeatureEdgesOff()
    feat.ManifoldEdgesOff()
    feat.NonManifoldEdgesOff()
    feat.Update()
    open_edge_count = feat.GetOutput().GetNumberOfLines()
    is_watertight   = (open_edge_count == 0)

    if not is_watertight:
        warnings.append(
            f"Malla no hermética: {open_edge_count} bordes abiertos detectados. "
            "Considera incrementar 'Tamaño máximo agujero' o usar reparación externa."
        )
    _progress(93)

    # ── 10. Scaled dimensions ─────────────────────────────────────────── #
    sb = mesh.GetBounds()
    dims = (
        round(sb[1] - sb[0], 3),
        round(sb[3] - sb[2], 3),
        round(sb[5] - sb[4], 3),
    )

    # Sanity warnings
    if max(dims) > 300.0:
        warnings.append(
            f"Dimensión máxima {max(dims):.1f} mm excede 300 mm. "
            "Verifica la escala o activa el escalado."
        )
    if volume_cm3 < 0.001:
        warnings.append("Volumen calculado muy pequeño — verificar malla.")

    _progress(100)

    result = PrintPrepResult(
        mesh=mesh,
        scale_factor=round(scale_factor, 6),
        dimensions_mm=dims,
        volume_cm3=round(volume_cm3, 4),
        surface_area_cm2=round(surface_area_cm2, 4),
        is_watertight=is_watertight,
        open_edge_count=open_edge_count,
        warnings=warnings,
    )
    logger.info(
        "PrintPrep: done — scale=%.4f  dims=%s  vol=%.3f cm³  watertight=%s",
        scale_factor, dims, volume_cm3, is_watertight,
    )
    return result
