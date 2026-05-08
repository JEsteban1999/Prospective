"""Vascular segmentation pipelines.

A-02-04: Interactive segmentation — umbralización + Marching Cubes
A-02-07: Mesh optimisation — suavizado WindowedSinc + decimación QuadricDecimation
S-3:     Grow-from-seeds — SimpleITK ConnectedThreshold + Marching Cubes

Threshold pipeline
------------------
numpy volume (z,y,x)
    │
    ▼  vtkImageGaussianSmooth   (noise reduction before iso-surface)
    │
    ▼  vtkMarchingCubes         (iso-surface at threshold HU)
    │
    ▼  vtkWindowedSincPolyDataFilter  (mesh smoothing)
    │
    ▼  vtkQuadricDecimation     (polygon count reduction)
    │
    ▼  vtkPolyDataNormals       (smooth normals for shading)
    │
    vtkPolyData  ←  result

Grow-from-seeds pipeline (S-3)
-------------------------------
numpy volume (z,y,x)
    │
    ▼  SimpleITK ConnectedThreshold (lower/upper HU, seed voxels)
    │    → binary mask (0/1 uint8)
    │
    ▼  vtkMarchingCubes at iso=0.5 on binary mask
    │
    ▼  vtkWindowedSincPolyDataFilter  (mesh smoothing)
    │
    ▼  vtkQuadricDecimation           (polygon count reduction)
    │
    ▼  vtkPolyDataNormals             (smooth normals for shading)
    │
    vtkPolyData  ←  result
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import vtk
from vtkmodules.util import numpy_support as ns

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    poly_data: vtk.vtkPolyData
    n_vertices: int
    n_triangles: int
    threshold_hu: float
    reduction_pct: float          # actual decimation achieved
    n_fragments_removed: int = 0  # small components removed by fragment filter
    is_preview: bool = False      # True when produced by run_fast_preview()


class SegmentationPipeline:
    """
    Extracts an iso-surface from a CT volume using VTK Marching Cubes.

    All parameters are set at construction time and can be changed
    between calls to ``run()``.

    Parameters
    ----------
    threshold_hu:
        Iso-surface value in Hounsfield units.
        ~150 HU  → contrast-enhanced vessels (CTA)
        ~300 HU  → bone / calcified structures
    smooth_iterations:
        WindowedSinc iterations (0 = skip smoothing).  Default 20.
    smooth_pass_band:
        Pass-band frequency [0–2].  Lower = smoother.  Default 0.06.
    target_reduction:
        Fraction of triangles to remove [0–0.99].  Default 0.70.
    gaussian_sigma:
        Std-dev of pre-smoothing Gaussian in voxels (0 = skip).  Default 0.5.
        IMPORTANT: large sigma blurs small vessels — use 0–0.5 to preserve
        fine vascular detail, 0.8–1.2 for cleaner surfaces on large vessels.
    min_component_verts:
        Remove disconnected mesh fragments with fewer vertices than this value.
        0 = keep all components.  Default 100.
        Lower = more small vessels retained (fewer noise islands filtered out).
        Raise to 500–2000 if too many floating fragments appear.
    vessel_lower_hu:
        Secondary (lower) HU threshold for thin-vessel enhancement.
        0 = disabled (standard single-threshold behaviour).
        When > 0, voxels in the range [vessel_lower_hu, threshold_hu) that
        fall within *vessel_dilation_mm* of the primary mask are included.
        Typical value: 30–50 HU below threshold_hu.
    vessel_dilation_mm:
        Spatial search radius (mm) around the primary mask used to decide
        whether a below-threshold voxel belongs to a thin vessel or is
        unrelated soft tissue.  Default 3.0 mm.
    morpho_closing_mm:
        Radius in mm for binary morphological closing applied to the mask
        before Marching Cubes.  Fills micro-gaps and removes pin-noise.
        0 = disabled.  0.5–1.0 mm recommended for typical CTA data.
        Values > 1.5 mm may fuse adjacent vessels.
    keep_top_n:
        When > 0, keep only the N largest connected components (by voxel
        count) from the binary mask.  Overrides min_component_verts when set.
        0 = use min_component_verts instead.  5–10 retains main vasculature.
    threshold_max_hu:
        Upper bound for the iso-surface threshold (band-pass filter).
        0 = disabled (standard lower-only threshold behaviour).
        When > threshold_hu, only voxels in [threshold_hu, threshold_max_hu]
        are segmented.  Critical for XA / 3DRA data where bone intensities
        sit ABOVE vessel intensities — setting an upper bound excludes the
        skull and imaging-table artefacts.
        Typical 3DRA: lower ~ WC − WW×0.4, upper ~ WC + WW×0.45.
    """

    def __init__(
        self,
        threshold_hu: float = 150.0,
        threshold_max_hu: float = 0.0,
        smooth_iterations: int = 20,
        smooth_pass_band: float = 0.06,
        target_reduction: float = 0.70,
        gaussian_sigma: float = 0.5,
        min_component_verts: int = 100,
        vessel_lower_hu: float = 0.0,
        vessel_dilation_mm: float = 3.0,
        morpho_closing_mm: float = 0.0,
        keep_top_n: int = 0,
    ) -> None:
        self.threshold_hu        = threshold_hu
        self.threshold_max_hu    = threshold_max_hu
        self.smooth_iterations   = smooth_iterations
        self.smooth_pass_band    = smooth_pass_band
        self.target_reduction    = target_reduction
        self.gaussian_sigma      = gaussian_sigma
        self.min_component_verts = min_component_verts
        self.vessel_lower_hu     = vessel_lower_hu
        self.vessel_dilation_mm  = vessel_dilation_mm
        self.morpho_closing_mm   = morpho_closing_mm
        self.keep_top_n          = keep_top_n

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> SegmentationResult:
        """
        Execute the full pipeline and return a ``SegmentationResult``.

        Parameters
        ----------
        volume:   float32 array (z, y, x) in HU.
        spacing:  (sz, sy, sx) in mm.
        """
        use_dual = (
            self.vessel_lower_hu > 0
            and self.vessel_lower_hu < self.threshold_hu
        )
        use_max = (
            self.threshold_max_hu > self.threshold_hu
        )
        logger.info(
            "Segmentation started — threshold=%.0f HU  max=%.0f HU  dual=%s  closing=%.1f mm  top_n=%d  shape=%s",
            self.threshold_hu,
            self.threshold_max_hu if use_max else float("inf"),
            use_dual, self.morpho_closing_mm, self.keep_top_n, volume.shape,
        )

        # ── 1. Pre-smooth (numpy) ──────────────────────────────────────────── #
        vol_f = volume.astype(np.float32)
        if self.gaussian_sigma > 0.0:
            vol_f = self._numpy_gaussian(vol_f, self.gaussian_sigma)

        # ── 2. Binary mask ─────────────────────────────────────────────────── #
        if use_dual:
            mask = self._dual_threshold_mask(
                vol_f, spacing, self.threshold_hu,
                self.vessel_lower_hu, self.vessel_dilation_mm,
            )
            # Apply upper threshold to dual-threshold result when requested
            if use_max:
                mask = mask * (vol_f <= self.threshold_max_hu).astype(np.float32)
        elif use_max:
            mask = (
                (vol_f >= self.threshold_hu) & (vol_f <= self.threshold_max_hu)
            ).astype(np.float32)
        else:
            mask = (vol_f >= self.threshold_hu).astype(np.float32)

        # ── 3. Morphological closing (fills micro-gaps, removes pin-noise) ─── #
        if self.morpho_closing_mm > 0.0:
            mask = self._morpho_closing(mask, spacing, self.morpho_closing_mm)

        # ── 4. Connected component filtering in mask space (fast) ─────────── #
        n_fragments_removed = 0
        if self.keep_top_n > 0 or self.min_component_verts > 0:
            mask, n_fragments_removed = self._filter_mask_components(
                mask, self.min_component_verts, self.keep_top_n
            )

        # ── 5. Marching Cubes on binary mask at iso=0.5 ───────────────────── #
        mc_input  = self._to_vtk_image(mask, spacing)
        iso_value = 0.5
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(mc_input)
        mc.SetValue(0, iso_value)
        mc.ComputeNormalsOff()
        mc.ComputeGradientsOff()
        mc.Update()

        n_raw = mc.GetOutput().GetNumberOfPolys()
        logger.info("Marching Cubes: %d triangles", n_raw)

        if n_raw == 0:
            raise ValueError(
                f"No iso-surface found at {self.threshold_hu:.0f} HU. "
                "Try lowering the threshold."
            )

        # ── 6. Mesh smoothing ─────────────────────────────────────────── #
        if self.smooth_iterations > 0:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(mc.GetOutputPort())
            smoother.SetNumberOfIterations(self.smooth_iterations)
            smoother.SetPassBand(self.smooth_pass_band)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()
            prev_port = smoother.GetOutputPort()
        else:
            prev_port = mc.GetOutputPort()

        # ── 7. Decimation (A-02-07) ───────────────────────────────────────── #
        if self.target_reduction > 0:
            decimate = vtk.vtkQuadricDecimation()
            decimate.SetInputConnection(prev_port)
            decimate.SetTargetReduction(self.target_reduction)
            decimate.Update()
            prev_port = decimate.GetOutputPort()

        # ── 8. Normals for smooth shading ────────────────────────────────── #
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(prev_port)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()

        poly = normals.GetOutput()
        n_verts = poly.GetNumberOfPoints()
        n_tris  = poly.GetNumberOfPolys()
        actual_reduction = 1.0 - n_tris / max(n_raw, 1)

        logger.info(
            "Segmentation done — %d verts, %d tris (%.0f%% reduction) "
            "%d fragments removed",
            n_verts, n_tris, actual_reduction * 100, n_fragments_removed,
        )

        return SegmentationResult(
            poly_data=poly,
            n_vertices=n_verts,
            n_triangles=n_tris,
            threshold_hu=self.threshold_hu,
            reduction_pct=actual_reduction * 100,
            n_fragments_removed=n_fragments_removed,
        )

    # ------------------------------------------------------------------ #
    # Fast preview                                                         #
    # ------------------------------------------------------------------ #

    def run_fast_preview(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        downsample: int = 2,
    ) -> SegmentationResult:
        """
        Rapid preview: downsample the volume, skip smoothing and decimation.

        ~8× faster than ``run()`` with the default ``downsample=2``.
        The resulting mesh is coarser but reflects the threshold choice
        accurately enough for interactive parameter tuning.

        Parameters
        ----------
        volume:     float32 array (z, y, x) in HU.
        spacing:    (sz, sy, sx) in mm.
        downsample: stride applied to every axis (2 → half resolution).
        """
        s     = max(1, int(downsample))
        vol_d = volume[::s, ::s, ::s]
        sp_d  = tuple(sp * s for sp in spacing)

        # Binary mask (bandpass when threshold_max_hu is set)
        _use_max = self.threshold_max_hu > self.threshold_hu
        if _use_max:
            mask = (
                (vol_d >= self.threshold_hu) & (vol_d <= self.threshold_max_hu)
            ).astype(np.float32)
        else:
            mask = (vol_d >= self.threshold_hu).astype(np.float32)

        # Fast component filter: use configured top-N, else default to top-20 for clean preview
        kn = self.keep_top_n if self.keep_top_n > 0 else 20
        mask, _ = self._filter_mask_components(mask, 0, kn)

        img = self._to_vtk_image(mask, sp_d)

        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(img)
        mc.SetValue(0, 0.5)
        mc.ComputeNormalsOff()
        mc.ComputeGradientsOff()
        mc.Update()

        n_raw = mc.GetOutput().GetNumberOfPolys()
        if n_raw == 0:
            raise ValueError(
                f"No iso-surface found at {self.threshold_hu:.0f} HU in preview. "
                "Try lowering the threshold."
            )

        # Light smoothing (5 iter) for a presentable preview
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(mc.GetOutputPort())
        smoother.SetNumberOfIterations(5)
        smoother.SetPassBand(0.10)
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()

        poly    = normals.GetOutput()
        n_verts = poly.GetNumberOfPoints()
        n_tris  = poly.GetNumberOfPolys()

        logger.debug(
            "Preview — %d verts, %d tris  (downsample %d×, threshold %.0f HU)",
            n_verts, n_tris, s, self.threshold_hu,
        )

        return SegmentationResult(
            poly_data           = poly,
            n_vertices          = n_verts,
            n_triangles         = n_tris,
            threshold_hu        = self.threshold_hu,
            reduction_pct       = 0.0,
            n_fragments_removed = 0,
            is_preview          = True,
        )

    # ------------------------------------------------------------------ #
    # Vessel-enhancement helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _morpho_closing(
        mask: np.ndarray,
        spacing: tuple[float, float, float],
        closing_mm: float,
    ) -> np.ndarray:
        """
        Binary morphological closing to fill micro-gaps and remove pin-noise.

        Uses SimpleITK BinaryMorphologicalClosing if available, falls back to
        scipy binary_closing.  Returns float32 mask (0 / 1).
        """
        min_sp = min(float(spacing[0]), float(spacing[1]), float(spacing[2]))
        radius = max(1, round(closing_mm / min_sp))

        try:
            import SimpleITK as sitk
            sz, sy, sx = spacing
            sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
            sitk_mask.SetSpacing((float(sx), float(sy), float(sz)))
            closed = sitk.BinaryMorphologicalClosing(sitk_mask, [radius, radius, radius])
            logger.debug("Morpho closing: radius=%d voxels (%.1f mm)", radius, closing_mm)
            return sitk.GetArrayFromImage(closed).astype(np.float32)
        except Exception as exc:
            logger.debug("SimpleITK closing failed (%s); trying scipy", exc)

        try:
            from scipy.ndimage import binary_closing, generate_binary_structure
            struct = generate_binary_structure(3, 1)  # 6-connectivity ball
            closed = binary_closing(mask.astype(bool), structure=struct, iterations=radius)
            return closed.astype(np.float32)
        except Exception as exc:
            logger.warning("Morphological closing skipped: %s", exc)
            return mask.astype(np.float32)

    @staticmethod
    def _filter_mask_components(
        mask: np.ndarray,
        min_voxels: int,
        keep_top_n: int,
    ) -> tuple[np.ndarray, int]:
        """
        Remove small / unwanted connected components from a binary mask.

        Works in voxel space using scipy.ndimage.label — O(n_voxels), much
        faster than the previous VTK-based per-component pass.

        Parameters
        ----------
        mask:        float32 or bool array (0 / 1).
        min_voxels:  When keep_top_n == 0: keep components with ≥ this many voxels.
        keep_top_n:  When > 0: keep the N largest components regardless of size.

        Returns
        -------
        (filtered_mask_float32, n_removed)
        """
        try:
            from scipy.ndimage import label as scipy_label
        except ImportError:
            logger.warning("scipy not available — component filtering skipped")
            return mask.astype(np.float32), 0

        mask_bool = mask.astype(bool)
        labeled, n_labels = scipy_label(mask_bool)

        if n_labels <= 1:
            return mask.astype(np.float32), 0

        # Voxel count per component (index 0 = background, skip it)
        sizes = np.bincount(labeled.ravel())
        component_list = [(int(sizes[i + 1]), i + 1) for i in range(n_labels)]

        if keep_top_n > 0:
            component_list.sort(key=lambda x: x[0], reverse=True)
            keep_set = {idx for _, idx in component_list[:keep_top_n]}
        else:
            keep_set = {idx for sz, idx in component_list if sz >= min_voxels}
            if not keep_set:
                # Always keep at least the largest component
                keep_set = {max(component_list, key=lambda x: x[0])[1]}

        n_removed = n_labels - len(keep_set)
        if n_removed == 0:
            return mask.astype(np.float32), 0

        result = np.isin(labeled, list(keep_set)).astype(np.float32)
        logger.info(
            "Component filter: kept %d/%d components (%s)",
            len(keep_set), n_labels,
            f"top-{keep_top_n}" if keep_top_n > 0 else f"≥{min_voxels} voxels",
        )
        return result, n_removed

    @staticmethod
    def _dual_threshold_mask(
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        main_hu: float,
        vessel_hu: float,
        dilation_mm: float,
    ) -> np.ndarray:
        """
        Return a float32 binary mask (0/1) for marching cubes at iso=0.5.

        The mask combines:
          • All voxels ≥ *main_hu*  (standard segmentation)
          • Voxels ≥ *vessel_hu* that are within *dilation_mm* of the
            primary mask  (thin-vessel recovery)

        Requires SimpleITK (already a project dependency).
        Falls back to standard threshold if SimpleITK is unavailable.
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            logger.warning("SimpleITK not found — dual threshold skipped")
            return (volume >= main_hu).astype(np.float32)

        main_mask   = (volume >= main_hu).astype(np.uint8)
        vessel_mask = (volume >= vessel_hu).astype(np.uint8)

        sz, sy, sx = spacing
        sitk_main  = sitk.GetImageFromArray(main_mask)
        sitk_main.SetSpacing((float(sx), float(sy), float(sz)))   # ITK: x,y,z

        radius  = max(1, round(dilation_mm / min(float(sz), float(sy), float(sx))))
        dilated = sitk.BinaryDilate(sitk_main, [radius, radius, radius])
        dil_np  = sitk.GetArrayFromImage(dilated).astype(bool)

        combined = main_mask.astype(bool) | (vessel_mask.astype(bool) & dil_np)

        n_extra = int(combined.sum()) - int(main_mask.sum())
        logger.info(
            "Dual-threshold: +%d voxels recovered (vessel_hu=%.0f, dilation=%.1f mm)",
            max(n_extra, 0), vessel_hu, dilation_mm,
        )
        return combined.astype(np.float32)

    @staticmethod
    def _numpy_gaussian(volume: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian blur on the numpy HU volume (used before masking)."""
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(volume.astype(np.float32), sigma=sigma)
        except ImportError:
            pass
        try:
            import SimpleITK as sitk
            img = sitk.GetImageFromArray(volume.astype(np.float32))
            blurred = sitk.SmoothingRecursiveGaussian(img, sigma)
            return sitk.GetArrayFromImage(blurred)
        except Exception:
            pass
        logger.warning("No Gaussian library found — skipping pre-smooth for dual-threshold")
        return volume.astype(np.float32)

    # ------------------------------------------------------------------ #
    # VTK image conversion                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_vtk_image(
        volume: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> vtk.vtkImageData:
        z, y, x = volume.shape
        sz, sy, sx = spacing

        img = vtk.vtkImageData()
        img.SetDimensions(x, y, z)
        img.SetSpacing(sx, sy, sz)
        img.SetOrigin(0.0, 0.0, 0.0)

        flat = np.ascontiguousarray(volume, dtype=np.float32).ravel()
        arr  = ns.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        arr.SetName("HU")
        img.GetPointData().SetScalars(arr)
        return img


# ============================================================================ #
# S-3 — Grow-from-seeds pipeline (SimpleITK ConnectedThreshold)                #
# ============================================================================ #

@dataclass
class GrowResult:
    """Result from GrowSegmentationPipeline.run()."""
    poly_data:          vtk.vtkPolyData
    n_vertices:         int
    n_triangles:        int
    lower_hu:           float
    upper_hu:           float
    seeds:              list[tuple[int, int, int]] = field(default_factory=list)
    n_voxels:           int = 0   # voxels in the segmented region
    n_fragments_removed: int = 0  # disconnected components removed by post-grow filter


class GrowSegmentationPipeline:
    """
    Region-growing segmentation using SimpleITK ConnectedThreshold.

    The algorithm starts from user-provided seed voxels and expands
    into all connected voxels whose HU value lies in [lower_hu, upper_hu].
    The resulting binary mask is then converted to a mesh via Marching Cubes.

    Parameters
    ----------
    lower_hu, upper_hu:
        HU range for the connected-threshold filter.
        Typical vessel values: lower=80, upper=600 (CTA with contrast).
    smooth_iterations:
        WindowedSinc iterations on the output mesh (0 = skip).
    target_reduction:
        Fraction of triangles to remove [0–0.99].
    keep_top_n:
        After ConnectedThreshold, keep only the N largest connected components
        in the binary mask before building the mesh.  0 = keep all.
        Recommended: 1–3 for vascular segmentation (grow should produce
        one connected tree; using 1 removes any leaked satellite regions).
    morpho_closing_mm:
        Morphological closing radius in mm applied to the grow mask before
        Marching Cubes.  Fills thin gaps that break vessel continuity.
        0 = disabled.  0.5–1.0 mm recommended when thin-vessel connectivity
        is lost between adjacent slices.
    """

    def __init__(
        self,
        lower_hu:          float = 80.0,
        upper_hu:          float = 600.0,
        smooth_iterations: int   = 15,
        smooth_pass_band:  float = 0.10,
        target_reduction:  float = 0.70,
        keep_top_n:        int   = 0,
        morpho_closing_mm: float = 0.0,
    ) -> None:
        self.lower_hu          = lower_hu
        self.upper_hu          = upper_hu
        self.smooth_iterations = smooth_iterations
        self.smooth_pass_band  = smooth_pass_band
        self.target_reduction  = target_reduction
        self.keep_top_n        = keep_top_n
        self.morpho_closing_mm = morpho_closing_mm

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        volume:  np.ndarray,
        spacing: tuple[float, float, float],
        seeds:   list[tuple[int, int, int]],
    ) -> GrowResult:
        """
        Execute the grow-from-seeds pipeline.

        Parameters
        ----------
        volume:   float32 / int16 array shape (Z, Y, X) in HU.
        spacing:  (sz, sy, sx) in mm.
        seeds:    list of (z, y, x) voxel coordinates — at least one required.

        Returns
        -------
        GrowResult with extracted mesh.
        """
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise RuntimeError(
                "SimpleITK is required for grow-from-seeds segmentation. "
                "Install it with: pip install SimpleITK"
            ) from exc

        if not seeds:
            raise ValueError("At least one seed voxel is required.")

        logger.info(
            "Grow-from-seeds: lower=%.0f  upper=%.0f  seeds=%s  shape=%s",
            self.lower_hu, self.upper_hu, seeds, volume.shape,
        )

        nz, ny, nx = volume.shape

        # ── 1. Convert numpy → SimpleITK ──────────────────────────────── #
        vol_f32 = np.ascontiguousarray(volume, dtype=np.float32)
        # sitk.GetImageFromArray expects (Z, Y, X) → result: (X, Y, Z) ITK image
        sitk_img = sitk.GetImageFromArray(vol_f32)
        sz, sy, sx = spacing
        sitk_img.SetSpacing((float(sx), float(sy), float(sz)))   # ITK spacing is (x, y, z)

        # ── 2. ConnectedThreshold ─────────────────────────────────────── #
        # ITK seed format: (x, y, z) in voxel indices
        sitk_seeds = [(int(x), int(y), int(z)) for z, y, x in seeds]

        seg = sitk.ConnectedThreshold(
            sitk_img,
            seedList   = sitk_seeds,
            lower      = float(self.lower_hu),
            upper      = float(self.upper_hu),
            replaceValue = 1,
        )
        # seg is uint8 binary mask; convert back to numpy (Z, Y, X)
        mask = sitk.GetArrayFromImage(seg).astype(np.uint8)
        n_voxels = int(mask.sum())
        logger.info("ConnectedThreshold: %d voxels segmented", n_voxels)

        if n_voxels == 0:
            raise ValueError(
                f"No voxels found in HU range [{self.lower_hu:.0f}, {self.upper_hu:.0f}] "
                "connected to the seed(s). Try adjusting the HU range or seed position."
            )

        # ── 2b. Morphological closing (optional — fills thin-vessel gaps) ── #
        mask_f = mask.astype(np.float32)
        if self.morpho_closing_mm > 0.0:
            mask_f = SegmentationPipeline._morpho_closing(
                mask_f, spacing, self.morpho_closing_mm
            )
            logger.info(
                "Post-grow morpho closing: %.1f mm", self.morpho_closing_mm
            )

        # ── 2c. Component filter (optional — discard satellite fragments) ── #
        # ConnectedThreshold grows only CONNECTED voxels, so isolated noise
        # cannot appear.  However, voxels connected through thin bridges
        # (partial-volume artifacts along bone edges) can pull in satellite
        # regions.  keep_top_n=1 keeps only the dominant vascular tree.
        n_fragments_removed = 0
        if self.keep_top_n > 0:
            mask_f, n_fragments_removed = SegmentationPipeline._filter_mask_components(
                mask_f, 0, self.keep_top_n
            )
            if n_fragments_removed > 0:
                logger.info(
                    "Post-grow component filter: removed %d satellite fragment(s) "
                    "(kept top-%d)",
                    n_fragments_removed, self.keep_top_n,
                )
        else:
            mask_f = mask_f  # already float32

        n_voxels = int(mask_f.sum())

        # ── 3. Convert binary mask → vtkImageData ─────────────────────── #
        img = vtk.vtkImageData()
        img.SetDimensions(nx, ny, nz)
        img.SetSpacing(float(sx), float(sy), float(sz))
        img.SetOrigin(0.0, 0.0, 0.0)

        flat = np.ascontiguousarray(mask_f, dtype=np.float32).ravel(order="C")
        arr  = ns.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        arr.SetName("mask")
        img.GetPointData().SetScalars(arr)

        # ── 4. Marching Cubes at iso = 0.5 ────────────────────────────── #
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(img)
        mc.SetValue(0, 0.5)
        mc.ComputeNormalsOff()
        mc.ComputeGradientsOff()
        mc.Update()

        n_raw = mc.GetOutput().GetNumberOfPolys()
        logger.info("Marching Cubes on mask: %d triangles", n_raw)

        if n_raw == 0:
            raise ValueError(
                "Marching Cubes produced no triangles from the segmented mask. "
                "The region may be too small or consist of isolated voxels."
            )

        # ── 5. Mesh smoothing ─────────────────────────────────────────── #
        if self.smooth_iterations > 0:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(mc.GetOutputPort())
            smoother.SetNumberOfIterations(self.smooth_iterations)
            smoother.SetPassBand(self.smooth_pass_band)
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()
            prev_port = smoother.GetOutputPort()
        else:
            prev_port = mc.GetOutputPort()

        # ── 6. Decimation ─────────────────────────────────────────────── #
        if self.target_reduction > 0:
            decimate = vtk.vtkQuadricDecimation()
            decimate.SetInputConnection(prev_port)
            decimate.SetTargetReduction(self.target_reduction)
            decimate.Update()
            prev_port = decimate.GetOutputPort()

        # ── 7. Normals for smooth shading ─────────────────────────────── #
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(prev_port)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()

        poly    = normals.GetOutput()
        n_verts = poly.GetNumberOfPoints()
        n_tris  = poly.GetNumberOfPolys()

        logger.info(
            "Grow-from-seeds done — %d verts, %d tris, %d voxels, %d fragments removed",
            n_verts, n_tris, n_voxels, n_fragments_removed,
        )

        return GrowResult(
            poly_data           = poly,
            n_vertices          = n_verts,
            n_triangles         = n_tris,
            lower_hu            = self.lower_hu,
            upper_hu            = self.upper_hu,
            seeds               = list(seeds),
            n_voxels            = n_voxels,
            n_fragments_removed = n_fragments_removed,
        )
