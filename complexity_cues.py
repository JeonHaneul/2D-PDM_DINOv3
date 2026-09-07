"""Target-independent, image-space visible-clutter supervision and depth cues.

The saved scene segmentation groups pixels by asset-name colour, not durable
physical instance ID. Current scenes spawn each asset once. Repeated copies of
the same asset would therefore be counted once, even if disconnected on screen.
These labels describe visible clutter only, never hidden object counts.

All windows are pixel squares centred on patch centres. Count / 16 is a fixed
bounded count score, not a probability or a cross-scale physical density.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DEFAULT_WINDOWS = (48, 96, 160)
COUNT_CAPACITY = 16.0


def depth_channel_names(windows=DEFAULT_WINDOWS):
    return (
        "normalized_depth", "foreground_occupancy", "depth_validity",
        *(f"plane_residual_rms_{w}px" for w in windows),
        *(f"gradient_variation_{w}px" for w in windows),
    )


FEATURES_CHANNEL_NAMES = depth_channel_names()
DEPTH_CHANNEL_NAMES = FEATURES_CHANNEL_NAMES
DEPTH_FEATURE_CHANNELS = len(FEATURES_CHANNEL_NAMES)


def _inputs(workspace_mask, windows, patch_size):
    raw = np.asarray(workspace_mask)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("workspace_mask must be a finite H x W array")
    if patch_size <= 0 or int(patch_size) != patch_size:
        raise ValueError("patch_size must be a positive integer")
    if any(s % patch_size for s in raw.shape):
        raise ValueError("image dimensions must be divisible by patch_size")
    windows = tuple(windows)
    if not windows or any(w <= 0 or int(w) != w or w % 2 for w in windows):
        raise ValueError("windows must contain positive even integer pixel widths")
    return raw > 0, windows


def _pool(a, patch_size):
    h, w = a.shape
    return np.asarray(a, dtype=np.float64).reshape(
        h // patch_size, patch_size, w // patch_size, patch_size
    ).mean(axis=(1, 3))


def _window_sums(a, windows, patch_size):
    """Zero-padding outside the image, with an exact even-width window."""
    h, w = a.shape
    integral = np.pad(np.asarray(a, np.float64), ((1, 0), (1, 0)))
    integral = integral.cumsum(0).cumsum(1)
    cy = np.arange(patch_size // 2, h, patch_size)
    cx = np.arange(patch_size // 2, w, patch_size)
    out = []
    for size in windows:
        y0, y1 = np.clip(cy - size // 2, 0, h), np.clip(cy + size // 2, 0, h)
        x0, x1 = np.clip(cx - size // 2, 0, w), np.clip(cx + size // 2, 0, w)
        out.append(integral[y1[:, None], x1] - integral[y0[:, None], x1]
                   - integral[y1[:, None], x0] + integral[y0[:, None], x0])
    return np.stack(out)


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
                     where=denominator > 0)


def _colour_groups(mapping):
    groups = {}
    legacy = 0
    for name, entry in mapping.items():
        if str(name).upper() in {"BACKGROUND", "UNLABELLED", "UNLABELED"}:
            continue
        if isinstance(entry, Mapping):
            if "color_bgr" in entry:
                colour = entry["color_bgr"]
            elif "legacy_color" in entry:
                colour = entry["legacy_color"]
            elif "color_rgb" in entry:
                # 260714 legacy field is misnamed: values are already OpenCV BGR.
                colour = entry["color_rgb"]
                legacy += 1
            else:
                raise ValueError(f"mapping entry {name!r} has no supported colour field")
        else:
            colour = entry  # already-resolved BGR triples, as in gt_similarity.
        colour = np.asarray(colour)
        if (colour.shape != (3,) or not np.isfinite(colour).all()
                or (colour < 0).any() or (colour > 255).any()
                or (colour != np.floor(colour)).any()):
            raise ValueError(f"mapping entry {name!r} must contain three uint8 colours")
        key = tuple(int(v) for v in colour)
        if key == (0, 0, 0):
            raise ValueError("black is reserved for background, not a countable label")
        groups.setdefault(key, []).append(str(name))
    return groups, legacy


def compute_supervision(seg_bgr, mapping, workspace_mask, windows=DEFAULT_WINDOWS, *,
                        patch_size=16, capacity=COUNT_CAPACITY, min_visible_area=32,
                        min_window_pixels=16, min_workspace_fraction=0.95):
    """Visible asset-label counts, occupancy and explicit validity masks.

    A colour needs >=32 visible pixels in the full workspace; each window counts
    it only when >=16 of those pixels intersect that window (defaults). Same-
    colour aliases and disconnected fragments count once. Density windows need
    >=95% workspace coverage of their FULL square, including off-image area.
    Unknown nonblack colours invalidate intersecting density windows; occupancy
    supervision excludes their pixels via ``occupancy_valid``. Small known
    labels remain foreground for occupancy, although omitted from count GT.
    """
    workspace, windows = _inputs(workspace_mask, windows, patch_size)
    seg = np.asarray(seg_bgr)
    if seg.shape != (*workspace.shape, 3) or seg.dtype != np.uint8:
        raise ValueError("seg_bgr must be a uint8 H x W x 3 OpenCV image")
    if (not np.isfinite(capacity) or capacity <= 0 or min_visible_area < 1
            or min_window_pixels < 1 or not 0 < min_workspace_fraction <= 1):
        raise ValueError("invalid count capacity, thresholds or workspace fraction")
    groups, legacy = _colour_groups(mapping)
    spatial = (len(windows), *(s // patch_size for s in workspace.shape))
    counts = np.zeros(spatial, np.float64)
    foreground = np.zeros_like(workspace)
    areas = {}
    eligible = []
    for colour, names in groups.items():
        pixels = np.all(seg == colour, axis=-1) & workspace
        foreground |= pixels
        area = int(pixels.sum())
        areas["|".join(names)] = area
        if area >= min_visible_area:
            eligible.append(names)
            counts += _window_sums(pixels, windows, patch_size) >= min_window_pixels
    unknown = workspace & np.any(seg != 0, axis=-1) & ~foreground
    known_workspace = workspace & ~unknown
    workspace_fraction = _pool(workspace, patch_size)
    occupancy_valid = _pool(known_workspace, patch_size)
    window_workspace = _window_sums(workspace, windows, patch_size)
    full_areas = np.square(np.asarray(windows, np.float64))[:, None, None]
    density_valid = ((window_workspace / full_areas >= min_workspace_fraction)
                     & (_window_sums(unknown, windows, patch_size) == 0))
    clip_fraction = [float(np.mean(c[v] > capacity)) if v.any() else 0.0
                     for c, v in zip(counts, density_valid)]
    summary = {
        "definition": "visible_asset_label_count_divided_by_fixed_capacity",
        "not_hidden_instance_count": True,
        "windows_pixels": list(windows), "capacity": float(capacity),
        "min_visible_area_pixels": int(min_visible_area),
        "min_window_intersection_pixels": int(min_window_pixels),
        "min_workspace_fraction": float(min_workspace_fraction),
        "mapping_colour_groups": len(groups), "mapping_names": len(mapping),
        "legacy_color_rgb_as_bgr_entries": legacy,
        "visible_area_pixels_by_label": areas, "eligible_label_groups": eligible,
        "unknown_nonblack_pixels": int(unknown.sum()),
        "workspace_pixels": int(workspace.sum()),
        "clipping_fraction_by_window": clip_fraction,
        "max_raw_count_by_window": [float(c.max()) for c in counts],
        "valid_window_count": [int(v.sum()) for v in density_valid],
    }
    return {
        "density": np.clip(counts / capacity, 0, 1).astype(np.float32),
        "density_valid": density_valid.astype(np.float32),
        "occupancy": _ratio(_pool(foreground, patch_size), occupancy_valid)[None].astype(np.float32),
        "occupancy_valid": occupancy_valid[None].astype(np.float32),
        "workspace": workspace_fraction[None].astype(np.float32),
        "summary": summary,
    }


def depth_geometry(depth, empty_depth, workspace_mask, windows=DEFAULT_WINDOWS, *,
                   patch_size=16, foreground_threshold_m=0.015,
                   roughness_scale_m=0.03, gradient_scale_m=0.02,
                   depth_range_m=(2.5, 3.5), min_window_valid_fraction=0.25,
                   roughness_reference="scene"):
    """Deterministic depth-only cues; no segmentation or target inputs.

    Empty depth must share the scene camera calibration. Positive finite scene
    AND empty depths define validity. Foreground is empty_depth-depth >15 mm.
    ``roughness_reference='scene'`` preserves the original raw-depth cues.
    ``'empty_difference'`` applies both geometric cues to the signed ray-depth
    displacement empty_depth-depth, removing fixed empty-drawer geometry. No
    clipping of negative displacement is applied before those computations.
    Roughness is RMS residual after least-squares affine-plane fitting in each
    image-space window; gradient variation is sqrt(var(dx)+var(dy)), using valid
    adjacent pairs. Both remove constant image-space slope in the chosen signal.
    They are not metric surface curvature or hidden-object estimates. Normalized
    scene depth, occupancy and validity are identical in both reference modes.
    """
    workspace, windows = _inputs(workspace_mask, windows, patch_size)
    depth, empty = np.asarray(depth, np.float64), np.asarray(empty_depth, np.float64)
    if depth.shape != workspace.shape or empty.shape != workspace.shape:
        raise ValueError("depth, empty_depth and workspace must have matching H x W shapes")
    if roughness_reference not in {"scene", "empty_difference"}:
        raise ValueError("roughness_reference must be 'scene' or 'empty_difference'")
    low, high = depth_range_m
    if (not np.isfinite([low, high, foreground_threshold_m, roughness_scale_m,
                         gradient_scale_m, min_window_valid_fraction]).all()
            or high <= low or foreground_threshold_m < 0 or roughness_scale_m <= 0
            or gradient_scale_m <= 0 or not 0 < min_window_valid_fraction <= 1):
        raise ValueError("invalid fixed depth normalization parameters")
    valid = workspace & np.isfinite(depth) & (depth > 0) & np.isfinite(empty) & (empty > 0)
    safe_depth = np.where(valid, depth, 0.0)
    delta = np.where(valid, empty, 0.0) - safe_depth
    foreground = valid & (delta > foreground_threshold_m)
    valid_patch = _pool(valid, patch_size)
    occupancy = _ratio(_pool(foreground, patch_size), valid_patch)
    mean_depth = _ratio(_pool(safe_depth, patch_size), valid_patch)
    normalized_depth = np.where(valid_patch > 0, np.clip((mean_depth - low) / (high - low), 0, 1), 0)
    # Subtracting the calibrated empty image removes fixed walls and curved floor.
    signal = safe_depth if roughness_reference == "scene" else delta
    # Centred signals and bounded coordinates avoid cancellation from metric z~3m.
    centre = float(np.median(signal[valid])) if valid.any() else 0.0
    z = np.where(valid, signal - centre, 0.0)
    y, x = np.indices(depth.shape, dtype=np.float64)
    x /= depth.shape[1]
    y /= depth.shape[0]
    n = _window_sums(valid, windows, patch_size)
    def moment(value):
        return _ratio(_window_sums(np.where(valid, value, 0), windows, patch_size), n)
    mx, my, mz = moment(x), moment(y), moment(z)
    xx, yy = np.maximum(moment(x*x) - mx*mx, 0), np.maximum(moment(y*y) - my*my, 0)
    xy, xz, yz = moment(x*y) - mx*my, moment(x*z) - mx*mz, moment(y*z) - my*mz
    zz = np.maximum(moment(z*z) - mz*mz, 0)
    determinant = xx*yy - xy*xy
    slope_x = _ratio(xz*yy - yz*xy, determinant)
    slope_y = _ratio(yz*xx - xz*xy, determinant)
    residual = np.sqrt(np.maximum(zz - slope_x*xz - slope_y*yz, 0))
    full_areas = np.square(np.asarray(windows, np.float64))[:, None, None]
    window_valid = (n / full_areas >= min_window_valid_fraction) & (n >= 6) & (determinant > 1e-12)
    residual = np.where(window_valid, residual, 0)
    gradient_variance = np.zeros_like(n)
    for axis in (0, 1):
        pair_valid = np.zeros_like(valid)
        gradient = np.zeros_like(depth)
        if axis == 0:
            pair_valid[:-1] = valid[:-1] & valid[1:]
            gradient[:-1] = signal[1:] - signal[:-1]
        else:
            pair_valid[:, :-1] = valid[:, :-1] & valid[:, 1:]
            gradient[:, :-1] = signal[:, 1:] - signal[:, :-1]
        gradient = np.where(pair_valid, gradient, 0)
        pairs = _window_sums(pair_valid, windows, patch_size)
        mean = _ratio(_window_sums(gradient, windows, patch_size), pairs)
        square = _ratio(_window_sums(gradient*gradient, windows, patch_size), pairs)
        gradient_variance += np.maximum(square - mean*mean, 0)
    variation = np.where(window_valid, np.sqrt(gradient_variance), 0)
    features = np.concatenate((normalized_depth[None], occupancy[None], valid_patch[None],
                               np.clip(residual / roughness_scale_m, 0, 1),
                               np.clip(variation / gradient_scale_m, 0, 1)))
    # Neighbourhood cues should not create input support outside the workspace.
    features *= (_pool(workspace, patch_size) > 0)[None]
    return {
        "features": features.astype(np.float32),
        "direct_occupancy": occupancy[None].astype(np.float32),
        "validity": valid_patch[None].astype(np.float32),
        "summary": {
            "channel_names": list(depth_channel_names(windows)),
            "roughness_reference": roughness_reference,
            "windows_pixels": list(windows), "valid_pixels": int(valid.sum()),
            "workspace_pixels": int(workspace.sum()),
            "foreground_threshold_m": foreground_threshold_m,
            "roughness_scale_m": roughness_scale_m, "gradient_scale_m": gradient_scale_m,
            "depth_range_m": list(depth_range_m),
            "min_window_valid_fraction": min_window_valid_fraction,
            "plane_residual_max_m": [float(r.max()) for r in residual],
            "gradient_variation_max_m": [float(v.max()) for v in variation],
        },
    }
