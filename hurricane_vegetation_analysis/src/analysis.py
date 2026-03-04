"""
analysis.py
===========
Scientific core of the hurricane vegetation impact pipeline.

Implements:

a. **Difference Map**
   Pixel-wise delta (post − pre) and percentage change images on GEE.

b. **Zonal Statistics**
   Per-feature reduction for multi-polygon ROIs using ``ee.Image.reduceRegions()``.

c. **Statistical Significance Testing** (local, on sampled pixels)
   - Paired t-test (parametric)
   - Wilcoxon signed-rank test (non-parametric, preferred for NDVI)
   - Cohen's d effect size
   - Plain-language conclusion with confidence intervals

d. **Impact Classification**
   Pixel-level severity classes (No Impact / Low / Moderate / Severe).

e. **Baseline Variability Check**
   Compares the observed delta against historical same-season variability
   from 2–3 prior hurricane-free years to detect false positives.

All heavy geospatial operations run server-side on GEE.
Only sampled pixel values and aggregated statistics are pulled locally.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .utils import InsufficientDataError, date_windows, historical_date_windows

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (overridden by config.yaml values)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "no_impact": -0.05,      # delta > -0.05
    "low_impact": -0.15,     # -0.05 to -0.15
    "moderate_impact": -0.30,  # -0.15 to -0.30
    # severe: < -0.30
}

CLASS_LABELS = {
    0: "No Impact",
    1: "Low Impact",
    2: "Moderate Impact",
    3: "Severe Impact",
}

CLASS_COLORS = {
    0: "#2ecc71",  # green
    1: "#f1c40f",  # yellow
    2: "#e67e22",  # orange
    3: "#c0392b",  # red
}


# ---------------------------------------------------------------------------
# a. Difference Map
# ---------------------------------------------------------------------------

def compute_difference(
    pre_img: "ee.Image",
    post_img: "ee.Image",
    index: str,
) -> "ee.Image":
    """
    Compute pixel-wise delta and percentage change between post and pre composites.

    Parameters
    ----------
    pre_img : ee.Image
        Pre-event composite with the vegetation index band present.
    post_img : ee.Image
        Post-event composite with the vegetation index band present.
    index : str
        Name of the index band (e.g. ``"NDVI"``).

    Returns
    -------
    ee.Image
        Image with two bands:

        - ``"delta"``      — absolute change (post − pre).
        - ``"pct_change"`` — relative change in percent ((delta / pre) × 100).

    Notes
    -----
    Division by near-zero pre values (|pre| < 0.001) is masked to avoid
    infinite percentage changes over bare soil or water.
    """
    import ee

    pre_band = pre_img.select(index)
    post_band = post_img.select(index)

    delta = post_band.subtract(pre_band).rename("delta")

    # Mask near-zero denominators
    valid_pre = pre_band.abs().gt(0.001)
    pct_change = (
        delta.divide(pre_band)
        .multiply(100)
        .updateMask(valid_pre)
        .rename("pct_change")
    )

    return delta.addBands(pct_change)


def export_geotiff(
    image: "ee.Image",
    roi: "ee.Geometry",
    output_path: str,
    scale: int = 100,
    crs: str = "EPSG:4326",
    max_pixels: int = 1_000_000,
) -> None:
    """
    Export a GEE image as a GeoTIFF to a local file path using ``geemap``.

    The scale is automatically increased if the ROI is too large for GEE's
    50 MB direct-download limit.  The function computes the minimum safe
    resolution from the ROI area and the ``max_pixels`` budget, then uses
    whichever is coarser (the caller's ``scale`` or the computed floor).

    Parameters
    ----------
    image : ee.Image
        Image to export (e.g. the difference image).
    roi : ee.Geometry
        Clipping/export geometry.
    output_path : str
        Full local path for the output ``.tif`` file.
    scale : int
        Desired spatial resolution in metres.  Pass the native sensor
        resolution (10 for Sentinel-2, 30 for Landsat).  The function
        coarsens automatically if the ROI is too large for direct download.
    crs : str
        Coordinate reference system (default ``"EPSG:4326"``).
    max_pixels : int
        Maximum pixel count used to derive the minimum safe scale.
        Honoured together with GEE's 50 MB byte limit; the stricter of
        the two constraints wins.  Default 1 000 000 is conservative;
        raise to 6 000 000 for large-ROI exports.

    Notes
    -----
    GEE's ``getDownloadURL`` endpoint enforces a 50 MB (50 331 648 byte) cap
    per request.  For single-float-band images, that allows roughly
    12 500 000 pixels.  For pixel-level precision on very large ROIs, use
    ``ee.batch.Export.image.toDrive()`` instead.
    """
    try:
        import geemap
    except ImportError as exc:
        raise ImportError(
            "geemap is required for GeoTIFF export. "
            "Install it with: pip install geemap"
        ) from exc

    import math

    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Derive the minimum safe scale from the ROI area and pixel budget.
    #
    # GEE hard limit: ~50 MB per download request.
    # Assuming a single float32 band (4 bytes/pixel):
    #   max_gee_pixels = 50_000_000 / 4 = 12_500_000
    # We take the stricter of max_pixels and the GEE limit, then solve:
    #   min_scale = ceil(sqrt(area_m2 / max_safe_pixels))
    GEE_BYTE_LIMIT = 50_000_000
    BYTES_PER_PIXEL = 4  # float32
    gee_pixel_limit = GEE_BYTE_LIMIT // BYTES_PER_PIXEL   # ~12.5 M for 1 band
    max_safe_pixels = min(max_pixels, gee_pixel_limit)

    try:
        area_m2 = roi.area(maxError=1000).getInfo()
        min_safe_scale = math.ceil(math.sqrt(area_m2 / max_safe_pixels))
        scale = max(scale, min_safe_scale)
        logger.info(
            "ROI area ≈ %.1f km²; exporting at %d m resolution.",
            area_m2 / 1e6,
            scale,
        )
    except Exception as exc:
        scale = max(scale, 250)  # conservative fallback if area query fails
        logger.warning(
            "Could not estimate ROI area (%s); using fallback scale = %d m.",
            exc, scale,
        )

    logger.info("Exporting GeoTIFF → %s (scale=%d m)", output_path, scale)
    geemap.ee_export_image(
        image,
        filename=output_path,
        scale=scale,
        region=roi,
        crs=crs,
        file_per_band=False,
    )

    if os.path.exists(output_path):
        logger.info("GeoTIFF export complete at %d m resolution.", scale)
    else:
        raise RuntimeError(
            f"GeoTIFF export silently failed at {scale} m — file not created. "
            "Check the terminal output above for the GEE error message."
        )


# ---------------------------------------------------------------------------
# b. Zonal Statistics
# ---------------------------------------------------------------------------

def compute_zonal_stats(
    delta_img: "ee.Image",
    pre_img: "ee.Image",
    post_img: "ee.Image",
    index: str,
    features: "ee.FeatureCollection",
    scale: int = 30,
) -> pd.DataFrame:
    """
    Compute per-feature statistics for pre, post, and delta index values.

    Parameters
    ----------
    delta_img : ee.Image
        Difference image produced by :func:`compute_difference`.
    pre_img : ee.Image
        Pre-event composite with index band.
    post_img : ee.Image
        Post-event composite with index band.
    index : str
        Name of the index band.
    features : ee.FeatureCollection
        Feature collection defining zones (counties, parcels, etc.).
    scale : int
        Spatial resolution in metres for the reduction.

    Returns
    -------
    pd.DataFrame
        Table with columns: ``feature_id``, ``pre_mean``, ``pre_std``,
        ``post_mean``, ``post_std``, ``delta_mean``, ``delta_std``,
        ``delta_min``, ``delta_max``.
    """
    import ee

    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
    )

    def _reduce(img: "ee.Image", prefix: str) -> "ee.FeatureCollection":
        return img.select(index).reduceRegions(
            collection=features,
            reducer=reducer,
            scale=scale,
        )

    pre_stats = _reduce(pre_img, "pre")
    post_stats = _reduce(post_img, "post")
    delta_stats = _reduce(delta_img.select("delta"), "delta")

    # Convert to DataFrames
    def _to_df(fc: "ee.FeatureCollection", prefix: str) -> pd.DataFrame:
        rows = fc.getInfo()["features"]
        records = []
        for row in rows:
            props = row["properties"]
            records.append({
                "feature_id": props.get("system:index", props.get("id", "unknown")),
                f"{prefix}_mean": props.get("mean"),
                f"{prefix}_std": props.get("stdDev"),
                f"{prefix}_min": props.get("min"),
                f"{prefix}_max": props.get("max"),
                f"{prefix}_median": props.get("median"),
            })
        return pd.DataFrame(records)

    pre_df = _to_df(pre_stats, "pre")
    post_df = _to_df(post_stats, "post")
    delta_df = _to_df(delta_stats, "delta")

    merged = pre_df.merge(post_df, on="feature_id").merge(delta_df, on="feature_id")
    logger.info("Zonal statistics computed for %d features.", len(merged))
    return merged


# ---------------------------------------------------------------------------
# c. Statistical significance testing
# ---------------------------------------------------------------------------

def sample_pixels(
    pre_img: "ee.Image",
    post_img: "ee.Image",
    index: str,
    roi: "ee.Geometry",
    n: int = 500,
    scale: int = 30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample matched pixel values from pre and post composites.

    Pixels are sampled at the same locations so that a paired statistical test
    can be applied.  Only pixels valid in **both** composites are included.

    Parameters
    ----------
    pre_img : ee.Image
        Pre-event composite with index band.
    post_img : ee.Image
        Post-event composite with index band.
    index : str
        Name of the index band (e.g. ``"NDVI"``).
    roi : ee.Geometry
        Sampling region.
    n : int
        Number of pixels to sample (default 500).
    scale : int
        Pixel size in metres for the sampler.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        ``(pre_values, post_values)`` — paired 1-D float arrays.

    Raises
    ------
    InsufficientDataError
        If fewer than 10 valid pixels are returned after masking.
    """
    import ee

    combined = pre_img.select(index).rename("pre").addBands(
        post_img.select(index).rename("post")
    )

    sample_fc = combined.sample(
        region=roi,
        scale=scale,
        numPixels=n,
        seed=seed,
        dropNulls=True,
    )

    data = sample_fc.getInfo()
    features = data.get("features", [])
    if len(features) < 10:
        raise InsufficientDataError(
            f"Only {len(features)} valid pixels sampled (need ≥ 10). "
            "Try a larger ROI or wider date window."
        )

    pre_vals = np.array([f["properties"]["pre"] for f in features], dtype=float)
    post_vals = np.array([f["properties"]["post"] for f in features], dtype=float)

    # Remove NaN pairs
    valid = ~(np.isnan(pre_vals) | np.isnan(post_vals))
    pre_vals = pre_vals[valid]
    post_vals = post_vals[valid]

    logger.info("Sampled %d valid pixel pairs for statistical testing.", len(pre_vals))
    return pre_vals, post_vals


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for the difference between two paired arrays.

    Cohen's d = (mean_b − mean_a) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large

    Parameters
    ----------
    a : np.ndarray
        Pre-event values.
    b : np.ndarray
        Post-event values.

    Returns
    -------
    float
        Cohen's d (positive = increase, negative = decrease).
    """
    diff = b - a
    pooled_std = np.std(diff, ddof=1)
    if pooled_std == 0:
        return 0.0
    return float(np.mean(diff) / pooled_std)


def _effect_size_label(d: float) -> str:
    """Return a human-readable effect size label for Cohen's d."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def run_statistical_tests(
    pre_vals: np.ndarray,
    post_vals: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Run paired t-test, Wilcoxon signed-rank test, and compute Cohen's d on
    sampled pixel pairs.

    The Wilcoxon test is the preferred primary result because NDVI distributions
    are typically non-normal.

    Parameters
    ----------
    pre_vals : np.ndarray
        Pre-event index values (1-D float array).
    post_vals : np.ndarray
        Post-event index values, paired with ``pre_vals``.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    dict
        Keys:

        - ``n``                   — number of pixel pairs
        - ``pre_mean``            — mean pre-event value
        - ``post_mean``           — mean post-event value
        - ``delta_mean``          — mean change (post − pre)
        - ``delta_pct``           — percentage change relative to pre mean
        - ``ttest_stat``          — paired t-test statistic
        - ``ttest_pvalue``        — p-value from paired t-test
        - ``ttest_ci``            — 95 % confidence interval of the mean diff
        - ``wilcoxon_stat``       — Wilcoxon test statistic
        - ``wilcoxon_pvalue``     — p-value from Wilcoxon test
        - ``cohens_d``            — Cohen's d effect size
        - ``effect_label``        — text description of effect size
        - ``significant``         — bool, based on Wilcoxon p < alpha
        - ``conclusion``          — plain-language summary string
    """
    n = len(pre_vals)
    if n < 2:
        raise InsufficientDataError(f"Need at least 2 pixel pairs; got {n}.")

    diff = post_vals - pre_vals
    pre_mean = float(np.mean(pre_vals))
    post_mean = float(np.mean(post_vals))
    delta_mean = float(np.mean(diff))

    # Percentage change
    if abs(pre_mean) > 1e-6:
        delta_pct = (delta_mean / abs(pre_mean)) * 100.0
    else:
        delta_pct = float("nan")

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(pre_vals, post_vals)
    # 95% CI of the mean difference
    se = stats.sem(diff)
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=delta_mean, scale=se)

    # Wilcoxon signed-rank test
    try:
        w_stat, w_pval = stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        # All differences are zero (no change at all)
        w_stat, w_pval = 0.0, 1.0

    # Effect size
    d = cohens_d(pre_vals, post_vals)
    effect_label = _effect_size_label(d)

    # Primary decision: Wilcoxon p-value
    significant = w_pval < alpha

    # Build plain-language conclusion
    if significant:
        direction = "decline" if delta_mean < 0 else "increase"
        pct_str = f"{delta_pct:+.1f}%" if not np.isnan(delta_pct) else "N/A"
        conclusion = (
            f"Statistically significant vegetation {direction} detected "
            f"(Wilcoxon p = {w_pval:.3g}, Cohen's d = {d:.2f}, {effect_label} effect). "
            f"Mean index changed from {pre_mean:.3f} to {post_mean:.3f} ({pct_str})."
        )
    else:
        conclusion = (
            f"No statistically significant change detected "
            f"(Wilcoxon p = {w_pval:.3f}). "
            f"Vegetation appears stable in this region following the event."
        )

    return {
        "n": n,
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "delta_mean": delta_mean,
        "delta_pct": delta_pct,
        "ttest_stat": float(t_stat),
        "ttest_pvalue": float(t_pval),
        "ttest_ci": (float(ci_low), float(ci_high)),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "cohens_d": d,
        "effect_label": effect_label,
        "significant": significant,
        "alpha": alpha,
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# d. Impact Classification
# ---------------------------------------------------------------------------

def classify_impact(
    delta_img: "ee.Image",
    thresholds: Optional[Dict[str, float]] = None,
) -> "ee.Image":
    """
    Classify each pixel in the delta image into four impact severity categories.

    Classification is based on the ``"delta"`` band of the difference image:

    ============  =============  ========================
    Class         Delta range    Label
    ============  =============  ========================
    0             > no_impact    No Impact
    1             no_impact → low  Low Impact
    2             low → moderate  Moderate Impact
    3             < moderate     Severe Impact
    ============  =============  ========================

    Parameters
    ----------
    delta_img : ee.Image
        Image with a ``"delta"`` band from :func:`compute_difference`.
    thresholds : dict, optional
        Override default thresholds.  Expected keys: ``"no_impact"``,
        ``"low_impact"``, ``"moderate_impact"``.

    Returns
    -------
    ee.Image
        Single-band integer image (0–3) named ``"impact_class"``.
    """
    import ee

    thresholds = thresholds or DEFAULT_THRESHOLDS
    t_no = thresholds["no_impact"]        # e.g. -0.05
    t_low = thresholds["low_impact"]      # e.g. -0.15
    t_mod = thresholds["moderate_impact"] # e.g. -0.30

    delta = delta_img.select("delta")

    # Build classification using conditional expressions.
    # NOTE: ee.Image(0) is an unmasked constant covering the entire Earth.
    # .where(masked_condition, value) keeps the base value (0) rather than
    # propagating the mask, so without the explicit updateMask() below, every
    # pixel that was water-masked in delta would appear as class 0 ("No Impact")
    # instead of being transparent.
    classified = (
        ee.Image(0)                            # default: No Impact
        .where(delta.lt(t_no), 1)             # Low Impact
        .where(delta.lt(t_low), 2)            # Moderate Impact
        .where(delta.lt(t_mod), 3)            # Severe Impact
    )

    # Re-apply the delta mask so that water-masked and cloud-masked pixels
    # remain transparent in the classification layer.
    return classified.rename("impact_class").toInt().updateMask(delta.mask())


def compute_area_by_class(
    classified_img: "ee.Image",
    roi: "ee.Geometry",
    scale: int = 30,
) -> Dict[str, float]:
    """
    Compute the area (km²) of each impact class within the ROI.

    Parameters
    ----------
    classified_img : ee.Image
        Classification image from :func:`classify_impact`.
    roi : ee.Geometry
        Region of interest.
    scale : int
        Spatial resolution in metres.

    Returns
    -------
    dict
        Mapping of class label → area in km².
    """
    import ee

    # Create an area image (m²) per pixel
    area_img = ee.Image.pixelArea().divide(1e6)  # → km²

    # Compute area per class using a grouped reducer
    areas_list = (
        area_img.addBands(classified_img)
        .reduceRegion(
            reducer=ee.Reducer.sum().group(
                groupField=1, groupName="class"
            ),
            geometry=roi,
            scale=scale,
            maxPixels=1e10,
        )
        .get("groups")
        .getInfo()
    )

    result = {}
    for entry in areas_list:
        cls_id = int(entry["class"])
        area_km2 = float(entry["sum"])
        label = CLASS_LABELS.get(cls_id, f"Class {cls_id}")
        result[label] = round(area_km2, 4)

    logger.info("Area by impact class: %s", result)
    return result


# ---------------------------------------------------------------------------
# e. Baseline Variability Check
# ---------------------------------------------------------------------------

def compute_historical_baselines(
    roi: "ee.Geometry",
    event_date: str,
    satellite: str,
    index: str,
    n_years: int = 3,
    pre_days: int = 60,
    post_days: int = 60,
    buffer_days: int = 5,
    sample_n: int = 300,
    scale: int = 30,
) -> List[float]:
    """
    Compute the mean vegetation index delta for the same seasonal windows in
    ``n_years`` prior to the event year.

    These historical deltas serve as a baseline to detect whether the observed
    post-hurricane change is anomalous or within normal seasonal variability.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    event_date : str
        Hurricane event date (YYYY-MM-DD).
    satellite, index : str
        Sensor and index to use (same as the main analysis).
    n_years : int
        Number of prior years (default 3).
    pre_days, post_days, buffer_days : int
        Window parameters (same as the main analysis).
    sample_n : int
        Number of pixels to sample per year.
    scale : int
        Sampling resolution in metres.

    Returns
    -------
    list of float
        Mean delta value for each prior year.
    """
    from .data_acquisition import get_composites
    from .vegetation_indices import compute_index

    windows = historical_date_windows(
        event_date, n_years=n_years,
        pre_days=pre_days, post_days=post_days, buffer_days=buffer_days,
    )

    historical_deltas = []
    for i, (pre_start, pre_end, post_start, post_end) in enumerate(windows):
        year_label = f"Year -{i + 1}"
        logger.info("Computing historical baseline for %s (%s → %s / %s → %s) …",
                    year_label, pre_start, pre_end, post_start, post_end)
        try:
            # Re-construct a fake event_date for the prior year
            from datetime import datetime, timedelta
            prior_event = datetime.strptime(event_date, "%Y-%m-%d").replace(
                year=datetime.strptime(event_date, "%Y-%m-%d").year - (i + 1)
            )
            pre_img, post_img, _, _ = get_composites(
                roi,
                prior_event.strftime("%Y-%m-%d"),
                satellite=satellite,
                pre_days=pre_days,
                post_days=post_days,
                buffer_days=buffer_days,
            )
            pre_idx = compute_index(pre_img, index, satellite)
            post_idx = compute_index(post_img, index, satellite)

            pre_vals, post_vals = sample_pixels(
                pre_idx, post_idx, index, roi, n=sample_n, scale=scale
            )
            delta_mean = float(np.mean(post_vals - pre_vals))
            historical_deltas.append(delta_mean)
            logger.info("  %s: mean delta = %.4f", year_label, delta_mean)
        except (InsufficientDataError, Exception) as exc:
            logger.warning("  Skipping %s: %s", year_label, exc)

    return historical_deltas


def check_baseline_variability(
    current_delta_mean: float,
    historical_deltas: List[float],
    z_threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    Determine whether the observed delta is within normal seasonal variability.

    If the z-score of the current delta relative to the historical distribution
    is within ±``z_threshold`` standard deviations, the change is flagged as
    potentially within normal range.

    Parameters
    ----------
    current_delta_mean : float
        Mean delta index value from the event analysis.
    historical_deltas : list of float
        Mean deltas from prior years (from :func:`compute_historical_baselines`).
    z_threshold : float
        Z-score threshold for "within normal range" (default 2.0).

    Returns
    -------
    dict
        Keys: ``mean_hist``, ``std_hist``, ``z_score``,
        ``within_normal_range``, ``interpretation``.
    """
    if not historical_deltas:
        return {
            "mean_hist": None,
            "std_hist": None,
            "z_score": None,
            "within_normal_range": None,
            "interpretation": "No historical baseline data available.",
        }

    mean_hist = float(np.mean(historical_deltas))
    std_hist = float(np.std(historical_deltas, ddof=1)) if len(historical_deltas) > 1 else 0.0

    if std_hist < 1e-6:
        z_score = float("inf") if abs(current_delta_mean - mean_hist) > 1e-6 else 0.0
    else:
        z_score = (current_delta_mean - mean_hist) / std_hist

    within_normal = abs(z_score) <= z_threshold

    if within_normal:
        interpretation = (
            f"The observed change (delta = {current_delta_mean:.4f}) falls within "
            f"±{z_threshold} SD of historical variability "
            f"(mean = {mean_hist:.4f}, SD = {std_hist:.4f}). "
            f"Treat impact detection results with caution — this may reflect "
            f"normal seasonal variation rather than hurricane damage."
        )
    else:
        interpretation = (
            f"The observed change (delta = {current_delta_mean:.4f}) is anomalous "
            f"(z = {z_score:.2f}), exceeding ±{z_threshold} SD of historical "
            f"seasonal variability (mean = {mean_hist:.4f}, SD = {std_hist:.4f}). "
            f"This strongly suggests hurricane-induced vegetation damage."
        )

    return {
        "mean_hist": mean_hist,
        "std_hist": std_hist,
        "z_score": float(z_score),
        "within_normal_range": within_normal,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_analysis(
    roi: "ee.Geometry",
    event_date: str,
    satellite: str,
    index: str,
    output_dir: str,
    config: Dict[str, Any],
    sensors: str = "optical",
    palsar_pre_year: Optional[int] = None,
    palsar_post_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the complete vegetation impact analysis pipeline.

    Orchestrates data acquisition, index computation, difference mapping,
    statistical testing, impact classification, and baseline comparison.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    event_date : str
        Hurricane event date (YYYY-MM-DD).
    satellite : str
        ``"sentinel2"`` or ``"landsat"``.
    index : str
        Vegetation index to analyse (``"NDVI"``, ``"EVI"``, ``"SAVI"``,
        ``"NDMI"``).
    output_dir : str
        Path to write GeoTIFF and report outputs.
    config : dict
        Merged configuration dictionary from config.yaml + CLI overrides.
    sensors : str
        Comma-separated sensor tokens: ``"optical"`` (default), ``"optical,sar"``,
        ``"optical,sar,gedi"``, or ``"all"``.  Non-optical sensors are run via
        :func:`src.structural_analysis.run_structural_analysis` and their
        results are stored under the ``"structural"`` key.

    Returns
    -------
    dict
        Complete results including GEE images (for visualization) and all
        statistical outputs.  When SAR/GEDI sensors are requested the dict
        also contains a ``"structural"`` sub-dict.
    """
    from .data_acquisition import get_composites
    from .vegetation_indices import compute_index
    from .utils import ensure_dir

    out = ensure_dir(output_dir)
    windows_cfg = config.get("windows", {})
    pre_days = windows_cfg.get("pre_days", 60)
    post_days = windows_cfg.get("post_days", 60)
    buffer_days = windows_cfg.get("buffer_days", 5)
    stat_cfg = config.get("statistics", {})
    alpha = stat_cfg.get("significance_level", 0.05)
    sample_size = stat_cfg.get("sample_size", 500)
    hist_years = stat_cfg.get("historical_years", 3)
    thresholds = config.get("thresholds", DEFAULT_THRESHOLDS)
    mask_water = config.get("processing", {}).get("mask_water", False)
    mask_water_threshold = int(config.get("processing", {}).get("mask_water_threshold", 80))
    export_cfg = config.get("export", {})
    scale = (
        export_cfg.get("scale_sentinel2", 10)
        if satellite == "sentinel2"
        else export_cfg.get("scale_landsat", 30)
    )

    logger.info("=== Hurricane Vegetation Impact Analysis ===")
    logger.info("Event date : %s", event_date)
    logger.info("Satellite  : %s", satellite)
    logger.info("Index      : %s", index)
    logger.info("Output dir : %s", out)

    # 1. Data acquisition — land mask applied per-image inside get_composites
    #    when mask_water=True, so water pixels never enter the median composite.
    logger.info("--- Step 1: Acquiring composites ---")
    pre_img, post_img, pre_optical_meta, post_optical_meta = get_composites(
        roi, event_date, satellite=satellite,
        pre_days=pre_days, post_days=post_days, buffer_days=buffer_days,
        mask_water=mask_water, mask_water_threshold=mask_water_threshold,
    )

    # 1b. Build land mask once for reuse in structural analysis and area reporting.
    #     (get_composites builds its own internally; we build a matching instance
    #      here so downstream callers don't need their own copy.)
    land_mask = None
    if mask_water:
        from .data_acquisition import build_jrc_land_mask, compute_land_area_km2
        land_mask = build_jrc_land_mask(mask_water_threshold)
        try:
            _total_area = roi.area(maxError=100).getInfo() / 1e6
            _land_area  = compute_land_area_km2(roi, land_mask, scale=500)
            logger.info(
                "ROI: %.1f km² total, %.1f km² land (%.0f%%). "
                "Water pixels excluded from analysis.",
                _total_area, _land_area, _land_area / max(_total_area, 1) * 100,
            )
        except Exception as _exc:
            logger.warning("Land area computation failed: %s", _exc)
            _land_area = None

    # 2. Vegetation index
    logger.info("--- Step 2: Computing %s ---", index)
    pre_idx = compute_index(pre_img, index, satellite)
    post_idx = compute_index(post_img, index, satellite)

    # 3. Difference map
    logger.info("--- Step 3: Computing difference map ---")
    diff_img = compute_difference(pre_idx, post_idx, index)

    # 4. Export GeoTIFFs
    logger.info("--- Step 4: Exporting GeoTIFFs ---")

    geotiff_path = str(out / f"{index}_difference.tif")
    try:
        export_geotiff(diff_img, roi, geotiff_path, scale=scale)
    except Exception as exc:
        logger.warning("Difference GeoTIFF export failed: %s", exc)
        geotiff_path = None

    pre_geotiff_path = str(out / f"{index}_pre.tif")
    try:
        export_geotiff(pre_idx.select(index), roi, pre_geotiff_path, scale=scale)
    except Exception as exc:
        logger.warning("Pre-event GeoTIFF export failed: %s", exc)
        pre_geotiff_path = None

    post_geotiff_path = str(out / f"{index}_post.tif")
    try:
        export_geotiff(post_idx.select(index), roi, post_geotiff_path, scale=scale)
    except Exception as exc:
        logger.warning("Post-event GeoTIFF export failed: %s", exc)
        post_geotiff_path = None

    # 5. Impact classification
    logger.info("--- Step 5: Classifying impact severity ---")
    classified_img = classify_impact(diff_img, thresholds=thresholds)
    area_by_class = compute_area_by_class(classified_img, roi, scale=scale)

    # 6. Statistical testing (local sampling)
    logger.info("--- Step 6: Statistical testing (sampling %d pixels) ---", sample_size)
    pre_vals, post_vals = sample_pixels(
        pre_idx, post_idx, index, roi, n=sample_size, scale=scale
    )
    stat_results = run_statistical_tests(pre_vals, post_vals, alpha=alpha)

    # 7. Baseline variability
    logger.info("--- Step 7: Historical baseline variability check ---")
    try:
        hist_deltas = compute_historical_baselines(
            roi, event_date, satellite=satellite, index=index,
            n_years=hist_years, pre_days=pre_days, post_days=post_days,
            buffer_days=buffer_days, scale=scale,
        )
        baseline = check_baseline_variability(stat_results["delta_mean"], hist_deltas)
    except Exception as exc:
        logger.warning("Baseline variability check failed: %s", exc)
        baseline = {"interpretation": f"Baseline check failed: {exc}"}
        hist_deltas = []

    # Compile results
    results = {
        "event_date": event_date,
        "satellite": satellite,
        "index": index,
        "output_dir": str(out),
        "geotiff_path": geotiff_path,
        "pre_geotiff_path": pre_geotiff_path,
        "post_geotiff_path": post_geotiff_path,
        # Scene metadata (populated here for optical; SAR/GEDI/PALSAR added below)
        "scene_metadata": {
            "optical": {"pre": pre_optical_meta, "post": post_optical_meta},
        },
        # Land area (set when mask_water=True)
        "land_area_km2": _land_area if mask_water else None,
        # GEE images (for visualization)
        "pre_img": pre_idx,
        "post_img": post_idx,
        "diff_img": diff_img,
        "classified_img": classified_img,
        # Arrays (for plotting)
        "pre_vals": pre_vals,
        "post_vals": post_vals,
        # Statistical results
        "statistics": stat_results,
        # Classification areas
        "area_by_class": area_by_class,
        # Baseline
        "historical_deltas": hist_deltas,
        "baseline": baseline,
    }

    logger.info("=== Analysis complete ===")
    logger.info(stat_results["conclusion"])

    # ── Multi-sensor structural analysis (SAR / GEDI / PALSAR) ───────────────
    sensor_list = [s.strip().lower() for s in sensors.split(",")]
    has_extra_sensors = any(s in sensor_list for s in ("sar", "gedi", "palsar", "all"))
    if has_extra_sensors:
        logger.info("--- Step 8: Structural analysis (%s) ---", sensors)
        try:
            from .structural_analysis import run_structural_analysis
            structural = run_structural_analysis(
                roi=roi,
                event_date=event_date,
                optical_results=results,
                config=config,
                sensors=sensors,
                palsar_pre_year=palsar_pre_year,
                palsar_post_year=palsar_post_year,
                land_mask=land_mask,
            )
            results["structural"] = structural
            # Merge structural scene_metadata (SAR / GEDI / PALSAR) into top-level dict
            for sensor_key in ("sar", "gedi", "palsar"):
                if sensor_key in structural.get("scene_metadata", {}):
                    results["scene_metadata"][sensor_key] = (
                        structural["scene_metadata"][sensor_key]
                    )
        except Exception as exc:
            logger.warning("Structural analysis failed: %s", exc)
            results["structural"] = {
                "error": str(exc),
                "sar_available": False,
                "gedi_available": False,
                "palsar_available": False,
            }

    # Compute acquisition warnings now that all metadata is assembled
    try:
        from .metadata_utils import check_warnings
        results["scene_metadata"]["warnings"] = check_warnings(
            results["scene_metadata"], event_date
        )
    except Exception as exc:
        logger.warning("Warning check failed: %s", exc)
        results["scene_metadata"]["warnings"] = []

    return results
