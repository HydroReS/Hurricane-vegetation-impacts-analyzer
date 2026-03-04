"""
time_series.py
==============
Continuous temporal monitoring and anomaly detection for vegetation indices.

Implements:

a. Time Series Extraction
   Extract pixel-level (point) or spatially-averaged (ROI) vegetation index
   time series from GEE, with optional temporal compositing.

b. Seasonal Decomposition & Trend Fitting
   STL decomposition (statsmodels) and harmonic regression (numpy).

c. Anomaly Detection
   Three complementary methods: z-score on model residuals, moving window,
   and historical climatology comparison.

d. Change Point Detection
   CUSUM algorithm (built-in); optional ruptures-based PELT detection.

e. Recovery Analysis
   Post-event recovery time, rate, completeness, and status classification.

f. Multi-Point / Spatial Comparison
   Parallel time series extraction for a list of point locations.

g. Visualization
   Interactive Plotly charts and static matplotlib figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Resample frequency strings (pandas)
_COMPOSITE_FREQ = {
    "weekly": "7D",
    "biweekly": "14D",
    "monthly": "ME",  # pandas ≥ 2.2; falls back to "M" at runtime
}

# Inferred STL period (observations per annual cycle) per composite
_STL_PERIOD = {
    "raw": 73,        # ~5-day Sentinel-2 revisit
    "weekly": 52,
    "biweekly": 26,
    "monthly": 12,
}

ANOMALY_METHODS = ("zscore", "moving_window", "climatology")


# ── a. Time Series Extraction ─────────────────────────────────────────────────

def _build_indexed_collection(
    roi: "ee.Geometry",
    start: str,
    end: str,
    satellite: str,
    index: str,
    max_cloud_pct: float = 80.0,
) -> "ee.ImageCollection":
    """Build a cloud-masked, vegetation-index-computed ImageCollection."""
    from .data_acquisition import get_sentinel2_collection, get_landsat_collection
    from .vegetation_indices import compute_index

    if satellite == "sentinel2":
        collection = get_sentinel2_collection(roi, start, end, max_cloud_pct)
    else:
        collection = get_landsat_collection(roi, start, end, max_cloud_pct)

    return collection.map(lambda img: compute_index(img, index, satellite))


def extract_point_time_series(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    satellite: str = "sentinel2",
    index: str = "NDVI",
    scale: int = 30,
    max_cloud_pct: float = 80.0,
) -> pd.DataFrame:
    """
    Extract a vegetation index time series at a single point location.

    Uses ``ee.ImageCollection.getRegion()`` to pull the index value from
    every available cloud-free image at the specified coordinate.

    Parameters
    ----------
    lat, lon : float
        Point coordinates (WGS84).
    start_date, end_date : str
        Date range (``YYYY-MM-DD``).
    satellite : str
        ``"sentinel2"`` or ``"landsat"``.
    index : str
        Vegetation index band name (e.g. ``"NDVI"``).
    scale : int
        Sampling resolution in metres.
    max_cloud_pct : float
        Maximum scene cloud percentage for pre-filtering.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``index_value``, ``satellite``.
        Sorted by date; rows with missing values are dropped.
    """
    import ee

    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(scale * 2)

    collection = _build_indexed_collection(
        roi, start_date, end_date, satellite, index, max_cloud_pct
    )

    logger.info(
        "Extracting point time series at (%.4f, %.4f) — %s %s → %s",
        lat, lon, index, start_date, end_date,
    )

    region_data = collection.select(index).getRegion(point, scale=scale).getInfo()

    if len(region_data) <= 1:
        logger.warning("getRegion returned no data.")
        return pd.DataFrame(columns=["date", "index_value", "satellite"])

    headers = region_data[0]
    df = pd.DataFrame(region_data[1:], columns=headers)
    df["date"] = pd.to_datetime(df["time"], unit="ms")
    df = df.rename(columns={index: "index_value"})
    df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")
    df = df[["date", "index_value"]].dropna()
    df["satellite"] = satellite
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Retrieved %d valid observations.", len(df))
    return df


def _date_chunks(
    start: str,
    end: str,
    chunk_months: int = 6,
) -> List[Tuple[str, str]]:
    """
    Split *start*→*end* into calendar-month chunks, returning (start, end) pairs.

    Each chunk is exactly *chunk_months* calendar months wide.  The final
    chunk is clipped to *end* so no data outside the requested range is
    fetched.
    """
    from datetime import datetime

    s_dt = datetime.strptime(start, "%Y-%m-%d")
    e_dt = datetime.strptime(end,   "%Y-%m-%d")

    chunks: List[Tuple[str, str]] = []
    cur = s_dt
    while cur < e_dt:
        new_month = cur.month + chunk_months
        new_year  = cur.year + (new_month - 1) // 12
        new_month = (new_month - 1) % 12 + 1
        nxt = min(datetime(new_year, new_month, 1), e_dt)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt

    return chunks


def extract_roi_time_series(
    roi: "ee.Geometry",
    start_date: str,
    end_date: str,
    satellite: str = "sentinel2",
    index: str = "NDVI",
    scale: int = 30,
    max_cloud_pct: float = 80.0,
    progress_callback: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Extract a spatially-averaged vegetation index time series over an ROI.

    Maps ``reduceRegion(mean, stdDev, count)`` over each image in the
    collection and returns results as a DataFrame.

    The full date range is split into 6-month chunks and each chunk is
    fetched with a separate ``getInfo()`` call.  This avoids the GEE
    user-memory limit that is hit when collecting results for 300 + images
    in a single request.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    start_date, end_date : str
        Date range (``YYYY-MM-DD``).
    satellite : str
        ``"sentinel2"`` or ``"landsat"``.
    index : str
        Vegetation index band name.
    scale : int
        Reduction resolution in metres.
    max_cloud_pct : float
        Maximum scene cloud percentage.
    progress_callback : callable, optional
        Called after each chunk is fetched with ``(chunks_done, total_chunks)``
        so callers can show a progress bar.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``index_value``, ``std``, ``pixel_count``,
        ``satellite``.
    """
    import ee

    collection = _build_indexed_collection(
        roi, start_date, end_date, satellite, index, max_cloud_pct
    )
    logger.info(
        "Extracting ROI time series for %s %s → %s (scale=%dm)",
        index, start_date, end_date, scale,
    )

    def _reduce(image):
        stats = image.select(index).reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.count(), sharedInputs=True),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,   # subdivide computation to avoid GEE timeout
        )
        # Combined reducer naming: <band>_mean, <band>_stdDev, <band>_count
        return ee.Feature(None, {
            "date": image.date().format("YYYY-MM-dd"),
            "mean": stats.get(f"{index}_mean"),
            "std":  stats.get(f"{index}_stdDev"),
            "count": stats.get(f"{index}_count"),
        })

    # Split the date range into 6-month chunks to stay under GEE memory limits.
    # Each chunk is fetched independently; results are concatenated locally.
    chunks = _date_chunks(start_date, end_date, chunk_months=6)
    all_features: List[Dict] = []

    for i, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_col  = collection.filterDate(chunk_start, chunk_end)
        chunk_info = chunk_col.map(_reduce).getInfo()
        chunk_feats = chunk_info.get("features", [])
        all_features.extend(chunk_feats)
        logger.info(
            "  Chunk %d/%d (%s→%s): %d images",
            i + 1, len(chunks), chunk_start, chunk_end, len(chunk_feats),
        )
        if progress_callback is not None:
            progress_callback(i + 1, len(chunks))

    if not all_features:
        logger.warning("No data returned from ROI reduction.")
        return pd.DataFrame(
            columns=["date", "index_value", "std", "pixel_count", "satellite"]
        )

    records = [
        {
            "date":        f["properties"].get("date"),
            "index_value": f["properties"].get("mean"),
            "std":         f["properties"].get("std"),
            "pixel_count": f["properties"].get("count"),
        }
        for f in all_features
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")
    df = df.dropna(subset=["index_value"])
    df["satellite"] = satellite
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Retrieved %d valid observations.", len(df))
    return df


def extract_sar_time_series(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    scale: int = 10,
    orbit: str = "DESCENDING",
) -> pd.DataFrame:
    """
    Extract a Sentinel-1 SAR time series at a single point.

    Applies focal-median speckle filtering (50 m radius) to each image,
    computes RVI in linear scale, and samples at the specified point.

    Parameters
    ----------
    lat, lon : float
        Point coordinates (WGS84).
    start_date, end_date : str
        Date range (``YYYY-MM-DD``).
    scale : int
        Sampling resolution in metres (default 10 for Sentinel-1).
    orbit : str
        ``"DESCENDING"`` (default) or ``"ASCENDING"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``VV_db``, ``VH_db``, ``RVI``, ``sensor``.
        Sorted by date; rows with missing values dropped.
    """
    import ee

    point = ee.Geometry.Point([lon, lat])
    roi   = point.buffer(scale * 5)  # small buffer for pixel sampling

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )

    kernel = ee.Kernel.circle(50, "meters")

    def _add_rvi(img):
        filtered = img.focal_median(kernel=kernel)
        # linear scale: 10^(dB/10)
        linear   = filtered.expression(
            "10 ** (db / 10)", {"db": filtered}
        )
        vv_lin = linear.select("VV")
        vh_lin = linear.select("VH")
        rvi    = vh_lin.multiply(4).divide(vv_lin.add(vh_lin)).rename("RVI")
        return filtered.addBands(rvi)

    col_with_rvi = col.map(_add_rvi)

    logger.info(
        "Extracting SAR time series at (%.4f, %.4f) %s→%s orbit=%s",
        lat, lon, start_date, end_date, orbit,
    )

    data = col_with_rvi.select(["VV", "VH", "RVI"]).getRegion(point, scale=scale).getInfo()

    if len(data) <= 1:
        logger.warning("SAR getRegion returned no data.")
        return pd.DataFrame(columns=["date", "VV_db", "VH_db", "RVI", "sensor"])

    headers = data[0]
    df = pd.DataFrame(data[1:], columns=headers)
    df["date"] = pd.to_datetime(df["time"], unit="ms")
    df = df.rename(columns={"VV": "VV_db", "VH": "VH_db"})
    for col_name in ["VV_db", "VH_db", "RVI"]:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        else:
            df[col_name] = float("nan")
    df = df[["date", "VV_db", "VH_db", "RVI"]].dropna()
    df["sensor"] = "SAR"
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Retrieved %d valid SAR observations.", len(df))
    return df


def extract_palsar_time_series(
    roi: "ee.Geometry",
    start_year: int = 2014,
    end_year: Optional[int] = None,
    scale: int = 25,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Extract an annual ALOS PALSAR-2 L-band time series for an ROI.

    Uses the JAXA ``JAXA/ALOS/PALSAR/YEARLY/SAR`` collection, which
    provides one calibrated mosaic per calendar year (25 m resolution) with
    HH and HV bands stored as DN values.  Calibration to dB is applied here.

    Available years are approximately 2007 (PALSAR-1) through present
    (PALSAR-2 from 2014).

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest (used for spatial mean reduction).
    start_year : int
        First mosaic year to include (default 2014 = PALSAR-2 era start).
    end_year : int, optional
        Last mosaic year to include (default: current year).
    scale : int
        Reduction scale in metres (default 25 = native PALSAR resolution).
    config : dict, optional
        Configuration dict; reads ``palsar_settings.calibration_offset``.

    Returns
    -------
    pd.DataFrame
        Columns: ``year``, ``date`` (Jan 1 of that year as Timestamp),
                 ``HV_dB``, ``HH_dB``, ``sensor`` ("PALSAR").
        Rows with no valid data are dropped.
    """
    import ee
    from datetime import datetime as _dt

    cfg = config or {}
    cal_offset = float(
        cfg.get("palsar_settings", {}).get("calibration_offset", -83.0)
    )
    if end_year is None:
        end_year = _dt.utcnow().year

    _PALSAR_COLLECTION = "JAXA/ALOS/PALSAR/YEARLY/SAR"

    records = []
    for year in range(start_year, end_year + 1):
        try:
            col = (
                ee.ImageCollection(_PALSAR_COLLECTION)
                .filterBounds(roi)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .select(["HH", "HV"])
            )
            # calibrate in dB: 20*log10(DN) + offset
            img = col.mosaic().log10().multiply(20).add(cal_offset)
            img = img.rename(["HH_dB", "HV_dB"])

            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=scale,
                maxPixels=1e7,
            ).getInfo()

            hh_db = stats.get("HH_dB")
            hv_db = stats.get("HV_dB")
            if hh_db is not None and hv_db is not None:
                records.append({
                    "year":   year,
                    "date":   pd.Timestamp(f"{year}-01-01"),
                    "HV_dB":  float(hv_db),
                    "HH_dB":  float(hh_db),
                    "sensor": "PALSAR",
                })
        except Exception as exc:
            logger.debug("PALSAR: no data for year %d (%s)", year, exc)

    if not records:
        logger.warning("PALSAR time series: no valid annual data retrieved.")
        return pd.DataFrame(columns=["year", "date", "HV_dB", "HH_dB", "sensor"])

    df = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    logger.info("PALSAR time series: %d annual observations (%d–%d).",
                len(df), df["year"].min(), df["year"].max())
    return df


def apply_temporal_composite(
    df: pd.DataFrame,
    composite: str = "monthly",
) -> pd.DataFrame:
    """
    Resample a raw time series to a regular temporal composite using the
    median aggregation function.

    Parameters
    ----------
    df : pd.DataFrame
        Raw time series with ``date`` and ``index_value`` columns.
    composite : str
        One of: ``"raw"``, ``"weekly"``, ``"biweekly"``, ``"monthly"``.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with median ``index_value`` per period.
    """
    if composite == "raw" or composite not in _COMPOSITE_FREQ:
        return df.copy()

    freq = _COMPOSITE_FREQ[composite]
    sat = df["satellite"].iloc[0] if "satellite" in df.columns else "unknown"

    numeric = df.set_index("date").select_dtypes(include=[np.number])

    try:
        resampled = numeric.resample(freq).median()
    except ValueError:
        # pandas < 2.2: "ME" not recognised, fall back to "M"
        resampled = numeric.resample("M").median()

    resampled = resampled.dropna(subset=["index_value"]).reset_index()
    resampled["satellite"] = sat

    logger.info(
        "Resampled %d raw obs → %d %s composites.",
        len(df), len(resampled), composite,
    )
    return resampled


# ── b. Seasonal Decomposition & Trend Fitting ─────────────────────────────────

def fit_harmonic_model(
    df: pd.DataFrame,
    harmonics: int = 1,
) -> Dict[str, Any]:
    """
    Fit a harmonic regression model to the time series.

    Model: y = β₀ + β₁·(t/365) + β₂·cos(2πt/365) + β₃·sin(2πt/365) + ε

    Standard in remote sensing phenology; provides a compact seasonal baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Time series with ``date`` and ``index_value`` columns.
    harmonics : int
        Number of harmonic pairs (1 = annual cycle only, 2 adds semi-annual).

    Returns
    -------
    dict
        Keys: ``coefficients``, ``fitted``, ``residuals``, ``r_squared``.
    """
    df = df.dropna(subset=["index_value"]).copy()
    if len(df) < max(4, 2 * harmonics + 2):
        return {
            "fitted": df["index_value"].values,
            "residuals": np.zeros(len(df)),
            "r_squared": 0.0,
            "coefficients": [],
        }

    t = (df["date"] - df["date"].min()).dt.days.values.astype(float)
    y = df["index_value"].values

    cols = [np.ones(len(t)), t / 365.25]
    for k in range(1, harmonics + 1):
        omega = 2 * np.pi * k / 365.25
        cols += [np.cos(omega * t), np.sin(omega * t)]

    X = np.column_stack(cols)
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        beta = np.zeros(X.shape[1])

    fitted = X @ beta
    residuals = y - fitted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "coefficients": beta.tolist(),
        "fitted": fitted,
        "residuals": residuals,
        "r_squared": r_squared,
        "t": t,
    }


def apply_stl_decomposition(
    df: pd.DataFrame,
    period: Optional[int] = None,
    composite: str = "monthly",
) -> Optional[Dict[str, Any]]:
    """
    Apply STL (Seasonal-Trend decomposition using Loess) decomposition.

    Separates the time series into trend, seasonal, and residual components.
    The residual is the primary signal used for anomaly detection.

    Parameters
    ----------
    df : pd.DataFrame
        Time series with ``date`` and ``index_value`` columns.
    period : int, optional
        Seasonal period in observations. Inferred from ``composite`` if omitted.
    composite : str
        Compositing interval (used to infer ``period``).

    Returns
    -------
    dict or None
        Keys: ``observed``, ``trend``, ``seasonal``, ``residual``, ``dates``.
        Returns ``None`` if statsmodels is unavailable or data insufficient.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        logger.warning("statsmodels not installed — STL unavailable.")
        return None

    if period is None:
        period = _STL_PERIOD.get(composite, 12)

    df = df.dropna(subset=["index_value"]).copy()
    if len(df) < 2 * period + 1:
        logger.warning(
            "Insufficient data for STL (need ≥ %d observations, got %d).",
            2 * period + 1, len(df),
        )
        return None

    series = pd.Series(
        df["index_value"].values,
        index=pd.DatetimeIndex(df["date"]),
    )
    series = series.interpolate(method="time").bfill().ffill()

    try:
        result = STL(series, period=period, robust=True).fit()
        return {
            "observed": result.observed,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
            "dates": df["date"].values,
        }
    except Exception as exc:
        logger.warning("STL decomposition failed: %s", exc)
        return None


# ── c. Anomaly Detection ──────────────────────────────────────────────────────

def _severity_label(z: float) -> str:
    """Classify anomaly severity from z-score magnitude."""
    if z < -3.5:
        return "extreme"
    if z < -2.5:
        return "moderate"
    if z < -2.0:
        return "mild"
    if z > 3.5:
        return "extreme_positive"
    if z > 2.5:
        return "moderate_positive"
    if z > 2.0:
        return "mild_positive"
    return "none"


def detect_anomalies_zscore(
    df: pd.DataFrame,
    stl_result: Optional[Dict] = None,
    harmonic_result: Optional[Dict] = None,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Detect anomalies using z-scores on seasonal model residuals.

    Uses STL residuals if available, harmonic residuals as fallback, or
    simple mean-subtracted residuals if neither model is available.

    Parameters
    ----------
    df : pd.DataFrame
        Time series.
    stl_result : dict, optional
        Output of :func:`apply_stl_decomposition`.
    harmonic_result : dict, optional
        Output of :func:`fit_harmonic_model`.
    threshold : float
        Z-score magnitude for anomaly flagging (default 2.0).

    Returns
    -------
    pd.DataFrame
        Anomaly records: ``date``, ``observed_value``, ``expected_value``,
        ``deviation``, ``z_score``, ``method``, ``severity``.
    """
    df = df.dropna(subset=["index_value"]).copy()

    if stl_result is not None and len(stl_result["residual"]) == len(df):
        residuals = np.asarray(stl_result["residual"])
        expected = np.asarray(stl_result["trend"]) + np.asarray(stl_result["seasonal"])
    elif harmonic_result is not None and len(harmonic_result.get("residuals", [])) == len(df):
        residuals = harmonic_result["residuals"]
        expected = harmonic_result["fitted"]
    else:
        y = df["index_value"].values
        expected = np.full(len(y), np.mean(y))
        residuals = y - expected

    std_res = float(np.std(residuals, ddof=1))
    if std_res < 1e-9:
        return pd.DataFrame(columns=["date", "observed_value", "expected_value",
                                     "deviation", "z_score", "method", "severity"])

    z_scores = residuals / std_res
    observed = df["index_value"].values

    records = [
        {
            "date": pd.Timestamp(df["date"].iloc[i]),
            "observed_value": float(observed[i]),
            "expected_value": float(expected[i]),
            "deviation": float(residuals[i]),
            "z_score": float(z_scores[i]),
            "method": "zscore",
            "severity": _severity_label(float(z_scores[i])),
        }
        for i in range(len(df))
        if abs(z_scores[i]) > threshold
    ]
    return pd.DataFrame(records)


def detect_anomalies_moving_window(
    df: pd.DataFrame,
    window_days: int = 90,
    k: float = 2.0,
) -> pd.DataFrame:
    """
    Detect anomalies using a trailing moving window comparison.

    Each observation is compared to the mean ± k·std of the preceding
    ``window_days`` days of data.

    Parameters
    ----------
    df : pd.DataFrame
        Time series.
    window_days : int
        Trailing window size in days (default 90).
    k : float
        Number of standard deviations for the anomaly band (default 2.0).

    Returns
    -------
    pd.DataFrame
        Anomaly records.
    """
    df = df.dropna(subset=["index_value"]).sort_values("date").copy()
    ts = df.set_index("date")["index_value"]

    roll = ts.rolling(f"{window_days}D", min_periods=5)
    roll_mean = roll.mean()
    roll_std = roll.std().replace(0, np.nan)
    z = (ts - roll_mean) / roll_std

    records = []
    for date in z[z.abs() > k].index:
        z_val = float(z[date])
        records.append({
            "date": pd.Timestamp(date),
            "observed_value": float(ts[date]),
            "expected_value": float(roll_mean[date]) if not np.isnan(roll_mean[date]) else np.nan,
            "deviation": float(ts[date] - roll_mean[date]),
            "z_score": z_val,
            "method": "moving_window",
            "severity": _severity_label(z_val),
        })
    return pd.DataFrame(records)


def detect_anomalies_climatology(
    df: pd.DataFrame,
    threshold_pct: float = 5.0,
    doy_window: int = 15,
) -> pd.DataFrame:
    """
    Detect anomalies by comparing each observation to its historical
    climatology — the distribution of same-day-of-year values across all
    prior years.

    Parameters
    ----------
    df : pd.DataFrame
        Time series spanning multiple years.
    threshold_pct : float
        Lower percentile for anomaly flag (default 5th percentile).
        Values above the (100 − threshold_pct)th percentile are also flagged.
    doy_window : int
        ±days around each day-of-year used to build the historical sample.

    Returns
    -------
    pd.DataFrame
        Anomaly records.
    """
    df = df.dropna(subset=["index_value"]).sort_values("date").copy()
    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year

    if df["year"].nunique() < 2:
        logger.warning("Need ≥ 2 years of data for climatology comparison.")
        return pd.DataFrame(columns=["date", "observed_value", "expected_value",
                                     "deviation", "z_score", "method", "severity"])

    records = []
    for _, row in df.iterrows():
        doy, year, val = int(row["doy"]), int(row["year"]), float(row["index_value"])

        # Build DOY window (wrapping around year boundary)
        lo, hi = doy - doy_window, doy + doy_window
        if lo < 1:
            mask = (
                ((df["doy"] >= lo + 365) | (df["doy"] <= hi)) & (df["year"] < year)
            )
        elif hi > 365:
            mask = (
                ((df["doy"] >= lo) | (df["doy"] <= hi - 365)) & (df["year"] < year)
            )
        else:
            mask = (df["doy"] >= lo) & (df["doy"] <= hi) & (df["year"] < year)

        hist = df.loc[mask, "index_value"]
        if len(hist) < 3:
            continue

        low_pct = float(np.percentile(hist, threshold_pct))
        high_pct = float(np.percentile(hist, 100 - threshold_pct))
        hist_mean = float(np.mean(hist))
        hist_std = float(np.std(hist, ddof=1))

        if val < low_pct or val > high_pct:
            z_val = (val - hist_mean) / hist_std if hist_std > 0 else 0.0
            records.append({
                "date": row["date"],
                "observed_value": val,
                "expected_value": hist_mean,
                "deviation": val - hist_mean,
                "z_score": float(z_val),
                "method": "climatology",
                "severity": _severity_label(float(z_val)),
            })
    return pd.DataFrame(records)


def detect_all_anomalies(
    df: pd.DataFrame,
    stl_result: Optional[Dict] = None,
    harmonic_result: Optional[Dict] = None,
    threshold: float = 2.0,
    window_days: int = 90,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run all configured anomaly detection methods and return combined results.

    Parameters
    ----------
    df : pd.DataFrame
        Time series.
    stl_result, harmonic_result : dict, optional
        Seasonal decomposition outputs.
    threshold : float
        Z-score threshold (applies to zscore and moving_window methods).
    window_days : int
        Moving window size in days.
    methods : list of str, optional
        Subset of methods to run. Defaults to all three:
        ``["zscore", "moving_window", "climatology"]``.

    Returns
    -------
    pd.DataFrame
        Combined anomaly records with ``method`` column.
    """
    if methods is None:
        methods = list(ANOMALY_METHODS)

    parts = []
    if "zscore" in methods:
        parts.append(detect_anomalies_zscore(df, stl_result, harmonic_result, threshold))
    if "moving_window" in methods:
        parts.append(detect_anomalies_moving_window(df, window_days=window_days, k=threshold))
    if "climatology" in methods:
        parts.append(detect_anomalies_climatology(df))

    non_empty = [p for p in parts if not p.empty]
    if not non_empty:
        return pd.DataFrame()

    combined = pd.concat(non_empty, ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


# ── d. Change Point Detection ─────────────────────────────────────────────────

def detect_changepoints_cusum(
    df: pd.DataFrame,
    threshold: float = 4.0,
    drift: float = 0.0,
) -> pd.DataFrame:
    """
    Detect sustained shifts in the vegetation index using CUSUM.

    CUSUM accumulates deviations from the series mean. When the cumulative
    sum exceeds ``threshold`` standard deviations, a change point is flagged.
    Useful for pinpointing when hurricane damage begins relative to the last
    pre-event observation.

    Parameters
    ----------
    df : pd.DataFrame
        Time series.
    threshold : float
        Detection threshold in units of series standard deviation (default 4.0).
    drift : float
        Allowable drift before accumulation begins (default 0 = no drift).

    Returns
    -------
    pd.DataFrame
        Change point records: ``date``, ``direction``, ``magnitude``,
        ``cusum_value``.
    """
    df = df.dropna(subset=["index_value"]).sort_values("date").copy().reset_index(drop=True)
    if len(df) < 10:
        return pd.DataFrame(columns=["date", "direction", "magnitude", "cusum_value"])

    y = df["index_value"].values
    sigma = float(np.std(y, ddof=1))
    if sigma < 1e-9:
        return pd.DataFrame(columns=["date", "direction", "magnitude", "cusum_value"])

    x = (y - np.mean(y)) / sigma  # normalised series
    n = len(x)
    pos = np.zeros(n)  # upward CUSUM
    neg = np.zeros(n)  # downward CUSUM

    for i in range(1, n):
        pos[i] = max(0.0, pos[i - 1] + x[i] - drift)
        neg[i] = min(0.0, neg[i - 1] + x[i] + drift)

    records = []
    in_up, in_down, start_idx = False, False, 0

    for i in range(n):
        if pos[i] > threshold and not in_up:
            in_up = True
            start_idx = i
        elif pos[i] <= threshold and in_up:
            in_up = False
            cp = max(0, start_idx - 1)
            records.append({
                "date": df["date"].iloc[cp],
                "direction": "up",
                "magnitude": float(pos[start_idx] * sigma),
                "cusum_value": float(pos[start_idx]),
            })

        if neg[i] < -threshold and not in_down:
            in_down = True
            start_idx = i
        elif neg[i] >= -threshold and in_down:
            in_down = False
            cp = max(0, start_idx - 1)
            records.append({
                "date": df["date"].iloc[cp],
                "direction": "down",
                "magnitude": float(abs(neg[start_idx]) * sigma),
                "cusum_value": float(neg[start_idx]),
            })

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def detect_changepoints_ruptures(
    df: pd.DataFrame,
    n_bkps: int = 3,
    model: str = "rbf",
) -> pd.DataFrame:
    """
    Detect change points using the ``ruptures`` library (PELT algorithm).

    Returns an empty DataFrame if ``ruptures`` is not installed.

    Parameters
    ----------
    df : pd.DataFrame
        Time series.
    n_bkps : int
        Maximum number of breakpoints to detect.
    model : str
        Cost model for ruptures (``"rbf"``, ``"l2"``, or ``"l1"``).

    Returns
    -------
    pd.DataFrame
        Change point records.
    """
    try:
        import ruptures as rpt
    except ImportError:
        logger.info("ruptures not installed — skipping PELT change point detection.")
        return pd.DataFrame(columns=["date", "direction", "magnitude", "cusum_value"])

    df = df.dropna(subset=["index_value"]).sort_values("date").copy().reset_index(drop=True)
    y = df["index_value"].values.reshape(-1, 1)

    try:
        algo = rpt.Pelt(model=model, min_size=3).fit(y)
        bkps = algo.predict(pen=3)
    except Exception as exc:
        logger.warning("ruptures detection failed: %s", exc)
        return pd.DataFrame(columns=["date", "direction", "magnitude", "cusum_value"])

    records = []
    for bp in bkps[:-1]:  # last element is len(y)
        if bp >= len(df):
            continue
        before = float(df["index_value"].iloc[max(0, bp - 5):bp].mean())
        after = float(df["index_value"].iloc[bp:min(len(df), bp + 5)].mean())
        records.append({
            "date": df["date"].iloc[bp],
            "direction": "down" if after < before else "up",
            "magnitude": float(abs(after - before)),
            "cusum_value": np.nan,
        })
    return pd.DataFrame(records)


# ── e. Recovery Analysis ──────────────────────────────────────────────────────

def analyze_recovery(
    df: pd.DataFrame,
    event_date: str,
    pre_window_days: int = 90,
    recovery_tolerance: float = 1.0,
) -> Dict[str, Any]:
    """
    Quantify vegetation recovery following a disturbance event.

    Tracks the post-event time series forward to determine when and how
    completely the vegetation index returns to its pre-event baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Time series with ``date`` and ``index_value`` columns.
    event_date : str
        Date of the disturbance event (``YYYY-MM-DD``).
    pre_window_days : int
        Days before the event used to define the pre-event baseline.
    recovery_tolerance : float
        Recovery is declared when the index returns within
        ``recovery_tolerance × pre_std`` of ``pre_mean``.

    Returns
    -------
    dict
        Keys: ``pre_mean``, ``pre_std``, ``post_min``, ``post_min_date``,
        ``recovery_date``, ``recovery_days``, ``recovery_rate``,
        ``recovery_pct``, ``recovery_status``, ``interpretation``.
    """
    _empty = {
        "pre_mean": None, "pre_std": None, "post_min": None,
        "post_min_date": None, "recovery_date": None, "recovery_days": None,
        "recovery_rate": None, "recovery_pct": None,
    }
    df = df.dropna(subset=["index_value"]).sort_values("date").copy()
    event_dt = pd.Timestamp(event_date)

    pre_window = df[
        (df["date"] < event_dt) &
        (df["date"] >= event_dt - pd.Timedelta(days=pre_window_days))
    ]
    post = df[df["date"] > event_dt].copy()

    if len(pre_window) < 3:
        return {**_empty, "recovery_status": "insufficient_data",
                "interpretation": "Insufficient pre-event data."}
    if len(post) < 2:
        return {**_empty, "recovery_status": "insufficient_data",
                "interpretation": "Insufficient post-event data."}

    pre_mean = float(pre_window["index_value"].mean())
    pre_std = float(pre_window["index_value"].std(ddof=1))
    recovery_level = pre_mean - recovery_tolerance * pre_std

    post_min_idx = post["index_value"].idxmin()
    post_min = float(post["index_value"].min())
    post_min_date = post.loc[post_min_idx, "date"]

    # First post-minimum date where index ≥ recovery_level
    recovery_date = None
    for _, row in post[post["date"] >= post_min_date].iterrows():
        if row["index_value"] >= recovery_level:
            recovery_date = row["date"]
            break

    recovery_days = int((recovery_date - event_dt).days) if recovery_date else None

    # Recovery rate: slope (per month) from post-minimum to recovery or end
    slope_end = recovery_date if recovery_date is not None else post["date"].iloc[-1]
    slope_data = post[(post["date"] >= post_min_date) & (post["date"] <= slope_end)]
    recovery_rate = None
    if len(slope_data) >= 2:
        t = (slope_data["date"] - slope_data["date"].iloc[0]).dt.days.values.astype(float)
        v = slope_data["index_value"].values
        if t[-1] > 0:
            recovery_rate = float(np.polyfit(t, v, 1)[0] * 30)  # per month

    # Recovery completeness
    current = float(post["index_value"].iloc[-1])
    denom = pre_mean - post_min
    recovery_pct = float(max(0.0, min(100.0,
        (current - post_min) / denom * 100 if abs(denom) > 1e-6 else 100.0
    )))

    if recovery_pct < 20:
        status, label = "no_recovery", "No Recovery"
    elif recovery_pct < 80:
        status, label = "partial_recovery", "Partial Recovery"
    else:
        status, label = "full_recovery", "Full Recovery"

    period_end  = recovery_date if recovery_date is not None else post["date"].iloc[-1]
    period_days = int((period_end - event_dt).days)
    start_str   = event_dt.strftime("%Y-%m-%d")
    end_str     = pd.Timestamp(period_end).strftime("%Y-%m-%d")
    end_human   = pd.Timestamp(period_end).strftime("%B %Y")   # e.g. "June 2023"

    if recovery_date is not None:
        interp = (
            f"Recovery period: {start_str} to {end_str} ({period_days} days). "
            f"Vegetation returned to pre-event baseline by {end_human} "
            f"with {recovery_pct:.1f}% completeness."
        )
    else:
        interp = (
            f"Recovery period: {start_str} to {end_str} "
            f"({period_days} days, monitoring ongoing). "
            f"Vegetation has not returned to pre-event baseline "
            f"({recovery_pct:.1f}% completeness at end of monitoring period)."
        )

    return {
        "pre_mean": pre_mean,
        "pre_std": pre_std,
        "post_min": post_min,
        "post_min_date": post_min_date,
        "recovery_date": recovery_date,
        "recovery_days": recovery_days,
        "recovery_rate": recovery_rate,
        "recovery_pct": recovery_pct,
        "recovery_status": status,
        "interpretation": interp,
    }


def analyze_recovery_seasonal(
    df: pd.DataFrame,
    event_date: str,
    consec_months: int = 3,
) -> Dict[str, Any]:
    """
    Seasonal-aware recovery analysis using monthly climatology built from all
    pre-event observations.

    Instead of comparing against a flat pre-event mean, the recovery baseline
    is the expected value *for each calendar month* derived from pre-event data.
    Recovery is declared when observations fall within ±1σ of the monthly
    expectation for ``consec_months`` consecutive months after the post-event
    minimum.

    Parameters
    ----------
    df : pd.DataFrame
        Time series with ``date`` and ``index_value`` columns.
    event_date : str
        Date of the disturbance event (``YYYY-MM-DD``).
    consec_months : int
        Number of consecutive months within the seasonal envelope required to
        declare recovery.  Default 3.

    Returns
    -------
    dict
        Includes all standard recovery keys plus ``clim_months``
        (monthly climatology) and ``seasonal_departures`` (DataFrame).
    """
    _empty: Dict[str, Any] = {
        "pre_mean": None, "pre_std": None, "post_min": None,
        "post_min_date": None, "recovery_date": None, "recovery_days": None,
        "recovery_rate": None, "recovery_pct": None,
        "clim_months": None, "seasonal_departures": None, "method": "seasonal",
    }

    df = df.dropna(subset=["index_value"]).sort_values("date").copy()
    event_dt = pd.Timestamp(event_date)

    pre = df[df["date"] < event_dt].copy()
    post = df[df["date"] > event_dt].copy()

    if len(pre) < 6:
        return {**_empty, "recovery_status": "insufficient_data",
                "interpretation": "Insufficient pre-event data for seasonal climatology."}
    if len(post) < 2:
        return {**_empty, "recovery_status": "insufficient_data",
                "interpretation": "Insufficient post-event data."}

    # ── Monthly climatology from ALL pre-event data ───────────────────────────
    pre["month"] = pre["date"].dt.month
    global_std = float(pre["index_value"].std(ddof=1))
    clim_months: Dict[int, Dict[str, float]] = {}
    for m in range(1, 13):
        vals = pre.loc[pre["month"] == m, "index_value"]
        if len(vals) >= 2:
            clim_months[m] = {"mean": float(vals.mean()),
                               "std": float(vals.std(ddof=1))}
        elif len(vals) == 1:
            clim_months[m] = {"mean": float(vals.iloc[0]), "std": global_std}

    present = sorted(clim_months.keys())
    if len(present) < 3:
        return {**_empty, "recovery_status": "insufficient_data",
                "interpretation": "Insufficient monthly coverage for seasonal climatology."}

    # Fill missing months by linear interpolation around the annual cycle
    for m in range(1, 13):
        if m in clim_months:
            continue
        # Circular neighbours
        before = sorted([p for p in present if p < m] or [p - 12 for p in present if p > m])
        after  = sorted([p for p in present if p > m] or [p + 12 for p in present if p < m])
        if before and after:
            mb, ma = before[-1], after[0]
            cb = clim_months[((mb - 1) % 12) + 1]
            ca = clim_months[((ma - 1) % 12) + 1]
            t = (m - mb) / (ma - mb) if ma != mb else 0.5
            clim_months[m] = {
                "mean": cb["mean"] * (1 - t) + ca["mean"] * t,
                "std":  cb["std"]  * (1 - t) + ca["std"]  * t,
            }
        else:
            nearest = min(present, key=lambda p: abs(p - m))
            clim_months[m] = dict(clim_months[nearest])

    # ── Compute departure for every observation ───────────────────────────────
    df["month"] = df["date"].dt.month
    df["expected"]     = df["month"].map(lambda m: clim_months[m]["mean"])
    df["expected_std"] = df["month"].map(lambda m: clim_months[m]["std"])
    df["departure"]    = df["index_value"] - df["expected"]

    post_df = df[df["date"] > event_dt].sort_values("date").copy()

    post_min_idx  = post_df["index_value"].idxmin()
    post_min      = float(post_df["index_value"].min())
    post_min_date = post_df.loc[post_min_idx, "date"]

    # ── Consecutive-month recovery detection ─────────────────────────────────
    recovery_date = None
    consec, first_of_run = 0, None
    for _, row in post_df.iterrows():
        if row["date"] < post_min_date:
            continue
        if abs(row["departure"]) <= row["expected_std"]:
            consec += 1
            if consec == 1:
                first_of_run = row["date"]
            if consec >= consec_months:
                recovery_date = first_of_run
                break
        else:
            consec, first_of_run = 0, None

    recovery_days = int((recovery_date - event_dt).days) if recovery_date else None

    # ── Recovery rate: slope of departure from post-min onward ───────────────
    slope_end  = recovery_date if recovery_date is not None else post_df["date"].iloc[-1]
    slope_data = post_df[(post_df["date"] >= post_min_date) &
                         (post_df["date"] <= slope_end)]
    recovery_rate = None
    if len(slope_data) >= 2:
        t = (slope_data["date"] - slope_data["date"].iloc[0]).dt.days.values.astype(float)
        if t[-1] > 0:
            recovery_rate = float(np.polyfit(t, slope_data["departure"].values, 1)[0] * 30)

    # ── Recovery completeness (seasonal-departure-based) ─────────────────────
    recent_n  = min(consec_months, len(post_df))
    mean_dep_recent = float(np.mean(np.abs(post_df["departure"].iloc[-recent_n:].values)))
    max_dep         = float(np.max(np.abs(post_df["departure"].values)))

    if max_dep > 1e-6:
        recovery_pct = float(max(0.0, min(100.0, (1 - mean_dep_recent / max_dep) * 100)))
    else:
        recovery_pct = 100.0

    if recovery_pct < 20:
        status, label = "no_recovery",      "No Recovery"
    elif recovery_pct < 80:
        status, label = "partial_recovery", "Partial Recovery"
    else:
        status, label = "full_recovery",    "Full Recovery"

    period_end  = recovery_date if recovery_date is not None else post_df["date"].iloc[-1]
    period_days = int((period_end - event_dt).days)
    start_str   = event_dt.strftime("%Y-%m-%d")
    end_str     = pd.Timestamp(period_end).strftime("%Y-%m-%d")
    end_human   = pd.Timestamp(period_end).strftime("%B %Y")

    if recovery_date is not None:
        interp = (
            f"Recovery period: {start_str} to {end_str} ({period_days} days). "
            f"Vegetation returned to expected seasonal levels by {end_human} "
            f"with {recovery_pct:.1f}% completeness (seasonal departure basis)."
        )
    else:
        interp = (
            f"Recovery period: {start_str} to {end_str} "
            f"({period_days} days, monitoring ongoing). "
            f"Vegetation has not consistently returned to expected seasonal levels "
            f"({recovery_pct:.1f}% completeness at end of monitoring period)."
        )

    pre_mean = float(pre["index_value"].mean())
    pre_std  = float(pre["index_value"].std(ddof=1))

    return {
        "pre_mean":          pre_mean,
        "pre_std":           pre_std,
        "post_min":          post_min,
        "post_min_date":     post_min_date,
        "recovery_date":     recovery_date,
        "recovery_days":     recovery_days,
        "recovery_rate":     recovery_rate,
        "recovery_pct":      recovery_pct,
        "recovery_status":   status,
        "interpretation":    interp,
        "clim_months":       clim_months,
        "seasonal_departures": df[["date", "index_value",
                                    "expected", "expected_std",
                                    "departure"]].copy(),
        "method": "seasonal",
    }


# ── f. Multi-Point / Spatial Comparison ──────────────────────────────────────

def points_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load point locations from a CSV file.

    Required columns: ``lat``, ``lon``. Optional: ``label``.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.

    Returns
    -------
    list of dict
        Each entry has ``lat``, ``lon``, and ``label`` keys.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    missing = {"lat", "lon"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if "label" not in df.columns:
        df["label"] = [f"Point_{i + 1}" for i in range(len(df))]
    return df[["lat", "lon", "label"]].to_dict("records")


def extract_multi_point_time_series(
    points: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    satellite: str = "sentinel2",
    index: str = "NDVI",
    composite: str = "monthly",
    scale: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Extract and composite time series for multiple point locations.

    Parameters
    ----------
    points : list of dict
        Each entry must have ``lat``, ``lon``, and ``label`` keys.
    start_date, end_date : str
        Date range (``YYYY-MM-DD``).
    satellite, index, composite, scale : str / int
        Analysis parameters (same as :func:`extract_point_time_series`).

    Returns
    -------
    dict
        Mapping of ``label`` → composited :class:`pandas.DataFrame`.
    """
    results: Dict[str, pd.DataFrame] = {}
    for pt in points:
        label = pt.get("label", f"({pt['lat']:.3f}, {pt['lon']:.3f})")
        logger.info("Processing point: %s …", label)
        try:
            raw = extract_point_time_series(
                lat=pt["lat"], lon=pt["lon"],
                start_date=start_date, end_date=end_date,
                satellite=satellite, index=index, scale=scale,
            )
            results[label] = apply_temporal_composite(raw, composite)
        except Exception as exc:
            logger.warning("Failed for point %s: %s", label, exc)
            results[label] = pd.DataFrame()
    return results


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def run_time_series_analysis(
    location: Union[Tuple[float, float], "ee.Geometry"],
    start_date: str,
    end_date: str,
    satellite: str = "sentinel2",
    index: str = "NDVI",
    composite: str = "monthly",
    anomaly_methods: Optional[List[str]] = None,
    anomaly_threshold: float = 2.0,
    detect_changepoints: bool = False,
    event_date: Optional[str] = None,
    recovery_analysis: bool = False,
    recovery_style: str = "seasonal",
    scale: int = 30,
    output_dir: Optional[str] = None,
    hurricane_events: Optional[List[Dict]] = None,
    plot_type: str = "all",
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the full time series analysis pipeline.

    Parameters
    ----------
    location : (lat, lon) tuple or ee.Geometry
        ``(lat, lon)`` → point extraction; ``ee.Geometry`` → ROI spatial mean.
    start_date, end_date : str
        Analysis date range (``YYYY-MM-DD``).
    satellite, index : str
        Sensor and vegetation index.
    composite : str
        Temporal compositing interval
        (``"raw"``, ``"weekly"``, ``"biweekly"``, ``"monthly"``).
    anomaly_methods : list of str, optional
        Methods to apply. Defaults to all three. Use ``["all"]`` for all.
    anomaly_threshold : float
        Z-score threshold for anomaly detection.
    detect_changepoints : bool
        Whether to run CUSUM (and ruptures if installed).
    event_date : str, optional
        Hurricane/event date for annotation and recovery analysis.
    recovery_analysis : bool
        Whether to run recovery analysis (requires ``event_date``).
    scale : int
        GEE reduction / sampling resolution in metres.
    output_dir : str, optional
        Directory for saving output files.
    hurricane_events : list of dict, optional
        Catalog entries (``name``, ``date``, ``category``) pre-filtered to the
        analysis date range.  Passed to all plot functions as visual markers.
    recovery_style : str
        Recovery visualization style: ``"seasonal"`` (default) uses a monthly
        climatological baseline with ±1σ seasonal envelope;
        ``"flat"`` uses the legacy flat pre-event mean ±1σ band;
        ``"all"`` generates both.
    plot_type : str
        Controls which plot families are generated when ``output_dir`` is set.
        Choices: ``"raw"`` (existing plots only), ``"residual"``,
        ``"zscore"``, ``"departure"``, ``"cusum"``, or ``"all"``
        (raw + all four detrended types + combined panel).  Default ``"all"``.

    Returns
    -------
    dict
        Full results including ``df``, ``anomalies``, ``changepoints``,
        ``recovery`` (primary style), ``recovery_flat``, ``recovery_seasonal``,
        and ``plot_paths``.
    """
    import ee as _ee

    logger.info("=== Time Series Analysis ===")
    logger.info("Date range: %s → %s | %s | %s | composite: %s",
                start_date, end_date, satellite, index, composite)

    # 1. Extract
    if isinstance(location, tuple):
        lat, lon = location
        raw_df = extract_point_time_series(
            lat, lon, start_date, end_date, satellite, index, scale=scale
        )
        location_label = f"Point ({lat:.4f}, {lon:.4f})"
    else:
        # For ROI spatial averages, a coarse scale is both accurate enough and
        # far faster — 30 m over a 100+ km² ROI generates millions of pixels
        # per image and reliably triggers GEE computation timeouts.  If the
        # caller passed scale=30 (the point default), silently upgrade to 250 m.
        roi_scale = scale if scale >= 100 else 250
        raw_df = extract_roi_time_series(
            location, start_date, end_date, satellite, index,
            scale=roi_scale,
            progress_callback=progress_callback,
        )
        location_label = "ROI spatial mean"

    if raw_df.empty:
        raise ValueError("No data extracted — check date range, ROI, and GEE connection.")

    # 2. Composite
    df = apply_temporal_composite(raw_df, composite)

    # 3. Seasonal decomposition
    stl_result = apply_stl_decomposition(df, composite=composite)
    harmonic_result = fit_harmonic_model(df)

    # 4. Anomaly detection
    methods = anomaly_methods
    if methods is None or methods == ["all"]:
        methods = list(ANOMALY_METHODS)
    anomalies = detect_all_anomalies(
        df,
        stl_result=stl_result,
        harmonic_result=harmonic_result,
        threshold=anomaly_threshold,
        methods=methods,
    )

    # 5. Change points
    changepoints = pd.DataFrame()
    if detect_changepoints:
        cp_cusum = detect_changepoints_cusum(df, threshold=4.0)
        cp_rpt = detect_changepoints_ruptures(df)
        changepoints = pd.concat([cp_cusum, cp_rpt], ignore_index=True)

    # 6. Recovery
    recovery_flat:     Dict[str, Any] = {}
    recovery_seasonal: Dict[str, Any] = {}
    if recovery_analysis and event_date:
        run_flat     = recovery_style in ("flat", "all")
        run_seasonal = recovery_style in ("seasonal", "all")
        if run_flat:
            recovery_flat = analyze_recovery(df, event_date)
        if run_seasonal:
            recovery_seasonal = analyze_recovery_seasonal(df, event_date)
    # `recovery` is the primary result exposed to callers (backward-compatible key)
    recovery = (
        recovery_seasonal if recovery_style in ("seasonal", "all")
        else recovery_flat
    )

    # 7. Visualizations
    plot_paths: Dict[str, str] = {}
    if output_dir:
        from .utils import ensure_dir
        out = ensure_dir(output_dir)

        _safe_plot(
            plot_time_series_interactive,
            plot_paths, "time_series_html",
            df, stl_result, anomalies, event_date, index,
            str(out / f"{index}_time_series_full.html"), location_label,
            hurricane_events=hurricane_events,
        )
        _safe_plot(
            plot_time_series_static,
            plot_paths, "time_series_png",
            df, stl_result, anomalies, event_date, index,
            str(out / f"{index}_time_series.png"), location_label,
            hurricane_events=hurricane_events,
        )
        if stl_result is not None:
            _safe_plot(
                plot_stl_decomposition,
                plot_paths, "stl_decomposition",
                stl_result, index, str(out / f"{index}_stl_decomposition.png"),
                hurricane_events=hurricane_events,
            )
        if not anomalies.empty:
            _safe_plot(
                plot_anomaly_timeline,
                plot_paths, "anomaly_timeline",
                anomalies, index, str(out / f"{index}_anomaly_timeline.png"), event_date,
                hurricane_events=hurricane_events,
            )
        if event_date:
            if recovery_seasonal:
                _safe_plot(
                    plot_recovery_trajectory_seasonal,
                    plot_paths, "recovery_trajectory_seasonal",
                    df, recovery_seasonal, event_date, index,
                    str(out / f"{index}_recovery_trajectory_seasonal.png"),
                    hurricane_events=hurricane_events,
                )
            if recovery_flat:
                _safe_plot(
                    plot_recovery_trajectory,
                    plot_paths, "recovery_trajectory",
                    df, recovery_flat, event_date, index,
                    str(out / f"{index}_recovery_trajectory.png"),
                    hurricane_events=hurricane_events,
                )

        # ── Detrended / normalised views ─────────────────────────────────────
        _detrended = (
            {"residual", "zscore", "departure", "cusum"}
            if plot_type == "all"
            else ({plot_type} - {"raw"})
        )
        if "residual" in _detrended:
            _safe_plot(
                plot_residual,
                plot_paths, "residual",
                df, stl_result, str(out / f"{index}_residual.png"),
                index, event_date, hurricane_events,
            )
        if "zscore" in _detrended:
            _safe_plot(
                plot_standardized_anomaly,
                plot_paths, "zscore",
                df, stl_result, str(out / f"{index}_zscore.png"),
                index, event_date, hurricane_events,
            )
        if "departure" in _detrended:
            _safe_plot(
                plot_seasonal_departure,
                plot_paths, "departure",
                df, str(out / f"{index}_seasonal_departure.png"),
                index, event_date, hurricane_events,
            )
        if "cusum" in _detrended:
            _safe_plot(
                plot_cusum,
                plot_paths, "cusum",
                df, changepoints, str(out / f"{index}_cusum.png"),
                index, event_date, hurricane_events,
            )
        if plot_type == "all":
            _safe_plot(
                plot_combined_panel,
                plot_paths, "combined_panel",
                df, stl_result, changepoints,
                str(out / f"{index}_combined_panel.png"),
                index, event_date, hurricane_events,
            )

    n_anom = len(anomalies) if not anomalies.empty else 0
    logger.info("Analysis complete. Detected %d anomalies.", n_anom)

    return {
        "raw_df": raw_df,
        "df": df,
        "location_label": location_label,
        "stl_result": stl_result,
        "harmonic_result": harmonic_result,
        "anomalies": anomalies,
        "changepoints": changepoints,
        "recovery": recovery,             # primary (selected style)
        "recovery_flat": recovery_flat,
        "recovery_seasonal": recovery_seasonal,
        "plot_paths": plot_paths,
        "satellite": satellite,
        "index": index,
        "composite": composite,
        "start_date": start_date,
        "end_date": end_date,
        "event_date": event_date,
        "recovery_style": recovery_style,
    }


def _safe_plot(fn, plot_paths, key, *args, **kwargs):
    """Call a plot function and store the path; log a warning on failure."""
    try:
        path = fn(*args, **kwargs)
        plot_paths[key] = path
    except Exception as exc:
        logger.warning("Plot %s failed: %s", key, exc)


# ── h. Visualization ──────────────────────────────────────────────────────────

_HURRICANE_COLOR = "#8B0000"  # dark red — distinct from crimson event lines


def _draw_hurricane_markers(ax, hurricane_events) -> None:
    """
    Draw dashed vertical lines with angled name/category labels for hurricane
    events on a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    hurricane_events : list of dict or None
        Each dict: ``{"name": str, "date": "YYYY-MM-DD", "category": int}``.
    """
    if not hurricane_events:
        return
    import matplotlib.dates as mdates
    for he in hurricane_events:
        dt = pd.Timestamp(he["date"])
        cat = he.get("category", "")
        name = he.get("name", "")
        label = f"{name} (Cat {cat})" if cat else name
        ax.axvline(dt, color=_HURRICANE_COLOR, linewidth=1.4,
                   linestyle="--", zorder=4, alpha=0.85)
        ax.text(
            mdates.date2num(dt.to_pydatetime()), 0.97,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=45, fontsize=7, color=_HURRICANE_COLOR,
            ha="left", va="top", clip_on=True,
        )


def _add_hurricane_vlines(fig, hurricane_events, has_trend: bool) -> None:
    """
    Add dashed vertical lines with angled name/category annotations for
    hurricane events to a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    hurricane_events : list of dict or None
    has_trend : bool
        Whether the figure has a second (trend) subplot row.
    """
    if not hurricane_events:
        return
    rows = [1, 2] if has_trend else [None]
    for he in hurricane_events:
        dt = pd.Timestamp(he["date"])
        cat = he.get("category", "")
        name = he.get("name", "")
        label = f"{name} (Cat {cat})" if cat else name
        x_ms = dt.timestamp() * 1000
        for i, row in enumerate(rows):
            kw = {"row": row, "col": 1} if row else {}
            # Only annotate on the first (main) panel to avoid duplicate labels
            ann_kw = dict(
                annotation_text=label,
                annotation_position="top left",
                annotation_textangle=-45,
                annotation_font=dict(size=9, color=_HURRICANE_COLOR),
            ) if i == 0 else {}
            fig.add_vline(
                x=x_ms,
                line_dash="dash", line_color=_HURRICANE_COLOR, line_width=1.5,
                **ann_kw,
                **kw,
            )


def plot_time_series_interactive(
    df: pd.DataFrame,
    stl_result: Optional[Dict],
    anomalies: pd.DataFrame,
    event_date: Optional[str],
    index: str,
    output_path: str,
    title: str = "",
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """
    Generate an interactive Plotly time series chart with anomaly markers
    and event date annotations.

    The chart has two panels when STL results are available:
    the full index series (top) and the trend component (bottom).

    Parameters
    ----------
    df : pd.DataFrame
        Composited time series.
    stl_result : dict, optional
        STL decomposition output.
    anomalies : pd.DataFrame
        Anomaly records from :func:`detect_all_anomalies`.
    event_date : str, optional
        Event date for vertical line annotation.
    index : str
        Index name for axis labels.
    output_path : str
        Path to save the ``.html`` file.
    title : str
        Chart subtitle (e.g. location label).
    hurricane_events : list of dict, optional
        Hurricane catalog entries (``name``, ``date``, ``category``).
        Events within the series date range are marked with dark-red dashed
        lines and angled ``"Name (Cat N)"`` annotations.

    Returns
    -------
    str
        Output path.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_trend = stl_result is not None

    if has_trend:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=[f"{index} Time Series", "Trend Component"],
            vertical_spacing=0.08,
        )
    else:
        fig = go.Figure()

    def _add(trace, row=None):
        if has_trend and row:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

    # Raw observations
    _add(go.Scatter(
        x=df["date"], y=df["index_value"],
        mode="markers",
        marker=dict(size=5, color="steelblue", opacity=0.7),
        name="Observed",
    ), row=1)

    # Seasonal fit (trend + seasonal)
    if has_trend:
        expected = np.asarray(stl_result["trend"]) + np.asarray(stl_result["seasonal"])
        _add(go.Scatter(
            x=df["date"], y=expected,
            mode="lines", line=dict(color="darkorange", width=2),
            name="Seasonal fit",
        ), row=1)
        _add(go.Scatter(
            x=df["date"], y=np.asarray(stl_result["trend"]),
            mode="lines", line=dict(color="darkgreen", width=2),
            name="Trend",
        ), row=2)

    # Anomaly markers per method
    method_styles = {
        "zscore": ("red", "x"),
        "moving_window": ("darkorange", "circle-x"),
        "climatology": ("purple", "diamond"),
    }
    if not anomalies.empty:
        for method, (color, symbol) in method_styles.items():
            sub = anomalies[anomalies["method"] == method]
            if sub.empty:
                continue
            _add(go.Scatter(
                x=sub["date"], y=sub["observed_value"],
                mode="markers",
                marker=dict(size=11, color=color, symbol=symbol,
                            line=dict(width=1.5, color="black")),
                name=f"Anomaly ({method})",
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    f"{method}<br>"
                    "Observed: %{y:.4f}<br>"
                    "Z: %{customdata:.2f}<extra></extra>"
                ),
                customdata=sub["z_score"].values,
            ), row=1)

    # Event date vertical line
    if event_date:
        event_ms = pd.Timestamp(event_date).timestamp() * 1000
        for row in ([1, 2] if has_trend else [None]):
            kw = {"row": row, "col": 1} if row else {}
            fig.add_vline(
                x=event_ms,
                line_dash="dash", line_color="crimson", line_width=2,
                annotation_text=f"Event: {event_date}",
                annotation_position="top right",
                **kw,
            )

    # Hurricane event markers (dark red, named)
    _add_hurricane_vlines(fig, hurricane_events, has_trend)

    fig.update_layout(
        title=f"{index} Time Series — {title}",
        yaxis_title=index,
        hovermode="x unified",
        height=580 if has_trend else 420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.write_html(output_path)
    logger.info("Interactive plot saved: %s", output_path)
    return output_path


def plot_time_series_static(
    df: pd.DataFrame,
    stl_result: Optional[Dict],
    anomalies: pd.DataFrame,
    event_date: Optional[str],
    index: str,
    output_path: str,
    title: str = "",
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """Generate a static matplotlib time series plot (publication quality)."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.scatter(df["date"], df["index_value"], s=15, alpha=0.6,
                color="steelblue", label="Observed", zorder=3)

    if stl_result is not None:
        expected = np.asarray(stl_result["trend"]) + np.asarray(stl_result["seasonal"])
        ax1.plot(df["date"], expected, color="darkorange", lw=2,
                 label="Seasonal fit", zorder=4)
        ax2.plot(df["date"], stl_result["trend"],
                 color="darkgreen", lw=2, label="Trend")

    anomaly_styles = {
        "zscore": ("red", "X"),
        "moving_window": ("darkorange", "P"),
        "climatology": ("purple", "D"),
    }
    if not anomalies.empty:
        for method, (color, marker) in anomaly_styles.items():
            sub = anomalies[anomalies["method"] == method]
            if not sub.empty:
                ax1.scatter(sub["date"], sub["observed_value"], s=80,
                            color=color, marker=marker, zorder=5,
                            label=f"Anomaly ({method})",
                            edgecolors="black", linewidths=0.5)

    if event_date:
        edt = pd.Timestamp(event_date)
        for ax in (ax1, ax2):
            ax.axvline(edt, color="crimson", lw=1.5, linestyle="--",
                       label="Event date" if ax is ax1 else None)

    # Hurricane event markers (dark red, named)
    _draw_hurricane_markers(ax1, hurricane_events)
    _draw_hurricane_markers(ax2, hurricane_events)

    ax1.set_ylabel(index)
    ax1.set_title(f"{index} Time Series — {title}", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Trend")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Static plot saved: %s", output_path)
    return output_path


def plot_stl_decomposition(
    stl_result: Dict,
    index: str,
    output_path: str,
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """Generate a 4-panel STL decomposition plot."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = stl_result["dates"]
    panels = [
        (stl_result["observed"],  "Observed",  "steelblue"),
        (stl_result["trend"],     "Trend",     "darkgreen"),
        (stl_result["seasonal"],  "Seasonal",  "darkorange"),
        (stl_result["residual"],  "Residual",  "gray"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, (vals, label, color) in zip(axes, panels):
        ax.plot(dates, vals, color=color, lw=1.5)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        if label == "Residual":
            ax.axhline(0, color="black", lw=0.8, linestyle="--")
        _draw_hurricane_markers(ax, hurricane_events)

    axes[0].set_title(f"STL Decomposition — {index}", fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("STL decomposition plot saved: %s", output_path)
    return output_path


def plot_anomaly_timeline(
    anomalies: pd.DataFrame,
    index: str,
    output_path: str,
    event_date: Optional[str] = None,
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a bar-chart timeline of anomaly z-scores coloured by severity.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    severity_colors = {
        "extreme": "#c0392b", "moderate": "#e67e22", "mild": "#f1c40f",
        "extreme_positive": "#2980b9", "moderate_positive": "#3498db",
        "mild_positive": "#85c1e9",
    }

    fig, ax = plt.subplots(figsize=(14, 4))
    for _, row in anomalies.iterrows():
        color = severity_colors.get(row.get("severity", "mild"), "gray")
        ax.bar(row["date"], row.get("z_score", 0), color=color, alpha=0.8, width=5)

    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(2.0, color="orange", lw=0.8, linestyle="--", label="+2σ")
    ax.axhline(-2.0, color="orange", lw=0.8, linestyle="--", label="−2σ")

    if event_date:
        ax.axvline(pd.Timestamp(event_date), color="crimson", lw=2,
                   linestyle="--", label="Event date")

    # Hurricane event markers (dark red, named)
    _draw_hurricane_markers(ax, hurricane_events)

    ax.set_ylabel("Z-Score")
    ax.set_title(f"Anomaly Timeline — {index}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Anomaly timeline saved: %s", output_path)
    return output_path


def plot_recovery_trajectory(
    df: pd.DataFrame,
    recovery: Dict,
    event_date: str,
    index: str,
    output_path: str,
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """
    Generate a recovery trajectory plot with pre-event baseline band and
    annotated recovery date.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    event_dt = pd.Timestamp(event_date)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(df["date"], df["index_value"], s=15, color="steelblue",
               alpha=0.5, label="Observed", zorder=3)

    pre_mean = recovery.get("pre_mean")
    pre_std = recovery.get("pre_std")
    if pre_mean is not None and pre_std is not None:
        ax.axhspan(pre_mean - pre_std, pre_mean + pre_std,
                   alpha=0.15, color="green", label="Pre-event ±1σ")
        ax.axhline(pre_mean, color="green", lw=1.5, linestyle="--",
                   label="Pre-event mean")

    post_min = recovery.get("post_min")
    if post_min is not None:
        ax.axhline(post_min, color="red", lw=1, linestyle=":",
                   label="Post-event minimum")

    ax.axvline(event_dt, color="crimson", lw=2, linestyle="--",
               label="Event date")

    rec_date = recovery.get("recovery_date")
    if rec_date is not None:
        ax.axvline(pd.Timestamp(rec_date), color="darkgreen", lw=2,
                   linestyle="--", label=f"Recovery: {str(rec_date)[:10]}")

    # Hurricane event markers (dark red, named)
    _draw_hurricane_markers(ax, hurricane_events)

    pct = recovery.get("recovery_pct", 0)
    status = recovery.get("recovery_status", "")
    ax.set_title(
        f"Recovery Trajectory — {index}\n"
        f"{status.replace('_', ' ').title()} | {pct:.1f}% completeness",
        fontweight="bold",
    )
    ax.set_ylabel(index)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Recovery trajectory saved: %s", output_path)
    return output_path


def plot_recovery_trajectory_seasonal(
    df: pd.DataFrame,
    recovery: Dict,
    event_date: str,
    index: str,
    output_path: str,
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """
    Two-panel seasonal recovery plot.

    Top panel
        Observed scatter coloured **blue** (within seasonal ±1σ envelope) or
        **red** (outside), overlaid on a smoothed seasonal baseline curve with
        shaded ±1σ band derived from pre-event monthly climatology.

    Bottom panel
        Departure from the seasonal expectation over time (observed minus
        expected for that calendar month).  A zero line and ±1σ reference band
        show the recovery target.  Recovery is achieved when departures return
        to near zero.

    Parameters
    ----------
    df : pd.DataFrame
        Full time series (``date``, ``index_value``).
    recovery : dict
        Output of :func:`analyze_recovery_seasonal`.
    event_date : str
        Event date ``YYYY-MM-DD``.
    index : str
        Index name for axis labels.
    output_path : str
        Path to save the figure.
    hurricane_events : list of dict, optional
        Hurricane catalog entries for event markers.

    Returns
    -------
    str
        Output path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import gaussian_filter1d

    clim_months       = recovery.get("clim_months")
    seasonal_dep      = recovery.get("seasonal_departures")

    if clim_months is None or seasonal_dep is None or seasonal_dep.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "Seasonal climatology not available\n"
                "(need pre-event data spanning ≥ 3 calendar months)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, style="italic", color="gray")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    event_dt = pd.Timestamp(event_date)

    # ── Build smooth seasonal envelope at daily resolution ───────────────────
    full_dates = pd.date_range(
        df["date"].min() - pd.Timedelta(days=15),
        df["date"].max() + pd.Timedelta(days=15),
        freq="D",
    )
    env_mean = np.array([clim_months.get(d.month, {}).get("mean", np.nan)
                         for d in full_dates], dtype=float)
    env_std  = np.array([clim_months.get(d.month, {}).get("std",  np.nan)
                         for d in full_dates], dtype=float)

    # Fill any NaNs by linear interpolation before smoothing
    for arr in (env_mean, env_std):
        nans = ~np.isfinite(arr)
        if nans.any() and (~nans).any():
            idx = np.arange(len(arr))
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])

    smooth_mean = gaussian_filter1d(env_mean, sigma=30)
    smooth_std  = gaussian_filter1d(env_std,  sigma=30)

    # ── Classify observations (inside / outside envelope) ────────────────────
    sd = seasonal_dep.copy()
    inside = sd["departure"].abs() <= sd["expected_std"]

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ── Top panel: observations + seasonal envelope ───────────────────────────
    ax1.fill_between(
        full_dates, smooth_mean - smooth_std, smooth_mean + smooth_std,
        alpha=0.20, color="green", label="Seasonal ±1σ",
    )
    ax1.plot(full_dates, smooth_mean, color="green", lw=1.5, label="Seasonal expected")

    ax1.scatter(sd.loc[inside,  "date"], sd.loc[inside,  "index_value"],
                s=22, color="steelblue", alpha=0.85, zorder=5,
                label="Within envelope")
    ax1.scatter(sd.loc[~inside, "date"], sd.loc[~inside, "index_value"],
                s=22, color="crimson",   alpha=0.90, zorder=5, marker="^",
                label="Outside envelope")

    ax1.axvline(event_dt, color="crimson", lw=2, linestyle="--", label="Event date")
    rec_date = recovery.get("recovery_date")
    if rec_date is not None:
        ax1.axvline(pd.Timestamp(rec_date), color="darkgreen", lw=2,
                    linestyle="--", label=f"Recovery: {str(rec_date)[:10]}")

    _draw_hurricane_markers(ax1, hurricane_events)

    pct    = recovery.get("recovery_pct", 0)
    status = recovery.get("recovery_status", "")
    ax1.set_title(
        f"Recovery Trajectory — {index} (Seasonal Baseline)\n"
        f"{status.replace('_', ' ').title()} | {pct:.1f}% completeness (seasonal)",
        fontweight="bold",
    )
    ax1.set_ylabel(index)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Bottom panel: seasonal anomaly trajectory ─────────────────────────────
    bar_w   = pd.Timedelta(days=15)
    mean_envelope_std = float(sd["expected_std"].mean())

    ax2.fill_between(
        [sd["date"].min() - pd.Timedelta(days=30),
         sd["date"].max() + pd.Timedelta(days=30)],
        -mean_envelope_std, mean_envelope_std,
        alpha=0.10, color="green", label="±1σ target zone",
    )
    ax2.bar(sd.loc[inside,  "date"], sd.loc[inside,  "departure"],
            width=bar_w, color="steelblue", alpha=0.75, label="Within ±1σ")
    ax2.bar(sd.loc[~inside, "date"], sd.loc[~inside, "departure"],
            width=bar_w, color="crimson",   alpha=0.85, label="Outside ±1σ")

    ax2.axhline(0,                  color="black", lw=0.9)
    ax2.axhline( mean_envelope_std, color="gray",  lw=0.8, linestyle="--", alpha=0.6,
                 label="+1σ ref")
    ax2.axhline(-mean_envelope_std, color="gray",  lw=0.8, linestyle="--", alpha=0.6,
                 label="−1σ ref")

    ax2.axvline(event_dt, color="crimson", lw=1.5, linestyle="--")
    if rec_date is not None:
        ax2.axvline(pd.Timestamp(rec_date), color="darkgreen", lw=1.5, linestyle="--")
    _draw_hurricane_markers(ax2, hurricane_events)

    ax2.set_ylabel("Departure from\nSeasonal Expected")
    ax2.set_title("Seasonal Anomaly Trajectory", fontweight="bold", fontsize=10)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.25)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Seasonal recovery trajectory saved: %s", output_path)
    return output_path


def plot_multi_point_comparison(
    point_data: Dict[str, pd.DataFrame],
    index: str,
    output_path: str,
    event_date: Optional[str] = None,
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """
    Plot overlaid time series for multiple point locations.

    Parameters
    ----------
    point_data : dict
        Mapping of label → DataFrame from
        :func:`extract_multi_point_time_series`.
    index : str
        Index name for axis labels.
    output_path : str
        Path to save the figure.
    event_date : str, optional
        Event date for annotation.
    hurricane_events : list of dict, optional
        Hurricane catalog entries; see :func:`_draw_hurricane_markers`.

    Returns
    -------
    str
        Output path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.cm as cm

    colors = cm.tab10(np.linspace(0, 1, max(len(point_data), 1)))

    fig, ax = plt.subplots(figsize=(14, 6))
    for (label, df), color in zip(point_data.items(), colors):
        if df.empty:
            continue
        ax.plot(df["date"], df["index_value"], lw=1.5, alpha=0.85,
                color=color, label=label)
        ax.scatter(df["date"], df["index_value"], s=10, color=color, alpha=0.4)

    if event_date:
        ax.axvline(pd.Timestamp(event_date), color="crimson", lw=2,
                   linestyle="--", label="Event date")

    # Hurricane event markers (dark red, named)
    _draw_hurricane_markers(ax, hurricane_events)

    ax.set_ylabel(index)
    ax.set_title(f"Multi-Point {index} Comparison", fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Multi-point comparison saved: %s", output_path)
    return output_path


# ── i. Detrended / Normalised Views ───────────────────────────────────────────


def _compute_seasonal_departure(df: pd.DataFrame, doy_window: int = 15) -> np.ndarray:
    """
    Compute leave-one-year-out seasonal departure for each observation.

    For each data point the climatological mean is estimated from all
    observations in *other* years whose day-of-year falls within
    ``±doy_window`` days (circular, wrapping at year boundary).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``date`` (Timestamp) and ``index_value`` columns.
    doy_window : int
        Half-width of the day-of-year search window (default 15).

    Returns
    -------
    np.ndarray
        Departure array aligned with ``df``.  Entries are ``NaN`` where
        fewer than 2 reference observations exist (e.g. first year only).
    """
    dates = pd.to_datetime(df["date"])
    values = df["index_value"].values.astype(float)
    doys = dates.dt.dayofyear.values.astype(int)
    years = dates.dt.year.values.astype(int)
    departures = np.full(len(df), np.nan)
    for i in range(len(df)):
        other = years != years[i]
        diff = np.abs(doys[other] - doys[i])
        diff = np.minimum(diff, 366 - diff)   # circular DOY distance
        neighbors = values[other][diff <= doy_window]
        if len(neighbors) >= 2:
            departures[i] = values[i] - neighbors.mean()
    return departures


def plot_residual(
    df: pd.DataFrame,
    stl_result: Optional[Dict],
    output_path: str,
    index: str = "NDVI",
    event_date: Optional[str] = None,
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """
    Plot STL residuals as a bar chart with a ±2σ grey band.

    Positive outliers (> +2σ) are shown in royal-blue; negative outliers
    (< -2σ) are shown in crimson so anomalous dips stand out immediately.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``date`` and ``index_value``.
    stl_result : dict or None
        STL decomposition result from :func:`apply_stl_decomposition`.
        If ``None`` an informational message is rendered instead.
    output_path : str
        Path to save the PNG.
    index : str
        Index name for axis labels.
    event_date : str, optional
        Event date for a vertical annotation.
    hurricane_events : list of dict, optional
        Hurricane catalog entries for vertical markers.

    Returns
    -------
    str
        Output path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(13, 4))
    if stl_result is None:
        try:
            from statsmodels.tsa.seasonal import STL as _STL  # noqa: F401
            _reason = "insufficient data (≥ 2 years / 25 monthly obs required)"
        except ImportError:
            _reason = "statsmodels not installed — run: pip install statsmodels"
        ax.text(0.5, 0.5, f"STL not available: {_reason}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, style="italic", color="gray")
    else:
        residuals = np.asarray(stl_result["residual"], dtype=float)
        dates = pd.to_datetime(df["date"])
        sigma = np.nanstd(residuals)

        ax.axhspan(-2 * sigma, 2 * sigma, alpha=0.12, color="gray", label="±2σ band")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.axhline(2 * sigma, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax.axhline(-2 * sigma, color="gray", lw=0.8, linestyle="--", alpha=0.5)

        pos_out = residuals > 2 * sigma
        neg_out = residuals < -2 * sigma
        normal = ~pos_out & ~neg_out
        bar_w = pd.Timedelta(days=15)
        ax.bar(dates[normal], residuals[normal], width=bar_w,
               color="steelblue", alpha=0.6, label="Residual")
        ax.bar(dates[pos_out], residuals[pos_out], width=bar_w,
               color="royalblue", alpha=0.9, label="Positive outlier (>+2σ)")
        ax.bar(dates[neg_out], residuals[neg_out], width=bar_w,
               color="crimson", alpha=0.9, label="Negative outlier (<-2σ)")

        if event_date:
            ax.axvline(pd.Timestamp(event_date), color="crimson", lw=1.5,
                       linestyle="--", label="Event date")
        _draw_hurricane_markers(ax, hurricane_events)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30, ha="right")

    ax.set_ylabel("Residual")
    ax.set_title(f"{index} STL Residuals (±2σ band)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Residual plot saved: %s", output_path)
    return output_path


def plot_standardized_anomaly(
    df: pd.DataFrame,
    stl_result: Optional[Dict],
    output_path: str,
    index: str = "NDVI",
    event_date: Optional[str] = None,
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """
    Plot z-scores of STL residuals with ±2σ / ±3σ significance bands.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``date`` and ``index_value``.
    stl_result : dict or None
        STL result; if ``None`` an informational message is shown.
    output_path : str
        Path to save the PNG.
    index : str
        Index name.
    event_date : str, optional
        Vertical event annotation.
    hurricane_events : list of dict, optional
        Hurricane markers.

    Returns
    -------
    str
        Output path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(13, 4))
    if stl_result is None:
        try:
            from statsmodels.tsa.seasonal import STL as _STL  # noqa: F401
            _reason = "insufficient data (≥ 2 years / 25 monthly obs required)"
        except ImportError:
            _reason = "statsmodels not installed — run: pip install statsmodels"
        ax.text(0.5, 0.5, f"STL not available: {_reason}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, style="italic", color="gray")
    else:
        residuals = np.asarray(stl_result["residual"], dtype=float)
        dates = pd.to_datetime(df["date"])
        sigma = np.nanstd(residuals)
        z = residuals / sigma if sigma > 0 else residuals

        ax.axhspan(-2, 2, alpha=0.08, color="steelblue", label="±2σ (95%)")
        ax.axhspan(-3, -2, alpha=0.12, color="darkorange", label="±3σ (99.7%)")
        ax.axhspan(2, 3, alpha=0.12, color="darkorange")
        ax.axhspan(-10, -3, alpha=0.12, color="crimson", label=">3σ")
        ax.axhspan(3, 10, alpha=0.12, color="crimson")
        z_min = float(np.nanmin(z)); z_max = float(np.nanmax(z))
        ax.set_ylim(max(-5.5, z_min - 0.5), min(5.5, z_max + 0.5))

        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        for lvl in (-3, -2, 2, 3):
            ax.axhline(lvl, color="gray", lw=0.7, linestyle="--", alpha=0.6)

        ax.scatter(dates, z, s=20, color="steelblue", alpha=0.7, zorder=4)
        ax.plot(dates, z, lw=0.8, color="steelblue", alpha=0.5)

        if event_date:
            ax.axvline(pd.Timestamp(event_date), color="crimson", lw=1.5,
                       linestyle="--", label="Event date")
        _draw_hurricane_markers(ax, hurricane_events)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30, ha="right")

    ax.set_ylabel("Standard Deviations from Expected")
    ax.set_title(f"{index} Standardized Anomaly (Z-Score)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Standardized anomaly plot saved: %s", output_path)
    return output_path


def plot_seasonal_departure(
    df: pd.DataFrame,
    output_path: str,
    index: str = "NDVI",
    event_date: Optional[str] = None,
    hurricane_events: Optional[List[Dict]] = None,
    doy_window: int = 15,
) -> str:
    """
    Plot leave-one-year-out seasonal departure from DOY climatology.

    Model-free: no STL required.  Each observation's departure is
    ``observed − mean(same-DOY obs from other years ± doy_window days)``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``date`` and ``index_value``.
    output_path : str
        Path to save the PNG.
    index : str
        Index name.
    event_date : str, optional
        Vertical event annotation.
    hurricane_events : list of dict, optional
        Hurricane markers.
    doy_window : int
        ±day-of-year half-window for climatological mean (default 15).

    Returns
    -------
    str
        Output path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    departures = _compute_seasonal_departure(df, doy_window=doy_window)
    dates = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(13, 4))
    valid = ~np.isnan(departures)
    if valid.sum() < 2:
        ax.text(
            0.5, 0.5,
            "Insufficient data for seasonal departure\n"
            "(need ≥ 2 years with ±15-day DOY overlap)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, style="italic", color="gray",
        )
    else:
        sigma = np.nanstd(departures)
        ax.axhspan(-2 * sigma, 2 * sigma, alpha=0.10, color="steelblue",
                   label="±2σ band")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)

        bar_w = pd.Timedelta(days=15)
        pos = departures >= 0
        neg = departures < 0
        ax.bar(dates[pos & valid], departures[pos & valid], width=bar_w,
               color="steelblue", alpha=0.7, label="Above climatology")
        ax.bar(dates[neg & valid], departures[neg & valid], width=bar_w,
               color="crimson", alpha=0.7, label="Below climatology")

        nan_mask = ~valid
        if nan_mask.any():
            ax.scatter(dates[nan_mask], np.zeros(nan_mask.sum()),
                       s=10, color="lightgray", alpha=0.5, zorder=2,
                       label="No reference data")

        if event_date:
            ax.axvline(pd.Timestamp(event_date), color="crimson", lw=1.5,
                       linestyle="--", label="Event date")
        _draw_hurricane_markers(ax, hurricane_events)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30, ha="right")

    ax.set_ylabel(f"Δ{index} (departure from DOY clim.)")
    ax.set_title(
        f"{index} Seasonal Departure (leave-one-year-out, DOY ±{doy_window} d)",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Seasonal departure plot saved: %s", output_path)
    return output_path


def plot_cusum(
    df: pd.DataFrame,
    changepoints: pd.DataFrame,
    output_path: str,
    index: str = "NDVI",
    event_date: Optional[str] = None,
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """
    Plot CUSUM (cumulative deviation from mean) with changepoint annotations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``date`` and ``index_value``.
    changepoints : pd.DataFrame
        DataFrame with ``date`` column from change-point detection.
        May be empty.
    output_path : str
        Path to save the PNG.
    index : str
        Index name.
    event_date : str, optional
        Vertical event annotation.
    hurricane_events : list of dict, optional
        Hurricane markers.

    Returns
    -------
    str
        Output path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = pd.to_datetime(df["date"])
    values = df["index_value"].values.astype(float)
    cusum = np.cumsum(values - np.nanmean(values))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(dates, cusum, lw=1.8, color="steelblue", label="CUSUM")
    ax.fill_between(dates, 0, cusum, alpha=0.15, color="steelblue")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)

    if changepoints is not None and not changepoints.empty:
        cp_dates = pd.to_datetime(changepoints["date"])
        for i, cpd in enumerate(cp_dates):
            ax.axvline(cpd, color="darkgreen", lw=1.5, linestyle="--",
                       label="Change point" if i == 0 else None)

    if event_date:
        ax.axvline(pd.Timestamp(event_date), color="crimson", lw=1.5,
                   linestyle="--", label="Event date")
    _draw_hurricane_markers(ax, hurricane_events)

    ax.set_ylabel("Cumulative Sum")
    ax.set_title(f"{index} CUSUM (Cumulative Deviation from Mean)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("CUSUM plot saved: %s", output_path)
    return output_path


def plot_combined_panel(
    df: pd.DataFrame,
    stl_result: Optional[Dict],
    changepoints: pd.DataFrame,
    output_path: str,
    index: str = "NDVI",
    event_date: Optional[str] = None,
    hurricane_events: Optional[List[Dict]] = None,
) -> str:
    """
    Combined 5-row figure with all detrended / normalised views aligned on a
    shared x-axis.

    Row 1: Raw observed + seasonal fit + trend
    Row 2: STL residuals ±2σ
    Row 3: Standardised z-scores ±2σ / ±3σ
    Row 4: Seasonal departure (leave-one-year-out DOY climatology)
    Row 5: CUSUM with changepoint markers

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``date`` and ``index_value``.
    stl_result : dict or None
        STL decomposition result.
    changepoints : pd.DataFrame
        Change-point DataFrame; may be empty.
    output_path : str
        Path to save the PNG.
    index : str
        Index name.
    event_date : str, optional
        Vertical event annotation.
    hurricane_events : list of dict, optional
        Hurricane markers.

    Returns
    -------
    str
        Output path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec

    dates = pd.to_datetime(df["date"])
    values = df["index_value"].values.astype(float)

    # Pre-compute derived series
    if stl_result is not None:
        residuals = np.asarray(stl_result["residual"], dtype=float)
        sigma_r = np.nanstd(residuals)
        z_scores = residuals / sigma_r if sigma_r > 0 else residuals.copy()
        seasonal_fit = np.asarray(stl_result["trend"]) + np.asarray(stl_result["seasonal"])
        trend = np.asarray(stl_result["trend"])
    else:
        residuals = z_scores = seasonal_fit = trend = None

    departures = _compute_seasonal_departure(df)
    cusum = np.cumsum(values - np.nanmean(values))

    # Layout
    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(5, 1, height_ratios=[1.8, 1, 1, 1, 1], hspace=0.45)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    # ── Panel 1: Raw observations ────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(dates, values, s=15, color="steelblue", alpha=0.6,
               label="Observed", zorder=3)
    if seasonal_fit is not None:
        ax.plot(dates, seasonal_fit, color="darkorange", lw=2,
                label="Seasonal fit", zorder=4)
    if trend is not None:
        ax.plot(dates, trend, color="darkgreen", lw=1.5, linestyle="--",
                label="Trend", zorder=4)
    if event_date:
        ax.axvline(pd.Timestamp(event_date), color="crimson", lw=1.5,
                   linestyle="--", label="Event date")
    _draw_hurricane_markers(ax, hurricane_events)
    ax.set_ylabel(index)
    ax.set_title(f"{index} — Raw Observations", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: STL residuals ───────────────────────────────────────────────
    ax = axes[1]
    if residuals is not None:
        sigma = np.nanstd(residuals)
        ax.axhspan(-2 * sigma, 2 * sigma, alpha=0.12, color="gray")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        bar_w = pd.Timedelta(days=15)
        pos_out = residuals > 2 * sigma
        neg_out = residuals < -2 * sigma
        normal = ~pos_out & ~neg_out
        ax.bar(dates[normal], residuals[normal], width=bar_w, color="steelblue", alpha=0.6)
        ax.bar(dates[pos_out], residuals[pos_out], width=bar_w, color="royalblue", alpha=0.9)
        ax.bar(dates[neg_out], residuals[neg_out], width=bar_w, color="crimson", alpha=0.9)
        if event_date:
            ax.axvline(pd.Timestamp(event_date), color="crimson", lw=1.5, linestyle="--")
        _draw_hurricane_markers(ax, hurricane_events)
    else:
        ax.text(0.5, 0.5, "STL not available\n(statsmodels required)",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=9, style="italic", color="gray")
    ax.set_ylabel("Residual")
    ax.set_title("STL Residuals (±2σ)", fontweight="bold")
    ax.grid(True, alpha=0.25)

    # ── Panel 3: Z-scores ────────────────────────────────────────────────────
    ax = axes[2]
    if z_scores is not None:
        ax.axhspan(-2, 2, alpha=0.08, color="steelblue")
        ax.axhspan(-3, -2, alpha=0.12, color="darkorange")
        ax.axhspan(2, 3, alpha=0.12, color="darkorange")
        ax.axhspan(-10, -3, alpha=0.12, color="crimson")
        ax.axhspan(3, 10, alpha=0.12, color="crimson")
        z_min = float(np.nanmin(z_scores)); z_max = float(np.nanmax(z_scores))
        ax.set_ylim(max(-5.5, z_min - 0.5), min(5.5, z_max + 0.5))
        for lvl in (-3, -2, 2, 3):
            ax.axhline(lvl, color="gray", lw=0.7, linestyle="--", alpha=0.6)
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.scatter(dates, z_scores, s=12, color="steelblue", alpha=0.7, zorder=4)
        ax.plot(dates, z_scores, lw=0.8, color="steelblue", alpha=0.5)
        if event_date:
            ax.axvline(pd.Timestamp(event_date), color="crimson", lw=1.5, linestyle="--")
        _draw_hurricane_markers(ax, hurricane_events)
    else:
        ax.text(0.5, 0.5, "STL not available\n(statsmodels required)",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=9, style="italic", color="gray")
    ax.set_ylabel("Std Devs")
    ax.set_title("Standardized Anomaly (Z-Score)", fontweight="bold")
    ax.grid(True, alpha=0.25)

    # ── Panel 4: Seasonal departure ──────────────────────────────────────────
    ax = axes[3]
    valid = ~np.isnan(departures)
    if valid.sum() >= 2:
        sigma_d = np.nanstd(departures)
        ax.axhspan(-2 * sigma_d, 2 * sigma_d, alpha=0.10, color="steelblue")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        bar_w = pd.Timedelta(days=15)
        pos = departures >= 0
        neg = departures < 0
        ax.bar(dates[pos & valid], departures[pos & valid], width=bar_w,
               color="steelblue", alpha=0.7)
        ax.bar(dates[neg & valid], departures[neg & valid], width=bar_w,
               color="crimson", alpha=0.7)
        if event_date:
            ax.axvline(pd.Timestamp(event_date), color="crimson", lw=1.5, linestyle="--")
        _draw_hurricane_markers(ax, hurricane_events)
    else:
        ax.text(0.5, 0.5, "Need ≥ 2 years for seasonal departure",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, style="italic", color="gray")
    ax.set_ylabel(f"Δ{index}")
    ax.set_title("Seasonal Departure (DOY climatology)", fontweight="bold")
    ax.grid(True, alpha=0.25)

    # ── Panel 5: CUSUM ───────────────────────────────────────────────────────
    ax = axes[4]
    ax.plot(dates, cusum, lw=1.8, color="steelblue", label="CUSUM")
    ax.fill_between(dates, 0, cusum, alpha=0.15, color="steelblue")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    if changepoints is not None and not changepoints.empty:
        cp_dates = pd.to_datetime(changepoints["date"])
        for i, cpd in enumerate(cp_dates):
            ax.axvline(cpd, color="darkgreen", lw=1.5, linestyle="--",
                       label="Change point" if i == 0 else None)
        ax.legend(fontsize=7)
    if event_date:
        ax.axvline(pd.Timestamp(event_date), color="crimson", lw=1.5, linestyle="--")
    _draw_hurricane_markers(ax, hurricane_events)
    ax.set_ylabel("CUSUM")
    ax.set_title("CUSUM (Cumulative Deviation from Mean)", fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Shared x-axis formatting on bottom panel only
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[4].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[4].xaxis.get_majorticklabels(), rotation=30, ha="right")
    for ax in axes[:4]:
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.suptitle(f"{index} — Detrended & Normalised Views", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Combined detrended panel saved: %s", output_path)
    return output_path
