"""
structural_analysis.py
======================
Sentinel-1 SAR radar and GEDI lidar structural damage detection.

Complements the optical NDVI/EVI analysis with:

- SAR backscatter change (VV, VH, Radar Vegetation Index / RVI)
- GEDI canopy height (rh95) and tree cover change
- Multi-sensor concordance map (4 classes)

GEDI operational window: April 2019 – March 2023.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# GEDI ISS operational window
_GEDI_START = "2019-04-01"
_GEDI_END   = "2023-03-31"

# ---------------------------------------------------------------------------
# SAR helpers (Sentinel-1 GRD, IW mode)
# ---------------------------------------------------------------------------

def _get_sar_collection(
    roi: "ee.Geometry",
    start: str,
    end: str,
    orbit: str = "DESCENDING",
) -> "ee.ImageCollection":
    """Return a Sentinel-1 IW GRD collection filtered to VV+VH over the ROI."""
    import ee

    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )


def _apply_speckle_filter(image: "ee.Image", radius: int = 50) -> "ee.Image":
    """Apply a circular focal-median speckle filter (radius in metres)."""
    import ee

    kernel = ee.Kernel.circle(radius, "meters")
    return image.focal_median(kernel=kernel)


def _db_to_linear(image: "ee.Image") -> "ee.Image":
    """Convert dB image to linear power scale (10^(dB/10))."""
    return image.expression("10 ** (db / 10)", {"db": image})


def _linear_to_db(image: "ee.Image") -> "ee.Image":
    """Convert linear power image to dB (10 * log10(linear))."""
    return image.log10().multiply(10)


def _compute_rvi_linear(image_linear: "ee.Image") -> "ee.Image":
    """
    Compute Radar Vegetation Index from a linear-scale VV+VH image.

    RVI = 4 * VH / (VV + VH)
    """
    vv = image_linear.select("VV")
    vh = image_linear.select("VH")
    return vh.multiply(4).divide(vv.add(vh)).rename("RVI")


def get_sar_composites(
    roi: "ee.Geometry",
    event_date: str,
    pre_days: int = 60,
    post_days: int = 60,
    buffer_days: int = 5,
    config: Optional[Dict] = None,
    land_mask: Optional["ee.Image"] = None,
) -> Tuple["ee.Image", "ee.Image", Dict[str, Any], Dict[str, Any]]:
    """
    Acquire pre- and post-event Sentinel-1 SAR composites.

    Processing pipeline per image:
    1. Select IW GRD images with VV+VH polarisations.
    2. Apply focal-median speckle filter (50 m radius).
    3. Convert dB → linear for averaging.
    4. Mean-composite in linear space.
    5. Convert back to dB for VV and VH output bands.
    6. Add RVI computed in linear space.

    Returns
    -------
    (pre_sar, post_sar, pre_meta, post_meta)
        Each image has bands: ``VV`` (dB), ``VH`` (dB), ``RVI``.
        Metadata dicts contain per-scene IDs, dates, orbit, and pass.
    """
    cfg = config or {}
    sar_cfg = cfg.get("sar", {})
    orbit  = sar_cfg.get("orbit", "DESCENDING")
    radius = int(sar_cfg.get("speckle_filter_radius", 50))

    event_dt   = datetime.strptime(event_date, "%Y-%m-%d")
    pre_end    = event_dt - timedelta(days=buffer_days)
    pre_start  = pre_end  - timedelta(days=pre_days)
    post_start = event_dt + timedelta(days=buffer_days)
    post_end   = post_start + timedelta(days=post_days)

    pre_start_s  = pre_start.strftime("%Y-%m-%d")
    pre_end_s    = pre_end.strftime("%Y-%m-%d")
    post_start_s = post_start.strftime("%Y-%m-%d")
    post_end_s   = post_end.strftime("%Y-%m-%d")

    logger.info(
        "Acquiring SAR composites: pre %s→%s, post %s→%s, orbit=%s",
        pre_start_s, pre_end_s, post_start_s, post_end_s, orbit,
    )

    # Build raw collections (needed for both metadata and compositing)
    pre_col_raw  = _get_sar_collection(roi, pre_start_s,  pre_end_s,  orbit)
    post_col_raw = _get_sar_collection(roi, post_start_s, post_end_s, orbit)

    # Extract metadata before compositing
    pre_meta  = extract_sar_metadata(pre_col_raw,  pre_start_s,  pre_end_s)
    post_meta = extract_sar_metadata(post_col_raw, post_start_s, post_end_s)

    def _make_composite(col: "ee.ImageCollection") -> "ee.Image":
        # Apply land mask per-image before speckle filtering so water
        # backscatter is excluded from both the speckle kernel and the mean.
        if land_mask is not None:
            col = col.map(lambda img: img.updateMask(land_mask))
        filtered = col.map(lambda img: _apply_speckle_filter(img, radius))
        linear   = filtered.map(_db_to_linear)
        mean_lin = linear.mean()
        vv_db    = _linear_to_db(mean_lin.select("VV")).rename("VV")
        vh_db    = _linear_to_db(mean_lin.select("VH")).rename("VH")
        rvi      = _compute_rvi_linear(mean_lin)
        return vv_db.addBands(vh_db).addBands(rvi)

    pre_sar  = _make_composite(pre_col_raw)
    post_sar = _make_composite(post_col_raw)
    return pre_sar, post_sar, pre_meta, post_meta


def compute_sar_change(
    pre_sar: "ee.Image",
    post_sar: "ee.Image",
) -> "ee.Image":
    """
    Compute SAR change (post - pre) for all bands.

    Returns
    -------
    ee.Image
        Bands: ``VV_delta`` (dB), ``VH_delta`` (dB), ``RVI_delta``.
    """
    vv_delta  = post_sar.select("VV").subtract(pre_sar.select("VV")).rename("VV_delta")
    vh_delta  = post_sar.select("VH").subtract(pre_sar.select("VH")).rename("VH_delta")
    rvi_delta = post_sar.select("RVI").subtract(pre_sar.select("RVI")).rename("RVI_delta")
    return vv_delta.addBands(vh_delta).addBands(rvi_delta)


# ---------------------------------------------------------------------------
# Metadata extraction helpers
# ---------------------------------------------------------------------------

def extract_sar_metadata(
    collection: "ee.ImageCollection",
    window_start: str = "",
    window_end: str = "",
) -> Dict[str, Any]:
    """
    Extract per-scene metadata from a Sentinel-1 SAR ImageCollection.

    Returns
    -------
    dict
        ``{sensor, window_start, window_end, count, scenes}`` where each
        scene entry has keys ``id``, ``date``, ``orbit``, ``pass``.
        On GEE failure the dict contains ``"error"`` and ``count=None``.
    """
    try:
        from .metadata_utils import ts_to_date

        ids    = collection.aggregate_array("system:index").getInfo()
        times  = collection.aggregate_array("system:time_start").getInfo()
        orbits = collection.aggregate_array("relativeOrbitNumber_start").getInfo()
        passes = collection.aggregate_array("orbitProperties_pass").getInfo()

        n = len(ids)
        scenes = [
            {
                "id":    ids[i],
                "date":  ts_to_date(times[i]),
                "orbit": int(orbits[i]) if (orbits and i < len(orbits) and orbits[i] is not None) else None,
                "pass":  passes[i]      if (passes  and i < len(passes))  else None,
            }
            for i in range(n)
        ]
        scenes.sort(key=lambda s: s["date"])
        return {
            "sensor":       "sentinel1",
            "window_start": window_start,
            "window_end":   window_end,
            "count":        n,
            "scenes":       scenes,
        }
    except Exception as exc:
        logger.warning("SAR metadata extraction failed: %s", exc)
        return {
            "sensor":  "sentinel1",
            "count":   None,
            "scenes":  [],
            "error":   str(exc),
        }


def extract_palsar_metadata(
    year: int,
    event_date: str,
    collection: Optional["ee.ImageCollection"] = None,
) -> Dict[str, Any]:
    """
    Build PALSAR metadata for an annual mosaic year.

    ``event_in_epoch`` is ``True`` when the event date falls within
    [Jan 1, Dec 31] of *year*, indicating that the mosaic may include
    post-hurricane vegetation observations.

    Parameters
    ----------
    year : int
        Calendar year of the PALSAR annual mosaic.
    event_date : str
        Hurricane event date (``YYYY-MM-DD``).
    collection : ee.ImageCollection, optional
        If provided, extract scene IDs and dates from the raw collection
        (useful for diagnostic purposes).

    Returns
    -------
    dict
        ``{sensor, year, epoch_start, epoch_end, event_in_epoch, count, scenes}``.
    """
    epoch_start = f"{year}-01-01"
    epoch_end   = f"{year}-12-31"
    event_dt    = datetime.strptime(event_date, "%Y-%m-%d")
    event_in_epoch = (
        datetime.strptime(epoch_start, "%Y-%m-%d")
        <= event_dt
        <= datetime.strptime(epoch_end, "%Y-%m-%d")
    )

    meta: Dict[str, Any] = {
        "sensor":         "palsar",
        "year":           year,
        "epoch_start":    epoch_start,
        "epoch_end":      epoch_end,
        "event_in_epoch": event_in_epoch,
        "count":          1,   # yearly mosaics have one aggregate image
        "scenes":         [],
    }

    if collection is not None:
        try:
            from .metadata_utils import ts_to_date

            ids   = collection.aggregate_array("system:index").getInfo()
            times = collection.aggregate_array("system:time_start").getInfo()
            meta["count"]  = len(ids)
            meta["scenes"] = [
                {"id": ids[i], "date": ts_to_date(times[i])}
                for i in range(len(ids))
            ]
        except Exception as exc:
            logger.warning("PALSAR metadata extraction failed (year=%d): %s", year, exc)

    return meta


def extract_gedi_metadata(
    height_col: "ee.ImageCollection",
    cover_col: "ee.ImageCollection",
    window_start: str = "",
    window_end: str = "",
) -> Dict[str, Any]:
    """
    Extract metadata from GEDI GEDI02_A and GEDI02_B monthly collections.

    Returns
    -------
    dict
        ``{sensor, window_start, window_end, height_months, cover_months,
        count, date_start, date_end}``.
    """
    try:
        from .metadata_utils import ts_to_date

        h_count = height_col.size().getInfo()
        c_count = cover_col.size().getInfo()
        h_times = (
            height_col.aggregate_array("system:time_start").getInfo()
            if h_count > 0 else []
        )
        dates = sorted(ts_to_date(t) for t in h_times) if h_times else []
        return {
            "sensor":        "gedi",
            "window_start":  window_start,
            "window_end":    window_end,
            "height_months": h_count,
            "cover_months":  c_count,
            "count":         h_count,
            "date_start":    dates[0]  if dates else None,
            "date_end":      dates[-1] if dates else None,
        }
    except Exception as exc:
        logger.warning("GEDI metadata extraction failed: %s", exc)
        return {
            "sensor": "gedi",
            "count":  None,
            "error":  str(exc),
        }


# ---------------------------------------------------------------------------
# GEDI helpers
# ---------------------------------------------------------------------------

def check_gedi_availability(event_date: str) -> Dict[str, Any]:
    """
    Check whether *event_date* falls within the GEDI operational window.

    GEDI was aboard the ISS from April 2019 to March 2023.

    Returns
    -------
    dict
        ``{"available": bool, "message": str}``
    """
    event_dt  = datetime.strptime(event_date, "%Y-%m-%d")
    gedi_start = datetime.strptime(_GEDI_START, "%Y-%m-%d")
    gedi_end   = datetime.strptime(_GEDI_END,   "%Y-%m-%d")

    if event_dt < gedi_start:
        return {
            "available": False,
            "message": (
                f"GEDI data unavailable for {event_date}. "
                f"GEDI was operational April 2019 – March 2023. "
                f"This event ({event_dt.strftime('%B %Y')}) predates GEDI. "
                "Skipping GEDI analysis."
            ),
        }
    if event_dt > gedi_end:
        return {
            "available": False,
            "message": (
                f"GEDI data unavailable for {event_date}. "
                f"GEDI ceased operations in March 2023. "
                f"This event ({event_dt.strftime('%B %Y')}) postdates GEDI. "
                "Skipping GEDI analysis."
            ),
        }
    return {
        "available": True,
        "message": f"GEDI data available for {event_date}.",
    }


def get_gedi_composites(
    roi: "ee.Geometry",
    event_date: str,
    pre_months: int = 6,
    post_months: int = 6,
    config: Optional[Dict] = None,
) -> Tuple[Optional["ee.Image"], Optional["ee.Image"], Dict[str, Any], Dict[str, Any]]:
    """
    Acquire pre- and post-event GEDI composites.

    Uses:
    - ``LARSE/GEDI/GEDI02_A_002_MONTHLY`` for height (rh95, rh50)
    - ``LARSE/GEDI/GEDI02_B_002_MONTHLY`` for tree cover

    Applies quality filters: ``quality_flag == 1``, ``degrade_flag == 0``,
    ``sensitivity ≥ threshold`` for GEDI02_A.

    Returns
    -------
    (pre_gedi, post_gedi, pre_meta, post_meta)
        Each image has bands: ``rh95`` (m), ``rh50`` (m), ``cover`` (0–1).
        Returns ``(None, None, {}, {})`` when outside the operational window.
    """
    avail = check_gedi_availability(event_date)
    if not avail["available"]:
        logger.warning(avail["message"])
        return None, None, {}, {}

    import ee

    cfg = config or {}
    sensitivity = float(cfg.get("gedi", {}).get("sensitivity_threshold", 0.9))

    event_dt   = datetime.strptime(event_date, "%Y-%m-%d")
    pre_start  = (event_dt - timedelta(days=pre_months * 30)).strftime("%Y-%m-%d")
    pre_end    = event_date
    post_start = event_date
    post_end   = (event_dt + timedelta(days=post_months * 30)).strftime("%Y-%m-%d")

    # Clamp to operational window
    pre_start = max(pre_start, _GEDI_START)
    post_end  = min(post_end, _GEDI_END)

    def _quality_mask_a(image: "ee.Image") -> "ee.Image":
        ok = (
            image.select("quality_flag").eq(1)
            .And(image.select("degrade_flag").eq(0))
            .And(image.select("sensitivity").gte(sensitivity))
        )
        return image.updateMask(ok)

    def _quality_mask_b(image: "ee.Image") -> "ee.Image":
        ok = (
            image.select("l2b_quality_flag").eq(1)
            .And(image.select("algorithmrun_flag").eq(1))
        )
        return image.updateMask(ok)

    def _build_gedi_cols(start: str, end: str):
        h_col = (
            ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
            .filterBounds(roi)
            .filterDate(start, end)
            .map(_quality_mask_a)
            .select(["rh95", "rh50"])
        )
        c_col = (
            ee.ImageCollection("LARSE/GEDI/GEDI02_B_002_MONTHLY")
            .filterBounds(roi)
            .filterDate(start, end)
            .map(_quality_mask_b)
            .select(["cover"])
        )
        return h_col, c_col

    def _composite_from_cols(h_col, c_col) -> "ee.Image":
        rh95  = h_col.select("rh95").median()
        rh50  = h_col.select("rh50").median()
        cover = c_col.select("cover").median()
        return rh95.addBands(rh50).addBands(cover)

    logger.info(
        "Acquiring GEDI composites: pre %s→%s, post %s→%s",
        pre_start, pre_end, post_start, post_end,
    )

    pre_h_col,  pre_c_col  = _build_gedi_cols(pre_start,  pre_end)
    post_h_col, post_c_col = _build_gedi_cols(post_start, post_end)

    pre_meta  = extract_gedi_metadata(pre_h_col,  pre_c_col,  pre_start,  pre_end)
    post_meta = extract_gedi_metadata(post_h_col, post_c_col, post_start, post_end)

    pre_gedi  = _composite_from_cols(pre_h_col,  pre_c_col)
    post_gedi = _composite_from_cols(post_h_col, post_c_col)
    return pre_gedi, post_gedi, pre_meta, post_meta


def compute_gedi_change(
    pre_gedi: "ee.Image",
    post_gedi: "ee.Image",
) -> "ee.Image":
    """
    Compute GEDI structural change (post - pre).

    Returns
    -------
    ee.Image
        Bands: ``rh95_delta`` (m), ``rh50_delta`` (m), ``cover_delta``.
    """
    rh95_d  = post_gedi.select("rh95").subtract(pre_gedi.select("rh95")).rename("rh95_delta")
    rh50_d  = post_gedi.select("rh50").subtract(pre_gedi.select("rh50")).rename("rh50_delta")
    cover_d = post_gedi.select("cover").subtract(pre_gedi.select("cover")).rename("cover_delta")
    return rh95_d.addBands(rh50_d).addBands(cover_d)


# ---------------------------------------------------------------------------
# Multi-sensor concordance classification
# ---------------------------------------------------------------------------

def classify_concordance(
    diff_optical: "ee.Image",
    diff_sar: "ee.Image",
    thresholds: Optional[Dict] = None,
) -> "ee.Image":
    """
    Classify each pixel into one of four concordance classes.

    Classes
    -------
    0 : No Change              — neither optical nor SAR indicates damage
    1 : Vegetation Stress Only — optical index drops, SAR stable
                                 (foliar loss / chlorophyll depletion)
    2 : Structural Damage Only — SAR decreases, optical stable
                                 (physical structure loss, fewer scatterers)
    3 : High-Confidence Damage — both optical and SAR decline simultaneously

    Parameters
    ----------
    diff_optical : ee.Image
        Optical difference image with band ``delta`` (post - pre index value).
    diff_sar : ee.Image
        SAR change image with band ``RVI_delta``.
    thresholds : dict, optional
        ``optical_change``: threshold for optical loss (default -0.05).
        ``rvi_change``: threshold for SAR loss (default -0.10).

    Returns
    -------
    ee.Image
        Single-band integer image (0–3) named ``concordance``.
    """
    t = thresholds or {}
    opt_thresh = t.get("optical_change", -0.05)
    rvi_thresh = t.get("rvi_change", -0.10)

    optical_loss = diff_optical.select("delta").lt(opt_thresh)
    sar_loss     = diff_sar.select("RVI_delta").lt(rvi_thresh)

    # class = 1*optical_loss + 2*sar_loss  →  0,1,2,3
    concordance = (
        optical_loss.toInt()
        .add(sar_loss.multiply(2).toInt())
        .rename("concordance")
    )
    return concordance


# ---------------------------------------------------------------------------
# ALOS PALSAR-2 L-band SAR helpers
# ---------------------------------------------------------------------------

_PALSAR_COLLECTION = "JAXA/ALOS/PALSAR/YEARLY/SAR"


def _calibrate_palsar(
    image: "ee.Image",
    calibration_offset: float = -83.0,
) -> "ee.Image":
    """
    Convert PALSAR DN to dB using the JAXA standard formula.

    dB = 10 * log10(DN²) + calibration_offset
       = 20 * log10(DN) + calibration_offset

    Parameters
    ----------
    image : ee.Image
        Raw PALSAR image with HH and HV DN bands.
    calibration_offset : float
        JAXA offset (default -83.0 dB).

    Returns
    -------
    ee.Image
        Image with bands HH_dB and HV_dB.
    """
    # DN values are stored as uint16; log10 on the DN
    hh_db = image.select("HH").log10().multiply(20).add(calibration_offset).rename("HH_dB")
    hv_db = image.select("HV").log10().multiply(20).add(calibration_offset).rename("HV_dB")
    return hh_db.addBands(hv_db)


def _determine_palsar_years(event_date: str) -> Tuple[int, int]:
    """
    Select pre- and post-event PALSAR mosaic years.

    PALSAR mosaics are annual (typically covering the calendar year).
    For hurricanes that occur after mid-year (month ≥ 7), the pre-event
    mosaic is from the same calendar year and the post-event mosaic from
    the following year.  For early-year events the pre-event mosaic is
    from the prior year.

    Examples
    --------
    Irma 2017-09-10 (month=9 ≥ 7)  → pre=2017, post=2018
    Ian  2022-09-28 (month=9 ≥ 7)  → pre=2022, post=2023
    Dorian 2019-09-03 (month=9 ≥ 7) → pre=2019, post=2020
    A hypothetical Jan event in 2021 → pre=2020, post=2021

    Returns
    -------
    (pre_year, post_year) : Tuple[int, int]
    """
    dt = datetime.strptime(event_date, "%Y-%m-%d")
    if dt.month >= 7:
        return dt.year, dt.year + 1
    else:
        return dt.year - 1, dt.year


def get_palsar_images(
    roi: "ee.Geometry",
    event_date: str,
    pre_year: Optional[int] = None,
    post_year: Optional[int] = None,
    config: Optional[Dict] = None,
    land_mask: Optional["ee.Image"] = None,
) -> Tuple["ee.Image", "ee.Image"]:
    """
    Retrieve calibrated (dB) PALSAR annual mosaic images for pre- and post-event.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    event_date : str
        Hurricane event date (YYYY-MM-DD).
    pre_year : int, optional
        Override auto-selected pre-event mosaic year.
    post_year : int, optional
        Override auto-selected post-event mosaic year.
    config : dict, optional
        Merged configuration dictionary.

    Returns
    -------
    (pre_palsar, post_palsar, pre_meta, post_meta)
        Calibrated dB images with bands HH_dB, HV_dB.
        Metadata dicts contain year, epoch dates, and event-in-epoch flag.
    """
    import ee

    cfg = config or {}
    cal_offset = float(
        cfg.get("palsar_settings", {}).get("calibration_offset", -83.0)
    )

    auto_pre, auto_post = _determine_palsar_years(event_date)
    pre_yr  = int(pre_year)  if pre_year  is not None else auto_pre
    post_yr = int(post_year) if post_year is not None else auto_post

    def _get_year_image(year: int):
        start = f"{year}-01-01"
        end   = f"{year}-12-31"
        col = (
            ee.ImageCollection(_PALSAR_COLLECTION)
            .filterBounds(roi)
            .filterDate(start, end)
            .select(["HH", "HV"])
        )
        meta = extract_palsar_metadata(year, event_date, col)
        # Use mosaic (yearly collection typically has one image per year)
        img = _calibrate_palsar(col.mosaic(), cal_offset)
        if land_mask is not None:
            img = img.updateMask(land_mask)
        return img, meta

    logger.info("Acquiring PALSAR: pre=%d, post=%d", pre_yr, post_yr)
    pre_palsar,  pre_meta  = _get_year_image(pre_yr)
    post_palsar, post_meta = _get_year_image(post_yr)
    return pre_palsar, post_palsar, pre_meta, post_meta


def compute_palsar_change(
    pre_palsar: "ee.Image",
    post_palsar: "ee.Image",
) -> "ee.Image":
    """
    Compute L-band SAR change (post − pre) in dB.

    Returns
    -------
    ee.Image
        Bands: ``HV_delta`` (dB), ``HH_delta`` (dB),
               ``HV_HH_ratio_delta`` (difference of the HV/HH ratio).
    """
    hv_delta = (
        post_palsar.select("HV_dB")
        .subtract(pre_palsar.select("HV_dB"))
        .rename("HV_delta")
    )
    hh_delta = (
        post_palsar.select("HH_dB")
        .subtract(pre_palsar.select("HH_dB"))
        .rename("HH_delta")
    )
    # HV/HH ratio (both in dB, so difference = ratio change in log space)
    pre_ratio  = pre_palsar.select("HV_dB").subtract(pre_palsar.select("HH_dB"))
    post_ratio = post_palsar.select("HV_dB").subtract(post_palsar.select("HH_dB"))
    ratio_delta = post_ratio.subtract(pre_ratio).rename("HV_HH_ratio_delta")

    return hv_delta.addBands(hh_delta).addBands(ratio_delta)


def classify_palsar_damage(
    diff_palsar: "ee.Image",
    thresholds: Optional[Dict] = None,
) -> "ee.Image":
    """
    Classify L-band structural damage from ∆HV (dB).

    Classes
    -------
    0 : No Damage      ∆HV > -1.0 dB
    1 : Light Damage   -1.0 ≥ ∆HV > -2.0 dB
    2 : Moderate       -2.0 ≥ ∆HV > -4.0 dB
    3 : Severe         ∆HV ≤ -4.0 dB

    Parameters
    ----------
    diff_palsar : ee.Image
        Output of ``compute_palsar_change()``; must contain band ``HV_delta``.
    thresholds : dict, optional
        Keys: ``no_damage`` (default -1.0), ``light_damage`` (-2.0),
        ``moderate_damage`` (-4.0).

    Returns
    -------
    ee.Image
        Single-band integer image (0–3) named ``palsar_damage``.
    """
    t = thresholds or {}
    t_none = float(t.get("no_damage",       -1.0))
    t_lt   = float(t.get("light_damage",    -2.0))
    t_mod  = float(t.get("moderate_damage", -4.0))

    hv = diff_palsar.select("HV_delta")

    # 0 = no damage, 1 = light, 2 = moderate, 3 = severe (additive bitmask)
    light    = hv.lte(t_none).And(hv.gt(t_lt))   # -1 to -2
    moderate = hv.lte(t_lt).And(hv.gt(t_mod))    # -2 to -4
    severe   = hv.lte(t_mod)                      # < -4

    damage = (
        light.toInt()
        .add(moderate.toInt().multiply(2))
        .add(severe.toInt().multiply(3))
        .rename("palsar_damage")
    )
    return damage


def sample_palsar_stats(
    pre_palsar: "ee.Image",
    post_palsar: "ee.Image",
    roi: "ee.Geometry",
    n: int = 500,
    scale: int = 25,
) -> Dict[str, Any]:
    """
    Sample HV dB values in pre/post images and run a Wilcoxon signed-rank test.

    Parameters
    ----------
    n : int
        Number of random pixels to sample.
    scale : int
        PALSAR native resolution in metres (25 m).

    Returns
    -------
    dict
        Keys: ``pre_hv_mean``, ``post_hv_mean``, ``hv_delta_mean``,
              ``wilcoxon_p``, ``cohens_d``, ``significant`` (bool).
    """
    sample = (
        pre_palsar.select("HV_dB")
        .addBands(post_palsar.select("HV_dB").rename("HV_dB_post"))
        .sample(region=roi, scale=scale, numPixels=n, seed=42, geometries=False)
        .toList(n)
        .getInfo()
    )

    try:
        import numpy as np
        from scipy import stats as scipy_stats

        pre_vals  = np.array([f["properties"]["HV_dB"]      for f in sample])
        post_vals = np.array([f["properties"]["HV_dB_post"] for f in sample])
        delta     = post_vals - pre_vals

        _, p_val = scipy_stats.wilcoxon(pre_vals, post_vals)
        pooled_std = float(np.sqrt((pre_vals.std() ** 2 + post_vals.std() ** 2) / 2))
        cohens_d = float(delta.mean() / pooled_std) if pooled_std > 0 else 0.0

        return {
            "pre_hv_mean":    float(pre_vals.mean()),
            "post_hv_mean":   float(post_vals.mean()),
            "hv_delta_mean":  float(delta.mean()),
            "wilcoxon_p":     float(p_val),
            "cohens_d":       cohens_d,
            "significant":    bool(p_val < 0.05),
        }
    except Exception as exc:
        logger.warning("PALSAR stats computation failed: %s", exc)
        return {"error": str(exc)}


def estimate_agb(
    palsar_db_image: "ee.Image",
    config: Optional[Dict] = None,
) -> "ee.Image":
    """
    Experimental above-ground biomass (AGB) estimate from PALSAR HV backscatter.

    Formula
    -------
    AGB (Mg/ha) ≈ 10 ^ ((HV_dB + 83) * A + B)

    where A and B are empirical allometric coefficients that vary by ecosystem.
    The defaults (A=0.04, B=-2.0) are illustrative placeholders — regional
    calibration with field plots is required before use in publications.

    Parameters
    ----------
    palsar_db_image : ee.Image
        Calibrated PALSAR image with band ``HV_dB``.
    config : dict, optional
        Configuration dictionary; reads ``palsar_settings.agb_A`` and
        ``palsar_settings.agb_B``.

    Returns
    -------
    ee.Image
        Single-band image ``AGB_est_MgHa`` (float, Mg/ha).
    """
    cfg = config or {}
    ps  = cfg.get("palsar_settings", {})
    A   = float(ps.get("agb_A", 0.04))
    B   = float(ps.get("agb_B", -2.0))

    hv_db = palsar_db_image.select("HV_dB")
    # Exponent: (HV_dB + 83) * A + B
    exponent = hv_db.add(83.0).multiply(A).add(B)
    agb = exponent.pow(10).rename("AGB_est_MgHa")
    return agb


def classify_concordance_extended(
    diff_optical: "ee.Image",
    diff_sar: "ee.Image",
    diff_palsar: "ee.Image",
    thresholds: Optional[Dict] = None,
) -> "ee.Image":
    """
    8-class extended concordance map combining optical, C-band, and L-band signals.

    Bitmask
    -------
    bit 0 (value 1) : Optical loss   (foliar / chlorophyll signal)
    bit 1 (value 2) : C-band loss    (Sentinel-1 RVI, small-branch / leaf scale)
    bit 2 (value 4) : L-band loss    (PALSAR HV, trunk / primary branch scale)

    Resulting classes (0–7)
    -----------------------
    0  No Change           — all three sensors agree: no significant loss
    1  Foliar Loss Only    — optical↓, C-band stable, L-band stable
                             (leaf-scale stress, no structural damage)
    2  C-band Only         — C-band↓, optical stable, L-band stable
                             (canopy surface roughness change, bark/small branch)
    3  Optical + C-band    — leaf + small-branch loss; classic storm stress
    4  L-band Only         — L-band↓, optical and C-band stable
                             (trunk damage beneath intact canopy)
    5  Optical + L-band    — foliar loss + structural damage; incomplete C-band
    6  C-band + L-band     — structural damage confirmed by both radars, canopy
                             partially intact (e.g. dead standing timber)
    7  Full Concordance    — all three sensors indicate damage simultaneously;
                             highest-confidence severe structural destruction

    Parameters
    ----------
    diff_optical : ee.Image
        Band ``delta`` (optical index change).
    diff_sar : ee.Image
        Band ``RVI_delta`` (Sentinel-1 RVI change).
    diff_palsar : ee.Image
        Band ``HV_delta`` (PALSAR L-band HV change in dB).
    thresholds : dict, optional
        ``optical_change`` (default -0.05),
        ``rvi_change``     (default -0.10),
        ``palsar_hv_change`` (default -1.0 dB).

    Returns
    -------
    ee.Image
        Single-band integer image (0–7) named ``concordance_ext``.
    """
    t = thresholds or {}
    opt_thresh    = float(t.get("optical_change",   -0.05))
    rvi_thresh    = float(t.get("rvi_change",       -0.10))
    palsar_thresh = float(t.get("palsar_hv_change", -1.0))

    optical_loss = diff_optical.select("delta").lt(opt_thresh)       # bit 0
    cband_loss   = diff_sar.select("RVI_delta").lt(rvi_thresh)       # bit 1
    lband_loss   = diff_palsar.select("HV_delta").lt(palsar_thresh)  # bit 2

    concordance_ext = (
        optical_loss.toInt()
        .add(cband_loss.multiply(2).toInt())
        .add(lband_loss.multiply(4).toInt())
        .rename("concordance_ext")
    )
    return concordance_ext


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_structural_analysis(
    roi: "ee.Geometry",
    event_date: str,
    optical_results: Dict[str, Any],
    config: Optional[Dict] = None,
    sensors: str = "optical,sar",
    palsar_pre_year: Optional[int] = None,
    palsar_post_year: Optional[int] = None,
    land_mask: Optional["ee.Image"] = None,
) -> Dict[str, Any]:
    """
    Run Sentinel-1 SAR, GEDI lidar, and/or ALOS PALSAR-2 structural analysis.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    event_date : str
        Hurricane event date (YYYY-MM-DD).
    optical_results : dict
        Output of ``run_analysis()`` (contains ``diff_img`` etc.).
    config : dict, optional
        Merged configuration dictionary from config.yaml.
    sensors : str
        Comma-separated sensor tokens: ``"sar"``, ``"gedi"``, ``"palsar"``,
        or ``"all"``.  ``"optical"`` tokens are silently ignored.
    palsar_pre_year : int, optional
        Override auto-selected PALSAR pre-event mosaic year.
    palsar_post_year : int, optional
        Override auto-selected PALSAR post-event mosaic year.

    Returns
    -------
    dict
        Keys:
        ``sensors``, ``event_date``,
        ``sar_available`` (bool), ``pre_sar``, ``post_sar``, ``diff_sar``,
        ``gedi_available`` (bool), ``pre_gedi``, ``post_gedi``, ``diff_gedi``,
        ``palsar_available`` (bool), ``pre_palsar``, ``post_palsar``,
        ``diff_palsar``, ``palsar_damage``, ``palsar_stats``,
        ``concordance_img`` (4-class optical+C-band),
        ``concordance_ext`` (8-class optical+C-band+L-band, when PALSAR available),
        ``gedi_info`` (message), ``sar_error`` / ``gedi_error`` / ``palsar_error``
        (if a sensor block fails).
    """
    cfg = config or {}
    sensor_list = [s.strip().lower() for s in sensors.split(",")]
    if "all" in sensor_list:
        sensor_list = ["sar", "gedi", "palsar"]

    windows     = cfg.get("windows", {})
    pre_days    = int(windows.get("pre_days", 60))
    post_days   = int(windows.get("post_days", 60))
    buffer_days = int(windows.get("buffer_days", 5))
    sar_cfg     = cfg.get("sar", {})
    sar_thresh  = sar_cfg.get("thresholds", {})
    palsar_thresh = cfg.get("palsar_thresholds", {})

    results: Dict[str, Any] = {
        "sensors":          sensor_list,
        "event_date":       event_date,
        "sar_available":    False,
        "gedi_available":   False,
        "palsar_available": False,
    }

    scene_metadata: Dict[str, Any] = {}
    diff_sar    = None
    diff_palsar = None

    # ── Sentinel-1 SAR (C-band) ───────────────────────────────────────────────
    if "sar" in sensor_list:
        try:
            pre_sar, post_sar, sar_pre_meta, sar_post_meta = get_sar_composites(
                roi, event_date, pre_days, post_days, buffer_days, cfg,
                land_mask=land_mask,
            )
            diff_sar = compute_sar_change(pre_sar, post_sar)
            results.update({
                "sar_available": True,
                "pre_sar":  pre_sar,
                "post_sar": post_sar,
                "diff_sar": diff_sar,
            })
            scene_metadata["sar"] = {"pre": sar_pre_meta, "post": sar_post_meta}
            logger.info("SAR analysis complete.")
        except Exception as exc:
            logger.warning("SAR analysis failed: %s", exc)
            results["sar_error"] = str(exc)

    # ── GEDI ──────────────────────────────────────────────────────────────────
    if "gedi" in sensor_list:
        gedi_check = check_gedi_availability(event_date)
        results["gedi_info"] = gedi_check["message"]
        if gedi_check["available"]:
            try:
                pre_gedi, post_gedi, gedi_pre_meta, gedi_post_meta = get_gedi_composites(
                    roi, event_date, config=cfg,
                )
                if pre_gedi is not None and post_gedi is not None:
                    diff_gedi = compute_gedi_change(pre_gedi, post_gedi)
                    results.update({
                        "gedi_available": True,
                        "pre_gedi":  pre_gedi,
                        "post_gedi": post_gedi,
                        "diff_gedi": diff_gedi,
                    })
                    scene_metadata["gedi"] = {"pre": gedi_pre_meta, "post": gedi_post_meta}
                    logger.info("GEDI analysis complete.")
            except Exception as exc:
                logger.warning("GEDI analysis failed: %s", exc)
                results["gedi_error"] = str(exc)

    # ── ALOS PALSAR-2 (L-band) ────────────────────────────────────────────────
    if "palsar" in sensor_list:
        try:
            pre_palsar, post_palsar, palsar_pre_meta, palsar_post_meta = get_palsar_images(
                roi, event_date,
                pre_year=palsar_pre_year,
                post_year=palsar_post_year,
                config=cfg,
                land_mask=land_mask,
            )
            diff_palsar   = compute_palsar_change(pre_palsar, post_palsar)
            palsar_damage = classify_palsar_damage(diff_palsar, palsar_thresh)
            palsar_stats  = sample_palsar_stats(pre_palsar, post_palsar, roi)
            results.update({
                "palsar_available": True,
                "pre_palsar":    pre_palsar,
                "post_palsar":   post_palsar,
                "diff_palsar":   diff_palsar,
                "palsar_damage": palsar_damage,
                "palsar_stats":  palsar_stats,
            })
            scene_metadata["palsar"] = {"pre": palsar_pre_meta, "post": palsar_post_meta}
            logger.info("PALSAR analysis complete.")
        except Exception as exc:
            logger.warning("PALSAR analysis failed: %s", exc)
            results["palsar_error"] = str(exc)

    results["scene_metadata"] = scene_metadata

    # ── Multi-sensor concordance ───────────────────────────────────────────────
    if results.get("sar_available") and optical_results.get("diff_img") is not None:
        opt_thresh = cfg.get("thresholds", {}).get("no_impact", -0.05)
        conc_thresholds = {
            "optical_change": opt_thresh,
            "rvi_change":     sar_thresh.get("rvi_change", -0.10),
        }
        try:
            concordance_img = classify_concordance(
                optical_results["diff_img"],
                diff_sar,
                conc_thresholds,
            )
            results["concordance_img"] = concordance_img
            logger.info("4-class concordance classification complete.")
        except Exception as exc:
            logger.warning("Concordance classification failed: %s", exc)

        # Extended 8-class concordance when PALSAR is also available
        if results.get("palsar_available"):
            try:
                ext_thresholds = dict(conc_thresholds)
                ext_thresholds["palsar_hv_change"] = float(
                    palsar_thresh.get("no_damage", -1.0)
                )
                concordance_ext = classify_concordance_extended(
                    optical_results["diff_img"],
                    diff_sar,
                    diff_palsar,
                    ext_thresholds,
                )
                results["concordance_ext"] = concordance_ext
                logger.info("8-class extended concordance classification complete.")
            except Exception as exc:
                logger.warning("Extended concordance classification failed: %s", exc)

    return results
