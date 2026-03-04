"""
data_acquisition.py
===================
Satellite data retrieval and preprocessing via Google Earth Engine (GEE),
with a Planetary Computer STAC fallback.

Primary workflow
----------------
1. Build a cloud-masked ``ee.ImageCollection`` for Sentinel-2 or Landsat 8/9.
2. Composite the collection into a single median image clipped to the ROI.
3. Return ``(pre_composite, post_composite)`` ee.Image objects ready for
   vegetation index computation.

Cloud masking
-------------
- **Sentinel-2**: QA60 band (bits 10/11) + Scene Classification Layer (SCL)
  — keeps only vegetation (4), not-vegetated (5), water (6), unclassified (7).
- **Landsat 8/9**: QA_PIXEL band with bit masks for cloud and cloud shadow.

Planetary Computer fallback
---------------------------
If GEE is unavailable, :func:`get_planetary_computer_fallback` queries the
Microsoft Planetary Computer STAC API for Sentinel-2 Level-2A assets and
returns local numpy arrays via ``stackstac``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from .utils import InsufficientDataError, date_windows

logger = logging.getLogger(__name__)

# Minimum cloud-free scenes in a composite before raising InsufficientDataError
_MIN_IMAGES = 1
_WARN_IMAGES = 3  # warn (but don't fail) below this count

# ---------------------------------------------------------------------------
# Sentinel-2 helpers
# ---------------------------------------------------------------------------

def mask_sentinel2_clouds(image: "ee.Image") -> "ee.Image":
    """
    Mask clouds and cloud shadows from a Sentinel-2 SR image.

    Two complementary masks are applied:

    1. **QA60** — bits 10 (opaque clouds) and 11 (cirrus clouds) must be 0.
    2. **SCL** (Scene Classification Layer) — keep only pixels classified as
       vegetation (4), not-vegetated (5), water (6), or unclassified (7).
       This removes cloud shadows (3), thin cirrus (10), snow/ice (11), etc.

    Parameters
    ----------
    image : ee.Image
        Single Sentinel-2 image from ``COPERNICUS/S2_SR_HARMONIZED``.

    Returns
    -------
    ee.Image
        Cloud-masked image; masked pixels are excluded from later compositing.
    """
    import ee

    # QA60 bitmask: bit 10 = opaque cloud, bit 11 = cirrus
    qa60 = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa60_mask = qa60.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa60.bitwiseAnd(cirrus_bit_mask).eq(0)
    )

    # SCL: keep vegetation(4), not-vegetated(5), water(6), unclassified(7)
    scl = image.select("SCL")
    scl_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))

    combined_mask = qa60_mask.And(scl_mask)
    return image.updateMask(combined_mask)


def build_jrc_land_mask(threshold: int = 80) -> "ee.Image":
    """
    Build a binary land mask from the JRC Global Surface Water dataset.

    Returns 1 for land (or unknown), 0 for permanent water.

    Two safeguards vs the naive implementation:

    * ``.unmask(0)`` — pixels with no JRC observations (small islands, data
      gaps) are treated as 0 % water occurrence (i.e. land) rather than
      being masked out.  Without this, island pixels that the JRC dataset
      never observed appear as masked and propagate through the ``.Not()``
      inversion.
    * Threshold 80 (not 50) — permanent water bodies have occurrence ≈
      90–100 %.  50 incorrectly removes tidal flats, mangroves, and coastal
      marshes — exactly the vegetation of interest.  80 keeps those habitats
      while reliably excluding open water.

    Parameters
    ----------
    threshold : int
        JRC occurrence percentage above which a pixel is classified as water
        (default 80).

    Returns
    -------
    ee.Image
        Single-band binary image; 1 = land, 0 = water.
    """
    import ee

    return (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(0)           # no-data → 0 % (treat as land)
        .gt(threshold)       # 1 = water, 0 = land
        .Not()               # invert → 1 = land, 0 = water
    )


def compute_land_area_km2(
    roi: "ee.Geometry",
    land_mask: "ee.Image",
    scale: int = 500,
) -> float:
    """
    Estimate the land area (km²) within *roi* using a binary land mask.

    Uses ``ee.Image.pixelArea()`` to accumulate the area of all unmasked
    (land) pixels at the given *scale*.  A coarser scale (default 500 m)
    keeps the ``reduceRegion`` call cheap while still giving a reliable
    estimate.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    land_mask : ee.Image
        Binary land mask (1 = land) as returned by :func:`build_jrc_land_mask`.
    scale : int
        Reduction scale in metres (default 500).

    Returns
    -------
    float
        Land area in square kilometres.
    """
    import ee

    area_m2 = (
        ee.Image(1)
        .updateMask(land_mask)
        .multiply(ee.Image.pixelArea())
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
        )
        .getInfo()
        .get("constant", 0)
    )
    return (area_m2 or 0) / 1_000_000.0


def get_sentinel2_collection(
    roi: "ee.Geometry",
    start: str,
    end: str,
    max_cloud_pct: float = 80.0,
    land_mask: Optional["ee.Image"] = None,
) -> "ee.ImageCollection":
    """
    Build a cloud-masked Sentinel-2 SR ImageCollection for a given ROI and
    date range.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    start : str
        Start date (inclusive), format ``YYYY-MM-DD``.
    end : str
        End date (exclusive), format ``YYYY-MM-DD``.
    max_cloud_pct : float
        Pre-filter by scene-level cloud percentage (default 80 %).  Lower
        values produce a smaller collection but more aggressively filtered
        composites.
    land_mask : ee.Image, optional
        Binary land mask (1 = land) applied to every image immediately after
        cloud masking, before any band math or compositing.  Water pixels are
        excluded from all downstream reductions.  If ``None``, no land
        masking is applied.

    Returns
    -------
    ee.ImageCollection
        Filtered and cloud-masked collection.
    """
    import ee

    def _preprocess(image):
        img = mask_sentinel2_clouds(image)
        if land_mask is not None:
            img = img.updateMask(land_mask)
        return img

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(_preprocess)
    )
    return collection


# ---------------------------------------------------------------------------
# Landsat helpers
# ---------------------------------------------------------------------------

def mask_landsat_clouds(image: "ee.Image") -> "ee.Image":
    """
    Mask clouds, cloud shadows, and snow from a Landsat 8/9 C2 L2 image.

    Uses the ``QA_PIXEL`` band:

    - Bit 3: Cloud shadow
    - Bit 4: Snow
    - Bit 5: Cloud

    Parameters
    ----------
    image : ee.Image
        Single Landsat 8 or 9 image from the Collection 2 Level-2 catalog.

    Returns
    -------
    ee.Image
        Cloud-masked image.
    """
    import ee

    qa = image.select("QA_PIXEL")
    # Bit 5 = cloud, bit 3 = cloud shadow, bit 4 = snow/ice
    cloud_mask = qa.bitwiseAnd(1 << 5).eq(0)
    shadow_mask = qa.bitwiseAnd(1 << 3).eq(0)
    snow_mask = qa.bitwiseAnd(1 << 4).eq(0)
    combined = cloud_mask.And(shadow_mask).And(snow_mask)
    return image.updateMask(combined)


def get_landsat_collection(
    roi: "ee.Geometry",
    start: str,
    end: str,
    max_cloud_pct: float = 80.0,
    land_mask: Optional["ee.Image"] = None,
) -> "ee.ImageCollection":
    """
    Build a merged, cloud-masked Landsat 8+9 Collection 2 Level-2
    ImageCollection.

    Both LC08 and LC09 collections are queried and merged so that the
    composite draws from the maximum available observations.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    start, end : str
        Date range (``YYYY-MM-DD``).
    max_cloud_pct : float
        Maximum scene-level cloud cover percentage.
    land_mask : ee.Image, optional
        Binary land mask applied per-image before compositing (see
        :func:`get_sentinel2_collection`).

    Returns
    -------
    ee.ImageCollection
        Merged, cloud-masked Landsat 8/9 collection.
    """
    import ee

    def _preprocess(image):
        img = mask_landsat_clouds(image)
        if land_mask is not None:
            img = img.updateMask(land_mask)
        return img

    def _filter_and_mask(collection_id: str) -> "ee.ImageCollection":
        return (
            ee.ImageCollection(collection_id)
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_pct))
            .map(_preprocess)
        )

    lc08 = _filter_and_mask("LANDSAT/LC08/C02/T1_L2")
    lc09 = _filter_and_mask("LANDSAT/LC09/C02/T1_L2")
    return lc08.merge(lc09)


# ---------------------------------------------------------------------------
# Collection size check
# ---------------------------------------------------------------------------

def check_collection_size(
    collection: "ee.ImageCollection",
    window_label: str = "",
    min_images: int = _MIN_IMAGES,
    warn_images: int = _WARN_IMAGES,
) -> int:
    """
    Verify that a collection contains enough cloud-free images to form a
    reliable composite.

    Parameters
    ----------
    collection : ee.ImageCollection
        Collection to inspect.
    window_label : str
        Human-readable label for the window (e.g. ``"pre-event"``), used in
        warning/error messages.
    min_images : int
        If the count is below this value, :exc:`.InsufficientDataError` is
        raised.
    warn_images : int
        If the count is below this value (but >= min_images), a warning is
        logged.

    Returns
    -------
    int
        Number of images in the collection.

    Raises
    ------
    InsufficientDataError
        When there are fewer than ``min_images`` cloud-free scenes.
    """
    count = collection.size().getInfo()
    label = f" ({window_label})" if window_label else ""
    if count < min_images:
        raise InsufficientDataError(
            f"No cloud-free images found{label}. "
            "Consider widening the date window (--pre-days / --post-days) "
            "or increasing the cloud percentage threshold."
        )
    if count < warn_images:
        logger.warning(
            "Only %d cloud-free image(s) found%s. "
            "The composite may be noisy; consider widening the window.",
            count,
            label,
        )
    else:
        logger.info("Found %d image(s)%s.", count, label)
    return count


# ---------------------------------------------------------------------------
# Composite construction
# ---------------------------------------------------------------------------

def get_median_composite(
    collection: "ee.ImageCollection",
    roi: "ee.Geometry",
) -> "ee.Image":
    """
    Create a median composite from a cloud-masked ImageCollection, clipped
    to the ROI.

    The median reducer is robust to remaining cloud artifacts and minimises
    the influence of atmospheric outliers, making it the preferred aggregation
    method for multi-date vegetation analysis.

    Parameters
    ----------
    collection : ee.ImageCollection
        Cloud-masked image collection.
    roi : ee.Geometry
        Region of interest for clipping.

    Returns
    -------
    ee.Image
        Median composite image clipped to the ROI.
    """
    return collection.median().clip(roi)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_optical_metadata(
    collection: "ee.ImageCollection",
    sensor: str,
    window_start: str = "",
    window_end: str = "",
) -> Dict:
    """
    Extract per-scene metadata from a cloud-masked optical ImageCollection.

    Parameters
    ----------
    collection : ee.ImageCollection
        A Sentinel-2 or Landsat collection (already filtered and cloud-masked).
    sensor : str
        ``"sentinel2"`` or ``"landsat"``.
    window_start, window_end : str
        Human-readable date-range labels stored in the returned dict.

    Returns
    -------
    dict
        ``{sensor, window_start, window_end, count, scenes}``.
        ``scenes`` is a list of ``{id, date, cloud_pct}`` dicts sorted by date.
        On any GEE failure the dict contains ``"error"`` and ``count=None``.
    """
    try:
        from .metadata_utils import ts_to_date

        cloud_prop = (
            "CLOUDY_PIXEL_PERCENTAGE" if sensor == "sentinel2" else "CLOUD_COVER"
        )
        ids    = collection.aggregate_array("system:index").getInfo()
        times  = collection.aggregate_array("system:time_start").getInfo()
        clouds = collection.aggregate_array(cloud_prop).getInfo()

        n = len(ids)
        scenes = [
            {
                "id":        ids[i],
                "date":      ts_to_date(times[i]),
                "cloud_pct": round(float(clouds[i]), 1)
                             if (clouds and i < len(clouds) and clouds[i] is not None)
                             else None,
            }
            for i in range(n)
        ]
        scenes.sort(key=lambda s: s["date"])
        return {
            "sensor":       sensor,
            "window_start": window_start,
            "window_end":   window_end,
            "count":        n,
            "scenes":       scenes,
        }
    except Exception as exc:
        logger.warning("Optical metadata extraction failed (%s): %s", sensor, exc)
        return {
            "sensor":       sensor,
            "window_start": window_start,
            "window_end":   window_end,
            "count":        None,
            "scenes":       [],
            "error":        str(exc),
        }


# ---------------------------------------------------------------------------
# High-level composite retrieval
# ---------------------------------------------------------------------------

def get_composites(
    roi: "ee.Geometry",
    event_date: str,
    satellite: str = "sentinel2",
    pre_days: int = 60,
    post_days: int = 60,
    buffer_days: int = 5,
    max_cloud_pct: float = 80.0,
    mask_water: bool = False,
    mask_water_threshold: int = 80,
) -> Tuple["ee.Image", "ee.Image", Dict, Dict]:
    """
    Retrieve pre-event and post-event median composites for a given ROI,
    event date, and sensor.

    This is the primary entry point for data acquisition in the pipeline.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    event_date : str
        Hurricane landfall/closest-approach date (``YYYY-MM-DD``).
    satellite : str
        ``"sentinel2"`` or ``"landsat"``.
    pre_days : int
        Number of days in the pre-event window (default 60).
    post_days : int
        Number of days in the post-event window (default 60).
    buffer_days : int
        Days to exclude immediately around the event (default 5).
    max_cloud_pct : float
        Maximum scene-level cloud cover for pre-filtering.
    mask_water : bool
        If ``True``, apply the JRC land mask to every image before
        compositing so that water pixels never contribute to the median.
        Default ``False``.
    mask_water_threshold : int
        JRC occurrence percentage above which a pixel is treated as water
        (default 80).

    Returns
    -------
    tuple
        ``(pre_composite, post_composite, pre_metadata, post_metadata)``
        where the composites are ``ee.Image`` objects clipped to ``roi``
        and the metadata dicts contain scene IDs, dates, and cloud cover.

    Raises
    ------
    ValueError
        If ``satellite`` is not recognised.
    InsufficientDataError
        If either window has no cloud-free images.
    """
    satellite = satellite.lower()
    if satellite not in {"sentinel2", "landsat"}:
        raise ValueError(
            f"Unknown satellite '{satellite}'. Use 'sentinel2' or 'landsat'."
        )

    # Build land mask once so every image in both collections shares the same
    # server-side expression (GEE deduplicates identical sub-graphs).
    land_mask = build_jrc_land_mask(mask_water_threshold) if mask_water else None
    if mask_water:
        logger.info(
            "Land mask enabled (JRC GSW occurrence > %d%%) — applied per-image "
            "before compositing.", mask_water_threshold
        )

    pre_start, pre_end, post_start, post_end = date_windows(
        event_date, pre_days=pre_days, post_days=post_days, buffer_days=buffer_days
    )
    logger.info(
        "Pre-event window : %s → %s", pre_start, pre_end
    )
    logger.info(
        "Post-event window: %s → %s", post_start, post_end
    )

    # Select collection builder
    if satellite == "sentinel2":
        _get_collection = lambda s, e: get_sentinel2_collection(
            roi, s, e, max_cloud_pct=max_cloud_pct, land_mask=land_mask
        )
    else:
        _get_collection = lambda s, e: get_landsat_collection(
            roi, s, e, max_cloud_pct=max_cloud_pct, land_mask=land_mask
        )

    pre_col = _get_collection(pre_start, pre_end)
    post_col = _get_collection(post_start, post_end)

    logger.info("Checking pre-event collection …")
    check_collection_size(pre_col, window_label="pre-event")
    logger.info("Checking post-event collection …")
    check_collection_size(post_col, window_label="post-event")

    # Extract scene-level metadata before compositing (aggregate_array batches
    # the GEE calls server-side, so this adds only two round-trips total).
    pre_meta  = extract_optical_metadata(pre_col,  satellite, pre_start,  pre_end)
    post_meta = extract_optical_metadata(post_col, satellite, post_start, post_end)

    pre_composite = get_median_composite(pre_col, roi)
    post_composite = get_median_composite(post_col, roi)

    logger.info("Composites built successfully.")
    return pre_composite, post_composite, pre_meta, post_meta


# ---------------------------------------------------------------------------
# Planetary Computer fallback
# ---------------------------------------------------------------------------

def get_planetary_computer_fallback(
    roi_bbox: list,
    start: str,
    end: str,
    max_cloud_pct: float = 80.0,
    bands: Optional[list] = None,
) -> "np.ndarray":
    """
    Retrieve Sentinel-2 Level-2A data from Microsoft Planetary Computer as a
    numpy array.  Used as a fallback when GEE is unavailable.

    Parameters
    ----------
    roi_bbox : list
        Bounding box [west, south, east, north] in EPSG:4326.
    start, end : str
        Date range (``YYYY-MM-DD``).
    max_cloud_pct : float
        Maximum scene cloud cover for filtering.
    bands : list of str, optional
        Band names to retrieve (default: B2, B4, B8, B11).

    Returns
    -------
    np.ndarray
        4-D array with shape ``(time, band, y, x)`` in float32 reflectance
        (0–1), or an empty array if no scenes are found.

    Notes
    -----
    Requires ``pystac-client`` and ``stackstac`` to be installed.
    The caller is responsible for any further compositing or masking.
    """
    try:
        import numpy as np
        import pystac_client
        import stackstac
    except ImportError as exc:
        raise ImportError(
            "Planetary Computer fallback requires 'pystac-client' and 'stackstac'. "
            "Install them with: pip install pystac-client stackstac"
        ) from exc

    if bands is None:
        bands = ["B02", "B04", "B08", "B11"]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    west, south, east, north = roi_bbox
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[west, south, east, north],
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud_pct}},
    )

    items = list(search.items())
    if not items:
        logger.warning(
            "Planetary Computer: no Sentinel-2 scenes found for %s → %s", start, end
        )
        return np.array([])

    logger.info(
        "Planetary Computer: found %d scenes for %s → %s", len(items), start, end
    )

    # Sign items for authenticated access
    try:
        import planetary_computer
        items = [planetary_computer.sign(i) for i in items]
    except ImportError:
        logger.warning(
            "planetary-computer package not installed; assets may be inaccessible."
        )

    stack = stackstac.stack(
        items,
        assets=bands,
        bounds_latlon=[west, south, east, north],
        dtype="float32",
        rescale=False,
    )

    # Sentinel-2 L2A is scaled 0–10000; convert to [0, 1]
    data = (stack / 10000.0).clip(0, 1)

    # Simple median composite across time
    median = np.nanmedian(data.values, axis=0)
    return median
