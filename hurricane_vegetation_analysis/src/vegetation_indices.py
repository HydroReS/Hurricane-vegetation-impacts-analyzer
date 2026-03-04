"""
vegetation_indices.py
=====================
Compute spectral vegetation indices on Google Earth Engine ``ee.Image`` objects.

Supported indices
-----------------
- **NDVI** — Normalized Difference Vegetation Index
  ``(NIR − Red) / (NIR + Red)``
- **EVI**  — Enhanced Vegetation Index
  ``2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1)``
- **SAVI** — Soil-Adjusted Vegetation Index
  ``((NIR − Red) / (NIR + Red + L)) × (1 + L)``, L = 0.5
- **NDMI** — Normalized Difference Moisture Index
  ``(NIR − SWIR1) / (NIR + SWIR1)``

Supported satellites
--------------------
- ``sentinel2`` — Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED)
- ``landsat``   — Landsat 8/9 Collection 2 Level-2 (SR_B* bands)

Scaling factors are applied before index computation so that band values are
in physical reflectance units (0–1).

Usage
-----
>>> import ee
>>> from src.vegetation_indices import compute_index
>>> ndvi_img = compute_index(s2_image, "NDVI", "sentinel2")
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band name mappings
# ---------------------------------------------------------------------------

#: Map from logical band name → actual GEE band name, per satellite.
BAND_MAPS: Dict[str, Dict[str, str]] = {
    "sentinel2": {
        "blue": "B2",
        "green": "B3",
        "red": "B4",
        "rededge": "B5",
        "nir": "B8",
        "swir1": "B11",
        "swir2": "B12",
    },
    "landsat": {
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "swir2": "SR_B7",
    },
}

#: Scaling factors to convert raw DN to surface reflectance (0–1).
#: Sentinel-2 SR values are stored as integers ×10 000.
#: Landsat C2 L2 uses gain = 2.75×10⁻⁵, offset = −0.2.
SCALE_FACTORS: Dict[str, Dict[str, float]] = {
    "sentinel2": {"multiply": 1e-4, "offset": 0.0},
    "landsat": {"multiply": 2.75e-5, "offset": -0.2},
}

#: All supported index names (case-insensitive on input).
SUPPORTED_INDICES = {"NDVI", "EVI", "SAVI", "NDMI"}


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def apply_scale(image: "ee.Image", satellite: str) -> "ee.Image":
    """
    Apply the radiometric scaling factors for a given satellite sensor.

    Converts raw integer DN values stored in a GEE ``ee.Image`` to
    surface reflectance in the range 0–1.

    Parameters
    ----------
    image : ee.Image
        Raw image from the GEE catalog (unscaled).
    satellite : str
        Sensor identifier — ``"sentinel2"`` or ``"landsat"``.

    Returns
    -------
    ee.Image
        Scaled image with the same band structure.

    Raises
    ------
    ValueError
        If ``satellite`` is not recognised.
    """
    satellite = satellite.lower()
    if satellite not in SCALE_FACTORS:
        raise ValueError(
            f"Unknown satellite '{satellite}'. "
            f"Choose from: {list(SCALE_FACTORS.keys())}"
        )

    import ee

    factors = SCALE_FACTORS[satellite]
    scaled = image.multiply(factors["multiply"]).add(factors["offset"])

    # Clamp to [0, 1] to remove any out-of-range artifacts
    scaled = scaled.clamp(0, 1)

    # copyProperties() returns ee.Element (base class), not ee.Image.
    # Cast back explicitly so callers can still call .select(), .addBands(), etc.
    return ee.Image(scaled.copyProperties(image, image.propertyNames()))


# ---------------------------------------------------------------------------
# Per-index computation helpers
# ---------------------------------------------------------------------------

def compute_ndvi(image: "ee.Image", bands: Dict[str, str]) -> "ee.Image":
    """
    Compute NDVI = (NIR − Red) / (NIR + Red).

    Parameters
    ----------
    image : ee.Image
        Scaled reflectance image.
    bands : dict
        Mapping of logical name → actual band name (from :data:`BAND_MAPS`).

    Returns
    -------
    ee.Image
        Image with an added band named ``"NDVI"`` in the range [−1, 1].
    """
    nir = image.select(bands["nir"])
    red = image.select(bands["red"])
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    return image.addBands(ndvi)


def compute_evi(image: "ee.Image", bands: Dict[str, str]) -> "ee.Image":
    """
    Compute EVI = 2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1).

    EVI corrects for atmospheric disturbances and is less prone to saturation
    in dense forest canopies compared with NDVI.

    Parameters
    ----------
    image : ee.Image
        Scaled reflectance image.
    bands : dict
        Mapping of logical name → actual band name.

    Returns
    -------
    ee.Image
        Image with an added band named ``"EVI"``.
    """
    nir = image.select(bands["nir"])
    red = image.select(bands["red"])
    blue = image.select(bands["blue"])

    evi = (
        nir.subtract(red)
        .multiply(2.5)
        .divide(
            nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
        )
        .rename("EVI")
    )
    # EVI can exceed [−1, 1] over bright surfaces; clamp to sensible range
    evi = evi.clamp(-1, 1)
    return image.addBands(evi)


def compute_savi(
    image: "ee.Image", bands: Dict[str, str], L: float = 0.5
) -> "ee.Image":
    """
    Compute SAVI = ((NIR − Red) / (NIR + Red + L)) × (1 + L).

    The soil adjustment factor *L* reduces the effect of bare soil background
    reflectance.  A value of 0.5 works well for intermediate vegetation density.

    Parameters
    ----------
    image : ee.Image
        Scaled reflectance image.
    bands : dict
        Mapping of logical name → actual band name.
    L : float
        Soil adjustment factor (default 0.5).

    Returns
    -------
    ee.Image
        Image with an added band named ``"SAVI"``.
    """
    nir = image.select(bands["nir"])
    red = image.select(bands["red"])

    savi = (
        nir.subtract(red)
        .divide(nir.add(red).add(L))
        .multiply(1 + L)
        .rename("SAVI")
    )
    return image.addBands(savi)


def compute_ndmi(image: "ee.Image", bands: Dict[str, str]) -> "ee.Image":
    """
    Compute NDMI = (NIR − SWIR1) / (NIR + SWIR1).

    NDMI is sensitive to vegetation water content and canopy moisture stress,
    which is particularly useful for detecting storm-surge-induced salt stress
    and inundation effects.

    Parameters
    ----------
    image : ee.Image
        Scaled reflectance image.
    bands : dict
        Mapping of logical name → actual band name.

    Returns
    -------
    ee.Image
        Image with an added band named ``"NDMI"`` in the range [−1, 1].
    """
    nir = image.select(bands["nir"])
    swir1 = image.select(bands["swir1"])
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")
    return image.addBands(ndmi)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def compute_index(
    image: "ee.Image",
    index_name: str,
    satellite: str,
    apply_scaling: bool = True,
) -> "ee.Image":
    """
    Compute a vegetation index on a GEE image, returning the image with the
    index added as a new band.

    This is the primary entry point used by the rest of the pipeline.

    Parameters
    ----------
    image : ee.Image
        Raw (unscaled) or already-scaled GEE image.
    index_name : str
        One of ``"NDVI"``, ``"EVI"``, ``"SAVI"``, ``"NDMI"``
        (case-insensitive).
    satellite : str
        ``"sentinel2"`` or ``"landsat"`` — controls band name mapping and
        scaling factors.
    apply_scaling : bool
        If True (default), apply :func:`apply_scale` before computing the
        index.  Set to False if the image is already in reflectance units.

    Returns
    -------
    ee.Image
        Image with the vegetation index band added.

    Raises
    ------
    ValueError
        If ``index_name`` or ``satellite`` is not recognised.

    Examples
    --------
    >>> from src.vegetation_indices import compute_index
    >>> ndvi_img = compute_index(sentinel2_raw_img, "NDVI", "sentinel2")
    >>> evi_img  = compute_index(landsat_raw_img,   "EVI",  "landsat")
    """
    index_name = index_name.upper()
    satellite = satellite.lower()

    if index_name not in SUPPORTED_INDICES:
        raise ValueError(
            f"Unsupported index '{index_name}'. "
            f"Choose from: {sorted(SUPPORTED_INDICES)}"
        )
    if satellite not in BAND_MAPS:
        raise ValueError(
            f"Unsupported satellite '{satellite}'. "
            f"Choose from: {list(BAND_MAPS.keys())}"
        )

    bands = BAND_MAPS[satellite]

    if apply_scaling:
        image = apply_scale(image, satellite)

    dispatcher = {
        "NDVI": compute_ndvi,
        "EVI": compute_evi,
        "SAVI": compute_savi,
        "NDMI": compute_ndmi,
    }

    if index_name == "SAVI":
        result = dispatcher[index_name](image, bands, L=0.5)
    else:
        result = dispatcher[index_name](image, bands)

    logger.debug("Computed %s for satellite '%s'", index_name, satellite)
    return result


def get_index_band(index_name: str) -> str:
    """
    Return the GEE band name that :func:`compute_index` adds to an image.

    Parameters
    ----------
    index_name : str
        Index identifier (case-insensitive).

    Returns
    -------
    str
        Band name string (e.g. ``"NDVI"``).
    """
    return index_name.upper()
