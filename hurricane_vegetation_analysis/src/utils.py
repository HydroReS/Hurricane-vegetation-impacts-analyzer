"""
utils.py
========
Utility helpers for the hurricane vegetation analysis pipeline.

Provides:
- Configuration loading from YAML
- ROI parsing (bounding box string or GeoJSON/Shapefile)
- Date window computation for pre/post event periods
- Google Earth Engine initialization with graceful auth prompting
- Output directory management
- Logging setup
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Raised when configuration values are invalid."""


class ROIParseError(Exception):
    """Raised when an ROI string or file cannot be parsed."""


class GEEAuthError(Exception):
    """Raised when Google Earth Engine cannot be initialized."""


class InsufficientDataError(Exception):
    """Raised when a GEE ImageCollection contains too few cloud-free images."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the config.yaml file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    ConfigurationError
        If the file does not exist or cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")
    try:
        with path.open("r") as fh:
            config = yaml.safe_load(fh)
        logger.debug("Loaded config from %s", path)
        return config
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse config YAML: {exc}") from exc


# ---------------------------------------------------------------------------
# Date windows
# ---------------------------------------------------------------------------

def date_windows(
    event_date: str,
    pre_days: int = 60,
    post_days: int = 60,
    buffer_days: int = 5,
) -> Tuple[str, str, str, str]:
    """
    Compute pre-event and post-event date windows relative to a hurricane date.

    A buffer of ``buffer_days`` is applied on both sides of the event date to
    avoid capturing the storm itself in composite images.

    Parameters
    ----------
    event_date : str
        Hurricane landfall or closest-approach date, format ``YYYY-MM-DD``.
    pre_days : int
        Length of the pre-event window (default 60).
    post_days : int
        Length of the post-event window (default 60).
    buffer_days : int
        Days to exclude immediately before and after the event (default 5).

    Returns
    -------
    tuple of str
        ``(pre_start, pre_end, post_start, post_end)`` as ``YYYY-MM-DD`` strings.

    Examples
    --------
    >>> date_windows("2022-09-28", pre_days=60, post_days=60, buffer_days=5)
    ('2022-07-30', '2022-09-23', '2022-10-03', '2022-11-27')
    """
    event = datetime.strptime(event_date, "%Y-%m-%d")
    pre_end = event - timedelta(days=buffer_days)
    pre_start = pre_end - timedelta(days=pre_days)
    post_start = event + timedelta(days=buffer_days)
    post_end = post_start + timedelta(days=post_days)
    return (
        pre_start.strftime("%Y-%m-%d"),
        pre_end.strftime("%Y-%m-%d"),
        post_start.strftime("%Y-%m-%d"),
        post_end.strftime("%Y-%m-%d"),
    )


def historical_date_windows(
    event_date: str,
    n_years: int,
    pre_days: int = 60,
    post_days: int = 60,
    buffer_days: int = 5,
) -> List[Tuple[str, str, str, str]]:
    """
    Compute the same seasonal windows for ``n_years`` prior to the event.

    Used for baseline variability comparison: same calendar window, but in
    hurricane-free years.

    Parameters
    ----------
    event_date : str
        Hurricane event date (YYYY-MM-DD).
    n_years : int
        Number of prior years to compute windows for.
    pre_days, post_days, buffer_days : int
        Same semantics as :func:`date_windows`.

    Returns
    -------
    list of tuple
        Each element is ``(pre_start, pre_end, post_start, post_end)`` strings.
    """
    event = datetime.strptime(event_date, "%Y-%m-%d")
    windows = []
    for offset in range(1, n_years + 1):
        prior_event = event.replace(year=event.year - offset)
        windows.append(
            date_windows(
                prior_event.strftime("%Y-%m-%d"),
                pre_days=pre_days,
                post_days=post_days,
                buffer_days=buffer_days,
            )
        )
    return windows


# ---------------------------------------------------------------------------
# ROI parsing
# ---------------------------------------------------------------------------

def parse_roi(roi_str: str) -> Any:
    """
    Parse an ROI specification into a GEE ``ee.Geometry``.

    Accepts two formats:

    ``bbox:W,S,E,N``
        Bounding box in EPSG:4326 decimal degrees.  Example::

            bbox:-82.2,26.4,-81.7,26.8

    ``file:path/to/region.geojson`` or ``file:path/to/region.shp``
        Path to a GeoJSON or Shapefile.  If the file contains multiple
        features their geometries are unioned into a single ee.Geometry.

    Parameters
    ----------
    roi_str : str
        ROI specification string.

    Returns
    -------
    ee.Geometry
        GEE geometry object.

    Raises
    ------
    ROIParseError
        If the string cannot be parsed or the file does not exist.
    """
    import ee  # deferred so the module can be imported without GEE installed

    roi_str = roi_str.strip()

    if roi_str.startswith("bbox:"):
        return _parse_bbox(roi_str[5:])
    elif roi_str.startswith("file:"):
        return _parse_file(roi_str[5:])
    else:
        # Try to interpret as a raw bbox (no prefix)
        try:
            return _parse_bbox(roi_str)
        except Exception:
            raise ROIParseError(
                f"Unrecognised ROI format: '{roi_str}'. "
                "Use 'bbox:W,S,E,N' or 'file:path/to/region.geojson'."
            )


def _parse_bbox(coords_str: str) -> Any:
    """Parse a comma-separated W,S,E,N bounding box string."""
    import ee

    try:
        parts = [float(x.strip()) for x in coords_str.split(",")]
    except ValueError as exc:
        raise ROIParseError(f"Invalid bbox coordinates: '{coords_str}'") from exc

    if len(parts) != 4:
        raise ROIParseError(
            f"Bounding box must have exactly 4 values (W,S,E,N), got {len(parts)}"
        )

    west, south, east, north = parts
    if west >= east:
        raise ROIParseError(f"West ({west}) must be less than East ({east})")
    if south >= north:
        raise ROIParseError(f"South ({south}) must be less than North ({north})")

    return ee.Geometry.BBox(west, south, east, north)


def _parse_file(file_path: str) -> Any:
    """Parse a GeoJSON or Shapefile into an ee.Geometry."""
    import ee
    import geopandas as gpd

    path = Path(file_path)
    if not path.exists():
        raise ROIParseError(f"ROI file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in {".geojson", ".json", ".shp", ".gpkg"}:
        raise ROIParseError(
            f"Unsupported file type '{suffix}'. Use .geojson, .json, .shp, or .gpkg."
        )

    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        raise ROIParseError(f"Failed to read '{path}': {exc}") from exc

    if gdf.empty:
        raise ROIParseError(f"ROI file '{path}' contains no features.")

    # Re-project to WGS84 if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Union all features into one geometry
    union = gdf.geometry.unary_union

    return ee.Geometry(union.__geo_interface__)


# ---------------------------------------------------------------------------
# GEE initialization
# ---------------------------------------------------------------------------

def ee_init(project: str = "vegetation-impact-analysis") -> None:
    """
    Initialize the Google Earth Engine API.

    Attempts ``ee.Initialize(project=project)``.  If credentials are missing
    or the project is not found, prompts the user to run ``ee.Authenticate()``
    and provides guidance.

    Parameters
    ----------
    project : str
        GEE Cloud project ID.

    Raises
    ------
    GEEAuthError
        If initialization fails after authentication attempt.
    """
    try:
        import ee
    except ImportError as exc:
        raise GEEAuthError(
            "The 'earthengine-api' package is not installed. "
            "Run: pip install earthengine-api"
        ) from exc

    try:
        ee.Initialize(project=project)
        logger.info("GEE initialized with project '%s'", project)
    except Exception as first_exc:
        logger.warning("GEE initialization failed: %s", first_exc)
        logger.info("Attempting ee.Authenticate() …")
        try:
            ee.Authenticate()
            ee.Initialize(project=project)
            logger.info("GEE initialized successfully after authentication.")
        except Exception as second_exc:
            raise GEEAuthError(
                f"Could not initialize Google Earth Engine.\n"
                f"  1. Make sure you have a GEE account: https://earthengine.google.com/\n"
                f"  2. Run 'earthengine authenticate' in your terminal.\n"
                f"  3. Ensure the project '{project}' exists and is enabled.\n"
                f"Original error: {second_exc}"
            ) from second_exc


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create ``path`` (and any parents) if it does not already exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        Resolved absolute path of the directory.
    """
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """
    Configure the root logger with a human-readable format.

    Parameters
    ----------
    verbose : bool
        If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def bbox_to_list(bbox: List[float]) -> List[float]:
    """
    Validate and return a bounding box as [west, south, east, north].

    Parameters
    ----------
    bbox : list of float
        Four-element list [W, S, E, N].

    Returns
    -------
    list of float

    Raises
    ------
    ConfigurationError
        If the list is malformed.
    """
    if len(bbox) != 4:
        raise ConfigurationError(f"Bounding box must have 4 elements, got {len(bbox)}")
    west, south, east, north = bbox
    if west >= east or south >= north:
        raise ConfigurationError(
            f"Invalid bounding box [{west}, {south}, {east}, {north}]: "
            "west must be < east and south must be < north."
        )
    return [float(west), float(south), float(east), float(north)]
