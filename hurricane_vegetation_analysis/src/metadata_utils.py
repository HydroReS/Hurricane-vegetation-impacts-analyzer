"""
metadata_utils.py
=================
Utility functions for extracting, summarising, and warning about
image acquisition metadata for every sensor and composite window.

Functions here are sensor-agnostic.  Sensor-specific extraction
functions live alongside their respective collections:

- ``extract_optical_metadata`` — in ``data_acquisition.py``
- ``extract_sar_metadata``     — in ``structural_analysis.py``
- ``extract_palsar_metadata``  — in ``structural_analysis.py``
- ``extract_gedi_metadata``    — in ``structural_analysis.py``
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def ts_to_date(ts_ms: float) -> str:
    """Convert a GEE millisecond timestamp to a ``YYYY-MM-DD`` string (UTC)."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _max_gap_days(dates: List[str]) -> int:
    """Return the largest gap (days) between consecutive dates in a list."""
    if len(dates) < 2:
        return 0
    sorted_dts = sorted(datetime.strptime(d, "%Y-%m-%d") for d in dates)
    return max(
        (sorted_dts[i + 1] - sorted_dts[i]).days
        for i in range(len(sorted_dts) - 1)
    )


# ---------------------------------------------------------------------------
# Warning generation
# ---------------------------------------------------------------------------

def check_warnings(scene_metadata: Dict[str, Any], event_date: str) -> List[str]:
    """
    Inspect collected scene metadata and return human-readable warning strings.

    Warning triggers
    ----------------
    - 0 images found for any sensor / window.
    - < 3 images for optical or SAR composites.
    - Largest consecutive inter-scene gap > 30 days within a window.
    - PALSAR annual mosaic epoch contains the event date (pre- or post-event
      window, meaning the mosaic may include post-hurricane vegetation state).
    """
    warnings: List[str] = []
    sensor_labels = {
        "sentinel2": "Sentinel-2",
        "landsat":   "Landsat 8/9",
        "sentinel1": "Sentinel-1 SAR",
        "palsar":    "PALSAR",
        "gedi":      "GEDI",
    }

    for window_label, window_key in [("pre-event", "pre"), ("post-event", "post")]:
        # ── Optical ──────────────────────────────────────────────────────────
        opt = scene_metadata.get("optical", {}).get(window_key, {})
        if opt:
            sname = sensor_labels.get(opt.get("sensor", ""), "Optical")
            count = opt.get("count") or 0
            if count == 0:
                warnings.append(
                    f"No {sname} images found for the {window_label} window."
                )
            elif count < 3:
                warnings.append(
                    f"Low {sname} image count: only {count} image(s) in the "
                    f"{window_label} window. Composite may be noisy."
                )
            dates = [s["date"] for s in opt.get("scenes", []) if s.get("date")]
            gap = _max_gap_days(dates)
            if gap > 30:
                warnings.append(
                    f"{sname} {window_label} window has a {gap}-day gap "
                    "between consecutive scenes."
                )

        # ── SAR ──────────────────────────────────────────────────────────────
        sar = scene_metadata.get("sar", {}).get(window_key, {})
        if sar:
            count = sar.get("count") or 0
            if count == 0:
                warnings.append(
                    f"No Sentinel-1 SAR images found for the {window_label} window."
                )
            elif count < 3:
                warnings.append(
                    f"Low SAR image count: only {count} Sentinel-1 image(s) "
                    f"in the {window_label} window."
                )
            dates = [s["date"] for s in sar.get("scenes", []) if s.get("date")]
            gap = _max_gap_days(dates)
            if gap > 30:
                warnings.append(
                    f"Sentinel-1 SAR {window_label} window has a {gap}-day "
                    "gap between consecutive scenes."
                )

        # ── PALSAR epoch overlap ──────────────────────────────────────────────
        palsar = scene_metadata.get("palsar", {}).get(window_key, {})
        if palsar and palsar.get("event_in_epoch"):
            yr = palsar.get("year", "?")
            warnings.append(
                f"PALSAR {window_label} mosaic (year {yr}) contains the event "
                f"date ({event_date}) — it may include post-hurricane "
                "vegetation observations."
            )

    return warnings


# ---------------------------------------------------------------------------
# Table builders for display
# ---------------------------------------------------------------------------

def build_optical_table(meta: Dict[str, Any]) -> Optional["pd.DataFrame"]:
    """Return a ``pandas.DataFrame`` of optical scene metadata, or ``None``."""
    scenes = meta.get("scenes", [])
    if not scenes:
        return None
    try:
        import pandas as pd
        return pd.DataFrame([
            {
                "Scene ID": s.get("id", ""),
                "Date":     s.get("date", ""),
                "Cloud %":  s.get("cloud_pct", "—"),
            }
            for s in scenes
        ])
    except ImportError:
        return None


def build_sar_table(meta: Dict[str, Any]) -> Optional["pd.DataFrame"]:
    """Return a ``pandas.DataFrame`` of SAR scene metadata, or ``None``."""
    scenes = meta.get("scenes", [])
    if not scenes:
        return None
    try:
        import pandas as pd
        return pd.DataFrame([
            {
                "Scene ID": s.get("id", ""),
                "Date":     s.get("date", ""),
                "Orbit":    s.get("orbit", "—"),
                "Pass":     s.get("pass", "—"),
            }
            for s in scenes
        ])
    except ImportError:
        return None
