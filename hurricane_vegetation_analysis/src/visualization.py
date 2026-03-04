"""
visualization.py
================
Generate maps, charts, and HTML reports for hurricane vegetation impact results.

Outputs
-------
- **Difference map** — interactive HTML folium map with pre/post index layers,
  diverging delta layer (red = loss, green = gain), classification overlay,
  and ROI boundary.
- **Distribution plot** — overlaid histograms + KDE of pre vs post index values.
- **Time series plot** — monthly index composites ±18 months around the event.
- **HTML report** — Jinja2-rendered summary with all figures embedded as base64.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML report template (inline Jinja2)
# ---------------------------------------------------------------------------

_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Hurricane Vegetation Impact Report</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px;
           margin: 0 auto; padding: 24px; background: #f8f9fa; color: #212529; }
    h1   { color: #c0392b; border-bottom: 2px solid #c0392b; padding-bottom: 8px; }
    h2   { color: #2c3e50; margin-top: 36px; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0; }
    th, td { border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }
    th   { background: #343a40; color: #fff; }
    tr:nth-child(even) { background: #e9ecef; }
    .metric { display: inline-block; background: #fff; border: 1px solid #dee2e6;
              border-radius: 6px; padding: 12px 20px; margin: 8px;
              box-shadow: 0 1px 3px rgba(0,0,0,.1); min-width: 160px; }
    .metric .label { font-size: 0.78em; color: #6c757d; text-transform: uppercase; }
    .metric .value { font-size: 1.5em; font-weight: 700; color: #343a40; }
    .conclusion { background: #d4edda; border-left: 5px solid #28a745;
                  padding: 14px; border-radius: 4px; margin: 16px 0; }
    .conclusion.warn { background: #fff3cd; border-color: #ffc107; }
    .conclusion.alert { background: #f8d7da; border-color: #dc3545; }
    img { max-width: 100%; border: 1px solid #dee2e6; border-radius: 4px;
          margin: 12px 0; }
    .footer { margin-top: 48px; font-size: 0.8em; color: #6c757d;
              border-top: 1px solid #dee2e6; padding-top: 12px; }
  </style>
</head>
<body>
<h1>Hurricane Vegetation Impact Analysis Report</h1>

<h2>Metadata</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Event Date</td><td>{{ event_date }}</td></tr>
  <tr><td>Satellite</td><td>{{ satellite }}</td></tr>
  <tr><td>Vegetation Index</td><td>{{ index }}</td></tr>
  <tr><td>Pre-event Window</td><td>{{ pre_window }}</td></tr>
  <tr><td>Post-event Window</td><td>{{ post_window }}</td></tr>
  <tr><td>Pixels Sampled</td><td>{{ n_pixels }}</td></tr>
  <tr><td>Generated</td><td>{{ generated }}</td></tr>
</table>

<h2>Key Metrics</h2>
<div>
  <div class="metric">
    <div class="label">Pre Mean {{ index }}</div>
    <div class="value">{{ "%.4f"|format(pre_mean) }}</div>
  </div>
  <div class="metric">
    <div class="label">Post Mean {{ index }}</div>
    <div class="value">{{ "%.4f"|format(post_mean) }}</div>
  </div>
  <div class="metric">
    <div class="label">Mean Delta</div>
    <div class="value" style="color: {{ 'green' if delta_mean > 0 else '#c0392b' }}">
      {{ "%+.4f"|format(delta_mean) }}
    </div>
  </div>
  <div class="metric">
    <div class="label">% Change</div>
    <div class="value">{{ "%+.1f%%"|format(delta_pct) if delta_pct is not none else "N/A" }}</div>
  </div>
  <div class="metric">
    <div class="label">Cohen's d</div>
    <div class="value">{{ "%.2f"|format(cohens_d) }} ({{ effect_label }})</div>
  </div>
  <div class="metric">
    <div class="label">Wilcoxon p-value</div>
    <div class="value">{{ "%.4f"|format(wilcoxon_pvalue) }}</div>
  </div>
</div>

<h2>Statistical Conclusion</h2>
<div class="conclusion {{ 'alert' if significant else '' }}">
  <strong>{{ conclusion }}</strong>
</div>

{% if baseline_interp %}
<h2>Baseline Variability Check</h2>
<div class="conclusion warn">{{ baseline_interp }}</div>
{% endif %}

<h2>Impact Area by Severity Class</h2>
<table>
  <tr><th>Class</th><th>Area (km²)</th></tr>
  {% for cls, area in area_by_class.items() %}
  <tr><td>{{ cls }}</td><td>{{ "%.2f"|format(area) }}</td></tr>
  {% endfor %}
</table>

{% if dist_fig_b64 %}
<h2>Index Distribution: Pre vs Post</h2>
<img src="data:image/png;base64,{{ dist_fig_b64 }}" alt="Distribution plot"/>
{% endif %}

{% if ts_fig_b64 %}
<h2>Vegetation Index Time Series</h2>
<img src="data:image/png;base64,{{ ts_fig_b64 }}" alt="Time series plot"/>
{% endif %}

{% if map_html_path %}
<h2>Interactive Difference Map</h2>
<p>See: <a href="{{ map_html_path }}">{{ map_html_path }}</a></p>
{% endif %}

<h2>Detailed Statistics</h2>
<table>
  <tr><th>Test</th><th>Statistic</th><th>p-value</th></tr>
  <tr><td>Paired t-test</td><td>{{ "%.4f"|format(ttest_stat) }}</td>
      <td>{{ "%.4f"|format(ttest_pvalue) }}</td></tr>
  <tr><td>Wilcoxon signed-rank</td><td>{{ "%.4f"|format(wilcoxon_stat) }}</td>
      <td>{{ "%.4f"|format(wilcoxon_pvalue) }}</td></tr>
</table>
<p>95% CI of mean delta: [{{ "%.4f"|format(ci_low) }}, {{ "%.4f"|format(ci_high) }}]</p>

<div class="footer">
  Generated by <strong>Hurricane Vegetation Analysis Tool</strong> •
  Data: {{ satellite_full }} • Analysis date: {{ generated }}
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Distribution Plot
# ---------------------------------------------------------------------------

def plot_distributions(
    pre_vals: np.ndarray,
    post_vals: np.ndarray,
    index: str,
    output_path: str,
    event_date: str = "",
) -> str:
    """
    Generate an overlaid histogram + KDE plot comparing pre vs post index distributions.

    Parameters
    ----------
    pre_vals : np.ndarray
        Pre-event index values.
    post_vals : np.ndarray
        Post-event index values.
    index : str
        Index name (used in labels).
    output_path : str
        Full path for the output PNG file.
    event_date : str, optional
        Used in the plot title.

    Returns
    -------
    str
        Path of the saved PNG file.
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histograms
    bins = np.linspace(
        min(pre_vals.min(), post_vals.min()),
        max(pre_vals.max(), post_vals.max()),
        50,
    )
    ax.hist(pre_vals, bins=bins, alpha=0.4, color="#2980b9", label="Pre-event", density=True)
    ax.hist(post_vals, bins=bins, alpha=0.4, color="#e74c3c", label="Post-event", density=True)

    # KDE overlays
    for vals, color in [(pre_vals, "#1a5f8a"), (post_vals, "#922b21")]:
        if len(vals) > 2:
            kde = gaussian_kde(vals)
            x = np.linspace(vals.min(), vals.max(), 300)
            ax.plot(x, kde(x), color=color, linewidth=2)

    # Mean lines
    pre_mean = np.mean(pre_vals)
    post_mean = np.mean(post_vals)
    ax.axvline(pre_mean, color="#1a5f8a", linestyle="--", linewidth=1.5,
               label=f"Pre mean = {pre_mean:.3f}")
    ax.axvline(post_mean, color="#922b21", linestyle="--", linewidth=1.5,
               label=f"Post mean = {post_mean:.3f}")

    title = f"{index} Distribution: Pre vs Post"
    if event_date:
        title += f" (Event: {event_date})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(index, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Distribution plot saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Interactive Difference Map (folium)
# ---------------------------------------------------------------------------

def create_difference_map(
    pre_img: "ee.Image",
    post_img: "ee.Image",
    diff_img: "ee.Image",
    classified_img: "ee.Image",
    roi: "ee.Geometry",
    index: str,
    output_path: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """
    Create an interactive HTML map showing pre/post index layers, the delta
    (diverging colormap), and the impact classification, using ``geemap``.

    Parameters
    ----------
    pre_img : ee.Image
        Pre-event composite with index band.
    post_img : ee.Image
        Post-event composite with index band.
    diff_img : ee.Image
        Difference image (bands: ``delta``, ``pct_change``).
    classified_img : ee.Image
        Impact classification image (0–3).
    roi : ee.Geometry
        Region of interest (shown as boundary overlay).
    index : str
        Index name (used in layer labels).
    output_path : str
        Full path for the output HTML file.

    Returns
    -------
    str
        Path of the saved HTML file.
    """
    try:
        import folium
        import ee
    except ImportError:
        logger.warning("folium not available. Skipping interactive map generation.")
        return ""

    # Centre map on ROI centroid
    centroid = roi.centroid(maxError=1).getInfo()["coordinates"]
    centre_lon, centre_lat = centroid

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=10)

    # Visualisation parameters (GEE palette colours must not have '#' prefix)
    ndvi_vis = {"min": 0, "max": 1, "palette": ["ffffff", "006400"]}
    diff_vis = {
        "min": -0.5, "max": 0.5,
        "palette": ["d73027", "f46d43", "fdae61", "ffffbf", "a6d96a", "1a9641"],
    }
    class_vis = {
        "min": 0, "max": 3,
        "palette": ["2ecc71", "f1c40f", "e67e22", "c0392b"],
    }

    def _add_ee_layer(image, vis_params, name, shown=True):
        """Fetch a GEE tile URL and add it as a folium TileLayer."""
        try:
            map_id = image.getMapId(vis_params)
            tile_url = map_id["tile_fetcher"].url_format
            folium.TileLayer(
                tiles=tile_url,
                attr="Google Earth Engine",
                name=name,
                overlay=True,
                control=True,
                show=shown,
            ).add_to(m)
        except Exception as exc:
            logger.warning("Could not add EE layer '%s': %s", name, exc)

    _add_ee_layer(pre_img.select(index), ndvi_vis, f"Pre-event {index}", shown=True)
    _add_ee_layer(post_img.select(index), ndvi_vis, f"Post-event {index}", shown=False)
    _add_ee_layer(diff_img.select("delta"), diff_vis, f"Δ{index} (post−pre)", shown=True)
    _add_ee_layer(classified_img, class_vis, "Impact Classification", shown=False)

    # ROI boundary
    try:
        roi_geojson = roi.getInfo()
        folium.GeoJson(
            roi_geojson,
            name="ROI Boundary",
            style_function=lambda _: {"color": "black", "fillOpacity": 0, "weight": 2},
        ).add_to(m)
    except Exception:
        pass

    folium.LayerControl().add_to(m)

    # --- Legend ---
    t = thresholds or {"no_impact": -0.05, "low_impact": -0.15, "moderate_impact": -0.30}
    t_no  = t.get("no_impact", -0.05)
    t_low = t.get("low_impact", -0.15)
    t_mod = t.get("moderate_impact", -0.30)

    legend_html = f"""
    <div style="
        position: fixed; bottom: 36px; left: 36px; z-index: 9999;
        background: rgba(255,255,255,0.93);
        padding: 12px 16px; border-radius: 8px;
        border: 1px solid #bbb;
        font-size: 12px; font-family: Arial, sans-serif;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        min-width: 210px;">

      <div style="font-weight:700; margin-bottom:6px; font-size:13px;">
        Impact Classification
      </div>
      <div style="display:flex;align-items:center;margin-bottom:3px;">
        <span style="background:#2ecc71;width:14px;height:14px;
                     display:inline-block;margin-right:7px;border:1px solid #aaa;"></span>
        No Impact &nbsp;<span style="color:#777;">(Δ &gt; {t_no})</span>
      </div>
      <div style="display:flex;align-items:center;margin-bottom:3px;">
        <span style="background:#f1c40f;width:14px;height:14px;
                     display:inline-block;margin-right:7px;border:1px solid #aaa;"></span>
        Low Impact &nbsp;<span style="color:#777;">({t_no} to {t_low})</span>
      </div>
      <div style="display:flex;align-items:center;margin-bottom:3px;">
        <span style="background:#e67e22;width:14px;height:14px;
                     display:inline-block;margin-right:7px;border:1px solid #aaa;"></span>
        Moderate Impact &nbsp;<span style="color:#777;">({t_low} to {t_mod})</span>
      </div>
      <div style="display:flex;align-items:center;margin-bottom:10px;">
        <span style="background:#c0392b;width:14px;height:14px;
                     display:inline-block;margin-right:7px;border:1px solid #aaa;"></span>
        Severe Impact &nbsp;<span style="color:#777;">(Δ &lt; {t_mod})</span>
      </div>

      <div style="font-weight:700; margin-bottom:4px; font-size:13px;">
        &Delta;{index} (post &minus; pre)
      </div>
      <div style="background: linear-gradient(to right,
            #d73027, #f46d43, #fdae61, #ffffbf, #a6d96a, #1a9641);
            width:178px; height:14px; border:1px solid #aaa;
            border-radius:2px; margin-bottom:2px;"></div>
      <div style="display:flex;justify-content:space-between;
                  width:178px;color:#555;font-size:11px;">
        <span>&minus;0.5</span><span>0</span><span>+0.5</span>
      </div>

      <div style="font-weight:700; margin-top:10px; margin-bottom:4px; font-size:13px;">
        {index} Value (pre / post)
      </div>
      <div style="background: linear-gradient(to right, #ffffff, #006400);
            width:178px; height:14px; border:1px solid #aaa;
            border-radius:2px; margin-bottom:2px;"></div>
      <div style="display:flex;justify-content:space-between;
                  width:178px;color:#555;font-size:11px;">
        <span>0</span><span>0.5</span><span>1.0</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    logger.info("Interactive map saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Time Series Plot
# ---------------------------------------------------------------------------

def plot_time_series(
    roi: "ee.Geometry",
    event_date: str,
    satellite: str,
    index: str,
    output_path: str,
    months_before: int = 12,
    months_after: int = 6,
    scale: int = 30,
) -> str:
    """
    Plot the monthly median vegetation index over time, showing decline and
    potential recovery trajectory.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    event_date : str
        Hurricane event date (YYYY-MM-DD).
    satellite, index : str
        Sensor and index to use.
    output_path : str
        Path for the output PNG.
    months_before : int
        Months to plot before the event (default 12).
    months_after : int
        Months to plot after the event (default 6).
    scale : int
        Sampling resolution in metres.

    Returns
    -------
    str
        Path of the saved PNG file.
    """
    import ee
    from .data_acquisition import get_sentinel2_collection, get_landsat_collection
    from .vegetation_indices import compute_index as _compute_index
    from datetime import timedelta
    import calendar

    event = datetime.strptime(event_date, "%Y-%m-%d")

    def _month_start_end(year: int, month: int):
        last_day = calendar.monthrange(year, month)[1]
        return (
            f"{year}-{month:02d}-01",
            f"{year}-{month:02d}-{last_day:02d}",
        )

    dates = []
    values = []

    total_months = months_before + months_after + 1

    for offset in range(-months_before, months_after + 1):
        # Compute target year/month
        month_idx = event.month + offset
        year = event.year + (month_idx - 1) // 12
        month = ((month_idx - 1) % 12) + 1
        start, end = _month_start_end(year, month)

        try:
            if satellite == "sentinel2":
                col = get_sentinel2_collection(roi, start, end)
            else:
                col = get_landsat_collection(roi, start, end)

            count = col.size().getInfo()
            if count == 0:
                dates.append(datetime(year, month, 15))
                values.append(np.nan)
                continue

            composite = col.median().clip(roi)
            indexed = _compute_index(composite, index, satellite)

            mean_val = (
                indexed.select(index)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=roi,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get(index)
                .getInfo()
            )
            dates.append(datetime(year, month, 15))
            values.append(float(mean_val) if mean_val is not None else np.nan)
        except Exception as exc:
            logger.debug("Time series: skipping %s-%02d: %s", year, month, exc)
            dates.append(datetime(year, month, 15))
            values.append(np.nan)

    if all(np.isnan(v) for v in values):
        logger.warning("Time series: no valid data points; skipping plot.")
        return ""

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, values, color="#2c3e50", linewidth=1.5, marker="o", markersize=5, label=index)

    # Shade missing data
    ax.fill_between(dates, values, alpha=0.15, color="#2c3e50")

    # Mark event
    ax.axvline(event, color="#c0392b", linewidth=2, linestyle="--", label=f"Event: {event_date}")

    ax.set_title(f"{index} Time Series — {event_date}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"Mean {index}", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Time series plot saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Time Series — Auxiliary Plot Functions
# ---------------------------------------------------------------------------

def _fmt_monthly_axis(ax: plt.Axes, interval: int = 6) -> None:
    """Apply monthly major ticks (every *interval* months) to *ax*."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)


_HURRICANE_COLOR = "#8B0000"  # dark red — distinct from crimson event lines


def _draw_hurricane_markers(
    ax: plt.Axes,
    hurricane_events: Optional[List[Dict[str, Any]]],
) -> None:
    """
    Draw a dashed vertical line and angled name label for each hurricane event.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    hurricane_events : list of dict, optional
        Each dict must have ``date`` (YYYY-MM-DD str), ``name`` (str), and
        optionally ``category`` (int).  Events outside the axes x-range are
        silently skipped.
    """
    if not hurricane_events:
        return
    for he in hurricane_events:
        dt = pd.Timestamp(he["date"])
        cat = he.get("category", "")
        name = he.get("name", "")
        label = f"{name} (Cat {cat})" if cat else name
        ax.axvline(dt, color=_HURRICANE_COLOR, linewidth=1.4,
                   linestyle="--", zorder=4, alpha=0.85)
        # Use xaxis transform: x in data units, y in axes fraction (0–1)
        ax.text(
            mdates.date2num(dt.to_pydatetime()), 0.97,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=45, fontsize=7, color=_HURRICANE_COLOR,
            ha="left", va="top", clip_on=True,
        )


def plot_stl_decomposition(
    stl_result: Dict[str, Any],
    index: str,
    output_path: str,
) -> str:
    """
    Generate a 4-panel STL decomposition figure.

    Parameters
    ----------
    stl_result : dict
        Output of ``src.time_series.apply_stl_decomposition``. Expected keys:
        ``observed``, ``trend``, ``seasonal``, ``residual``, ``dates``.
    index : str
        Index name used in axis labels.
    output_path : str
        Full path for the output PNG file.

    Returns
    -------
    str
        Path of the saved PNG file.
    """
    dates = list(stl_result["dates"])
    observed = np.asarray(stl_result["observed"])
    trend = np.asarray(stl_result["trend"])
    seasonal = np.asarray(stl_result["seasonal"])
    residual = np.asarray(stl_result["residual"])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{index} — STL Decomposition", fontsize=13, fontweight="bold", y=0.98)

    panels = [
        (observed, "Observed", "steelblue"),
        (trend,    "Trend",    "darkgreen"),
        (seasonal, "Seasonal", "darkorange"),
        (residual, "Residual", "gray"),
    ]
    for ax, (vals, label, color) in zip(axes, panels):
        if label == "Residual":
            ax.bar(dates, vals, color=color, alpha=0.6, width=20)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        else:
            ax.plot(dates, vals, color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)

    _fmt_monthly_axis(axes[-1], interval=6)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("STL decomposition plot saved → %s", output_path)
    return output_path


def plot_timeseries_with_anomalies(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    index: str,
    output_path: str,
    event_dates: Optional[List[str]] = None,
    stl_result: Optional[Dict[str, Any]] = None,
    title: str = "",
    hurricane_events: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Plot a vegetation index time series with anomalies highlighted and
    optional hurricane event date annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Composited time series. Must have columns ``date`` and ``index_value``.
    anomalies : pd.DataFrame
        Anomaly records from :func:`src.time_series.detect_all_anomalies`.
        Columns: ``date``, ``observed_value``, ``method``, ``severity``.
    index : str
        Index name for axis labels.
    output_path : str
        Full path for the output PNG.
    event_dates : list of str, optional
        YYYY-MM-DD strings; each drawn as a crimson dashed vertical line.
    stl_result : dict, optional
        If provided, a second panel shows the trend component and the main
        panel overlays the seasonal fit (trend + seasonal).
    title : str, optional
        Subtitle appended to the figure title.
    hurricane_events : list of dict, optional
        Hurricane catalog entries (``name``, ``date``, ``category``).  Events
        whose dates fall within the series range are marked with dark-red
        dashed lines and angled name labels.

    Returns
    -------
    str
        Path of the saved PNG.
    """
    event_dates = event_dates or []
    dates = pd.to_datetime(df["date"])
    values = df["index_value"].values

    has_stl = stl_result is not None
    if has_stl:
        fig, (ax_main, ax_trend) = plt.subplots(
            2, 1, figsize=(13, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax_main = plt.subplots(figsize=(13, 5))
        ax_trend = None

    # Observations
    ax_main.scatter(dates, values, color="#2980b9", s=18, zorder=3, label=f"Observed {index}")

    # Seasonal fit overlay
    if has_stl:
        fit = np.asarray(stl_result["trend"]) + np.asarray(stl_result["seasonal"])
        ax_main.plot(
            list(stl_result["dates"]), fit,
            color="#e67e22", linewidth=1.5, label="Seasonal fit", zorder=2,
        )

    # Anomaly markers
    _markers = {
        "zscore":        ("X", "#c0392b",   "z-score anomaly"),
        "moving_window": ("P", "darkorange", "moving-window anomaly"),
        "climatology":   ("D", "#8e44ad",   "climatology anomaly"),
    }
    if anomalies is not None and not anomalies.empty:
        for method, (marker, color, label) in _markers.items():
            sub = anomalies[anomalies["method"] == method]
            if sub.empty:
                continue
            ax_main.scatter(
                pd.to_datetime(sub["date"]),
                sub["observed_value"],
                marker=marker, color=color, s=80, zorder=5,
                label=label, edgecolors="white", linewidths=0.5,
            )

    # Event date vertical lines
    for ed in event_dates:
        ax_main.axvline(pd.Timestamp(ed), color="crimson", linewidth=1.8,
                        linestyle="--", label=f"Event: {ed}", zorder=4)

    # Hurricane event markers (dark red, named)
    _draw_hurricane_markers(ax_main, hurricane_events)

    main_title = f"{index} Time Series with Anomaly Detection"
    if title:
        main_title += f"\n{title}"
    ax_main.set_title(main_title, fontsize=12, fontweight="bold")
    ax_main.set_ylabel(index, fontsize=10)
    ax_main.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_main.grid(alpha=0.25)

    # Trend panel
    if has_stl and ax_trend is not None:
        ax_trend.plot(
            list(stl_result["dates"]),
            np.asarray(stl_result["trend"]),
            color="darkgreen", linewidth=1.5,
        )
        for ed in event_dates:
            ax_trend.axvline(pd.Timestamp(ed), color="crimson", linewidth=1.4,
                             linestyle="--", zorder=4)
        _draw_hurricane_markers(ax_trend, hurricane_events)
        ax_trend.set_ylabel("Trend", fontsize=9)
        ax_trend.grid(alpha=0.25)
        _fmt_monthly_axis(ax_trend, interval=6)
    else:
        _fmt_monthly_axis(ax_main, interval=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Time series anomaly plot saved → %s", output_path)
    return output_path


def plot_recovery_trajectory(
    df: pd.DataFrame,
    recovery: Dict[str, Any],
    event_date: str,
    index: str,
    output_path: str,
    hurricane_events: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Plot the post-hurricane vegetation recovery trajectory with a
    pre-event baseline band.

    Parameters
    ----------
    df : pd.DataFrame
        Full time series (``date``, ``index_value`` columns).
    recovery : dict
        Output of :func:`src.time_series.analyze_recovery`. Keys used:
        ``pre_mean``, ``pre_std``, ``post_min``, ``recovery_date``,
        ``recovery_pct``, ``recovery_status``, ``interpretation``.
    event_date : str
        YYYY-MM-DD hurricane date.
    index : str
        Index name.
    output_path : str
        Full path for the output PNG.
    hurricane_events : list of dict, optional
        Hurricane catalog entries; see :func:`_draw_hurricane_markers`.

    Returns
    -------
    str
        Path of the saved PNG.
    """
    dates = pd.to_datetime(df["date"])
    values = df["index_value"].values

    pre_mean = float(recovery.get("pre_mean") or 0.0)
    pre_std  = float(recovery.get("pre_std")  or 0.0)
    post_min = recovery.get("post_min")
    rec_date = recovery.get("recovery_date")
    rec_pct  = recovery.get("recovery_pct", 0.0)
    status   = recovery.get("recovery_status", "")

    fig, ax = plt.subplots(figsize=(13, 5))

    # Observations
    ax.scatter(dates, values, color="#2980b9", s=18, zorder=3, label=f"Observed {index}")

    # Pre-event baseline band
    ax.axhspan(pre_mean - pre_std, pre_mean + pre_std,
               color="green", alpha=0.12, label="Pre-event baseline ±1σ")
    ax.axhline(pre_mean, color="darkgreen", linewidth=1.5, linestyle="--",
               label=f"Pre-event mean ({pre_mean:.3f})")

    # Post-event minimum
    if post_min is not None:
        ax.axhline(float(post_min), color="#c0392b", linewidth=1.2, linestyle=":",
                   label=f"Post-event min ({float(post_min):.3f})")

    # Event date
    ax.axvline(pd.Timestamp(event_date), color="crimson", linewidth=1.8,
               linestyle="--", label=f"Event: {event_date}", zorder=4)

    # Recovery date
    if rec_date is not None:
        ax.axvline(pd.Timestamp(rec_date), color="darkgreen", linewidth=1.8,
                   linestyle="--", label=f"Recovery: {rec_date}", zorder=4)

    # Hurricane event markers (dark red, named)
    _draw_hurricane_markers(ax, hurricane_events)

    title = f"{index} Recovery Trajectory"
    if status:
        title += f"  [{status.replace('_', ' ').title()}"
        if rec_pct:
            title += f" — {float(rec_pct):.1f}% recovered"
        title += "]"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(index, fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(alpha=0.25)

    _fmt_monthly_axis(ax, interval=3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Recovery trajectory plot saved → %s", output_path)
    return output_path


def plot_multi_point_comparison(
    point_data: Dict[str, pd.DataFrame],
    index: str,
    output_path: str,
    event_date: Optional[str] = None,
    hurricane_events: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Plot overlaid vegetation index time series for multiple point locations.

    Parameters
    ----------
    point_data : dict
        Mapping of location label → DataFrame with ``date`` and
        ``index_value`` columns.  Output of
        :func:`src.time_series.extract_multi_point_time_series`.
    index : str
        Index name for axis labels.
    output_path : str
        Full path for the output PNG.
    event_date : str, optional
        YYYY-MM-DD event date for a vertical annotation line.
    hurricane_events : list of dict, optional
        Hurricane catalog entries; see :func:`_draw_hurricane_markers`.

    Returns
    -------
    str
        Path of the saved PNG.
    """
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (label, df_pt) in enumerate(point_data.items()):
        color = cmap(i % 10)
        dates = pd.to_datetime(df_pt["date"])
        vals  = df_pt["index_value"].values
        ax.plot(dates, vals, color=color, linewidth=1.4, label=label)
        ax.scatter(dates, vals, color=color, s=12, zorder=3)

    if event_date:
        ax.axvline(pd.Timestamp(event_date), color="crimson", linewidth=1.8,
                   linestyle="--", label=f"Event: {event_date}", zorder=4)

    # Hurricane event markers (dark red, named)
    _draw_hurricane_markers(ax, hurricane_events)

    ax.set_title(f"{index} — Multi-Point Comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel(index, fontsize=10)
    ax.legend(fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.85)
    ax.grid(alpha=0.25)

    _fmt_monthly_axis(ax, interval=6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Multi-point comparison plot saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def _fig_to_base64(path: str) -> str:
    """Read a PNG file and return its base64-encoded string."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def generate_report(
    results: Dict[str, Any],
    output_dir: str,
    dist_plot_path: str = "",
    ts_plot_path: str = "",
    map_html_path: str = "",
    fmt: str = "html",
) -> str:
    """
    Generate a summary Markdown or HTML report embedding all figures and
    statistical conclusions.

    Parameters
    ----------
    results : dict
        Full results dict from :func:`analysis.run_analysis`.
    output_dir : str
        Directory in which to write the report.
    dist_plot_path : str
        Path to the distribution PNG (embedded as base64).
    ts_plot_path : str
        Path to the time series PNG (embedded as base64).
    map_html_path : str
        Path to the interactive HTML map (linked, not embedded).
    fmt : str
        Output format: ``"html"`` (default) or ``"markdown"``.

    Returns
    -------
    str
        Path of the generated report file.
    """
    if fmt == "markdown":
        return _generate_plain_report(results, output_dir)

    try:
        from jinja2 import Environment
    except ImportError:
        logger.warning("jinja2 not installed; falling back to plain-text report.")
        return _generate_plain_report(results, output_dir)

    stat = results.get("statistics", {})
    baseline = results.get("baseline", {})
    event_date = results.get("event_date", "Unknown")
    satellite = results.get("satellite", "Unknown")
    index = results.get("index", "Unknown")

    pre_window = ""
    post_window = ""
    try:
        from .utils import date_windows
        cfg = results.get("config", {})
        w = cfg.get("windows", {})
        pre_start, pre_end, post_start, post_end = date_windows(
            event_date,
            pre_days=w.get("pre_days", 60),
            post_days=w.get("post_days", 60),
            buffer_days=w.get("buffer_days", 5),
        )
        pre_window = f"{pre_start} → {pre_end}"
        post_window = f"{post_start} → {post_end}"
    except Exception:
        pass

    satellite_full_map = {
        "sentinel2": "Sentinel-2 SR Harmonized (COPERNICUS/S2_SR_HARMONIZED)",
        "landsat": "Landsat 8/9 Collection 2 Level-2",
    }

    context = {
        "event_date": event_date,
        "satellite": satellite,
        "satellite_full": satellite_full_map.get(satellite, satellite),
        "index": index,
        "pre_window": pre_window or "N/A",
        "post_window": post_window or "N/A",
        "n_pixels": stat.get("n", "N/A"),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pre_mean": stat.get("pre_mean", 0.0),
        "post_mean": stat.get("post_mean", 0.0),
        "delta_mean": stat.get("delta_mean", 0.0),
        "delta_pct": stat.get("delta_pct"),
        "cohens_d": stat.get("cohens_d", 0.0),
        "effect_label": stat.get("effect_label", "N/A"),
        "wilcoxon_pvalue": stat.get("wilcoxon_pvalue", 1.0),
        "wilcoxon_stat": stat.get("wilcoxon_stat", 0.0),
        "ttest_stat": stat.get("ttest_stat", 0.0),
        "ttest_pvalue": stat.get("ttest_pvalue", 1.0),
        "ci_low": stat.get("ttest_ci", (0.0, 0.0))[0],
        "ci_high": stat.get("ttest_ci", (0.0, 0.0))[1],
        "significant": stat.get("significant", False),
        "conclusion": stat.get("conclusion", ""),
        "area_by_class": results.get("area_by_class", {}),
        "baseline_interp": baseline.get("interpretation", ""),
        "dist_fig_b64": _fig_to_base64(dist_plot_path),
        "ts_fig_b64": _fig_to_base64(ts_plot_path),
        "map_html_path": os.path.basename(map_html_path) if map_html_path else "",
    }

    env = Environment()
    rendered = env.from_string(_REPORT_TEMPLATE).render(**context)

    out_path = str(Path(output_dir) / "impact_report.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(rendered)

    logger.info("HTML report saved → %s", out_path)
    return out_path


def _generate_plain_report(results: Dict[str, Any], output_dir: str) -> str:
    """Fallback plain-text report when Jinja2 is unavailable."""
    stat = results.get("statistics", {})
    lines = [
        "# Hurricane Vegetation Impact Analysis Report",
        f"Event Date : {results.get('event_date', 'N/A')}",
        f"Satellite  : {results.get('satellite', 'N/A')}",
        f"Index      : {results.get('index', 'N/A')}",
        "",
        "## Statistical Results",
        stat.get("conclusion", "No results available."),
        "",
        "## Impact Area by Class",
    ]
    for cls, area in results.get("area_by_class", {}).items():
        lines.append(f"  {cls}: {area:.2f} km²")

    baseline = results.get("baseline", {})
    if baseline.get("interpretation"):
        lines += ["", "## Baseline Variability", baseline["interpretation"]]

    out_path = str(Path(output_dir) / "impact_report.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("Markdown report saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Multi-sensor maps (SAR + concordance)
# ---------------------------------------------------------------------------

def create_sar_change_map(
    pre_sar: "ee.Image",
    post_sar: "ee.Image",
    diff_sar: "ee.Image",
    roi: "ee.Geometry",
    output_path: str,
) -> str:
    """
    Generate an interactive folium map showing Sentinel-1 SAR change layers.

    Layers (toggle via LayerControl):
    - Pre-event RVI
    - Post-event RVI
    - ∆RVI (diverging palette: brown = loss, teal = gain)
    - ∆VV dB
    - ∆VH dB

    Parameters
    ----------
    pre_sar, post_sar : ee.Image
        SAR composites with bands ``VV``, ``VH``, ``RVI``.
    diff_sar : ee.Image
        SAR change image with bands ``VV_delta``, ``VH_delta``, ``RVI_delta``.
    roi : ee.Geometry
        Region of interest (for centering and boundary).
    output_path : str
        Full path for the output HTML file.

    Returns
    -------
    str
        Path of the saved HTML file, or empty string on failure.
    """
    try:
        import folium
    except ImportError:
        logger.warning("folium not available. Skipping SAR map generation.")
        return ""

    try:
        centroid = roi.centroid(maxError=1).getInfo()["coordinates"]
        centre_lon, centre_lat = centroid
    except Exception:
        centre_lat, centre_lon = 27.5, -81.5

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=10,
                   tiles="CartoDB positron")

    rvi_vis  = {"min": 0, "max": 0.5, "palette": ["white", "darkgreen"]}
    drvi_vis = {"min": -0.2, "max": 0.2,
                "palette": ["8B4513", "D2691E", "FFFACD", "48D1CC", "008080"]}
    ddb_vis  = {"min": -3, "max": 1,
                "palette": ["8B0000", "FF6347", "FFFACD", "90EE90", "006400"]}

    def _add(image, vis, name, shown=True):
        try:
            map_id   = image.getMapId(vis)
            tile_url = map_id["tile_fetcher"].url_format
            folium.TileLayer(
                tiles=tile_url, attr="Google Earth Engine",
                name=name, overlay=True, control=True, show=shown,
            ).add_to(m)
        except Exception as exc:
            logger.warning("Could not add EE layer '%s': %s", name, exc)

    _add(pre_sar.select("RVI"),       rvi_vis,  "Pre-event RVI",   shown=False)
    _add(post_sar.select("RVI"),      rvi_vis,  "Post-event RVI",  shown=False)
    _add(diff_sar.select("RVI_delta"), drvi_vis, "∆RVI (post−pre)", shown=True)
    _add(diff_sar.select("VV_delta"),  ddb_vis,  "∆VV dB",          shown=False)
    _add(diff_sar.select("VH_delta"),  ddb_vis,  "∆VH dB",          shown=False)

    try:
        folium.GeoJson(
            roi.getInfo(), name="ROI Boundary",
            style_function=lambda _: {"color": "black", "fillOpacity": 0, "weight": 2},
        ).add_to(m)
    except Exception:
        pass

    folium.LayerControl().add_to(m)

    legend_html = """
    <div style="
        position:fixed; bottom:36px; left:36px; z-index:9999;
        background:rgba(255,255,255,0.93); padding:12px 16px;
        border-radius:8px; border:1px solid #bbb;
        font-size:12px; font-family:Arial,sans-serif;
        box-shadow:2px 2px 6px rgba(0,0,0,.2); min-width:210px;">
      <div style="font-weight:700; margin-bottom:6px; font-size:13px;">∆RVI (SAR change)</div>
      <div style="background:linear-gradient(to right,#8B4513,#D2691E,#FFFACD,#48D1CC,#008080);
                  width:178px; height:14px; border:1px solid #aaa; border-radius:2px;
                  margin-bottom:2px;"></div>
      <div style="display:flex; justify-content:space-between; width:178px;
                  color:#555; font-size:11px;">
        <span>−0.2 (loss)</span><span>0</span><span>+0.2 (gain)</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    logger.info("SAR change map saved → %s", output_path)
    return output_path


def create_concordance_map(
    concordance_img: "ee.Image",
    diff_img: "ee.Image",
    roi: "ee.Geometry",
    output_path: str,
    event_date: str = "",
    index: str = "NDVI",
) -> str:
    """
    Generate an interactive folium map showing the multi-sensor concordance.

    Classes
    -------
    0 : No Change              (light grey)
    1 : Vegetation Stress Only (yellow)   — optical loss, SAR stable
    2 : Structural Damage Only (orange)   — SAR loss, optical stable
    3 : High-Confidence Damage (crimson)  — both signals

    Parameters
    ----------
    concordance_img : ee.Image
        Single-band classification image (0–3).
    diff_img : ee.Image
        Optical difference image (for optional overlay).
    roi : ee.Geometry
        Region of interest.
    output_path : str
        Full path for the output HTML file.
    event_date : str
        Used in map title only.
    index : str
        Optical index name for labels.

    Returns
    -------
    str
        Path of the saved HTML file, or empty string on failure.
    """
    try:
        import folium
    except ImportError:
        logger.warning("folium not available. Skipping concordance map.")
        return ""

    try:
        centroid = roi.centroid(maxError=1).getInfo()["coordinates"]
        centre_lon, centre_lat = centroid
    except Exception:
        centre_lat, centre_lon = 27.5, -81.5

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=10,
                   tiles="CartoDB positron")

    conc_vis = {
        "min": 0, "max": 3,
        "palette": ["D3D3D3", "FFD700", "FF8C00", "DC143C"],
    }
    diff_vis = {
        "min": -0.5, "max": 0.5,
        "palette": ["d73027", "f46d43", "fdae61", "ffffbf", "a6d96a", "1a9641"],
    }

    def _add(image, vis, name, shown=True):
        try:
            map_id   = image.getMapId(vis)
            tile_url = map_id["tile_fetcher"].url_format
            folium.TileLayer(
                tiles=tile_url, attr="Google Earth Engine",
                name=name, overlay=True, control=True, show=shown,
            ).add_to(m)
        except Exception as exc:
            logger.warning("Could not add EE layer '%s': %s", name, exc)

    _add(concordance_img, conc_vis,
         "Multi-Sensor Concordance", shown=True)
    _add(diff_img.select("delta"), diff_vis,
         f"Δ{index} (optical)", shown=False)

    try:
        folium.GeoJson(
            roi.getInfo(), name="ROI Boundary",
            style_function=lambda _: {"color": "black", "fillOpacity": 0, "weight": 2},
        ).add_to(m)
    except Exception:
        pass

    folium.LayerControl().add_to(m)

    title = f"Multi-Sensor Concordance — {event_date}" if event_date else "Multi-Sensor Concordance"
    legend_html = f"""
    <div style="
        position:fixed; bottom:36px; left:36px; z-index:9999;
        background:rgba(255,255,255,0.93); padding:12px 16px;
        border-radius:8px; border:1px solid #bbb;
        font-size:12px; font-family:Arial,sans-serif;
        box-shadow:2px 2px 6px rgba(0,0,0,.2); min-width:230px;">
      <div style="font-weight:700; margin-bottom:8px; font-size:13px;">{title}</div>
      <div style="display:flex;align-items:center;margin-bottom:4px;">
        <span style="background:#D3D3D3;width:14px;height:14px;display:inline-block;
                     margin-right:7px;border:1px solid #aaa;"></span>
        No Change
      </div>
      <div style="display:flex;align-items:center;margin-bottom:4px;">
        <span style="background:#FFD700;width:14px;height:14px;display:inline-block;
                     margin-right:7px;border:1px solid #aaa;"></span>
        Vegetation Stress (optical loss, SAR stable)
      </div>
      <div style="display:flex;align-items:center;margin-bottom:4px;">
        <span style="background:#FF8C00;width:14px;height:14px;display:inline-block;
                     margin-right:7px;border:1px solid #aaa;"></span>
        Structural Damage (SAR loss, optical stable)
      </div>
      <div style="display:flex;align-items:center;">
        <span style="background:#DC143C;width:14px;height:14px;display:inline-block;
                     margin-right:7px;border:1px solid #aaa;"></span>
        High-Confidence Damage (both signals)
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    logger.info("Concordance map saved → %s", output_path)
    return output_path


def create_palsar_change_map(
    pre_palsar: "ee.Image",
    post_palsar: "ee.Image",
    diff_palsar: "ee.Image",
    roi: "ee.Geometry",
    output_path: str,
    event_date: str = "",
) -> str:
    """
    Generate an interactive folium map showing ALOS PALSAR-2 L-band HV change.

    Layers
    ------
    ΔHV (dB)   — sequential red colourmap; negative = backscatter decrease = damage
    Pre HV dB  — greyscale reference layer (hidden by default)
    Post HV dB — greyscale reference layer (hidden by default)

    Parameters
    ----------
    pre_palsar, post_palsar : ee.Image
        Calibrated PALSAR images (bands HH_dB, HV_dB).
    diff_palsar : ee.Image
        Output of ``compute_palsar_change()`` (bands HV_delta, HH_delta, …).
    roi : ee.Geometry
        Region of interest.
    output_path : str
        Full path for the output HTML file.
    event_date : str
        Used in map title only.

    Returns
    -------
    str
        Path of the saved HTML file, or empty string on failure.
    """
    try:
        import folium
    except ImportError:
        logger.warning("folium not available. Skipping PALSAR change map.")
        return ""

    try:
        centroid = roi.centroid(maxError=1).getInfo()["coordinates"]
        centre_lon, centre_lat = centroid
    except Exception:
        centre_lat, centre_lon = 27.5, -81.5

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=10,
                   tiles="CartoDB positron")

    hv_diff_vis = {
        "min": -6, "max": 0,
        "palette": ["FFFFFF", "FEE5D9", "FCAE91", "FB6A4A", "DE2D26", "A50F15"],
    }
    hv_abs_vis = {"min": -25, "max": -5, "palette": ["000000", "FFFFFF"]}

    def _add(image, vis, name, shown=True):
        try:
            map_id   = image.getMapId(vis)
            tile_url = map_id["tile_fetcher"].url_format
            folium.TileLayer(
                tiles=tile_url, attr="Google Earth Engine",
                name=name, overlay=True, control=True, show=shown,
            ).add_to(m)
        except Exception as exc:
            logger.warning("Could not add EE layer '%s': %s", name, exc)

    _add(diff_palsar.select("HV_delta"), hv_diff_vis, "ΔHV dB (L-band)", shown=True)
    _add(pre_palsar.select("HV_dB"),   hv_abs_vis,   "Pre HV dB",        shown=False)
    _add(post_palsar.select("HV_dB"),  hv_abs_vis,   "Post HV dB",       shown=False)

    try:
        folium.GeoJson(
            roi.getInfo(), name="ROI Boundary",
            style_function=lambda _: {"color": "black", "fillOpacity": 0, "weight": 2},
        ).add_to(m)
    except Exception:
        pass

    folium.LayerControl().add_to(m)

    title = f"PALSAR L-band HV Change — {event_date}" if event_date else "PALSAR L-band HV Change"
    legend_html = f"""
    <div style="
        position:fixed; bottom:36px; left:36px; z-index:9999;
        background:rgba(255,255,255,0.93); padding:12px 16px;
        border-radius:8px; border:1px solid #bbb;
        font-size:12px; font-family:Arial,sans-serif;
        box-shadow:2px 2px 6px rgba(0,0,0,.2); min-width:240px;">
      <div style="font-weight:700; margin-bottom:8px; font-size:13px;">{title}</div>
      <div style="margin-bottom:4px;">ΔHV dB (negative = backscatter loss)</div>
      <div style="display:flex; align-items:center; margin-bottom:4px;">
        <div style="background:linear-gradient(to right,#FFFFFF,#A50F15);
                    width:100px;height:12px;margin-right:8px;border:1px solid #ccc;"></div>
        <span>0 → −6 dB</span>
      </div>
      <div style="margin-top:8px; font-size:11px; color:#555;">
        L-band (~24 cm) sensitive to trunk and primary branch structure.<br>
        Deep red = severe structural loss (≤ −4 dB).
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    logger.info("PALSAR change map saved → %s", output_path)
    return output_path


def create_extended_concordance_map(
    concordance_ext: "ee.Image",
    roi: "ee.Geometry",
    output_path: str,
    event_date: str = "",
    index: str = "NDVI",
) -> str:
    """
    Generate an interactive folium map for the 8-class extended concordance.

    Classes (bitmask: bit0=optical, bit1=C-band, bit2=L-band)
    ----------------------------------------------------------
    0  No Change            (light grey)
    1  Foliar Loss Only     (yellow-green)
    2  C-band Only          (light orange)
    3  Optical + C-band     (orange)
    4  L-band Only          (sky blue)  — trunk damage beneath intact canopy
    5  Optical + L-band     (purple)
    6  C-band + L-band      (deep blue) — dead standing timber
    7  Full Concordance     (crimson)   — all three sensors

    Parameters
    ----------
    concordance_ext : ee.Image
        Single-band image (0–7) named ``concordance_ext``.
    roi : ee.Geometry
        Region of interest.
    output_path : str
        Full path for the output HTML file.
    event_date, index : str
        Used in labels only.

    Returns
    -------
    str
        Path of the saved HTML file, or empty string on failure.
    """
    try:
        import folium
    except ImportError:
        logger.warning("folium not available. Skipping extended concordance map.")
        return ""

    try:
        centroid = roi.centroid(maxError=1).getInfo()["coordinates"]
        centre_lon, centre_lat = centroid
    except Exception:
        centre_lat, centre_lon = 27.5, -81.5

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=10,
                   tiles="CartoDB positron")

    # 8 colours for classes 0-7
    palette = ["D3D3D3", "ADFF2F", "FFD580", "FF8C00", "87CEEB", "9370DB", "1E3A8A", "DC143C"]
    vis = {"min": 0, "max": 7, "palette": palette}

    try:
        map_id   = concordance_ext.getMapId(vis)
        tile_url = map_id["tile_fetcher"].url_format
        folium.TileLayer(
            tiles=tile_url, attr="Google Earth Engine",
            name="Extended Concordance (8-class)",
            overlay=True, control=True, show=True,
        ).add_to(m)
    except Exception as exc:
        logger.warning("Could not add extended concordance EE layer: %s", exc)

    try:
        folium.GeoJson(
            roi.getInfo(), name="ROI Boundary",
            style_function=lambda _: {"color": "black", "fillOpacity": 0, "weight": 2},
        ).add_to(m)
    except Exception:
        pass

    folium.LayerControl().add_to(m)

    title = f"Extended Concordance — {event_date}" if event_date else "Extended Concordance"
    _legend_items = [
        ("#D3D3D3", "0 – No Change"),
        ("#ADFF2F", "1 – Foliar Loss Only (optical↓)"),
        ("#FFD580", "2 – C-band Only (surface roughness)"),
        ("#FF8C00", f"3 – Optical + C-band (leaf + small-branch)"),
        ("#87CEEB", "4 – L-band Only (trunk beneath intact canopy)"),
        ("#9370DB", "5 – Optical + L-band"),
        ("#1E3A8A", "6 – C-band + L-band (dead standing timber)"),
        ("#DC143C", "7 – Full Concordance (all sensors)"),
    ]
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:3px;">'
        f'<span style="background:{c};width:14px;height:14px;display:inline-block;'
        f'margin-right:7px;border:1px solid #aaa;flex-shrink:0;"></span>'
        f'{lbl}</div>'
        for c, lbl in _legend_items
    )
    legend_html = f"""
    <div style="
        position:fixed; bottom:36px; left:36px; z-index:9999;
        background:rgba(255,255,255,0.93); padding:12px 16px;
        border-radius:8px; border:1px solid #bbb;
        font-size:12px; font-family:Arial,sans-serif;
        box-shadow:2px 2px 6px rgba(0,0,0,.2); min-width:300px;">
      <div style="font-weight:700; margin-bottom:8px; font-size:13px;">{title}</div>
      <div style="margin-bottom:6px; color:#555; font-size:11px;">
        Optical ({index}) · C-band (Sentinel-1) · L-band (PALSAR-2)
      </div>
      {rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    logger.info("Extended concordance map saved → %s", output_path)
    return output_path
