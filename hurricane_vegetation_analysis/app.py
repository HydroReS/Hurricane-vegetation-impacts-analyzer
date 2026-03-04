"""
app.py
======
Streamlit dashboard for the Hurricane Vegetation Impact Analysis tool.

Features
--------
- Interactive map to draw or upload an ROI (via streamlit-folium).
- Hurricane preset selector (Ian, Idalia, Milton, Michael, Irma) or custom date.
- Vegetation index and satellite selector.
- "Analyze" button that triggers the full GEE pipeline.
- Tabbed results: interactive difference map, statistics, classification, downloads.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

_nullcontext = contextlib.nullcontext

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hurricane Vegetation Impact Analyzer",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

HURRICANE_PRESETS = {
    "Custom Date / ROI": None,
    # ── 2004 ──────────────────────────────────────────────────────────────────
    "Hurricane Frances (2004-09-05) — Stuart / Treasure Coast": {
        "date": "2004-09-05",
        "bbox": [-80.5, 27.0, -80.0, 27.5],
        "description": "Cat 2 at landfall near Hutchinson Island. Second major storm of the 2004 season.",
    },
    "Hurricane Jeanne (2004-09-26) — Stuart / Treasure Coast": {
        "date": "2004-09-26",
        "bbox": [-80.5, 27.0, -80.0, 27.5],
        "description": "Cat 3 at landfall — nearly identical track to Frances three weeks earlier.",
    },
    # ── 2016 ──────────────────────────────────────────────────────────────────
    "Hurricane Hermine (2016-09-02) — St. Marks / Big Bend": {
        "date": "2016-09-02",
        "bbox": [-84.5, 29.8, -84.0, 30.3],
        "description": "Cat 1 at landfall near St. Marks. First FL landfall since Wilma (2005).",
    },
    # ── 2017 ──────────────────────────────────────────────────────────────────
    "Hurricane Irma (2017-09-10) — Florida Keys": {
        "date": "2017-09-10",
        "bbox": [-81.5, 24.5, -80.3, 25.5],
        "description": "Cat 4 at landfall. Expected: mangrove and coastal vegetation damage.",
    },
    # ── 2018 ──────────────────────────────────────────────────────────────────
    "Hurricane Michael (2018-10-10) — Mexico Beach": {
        "date": "2018-10-10",
        "bbox": [-85.8, 29.9, -85.3, 30.3],
        "description": "Cat 5 at landfall. Expected: near-total canopy destruction.",
    },
    # ── 2019 ──────────────────────────────────────────────────────────────────
    "Hurricane Dorian (2019-09-03) — NE Florida coast": {
        "date": "2019-09-03",
        "bbox": [-81.6, 30.3, -81.1, 30.7],
        "description": "Cat 5 in Bahamas; grazed NE FL barrier islands with surge and wind damage.",
    },
    # ── 2022 ──────────────────────────────────────────────────────────────────
    "Hurricane Ian (2022-09-28) — Fort Myers": {
        "date": "2022-09-28",
        "bbox": [-82.2, 26.4, -81.7, 26.8],
        "description": "Cat 4 at landfall on SW Florida coast. Expected: severe coastal surge damage.",
    },
    "Hurricane Nicole (2022-11-10) — Vero Beach": {
        "date": "2022-11-10",
        "bbox": [-80.7, 27.4, -80.2, 27.8],
        "description": "Cat 1 at landfall. Rare November hurricane causing coastal erosion and surge.",
    },
    # ── 2023 ──────────────────────────────────────────────────────────────────
    "Hurricane Idalia (2023-08-30) — Cedar Key": {
        "date": "2023-08-30",
        "bbox": [-83.3, 29.0, -82.8, 29.5],
        "description": "Cat 3 at landfall. Expected: surge impacts in coastal marshes.",
    },
    # ── 2024 ──────────────────────────────────────────────────────────────────
    "Hurricane Debby (2024-08-05) — Steinhatchee / Big Bend": {
        "date": "2024-08-05",
        "bbox": [-83.8, 29.5, -83.3, 30.0],
        "description": "Cat 1 at landfall. Slow-moving; extreme rainfall and surge in Big Bend region.",
    },
    "Hurricane Milton (2024-10-09) — Sarasota": {
        "date": "2024-10-09",
        "bbox": [-82.5, 27.8, -81.8, 28.4],
        "description": "Cat 3 at landfall. Rapid intensification storm on the central Gulf coast.",
    },
    # ── Control ───────────────────────────────────────────────────────────────
    "Control — Orlando (no storm, Ian date)": {
        "date": "2022-09-28",
        "bbox": [-81.6, 28.3, -81.2, 28.7],
        "description": "Same date as Ian but inland. Expected: no significant change.",
    },
}

INDEX_INFO = {
    "NDVI": "General vegetation health — Normalized Difference Vegetation Index",
    "EVI": "Dense canopy — Enhanced Vegetation Index (atmospheric correction)",
    "SAVI": "Sparse vegetation — Soil-Adjusted Vegetation Index",
    "NDMI": "Moisture stress — Normalized Difference Moisture Index",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _fig_to_bytes(fig) -> bytes:
    """Convert a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _cleanup_temp_dir(path: Optional[str]) -> None:
    """Remove a temp directory left over from a previous analysis run."""
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _load_config() -> Dict[str, Any]:
    """Load config.yaml with a fallback empty dict."""
    try:
        from src.utils import load_config
        return load_config(_CONFIG_PATH)
    except Exception as exc:
        st.warning(f"Could not load config.yaml: {exc}")
        return {}


@st.cache_resource(show_spinner="Initializing Google Earth Engine …")
def _init_gee(project: str) -> bool:
    """Initialize GEE (cached so it only runs once per session)."""
    try:
        from src.utils import ee_init
        ee_init(project=project)
        return True
    except Exception as exc:
        st.error(f"GEE initialization failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> Dict[str, Any]:
    """Render the sidebar controls and return the collected parameters."""
    st.sidebar.title("🌪️ Hurricane Vegetation Analyzer")
    st.sidebar.markdown("---")

    # GEE project
    cfg = _load_config()
    default_project = cfg.get("gee", {}).get("project", "vegetation-impact-analysis")
    gee_project = st.sidebar.text_input("GEE Project ID", value=default_project)

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Select Event")

    preset_name = st.sidebar.selectbox(
        "Hurricane Preset",
        options=list(HURRICANE_PRESETS.keys()),
        index=1,  # default to Ian
    )

    preset = HURRICANE_PRESETS[preset_name]

    if preset is None:
        # Custom input
        event_date = st.sidebar.text_input("Event Date (YYYY-MM-DD)", value="2022-09-28")
        st.sidebar.markdown("**Bounding Box** [W, S, E, N]")
        col1, col2 = st.sidebar.columns(2)
        west = col1.number_input("West", value=-82.2, format="%.4f")
        south = col2.number_input("South", value=26.4, format="%.4f")
        east = col1.number_input("East", value=-81.7, format="%.4f")
        north = col2.number_input("North", value=26.8, format="%.4f")
        bbox = [west, south, east, north]
        description = "Custom ROI"
    else:
        event_date = preset["date"]
        bbox = preset["bbox"]
        description = preset["description"]
        st.sidebar.info(f"📅 **{event_date}**\n\n{description}")
        # Allow ROI customisation without switching to "Custom" mode
        with st.sidebar.expander("✏️ Customize ROI (optional)"):
            st.caption(
                f"Default: [{', '.join(str(v) for v in bbox)}]\n\n"
                "Edit below to narrow or shift the analysis area."
            )
            # Keys include preset_name so inputs reset when a different preset is selected
            west  = st.number_input("West",  value=float(bbox[0]), format="%.3f",
                                    key=f"roi_w_{preset_name}")
            south = st.number_input("South", value=float(bbox[1]), format="%.3f",
                                    key=f"roi_s_{preset_name}")
            east  = st.number_input("East",  value=float(bbox[2]), format="%.3f",
                                    key=f"roi_e_{preset_name}")
            north = st.number_input("North", value=float(bbox[3]), format="%.3f",
                                    key=f"roi_n_{preset_name}")
            bbox = [west, south, east, north]

    # GeoJSON / Shapefile upload (optional)
    uploaded_file = st.sidebar.file_uploader(
        "Upload ROI (GeoJSON / Shapefile) — overrides bbox",
        type=["geojson", "json", "zip"],
        help=(
            "Upload a GeoJSON file **or** a zipped shapefile (.zip containing "
            ".shp/.dbf/.shx) to use as the region of interest."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Analysis Parameters")

    index = st.sidebar.selectbox(
        "Vegetation Index",
        options=list(INDEX_INFO.keys()),
        index=0,
        help="\n".join(f"**{k}**: {v}" for k, v in INDEX_INFO.items()),
    )
    st.sidebar.caption(INDEX_INFO[index])

    satellite = st.sidebar.selectbox(
        "Satellite",
        options=["sentinel2", "landsat"],
        format_func=lambda x: "Sentinel-2 (10 m)" if x == "sentinel2" else "Landsat 8/9 (30 m)",
    )

    _SENSOR_LABELS = [
        "Optical only",
        "Optical + SAR (Sentinel-1)",
        "Optical + SAR + GEDI lidar",
    ]
    _SENSOR_VALUES = {
        "Optical only":               "optical",
        "Optical + SAR (Sentinel-1)": "optical,sar",
        "Optical + SAR + GEDI lidar": "optical,sar,gedi",
    }
    sensors_label = st.sidebar.radio(
        "Sensors",
        options=_SENSOR_LABELS,
        index=0,
        help=(
            "**Optical only**: NDVI/EVI/SAVI/NDMI from Sentinel-2 or Landsat.\n\n"
            "**+ SAR**: Adds Sentinel-1 C-band Radar Vegetation Index (RVI) change and "
            "a multi-sensor concordance map.\n\n"
            "**+ GEDI**: Adds GEDI lidar canopy height and cover change "
            "(available for events April 2019 – March 2023 only)."
        ),
    )
    sensors_val = _SENSOR_VALUES[sensors_label]

    use_palsar = st.sidebar.checkbox(
        "🛰️ Add PALSAR-2 L-band SAR",
        value=False,
        help=(
            "Adds ALOS PALSAR-2 L-band (~24 cm wavelength) annual mosaic analysis. "
            "L-band penetrates foliage to detect trunk and primary-branch damage — "
            "the most reliable radar indicator of structural biomass loss. "
            "Available from 2014 onwards (JAXA PALSAR-2). "
            "When combined with Sentinel-1 C-band, generates an 8-class extended "
            "concordance map (optical + C-band + L-band)."
        ),
    )
    if use_palsar:
        sensors_val = sensors_val + ",palsar"
        with st.sidebar.expander("PALSAR year overrides (optional)", expanded=False):
            st.caption(
                "PALSAR uses annual mosaics. Years are auto-selected from the event "
                "date (month ≥ 7 → same year / next year). Override if needed."
            )
            palsar_pre_year_val: Optional[int] = None
            palsar_post_year_val: Optional[int] = None
            _pre_override  = st.number_input("Pre-event mosaic year",  min_value=2007, max_value=2030, value=None, placeholder="auto")
            _post_override = st.number_input("Post-event mosaic year", min_value=2007, max_value=2030, value=None, placeholder="auto")
            if _pre_override:
                palsar_pre_year_val  = int(_pre_override)
            if _post_override:
                palsar_post_year_val = int(_post_override)
    else:
        palsar_pre_year_val  = None
        palsar_post_year_val = None

    mask_water = st.sidebar.checkbox(
        "Mask ocean / water bodies",
        value=cfg.get("processing", {}).get("mask_water", False),
        help=(
            "Excludes permanent water pixels using the JRC Global Surface Water dataset. "
            "Recommended for coastal ROIs to avoid skewing statistics."
        ),
    )
    mask_water_threshold = cfg.get("processing", {}).get("mask_water_threshold", 80)
    if mask_water:
        mask_water_threshold = st.sidebar.slider(
            "Water occurrence threshold (%)",
            min_value=50, max_value=99, value=int(mask_water_threshold), step=5,
            help=(
                "Pixels with JRC water occurrence above this value are masked. "
                "80 (default): removes open ocean and large lakes while preserving "
                "tidal flats and mangroves. "
                "Raise toward 95 to mask only open water; "
                "lower toward 60 for stricter coastal filtering."
            ),
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Window Settings")

    pre_days = st.sidebar.slider("Pre-event days", 30, 120,
                                  cfg.get("windows", {}).get("pre_days", 60), 10)
    post_days = st.sidebar.slider("Post-event days", 30, 120,
                                   cfg.get("windows", {}).get("post_days", 60), 10)
    buffer_days = st.sidebar.slider("Buffer days (around event)", 0, 15,
                                     cfg.get("windows", {}).get("buffer_days", 5), 1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Statistical Settings")

    alpha = st.sidebar.select_slider(
        "Significance level (α)",
        options=[0.01, 0.05, 0.10],
        value=cfg.get("statistics", {}).get("significance_level", 0.05),
    )
    hist_years = st.sidebar.slider(
        "Historical baseline years", 0, 5,
        cfg.get("statistics", {}).get("historical_years", 3),
        help="Prior years used for baseline variability check. Set to 0 to skip.",
    )
    sample_size = st.sidebar.slider(
        "Pixel sample size", 100, 1000,
        cfg.get("statistics", {}).get("sample_size", 500), 100,
    )

    st.sidebar.markdown("---")
    run = st.sidebar.button("🔍 Run Analysis", type="primary", width="stretch")

    return {
        "gee_project": gee_project,
        "event_date": event_date,
        "bbox": bbox,
        "uploaded_file": uploaded_file,
        "index": index,
        "satellite": satellite,
        "sensors": sensors_val,
        "palsar_pre_year":  palsar_pre_year_val,
        "palsar_post_year": palsar_post_year_val,
        "pre_days": pre_days,
        "post_days": post_days,
        "buffer_days": buffer_days,
        "alpha": alpha,
        "hist_years": hist_years,
        "sample_size": sample_size,
        "mask_water": mask_water,
        "mask_water_threshold": mask_water_threshold,
        "run": run,
        "config": cfg,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def _run_analysis(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute the full analysis pipeline and return results."""
    from src.utils import parse_roi
    from src.analysis import run_analysis

    # Build ROI string
    if params["uploaded_file"] is not None:
        uploaded = params["uploaded_file"]
        fname = uploaded.name.lower()
        if fname.endswith(".zip"):
            # Zipped shapefile: extract to a temp dir so geopandas can find sidecar files
            tmp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(tmp_dir, "roi.zip")
            with open(zip_path, "wb") as fh:
                fh.write(uploaded.read())
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            shp_files = [
                os.path.join(root, f)
                for root, _dirs, files in os.walk(tmp_dir)
                for f in files
                if f.lower().endswith(".shp")
            ]
            if not shp_files:
                st.error(
                    "No .shp file found inside the uploaded ZIP archive. "
                    "Please include the .shp, .dbf, and .shx files in the ZIP."
                )
                return None
            roi_str = f"file:{shp_files[0]}"
        else:
            # GeoJSON / JSON
            suffix = Path(uploaded.name).suffix or ".geojson"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.read())
                roi_str = f"file:{tmp.name}"
    else:
        bbox = params["bbox"]
        roi_str = f"bbox:{','.join(str(v) for v in bbox)}"

    try:
        roi_geom = parse_roi(roi_str)
    except Exception as exc:
        st.error(f"Failed to parse ROI: {exc}")
        return None

    # Build config override — merge into sub-dicts to preserve keys from
    # config.yaml that are not explicitly set by the sidebar (e.g. min_sample_size).
    import copy
    cfg = copy.deepcopy(params["config"])
    cfg.setdefault("windows", {}).update({
        "pre_days":    params["pre_days"],
        "post_days":   params["post_days"],
        "buffer_days": params["buffer_days"],
    })
    cfg.setdefault("statistics", {}).update({
        "significance_level": params["alpha"],
        "historical_years":   params["hist_years"],
        "sample_size":        params["sample_size"],
    })
    cfg.setdefault("processing", {}).update({
        "mask_water":           params["mask_water"],
        "mask_water_threshold": params["mask_water_threshold"],
    })

    # Temporary output directory
    out_dir = tempfile.mkdtemp(prefix="hurricane_veg_")

    sensors = params.get("sensors", "optical")
    spinner_msg = f"Running {params['index']} analysis on {params['satellite']} data"
    if sensors != "optical":
        spinner_msg += f" + {sensors.replace('optical,', '').replace(',', ' + ').upper()}"
    spinner_msg += " …"

    with st.spinner(spinner_msg):
        try:
            results = run_analysis(
                roi=roi_geom,
                event_date=params["event_date"],
                satellite=params["satellite"],
                index=params["index"],
                output_dir=out_dir,
                config=cfg,
                sensors=sensors,
                palsar_pre_year=params.get("palsar_pre_year"),
                palsar_post_year=params.get("palsar_post_year"),
            )
            results["config"] = cfg
            results["roi_geom"] = roi_geom
            results["output_dir"] = out_dir
            try:
                results["roi_area_km2"] = roi_geom.area(maxError=100).getInfo() / 1e6
            except Exception:
                results["roi_area_km2"] = None
            return results
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            logging.getLogger("app").exception("Analysis error")
            return None


# ---------------------------------------------------------------------------
# Metadata expander
# ---------------------------------------------------------------------------

def _render_metadata_expander(results: Dict[str, Any]) -> None:
    """Render the Image Acquisition Details expander below the summary banner."""
    sm = results.get("scene_metadata")
    if not sm:
        return

    with st.expander("🛰️ Image Acquisition Details", expanded=False):
        # ── Warnings ─────────────────────────────────────────────────────────
        warnings = sm.get("warnings", [])
        if warnings:
            for w in warnings:
                st.warning(f"⚠️ {w}")

        # ── Summary table ─────────────────────────────────────────────────────
        sensor_configs = [
            ("optical", "Optical (Sentinel-2 / Landsat)"),
            ("sar",     "Sentinel-1 SAR"),
            ("palsar",  "ALOS PALSAR"),
            ("gedi",    "GEDI Lidar"),
        ]
        rows = []
        for key, label in sensor_configs:
            smeta = sm.get(key)
            if not smeta:
                continue
            pre  = smeta.get("pre",  {})
            post = smeta.get("post", {})

            def _count_str(m):
                c = m.get("count")
                return str(c) if c is not None else "—"

            def _range_str(m):
                if key == "palsar":
                    yr = m.get("year")
                    return f"Mosaic {yr}" if yr else "—"
                if key == "gedi":
                    ds = m.get("date_start")
                    de = m.get("date_end")
                    return f"{ds} → {de}" if ds else "—"
                ws = m.get("window_start", "")
                we = m.get("window_end",   "")
                return f"{ws} → {we}" if ws else "—"

            rows.append({
                "Sensor":             label,
                "Pre-Event Scenes":   _count_str(pre),
                "Pre-Event Window":   _range_str(pre),
                "Post-Event Scenes":  _count_str(post),
                "Post-Event Window":  _range_str(post),
            })

        if rows:
            st.subheader("Acquisition Summary")
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        # ── Optical scene details ─────────────────────────────────────────────
        from src.metadata_utils import build_optical_table, build_sar_table

        opt = sm.get("optical", {})
        if opt:
            st.subheader("Optical Scenes")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Pre-Event**")
                df = build_optical_table(opt.get("pre", {}))
                if df is not None:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No scene details available.")
            with c2:
                st.write("**Post-Event**")
                df = build_optical_table(opt.get("post", {}))
                if df is not None:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No scene details available.")

        # ── SAR scene details ─────────────────────────────────────────────────
        sar = sm.get("sar", {})
        if sar:
            st.subheader("Sentinel-1 SAR Scenes")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Pre-Event**")
                df = build_sar_table(sar.get("pre", {}))
                if df is not None:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No scene details available.")
            with c2:
                st.write("**Post-Event**")
                df = build_sar_table(sar.get("post", {}))
                if df is not None:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No scene details available.")

        # ── PALSAR mosaic details ─────────────────────────────────────────────
        palsar = sm.get("palsar", {})
        if palsar:
            st.subheader("ALOS PALSAR Annual Mosaics")
            c1, c2 = st.columns(2)
            for col_ctx, window_key, label in [(c1, "pre", "Pre-Event"), (c2, "post", "Post-Event")]:
                with col_ctx:
                    st.write(f"**{label}**")
                    p = palsar.get(window_key, {})
                    if p:
                        st.write(f"Year: **{p.get('year', '?')}**")
                        st.write(f"Epoch: {p.get('epoch_start', '?')} → {p.get('epoch_end', '?')}")
                        if p.get("event_in_epoch"):
                            st.caption("⚠️ Event date falls within this epoch.")
                    else:
                        st.caption("Not available.")

        # ── GEDI details ──────────────────────────────────────────────────────
        gedi = sm.get("gedi", {})
        if gedi:
            st.subheader("GEDI Monthly Products")
            c1, c2 = st.columns(2)
            for col_ctx, window_key, label in [(c1, "pre", "Pre-Event"), (c2, "post", "Post-Event")]:
                with col_ctx:
                    st.write(f"**{label}**")
                    g = gedi.get(window_key, {})
                    if g:
                        st.write(f"Height granules (GEDI02_A): {g.get('height_months', '?')}")
                        st.write(f"Cover granules (GEDI02_B): {g.get('cover_months', '?')}")
                        ds = g.get("date_start")
                        de = g.get("date_end")
                        if ds:
                            st.write(f"Date range: {ds} → {de}")
                    else:
                        st.caption("Not available.")


# ---------------------------------------------------------------------------
# Result tabs
# ---------------------------------------------------------------------------

def _render_statistics_tab(results: Dict[str, Any]) -> None:
    """Render the Statistics tab."""
    stat = results.get("statistics", {})
    if not stat:
        st.warning("No statistical results available.")
        return

    # Key metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Pre Mean", f"{stat['pre_mean']:.4f}")
    col2.metric("Post Mean", f"{stat['post_mean']:.4f}",
                delta=f"{stat['delta_mean']:+.4f}")
    pct = stat.get("delta_pct")
    col3.metric("% Change", f"{pct:+.1f}%" if pct is not None else "N/A")
    col4.metric("Cohen's d", f"{stat['cohens_d']:.3f}",
                help=f"Effect size: {stat.get('effect_label', 'N/A')}")
    col5.metric("Wilcoxon p", f"{stat['wilcoxon_pvalue']:.4f}",
                help=f"α = {stat.get('alpha', 0.05)}")
    col6.metric("Pixels (n)", f"{stat['n']:,}")

    st.markdown("---")

    # Plain-language conclusion
    if stat.get("significant"):
        st.error(f"**{stat['conclusion']}**")
    else:
        st.success(f"**{stat['conclusion']}**")

    # Distribution plot
    st.subheader("Index Distribution: Pre vs Post")
    try:
        import matplotlib.pyplot as plt
        from src.visualization import plot_distributions
        import tempfile

        pre_vals = results.get("pre_vals", np.array([]))
        post_vals = results.get("post_vals", np.array([]))
        if len(pre_vals) > 0:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plot_distributions(
                    pre_vals, post_vals,
                    index=results["index"],
                    output_path=tmp.name,
                    event_date=results["event_date"],
                )
                st.image(tmp.name, width="stretch")
    except Exception as exc:
        st.warning(f"Distribution plot failed: {exc}")

    # Detailed stats table
    st.subheader("Detailed Test Results")
    import pandas as pd
    ci = stat.get("ttest_ci", (float("nan"), float("nan")))
    detail_df = pd.DataFrame({
        "Test": ["Paired t-test", "Wilcoxon signed-rank"],
        "Statistic": [f"{stat.get('ttest_stat', 0):.4f}", f"{stat.get('wilcoxon_stat', 0):.4f}"],
        "p-value": [f"{stat.get('ttest_pvalue', 1):.4f}", f"{stat.get('wilcoxon_pvalue', 1):.4f}"],
        "Significant?": [
            "✅" if stat.get("ttest_pvalue", 1) < stat.get("alpha", 0.05) else "❌",
            "✅" if stat.get("wilcoxon_pvalue", 1) < stat.get("alpha", 0.05) else "❌",
        ],
    })
    st.table(detail_df)
    st.caption(f"95% CI of mean delta: [{ci[0]:.4f}, {ci[1]:.4f}]  "
               f"| Cohen's d = {stat.get('cohens_d', 0):.3f} "
               f"({stat.get('effect_label', 'N/A')} effect size)")


def _render_map_tab(results: Dict[str, Any]) -> None:
    """Render the Map tab with the interactive difference map."""
    try:
        import streamlit_folium as st_folium
        from src.visualization import create_difference_map
        import tempfile

        out_path = tempfile.mktemp(suffix=".html")
        html_path = create_difference_map(
            pre_img=results["pre_img"],
            post_img=results["post_img"],
            diff_img=results["diff_img"],
            classified_img=results["classified_img"],
            roi=results["roi_geom"],
            index=results["index"],
            output_path=out_path,
            thresholds=results.get("config", {}).get("thresholds"),
        )
        if html_path and os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            # Embed using streamlit components
            import streamlit.components.v1 as components
            components.html(html_content, height=600, scrolling=False)
        else:
            st.warning("Interactive map could not be generated.")
    except ImportError:
        st.info(
            "Install `streamlit-folium` and `geemap` to enable the interactive map. "
            "The difference GeoTIFF is available in the Downloads tab."
        )
    except Exception as exc:
        st.warning(f"Map rendering failed: {exc}")


def _render_classification_tab(results: Dict[str, Any]) -> None:
    """Render the Classification tab with severity area breakdown."""
    area_by_class = results.get("area_by_class", {})
    if not area_by_class:
        st.warning("No classification data available.")
        return

    st.subheader("Impact Area by Severity Class")

    import pandas as pd

    class_colors = {
        "No Impact": "#2ecc71",
        "Low Impact": "#f1c40f",
        "Moderate Impact": "#e67e22",
        "Severe Impact": "#c0392b",
    }
    total = sum(area_by_class.values()) or 1

    # Metrics
    cols = st.columns(len(area_by_class))
    for i, (cls, area_km2) in enumerate(area_by_class.items()):
        pct = area_km2 / total * 100
        cols[i].metric(cls, f"{area_km2:.2f} km²", f"{pct:.1f}%")

    # Pie chart
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        labels = list(area_by_class.keys())
        values = list(area_by_class.values())
        colors = [class_colors.get(l, "#95a5a6") for l in labels]

        ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, pctdistance=0.82)
        ax.set_title("Impact Severity Distribution", fontsize=13, fontweight="bold")

        col1, col2 = st.columns([1, 1])
        col1.pyplot(fig)

        # Table
        df = pd.DataFrame({
            "Class": labels,
            "Area (km²)": [f"{v:.4f}" for v in values],
            "Share (%)": [f"{v/total*100:.1f}%" for v in values],
        })
        col2.dataframe(df, width="stretch", hide_index=True)

    except Exception as exc:
        st.warning(f"Pie chart failed: {exc}")

    # Baseline variability
    baseline = results.get("baseline", {})
    if baseline.get("interpretation"):
        st.markdown("---")
        st.subheader("Baseline Variability Check")
        within = baseline.get("within_normal_range")
        if within is True:
            st.warning(baseline["interpretation"])
        elif within is False:
            st.info(baseline["interpretation"])
        else:
            st.caption(baseline["interpretation"])


def _render_downloads_tab(results: Dict[str, Any]) -> None:
    """Render the Downloads tab."""
    from src.visualization import generate_report, plot_distributions
    import tempfile

    st.subheader("Download Outputs")
    out_dir = results.get("output_dir", tempfile.mkdtemp())

    # Distribution plot PNG
    pre_vals = results.get("pre_vals", np.array([]))
    post_vals = results.get("post_vals", np.array([]))
    dist_path = ""
    if len(pre_vals) > 0:
        dist_path = os.path.join(out_dir, f"{results['index']}_distribution.png")
        try:
            plot_distributions(pre_vals, post_vals, results["index"], dist_path,
                               event_date=results["event_date"])
        except Exception:
            dist_path = ""

    if dist_path and os.path.exists(dist_path):
        with open(dist_path, "rb") as f:
            st.download_button(
                "📊 Download Distribution Plot (PNG)",
                data=f.read(),
                file_name=f"{results['index']}_distribution.png",
                mime="image/png",
            )

    # GeoTIFFs
    idx = results["index"]
    geotiff_specs = [
        (results.get("pre_geotiff_path"),  f"🗺️ Download Pre-event {idx} GeoTIFF",  f"{idx}_pre.tif"),
        (results.get("post_geotiff_path"), f"🗺️ Download Post-event {idx} GeoTIFF", f"{idx}_post.tif"),
        (results.get("geotiff_path"),      f"🗺️ Download Difference GeoTIFF (Δ{idx})", f"{idx}_difference.tif"),
    ]
    any_geotiff = False
    for path, label, fname in geotiff_specs:
        if path and os.path.exists(path):
            any_geotiff = True
            with open(path, "rb") as f:
                st.download_button(label, data=f.read(), file_name=fname, mime="image/tiff")
    if not any_geotiff:
        st.caption("GeoTIFFs not available (geemap export may have failed).")

    # HTML Report
    report_path = ""
    try:
        report_path = generate_report(
            results=results,
            output_dir=out_dir,
            dist_plot_path=dist_path,
        )
    except Exception as exc:
        st.warning(f"Report generation failed: {exc}")

    if report_path and os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                "📄 Download HTML Report",
                data=f.read(),
                file_name="impact_report.html",
                mime="text/html",
            )

    # CSV statistics
    stat = results.get("statistics", {})
    if stat:
        import pandas as pd
        stat_df = pd.DataFrame([{
            "event_date": results.get("event_date"),
            "satellite": results.get("satellite"),
            "index": results.get("index"),
            **{k: v for k, v in stat.items() if k != "ttest_ci" and k != "conclusion"},
            "ttest_ci_low": stat.get("ttest_ci", (None, None))[0],
            "ttest_ci_high": stat.get("ttest_ci", (None, None))[1],
            **{f"area_{cls.replace(' ', '_').lower()}": km2
               for cls, km2 in results.get("area_by_class", {}).items()},
        }])
        csv_bytes = stat_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📋 Download Statistics (CSV)",
            data=csv_bytes,
            file_name="analysis_statistics.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Multi-Sensor tab
# ---------------------------------------------------------------------------

def _render_multisensor_tab(results: Dict[str, Any]) -> None:
    """Render the Multi-Sensor tab (SAR + GEDI + PALSAR + concordance)."""
    structural = results.get("structural", {})
    if not structural:
        st.info("No structural analysis results available. Re-run with SAR or GEDI sensors enabled.")
        return

    event_date    = results.get("event_date", "")
    index         = results.get("index", "NDVI")
    sar_map_path  = ""
    conc_map_path = ""
    palsar_map_path = ""
    ext_conc_path   = ""

    # ── GEDI availability note ─────────────────────────────────────────────
    gedi_info = structural.get("gedi_info", "")
    if gedi_info and not structural.get("gedi_available"):
        st.warning(f"ℹ️ {gedi_info}")

    # ── SAR results ────────────────────────────────────────────────────────
    if structural.get("sar_available"):
        st.subheader("Sentinel-1 C-band SAR Change")
        st.caption(
            "Radar Vegetation Index (RVI = 4·VH / (VV + VH)) is computed "
            "in linear scale before compositing. Negative ∆RVI indicates "
            "vegetation structural loss (fewer scatterers, more specular return)."
        )

        try:
            from src.visualization import create_sar_change_map
            import streamlit.components.v1 as components

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                sar_map_path = tmp.name
            create_sar_change_map(
                structural["pre_sar"], structural["post_sar"],
                structural["diff_sar"],
                results["roi_geom"],
                sar_map_path,
            )
            if os.path.exists(sar_map_path):
                with open(sar_map_path, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=520, scrolling=False)
        except ImportError:
            st.info("Install `streamlit-folium` to view the interactive SAR map.")
        except Exception as exc:
            st.warning(f"SAR map failed: {exc}")

    # ── PALSAR L-band results ─────────────────────────────────────────────
    if structural.get("palsar_available"):
        st.markdown("---")
        st.subheader("ALOS PALSAR-2 L-band SAR Change")
        st.caption(
            "L-band (~24 cm wavelength) penetrates foliage and interacts "
            "directly with trunks and primary branches. Negative ∆HV (dB) "
            "indicates backscatter decrease — loss of woody biomass scatterers."
        )

        # PALSAR stats
        ps = structural.get("palsar_stats", {})
        if ps and "hv_delta_mean" in ps:
            _pre  = ps.get("pre_hv_mean",   "—")
            _post = ps.get("post_hv_mean",  "—")
            _d    = ps.get("hv_delta_mean", "—")
            _p    = ps.get("wilcoxon_p",    "—")
            _cd   = ps.get("cohens_d",      "—")
            _sig  = "Yes ✓" if ps.get("significant") else "No"
            import pandas as pd
            _stats_df = pd.DataFrame([{
                "Pre HV (dB)": f"{_pre:.2f}" if isinstance(_pre, float) else _pre,
                "Post HV (dB)": f"{_post:.2f}" if isinstance(_post, float) else _post,
                "ΔHV mean (dB)": f"{_d:+.2f}" if isinstance(_d, float) else _d,
                "Wilcoxon p": f"{_p:.4f}" if isinstance(_p, float) else _p,
                "Cohen's d": f"{_cd:.3f}" if isinstance(_cd, float) else _cd,
                "Significant": _sig,
            }])
            st.dataframe(_stats_df, hide_index=True, use_container_width=True)

        # Determine which auto years were used for display
        from src.structural_analysis import _determine_palsar_years
        _auto_pre, _auto_post = _determine_palsar_years(event_date)
        st.caption(f"PALSAR mosaics: pre={_auto_pre}, post={_auto_post} (auto-selected from event date).")

        try:
            from src.visualization import create_palsar_change_map
            import streamlit.components.v1 as components

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                palsar_map_path = tmp.name
            create_palsar_change_map(
                structural["pre_palsar"], structural["post_palsar"],
                structural["diff_palsar"],
                results["roi_geom"],
                palsar_map_path,
                event_date=event_date,
            )
            if os.path.exists(palsar_map_path):
                with open(palsar_map_path, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=520, scrolling=False)
        except ImportError:
            st.info("Install `streamlit-folium` to view the interactive PALSAR map.")
        except Exception as exc:
            st.warning(f"PALSAR map failed: {exc}")

        # AGB note
        st.info(
            "**Experimental AGB estimate** (Above-Ground Biomass): "
            "Allometric formula AGB ≈ 10^((HV_dB + 83) × A + B) can be computed "
            "via the CLI. Default coefficients (A=0.04, B=−2.0) are illustrative "
            "placeholders — regional calibration with field plots is required."
        )

    # ── Concordance results ───────────────────────────────────────────────
    # Prefer 8-class extended concordance if PALSAR available
    if structural.get("concordance_ext") is not None:
        st.markdown("---")
        st.subheader("Extended Multi-Sensor Concordance (8-class)")
        st.caption(
            "Bitmask combining optical, C-band (Sentinel-1), and L-band (PALSAR-2) signals:\n\n"
            "- **0** No Change  \n"
            "- **1** Foliar Loss Only (optical↓)  \n"
            "- **2** C-band Only (canopy surface roughness)  \n"
            "- **3** Optical + C-band (leaf + small-branch)  \n"
            "- **4** L-band Only (trunk damage beneath intact canopy)  \n"
            "- **5** Optical + L-band  \n"
            "- **6** C-band + L-band (dead standing timber)  \n"
            "- **7** Full Concordance — all three sensors (highest confidence)"
        )
        try:
            from src.visualization import create_extended_concordance_map
            import streamlit.components.v1 as components

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                ext_conc_path = tmp.name
            create_extended_concordance_map(
                structural["concordance_ext"],
                results["roi_geom"],
                ext_conc_path,
                event_date=event_date,
                index=index,
            )
            if os.path.exists(ext_conc_path):
                with open(ext_conc_path, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=520, scrolling=False)
        except ImportError:
            st.info("Install `streamlit-folium` to view the interactive extended concordance map.")
        except Exception as exc:
            st.warning(f"Extended concordance map failed: {exc}")

    elif structural.get("concordance_img") is not None:
        st.markdown("---")
        st.subheader("Multi-Sensor Concordance Map (4-class)")
        st.caption(
            "Each pixel is classified by the agreement between optical "
            f"({index}) and SAR (RVI) change signals:\n\n"
            "- **No Change** — neither sensor detects impact  \n"
            "- **Vegetation Stress Only** — optical loss, SAR stable (foliar change)  \n"
            "- **Structural Damage Only** — SAR loss, optical stable (physical structure)  \n"
            "- **High-Confidence Damage** — both optical and SAR decline"
        )

        try:
            from src.visualization import create_concordance_map
            import streamlit.components.v1 as components

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                conc_map_path = tmp.name
            create_concordance_map(
                structural["concordance_img"],
                results["diff_img"],
                results["roi_geom"],
                conc_map_path,
                event_date=event_date,
                index=index,
            )
            if os.path.exists(conc_map_path):
                with open(conc_map_path, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=520, scrolling=False)
        except ImportError:
            st.info("Install `streamlit-folium` to view the interactive concordance map.")
        except Exception as exc:
            st.warning(f"Concordance map failed: {exc}")

    # ── GEDI results ───────────────────────────────────────────────────────
    if structural.get("gedi_available"):
        st.markdown("---")
        st.subheader("GEDI Lidar Structural Metrics")
        st.caption(
            "GEDI (Global Ecosystem Dynamics Investigation) lidar measures canopy "
            "structure from the International Space Station. Metrics shown are "
            "post-event minus pre-event composites (negative = loss)."
        )
        st.info(
            "GEDI change layers (∆rh95, ∆rh50, ∆cover) are available as GEE image "
            "objects. To inspect pixel values, use the GEE Code Editor or export "
            "via the CLI (`python cli.py analyze --sensors optical,sar,gedi`)."
        )

    # ── Errors ─────────────────────────────────────────────────────────────
    if not structural.get("sar_available") and "sar_error" in structural:
        st.error(f"SAR analysis error: {structural['sar_error']}")
    if not structural.get("gedi_available") and "gedi_error" in structural:
        st.error(f"GEDI analysis error: {structural['gedi_error']}")
    if not structural.get("palsar_available") and "palsar_error" in structural:
        st.error(f"PALSAR analysis error: {structural['palsar_error']}")

    # ── Downloads ──────────────────────────────────────────────────────────
    _has_downloads = any([
        structural.get("sar_available"),
        structural.get("palsar_available"),
        structural.get("concordance_img") is not None,
        structural.get("concordance_ext") is not None,
    ])
    if _has_downloads:
        st.markdown("---")
        st.subheader("Downloads")
        col1, col2, col3, col4 = st.columns(4)

        if sar_map_path and os.path.exists(sar_map_path):
            with open(sar_map_path, "rb") as f:
                col1.download_button(
                    "📊 SAR Change Map",
                    data=f.read(),
                    file_name="sar_change_map.html",
                    mime="text/html",
                )
        if conc_map_path and os.path.exists(conc_map_path):
            with open(conc_map_path, "rb") as f:
                col2.download_button(
                    "📊 Concordance Map",
                    data=f.read(),
                    file_name="concordance_map.html",
                    mime="text/html",
                )
        if palsar_map_path and os.path.exists(palsar_map_path):
            with open(palsar_map_path, "rb") as f:
                col3.download_button(
                    "📊 PALSAR HV Map",
                    data=f.read(),
                    file_name="palsar_change_map.html",
                    mime="text/html",
                )
        if ext_conc_path and os.path.exists(ext_conc_path):
            with open(ext_conc_path, "rb") as f:
                col4.download_button(
                    "📊 Extended Concordance",
                    data=f.read(),
                    file_name="concordance_ext_map.html",
                    mime="text/html",
                )


# ---------------------------------------------------------------------------
# Time Series tab helpers
# ---------------------------------------------------------------------------

def _roi_center(roi_geom) -> tuple:
    """Return (lat, lon) centroid of *roi_geom*, or a Florida fallback."""
    try:
        coords = roi_geom.centroid(maxError=1).getInfo()["coordinates"]
        return coords[1], coords[0]   # GEE returns [lon, lat]
    except Exception:
        return 27.5, -81.5            # central Florida


# ---------------------------------------------------------------------------
# Time Series tab
# ---------------------------------------------------------------------------

def _render_time_series_tab(impact_results: Dict[str, Any]) -> None:
    """
    Render the Time Series tab.

    Lets the user run an independent temporal analysis over a configurable
    date range, then shows an interactive Plotly chart, anomaly table, and
    recovery metrics.
    """
    st.subheader("⏱️ Time Series Analysis & Anomaly Detection")
    st.markdown(
        "Analyse the vegetation index trajectory over a multi-year period. "
        "Seasonal decomposition (STL) separates the long-term trend and "
        "annual phenological cycle from anomalous deviations."
    )

    _default_index = impact_results.get("index", "NDVI")
    satellite = impact_results.get("satellite", "sentinel2")
    event_date = impact_results.get("event_date")

    # Load hurricane event catalog from config for auto-overlay markers
    _cfg = _load_config()
    _all_hurricanes: list = _cfg.get("hurricane_events", [])

    # ── Configuration ──────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Date range**")
        start_date = st.date_input(
            "Start date", value=pd.Timestamp("2019-01-01"), key="ts_start"
        ).strftime("%Y-%m-%d")
        end_date = st.date_input(
            "End date", value=pd.Timestamp("2024-12-31"), key="ts_end"
        ).strftime("%Y-%m-%d")

        _index_options = ["NDVI", "EVI", "SAVI", "NDMI"]
        index = st.selectbox(
            "Vegetation index",
            options=_index_options,
            index=_index_options.index(_default_index) if _default_index in _index_options else 0,
            key="ts_index",
            help="Index to extract and analyse. Defaults to the index used in the impact analysis.",
        )

        composite = st.selectbox(
            "Temporal compositing",
            options=["monthly", "biweekly", "weekly", "raw"],
            help=(
                "Aggregation period for each data point. Monthly (12 obs/yr) "
                "is recommended for multi-year STL analysis."
            ),
            key="ts_composite",
        )

    with col_right:
        st.markdown("**Location**")
        use_roi = st.radio(
            "Source",
            options=[
                "Use impact analysis ROI (spatial mean)",
                "Enter a point (lat, lon)",
                "Click a point on the map",
            ],
            key="ts_loc_mode",
        )

        if "Enter a point" in use_roi:
            lat = st.number_input("Latitude", value=26.45, format="%.4f", key="ts_lat")
            lon = st.number_input("Longitude", value=-81.95, format="%.4f", key="ts_lon")
            location = (lat, lon)

        elif "Click" in use_roi:
            center = _roi_center(impact_results.get("roi_geom"))
            try:
                from streamlit_folium import st_folium
                import folium

                m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

                # Mark previously selected point
                saved_lat = st.session_state.get("ts_clicked_lat")
                saved_lon = st.session_state.get("ts_clicked_lon")
                if saved_lat is not None:
                    folium.Marker(
                        [saved_lat, saved_lon],
                        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
                        tooltip="Selected point",
                    ).add_to(m)

                map_data = st_folium(
                    m, height=300, width=None,
                    key="ts_map_click",
                    returned_objects=["last_clicked"],
                )

                if map_data and map_data.get("last_clicked"):
                    st.session_state["ts_clicked_lat"] = map_data["last_clicked"]["lat"]
                    st.session_state["ts_clicked_lon"] = map_data["last_clicked"]["lng"]

                lat = st.session_state.get("ts_clicked_lat", center[0])
                lon = st.session_state.get("ts_clicked_lon", center[1])
                location = (lat, lon)
                _lat_hem = "N" if lat >= 0 else "S"
                _lon_hem = "E" if lon >= 0 else "W"
                st.caption(
                    f"Selected: {abs(lat):.4f}°{_lat_hem}, {abs(lon):.4f}°{_lon_hem}"
                    " — click map to change"
                )

            except ImportError:
                st.error(
                    "Map click requires `streamlit-folium`. "
                    "Run `pip install streamlit-folium` and restart."
                )
                location = None

        else:
            location = impact_results.get("roi_geom")

        st.markdown("**Anomaly detection**")
        anomaly_methods = st.multiselect(
            "Methods",
            options=["zscore", "moving_window", "climatology"],
            default=["zscore"],
            key="ts_methods",
        )
        anomaly_threshold = st.slider(
            "Z-score threshold", 1.5, 4.0, 2.0, 0.5, key="ts_thresh"
        )

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        detect_cp = st.checkbox(
            "Detect change points (CUSUM)",
            value=False, key="ts_cp",
            help="Detect sustained regime shifts using CUSUM algorithm.",
        )
    with col_opt2:
        do_recovery = st.checkbox(
            "Recovery analysis",
            value=bool(event_date),
            key="ts_recovery",
            help=(
                "Track vegetation recovery after the event date. "
                "Requires the impact analysis event date to be set."
            ),
        )
    with col_opt3:
        ts_scale = st.selectbox(
            "Spatial resolution (ROI mode)",
            options=[250, 100, 500, 30],
            index=0,
            key="ts_scale",
            help=(
                "Pixel size for ROI spatial averaging. "
                "250 m is recommended — finer resolutions may time out for large ROIs."
            ),
        )

    recovery_style = "seasonal"
    if do_recovery:
        recovery_style = st.radio(
            "Recovery baseline",
            options=["seasonal", "flat", "all"],
            format_func=lambda x: {
                "seasonal": "Seasonal envelope (default)",
                "flat":     "Flat pre-event mean",
                "all":      "Both (seasonal + flat)",
            }[x],
            index=0,
            key="ts_recovery_style",
            horizontal=True,
            help=(
                "**Seasonal**: monthly climatology ±1σ band from pre-event data — "
                "recovery requires returning to the expected seasonal level. "
                "**Flat**: legacy flat pre-event mean ±1σ baseline."
            ),
        )

    # Filter hurricane catalog to events within the selected date range
    visible_hurricanes = [
        he for he in _all_hurricanes
        if start_date <= str(he.get("date", "")) <= end_date
    ]

    # ── ROI size advisory ────────────────────────────────────────────────────
    # Only applies when the user is working with the ROI spatial mean.
    _is_roi_mode = location is not None and not isinstance(location, tuple)
    _roi_tier_info = None
    _large_roi_confirmed = True   # default: no confirmation needed

    if _is_roi_mode:
        _roi_area_km2 = impact_results.get("roi_area_km2")
        if _roi_area_km2 is not None:
            from src.utils import classify_roi_size as _classify_roi
            _roi_tier_info = _classify_roi(_roi_area_km2)
            if _roi_tier_info["note"]:
                st.info(_roi_tier_info["note"])
            if _roi_tier_info["warning"]:
                st.warning(_roi_tier_info["warning"])
            if _roi_tier_info["requires_confirm"]:
                _large_roi_confirmed = st.checkbox(
                    "I understand this ROI is very large — proceed anyway",
                    key="ts_large_roi_confirm",
                )

    run_ts = st.button(
        "▶ Run Time Series Analysis",
        type="primary",
        key="ts_run",
        disabled=not _large_roi_confirmed,
    )

    if not run_ts:
        st.info(
            "Configure parameters above and click **Run Time Series Analysis**.\n\n"
            "The analysis runs independently of the impact analysis — "
            "it queries GEE again for the full time range."
        )
        return

    if location is None:
        st.error("No ROI available from the impact analysis. Select 'Enter a point' instead.")
        return

    if not anomaly_methods:
        st.warning("Select at least one anomaly detection method.")
        return

    from src.time_series import run_time_series_analysis

    # ── Effective scale (tier override) ──────────────────────────────────────
    # For tier 3+ the recommended scale is ≥ 500 m; we take the larger of
    # the user's choice and the tier minimum to avoid GEE timeouts.
    _effective_scale = ts_scale
    if _is_roi_mode and _roi_tier_info is not None:
        _rec = _roi_tier_info["recommended_scale"]
        if _rec > ts_scale:
            _effective_scale = _rec
            st.info(
                f"Spatial resolution raised from {ts_scale} m to {_effective_scale} m "
                "for this ROI size. Change the selector above to override."
            )

    # ── Progress UI ──────────────────────────────────────────────────────────
    # For point mode a single getRegion() call is used (no chunking needed).
    # For ROI mode the date range is split into 6-month chunks; we show a
    # progress bar that updates after each chunk is fetched from GEE.
    _is_roi_mode = not isinstance(location, tuple)
    _prog_bar   = st.progress(0, text="Connecting to Google Earth Engine…") \
                  if _is_roi_mode else None
    _prog_text  = st.empty() if _is_roi_mode else None

    def _progress_callback(done: int, total: int) -> None:
        pct = int(done / total * 100)
        _prog_bar.progress(
            pct,
            text=f"Fetching time series from GEE… chunk {done}/{total}",
        )
        if done == total:
            _prog_text.text("Running decomposition and anomaly detection…")

    # Inherit water mask setting from the impact analysis configuration
    _ts_proc = impact_results.get("config", {}).get("processing", {})
    _ts_mask_water     = bool(_ts_proc.get("mask_water", False))
    _ts_mask_threshold = int(_ts_proc.get("mask_water_threshold", 80))

    try:
        with st.spinner("Running decomposition and anomaly detection…") \
                if not _is_roi_mode else _nullcontext():
            ts = run_time_series_analysis(
                location=location,
                start_date=start_date,
                end_date=end_date,
                satellite=satellite,
                index=index,
                composite=composite,
                anomaly_methods=anomaly_methods,
                anomaly_threshold=anomaly_threshold,
                detect_changepoints=detect_cp,
                event_date=event_date,
                recovery_analysis=do_recovery,
                recovery_style=recovery_style,
                scale=_effective_scale,
                output_dir=None,  # no file output; display inline
                progress_callback=_progress_callback if _is_roi_mode else None,
                mask_water=_ts_mask_water,
                mask_water_threshold=_ts_mask_threshold,
            )
    except Exception as exc:
        if _prog_bar:
            _prog_bar.empty()
        if _prog_text:
            _prog_text.empty()
        st.error(f"Time series analysis failed: {exc}")
        logging.getLogger("app").exception("Time series error")
        return
    finally:
        if _prog_bar:
            _prog_bar.empty()
        if _prog_text:
            _prog_text.empty()

    df = ts["df"]
    anomalies = ts["anomalies"]
    stl_result = ts["stl_result"]
    changepoints = ts.get("changepoints", pd.DataFrame())
    recovery = ts.get("recovery", {})

    # ── Summary metrics ─────────────────────────────────────────────────────
    n_anom = len(anomalies) if not anomalies.empty else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", f"{len(df)}")
    m2.metric(f"Mean {index}", f"{df['index_value'].mean():.4f}")
    m3.metric("Std dev", f"{df['index_value'].std():.4f}")
    m4.metric("Anomalies detected", f"{n_anom}",
              delta="⚠ check results" if n_anom > 0 else None,
              delta_color="inverse")

    # ── Interactive Plotly chart ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Interactive Time Series Chart")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        html_path = tmp.name

    try:
        from src.time_series import plot_time_series_interactive
        plot_time_series_interactive(
            df, stl_result, anomalies, event_date,
            index, html_path, title=ts["location_label"],
            hurricane_events=visible_hurricanes,
        )
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        import streamlit.components.v1 as components
        components.html(html_content, height=600, scrolling=False)
    except Exception as exc:
        st.warning(f"Interactive chart failed: {exc}")

    # ── STL decomposition ───────────────────────────────────────────────────
    if stl_result is not None:
        st.markdown("---")
        with st.expander("📉 STL Decomposition (Trend / Seasonal / Residual)", expanded=False):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                stl_path = tmp.name
            try:
                from src.time_series import plot_stl_decomposition
                plot_stl_decomposition(stl_result, index, stl_path,
                                       hurricane_events=visible_hurricanes)
                st.image(stl_path, width="stretch")
            except Exception as exc:
                st.warning(f"STL plot failed: {exc}")

    # ── Detrended & Normalised Views ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Detrended & Normalised Views")
    dt_tab1, dt_tab2, dt_tab3, dt_tab4, dt_tab5 = st.tabs(
        ["Residual", "Z-Score", "Seasonal Departure", "CUSUM", "Combined Panel"]
    )

    def _render_detrended_plot(plot_fn, tab, *args, **kwargs):
        with tab:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                plot_fn(*args, output_path=tmp_path, **kwargs)
                st.image(tmp_path, width="stretch")
            except Exception as exc:
                st.warning(f"Plot failed: {exc}")

    from src.time_series import (
        plot_residual as _plot_residual,
        plot_standardized_anomaly as _plot_zscore,
        plot_seasonal_departure as _plot_departure,
        plot_cusum as _plot_cusum,
        plot_combined_panel as _plot_combined,
    )

    _detrended_kwargs = dict(
        index=index,
        event_date=event_date,
        hurricane_events=visible_hurricanes,
    )

    _render_detrended_plot(
        _plot_residual, dt_tab1, df, stl_result, **_detrended_kwargs
    )
    _render_detrended_plot(
        _plot_zscore, dt_tab2, df, stl_result, **_detrended_kwargs
    )
    with dt_tab3:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            dep_path = tmp.name
        try:
            _plot_departure(df, output_path=dep_path, **_detrended_kwargs)
            st.image(dep_path, width="stretch")
        except Exception as exc:
            st.warning(f"Seasonal departure plot failed: {exc}")
    with dt_tab4:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cusum_path = tmp.name
        try:
            _plot_cusum(df, changepoints, output_path=cusum_path, **_detrended_kwargs)
            st.image(cusum_path, width="stretch")
        except Exception as exc:
            st.warning(f"CUSUM plot failed: {exc}")
    with dt_tab5:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            comb_path = tmp.name
        try:
            _plot_combined(df, stl_result, changepoints, output_path=comb_path,
                           **_detrended_kwargs)
            st.image(comb_path, width="stretch")
        except Exception as exc:
            st.warning(f"Combined panel failed: {exc}")

    # ── Anomaly table ───────────────────────────────────────────────────────
    if not anomalies.empty:
        st.markdown("---")
        st.subheader(f"Detected Anomalies ({len(anomalies)})")
        display_cols = ["date", "observed_value", "expected_value",
                        "deviation", "z_score", "method", "severity"]
        display_cols = [c for c in display_cols if c in anomalies.columns]
        st.dataframe(
            anomalies[display_cols]
            .assign(date=anomalies["date"].dt.strftime("%Y-%m-%d"))
            .round(4),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("No anomalies detected in the time series.")

    # ── Change points ───────────────────────────────────────────────────────
    if detect_cp and changepoints is not None and not changepoints.empty:
        st.markdown("---")
        st.subheader(f"Change Points ({len(changepoints)})")
        st.dataframe(
            changepoints
            .assign(date=changepoints["date"].dt.strftime("%Y-%m-%d"))
            .round(4),
            width="stretch",
            hide_index=True,
        )

    # ── Recovery analysis ───────────────────────────────────────────────────
    _rs = ts.get("recovery_style", "seasonal")
    recovery_seasonal_res = ts.get("recovery_seasonal", {})
    recovery_flat_res     = ts.get("recovery_flat", {})

    if recovery and recovery.get("pre_mean") is not None:
        st.markdown("---")
        st.subheader("Recovery Analysis")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Pre-event mean", f"{recovery['pre_mean']:.4f}")
        r2.metric("Post-event min", f"{recovery['post_min']:.4f}")
        pct = recovery.get("recovery_pct", 0)
        r3.metric(
            "Recovery %",
            f"{pct:.1f}%",
            help="Seasonal-departure-based" if recovery.get("method") == "seasonal"
                 else "Flat-baseline-based",
        )
        days = recovery.get("recovery_days")
        r4.metric("Recovery days", f"{days}" if days else "Not yet")

        status = recovery.get("recovery_status", "")
        interp = recovery.get("interpretation", "")
        if "full" in status:
            st.success(interp)
        elif "partial" in status:
            st.warning(interp)
        else:
            st.error(interp)

        # Recovery trajectory plot(s)
        from src.time_series import (
            plot_recovery_trajectory,
            plot_recovery_trajectory_seasonal,
        )

        def _show_rec_plot(plot_fn, rec_dict, caption=""):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                rec_path = tmp.name
            try:
                plot_fn(df, rec_dict, event_date, index, rec_path,
                        hurricane_events=visible_hurricanes)
                st.image(rec_path, width="stretch")
                if caption:
                    st.caption(caption)
            except Exception as exc:
                st.warning(f"Recovery plot failed: {exc}")

        if _rs == "all":
            tab_seas, tab_flat = st.tabs(["Seasonal Baseline", "Flat Baseline"])
            with tab_seas:
                if recovery_seasonal_res.get("pre_mean") is not None:
                    _show_rec_plot(plot_recovery_trajectory_seasonal,
                                   recovery_seasonal_res,
                                   "Monthly climatology ±1σ envelope")
            with tab_flat:
                if recovery_flat_res.get("pre_mean") is not None:
                    _show_rec_plot(plot_recovery_trajectory,
                                   recovery_flat_res,
                                   "Flat pre-event mean ±1σ")
        elif _rs == "seasonal":
            _show_rec_plot(plot_recovery_trajectory_seasonal, recovery)
        else:
            _show_rec_plot(plot_recovery_trajectory, recovery)

    # ── SAR time series overlay ──────────────────────────────────────────────
    # Available in point mode so we have a concrete lat/lon
    if isinstance(location, tuple):
        st.markdown("---")
        with st.expander("📡 SAR Time Series Overlay (Sentinel-1 RVI)", expanded=False):
            st.caption(
                "Extract a Sentinel-1 RVI time series at the same point and "
                "compare it with the optical vegetation index on a dual-axis chart. "
                "Divergence between the two signals distinguishes structural damage "
                "(SAR ↓, NDVI stable) from foliar stress (NDVI ↓, SAR stable)."
            )
            _sar_orbit = _load_config().get("sar", {}).get("orbit", "DESCENDING")
            if st.button("Extract SAR time series at this point", key="ts_sar_btn"):
                with st.spinner("Extracting SAR observations …"):
                    try:
                        from src.time_series import extract_sar_time_series as _extr_sar
                        _sar_df = _extr_sar(
                            lat=location[0], lon=location[1],
                            start_date=start_date, end_date=end_date,
                            orbit=_sar_orbit,
                        )
                        st.session_state["ts_sar_df"] = _sar_df
                        st.session_state["ts_sar_loc"] = location
                    except Exception as exc:
                        st.error(f"SAR extraction failed: {exc}")

            _sar_series = st.session_state.get("ts_sar_df")
            if _sar_series is not None and not _sar_series.empty:
                st.success(f"{len(_sar_series)} SAR observations retrieved.")

                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates

                fig, ax1 = plt.subplots(figsize=(14, 5))
                ax1.plot(df["date"], df["index_value"],
                         color="steelblue", linewidth=1.5, label=f"{index} (left axis)")
                ax1.set_ylabel(index, color="steelblue", fontsize=11)
                ax1.tick_params(axis="y", labelcolor="steelblue")

                ax2 = ax1.twinx()
                ax2.plot(_sar_series["date"], _sar_series["RVI"],
                         color="darkorange", linewidth=1.4, linestyle="--",
                         label="RVI (right axis)")
                ax2.set_ylabel("Radar Vegetation Index (RVI)", color="darkorange", fontsize=11)
                ax2.tick_params(axis="y", labelcolor="darkorange")

                if event_date:
                    ax1.axvline(pd.Timestamp(event_date), color="crimson",
                                linewidth=1.5, linestyle="--", label="Event date")
                for _he in visible_hurricanes:
                    _hdt = pd.Timestamp(_he["date"])
                    if pd.Timestamp(start_date) <= _hdt <= pd.Timestamp(end_date):
                        ax1.axvline(_hdt, color="#8B0000", linewidth=1.2,
                                    linestyle="--", alpha=0.7)

                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

                _l1, _lb1 = ax1.get_legend_handles_labels()
                _l2, _lb2 = ax2.get_legend_handles_labels()
                ax1.legend(_l1 + _l2, _lb1 + _lb2, fontsize=9, loc="upper left")
                ax1.set_title(
                    f"{index} vs SAR RVI — dual-sensor temporal comparison",
                    fontsize=12, fontweight="bold",
                )
                ax1.grid(alpha=0.25)
                plt.tight_layout()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    _dual_path = tmp.name
                fig.savefig(_dual_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                st.image(_dual_path, width="stretch")

                _sar_csv = _sar_series.assign(
                    date=_sar_series["date"].dt.strftime("%Y-%m-%d")
                ).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📋 Download SAR Time Series CSV",
                    data=_sar_csv,
                    file_name="sar_time_series.csv",
                    mime="text/csv",
                )

        # ── PALSAR annual time series overlay ─────────────────────────────
        st.markdown("---")
        with st.expander("🛰️ PALSAR-2 L-band Annual Time Series", expanded=False):
            st.caption(
                "Extract annual ALOS PALSAR-2 HV dB values for the analysis ROI "
                "(spatial mean). Annual markers are overlaid on a timeline to show "
                "long-term structural biomass trends. Available from 2014 onwards."
            )
            _cfg_ts = _load_config()
            _palsar_start_yr = st.number_input(
                "First year", min_value=2007, max_value=2030,
                value=2014, key="ts_palsar_start_yr",
            )
            if st.button("Extract PALSAR time series for ROI", key="ts_palsar_btn"):
                with st.spinner("Extracting annual PALSAR mosaics …"):
                    try:
                        from src.time_series import extract_palsar_time_series as _extr_palsar
                        _palsar_df = _extr_palsar(
                            roi=impact_results.get("roi_geom"),
                            start_year=int(_palsar_start_yr),
                            config=_cfg_ts,
                        )
                        st.session_state["ts_palsar_df"] = _palsar_df
                    except Exception as exc:
                        st.error(f"PALSAR extraction failed: {exc}")

            _palsar_series = st.session_state.get("ts_palsar_df")
            if _palsar_series is not None and not _palsar_series.empty:
                st.success(f"{len(_palsar_series)} annual PALSAR observations.")

                import matplotlib.pyplot as _mplt
                import matplotlib.dates as _mdates

                _fig_p, _ax_p = _mplt.subplots(figsize=(12, 4))
                _ax_p.plot(
                    _palsar_series["date"], _palsar_series["HV_dB"],
                    color="saddlebrown", linewidth=2, marker="o", markersize=6,
                    label="HV dB (L-band)",
                )
                _ax_p.fill_between(
                    _palsar_series["date"], _palsar_series["HV_dB"],
                    alpha=0.15, color="saddlebrown",
                )
                if event_date:
                    _ax_p.axvline(pd.Timestamp(event_date), color="crimson",
                                  linewidth=1.5, linestyle="--", label="Event date")
                _ax_p.set_ylabel("HV Backscatter (dB)", fontsize=11)
                _ax_p.set_xlabel("Year")
                _ax_p.xaxis.set_major_locator(_mdates.YearLocator())
                _ax_p.xaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
                _mplt.setp(_ax_p.get_xticklabels(), rotation=45, ha="right")
                _ax_p.legend(fontsize=9)
                _ax_p.set_title(
                    "PALSAR-2 Annual HV Backscatter — ROI Spatial Mean",
                    fontsize=12, fontweight="bold",
                )
                _ax_p.grid(alpha=0.25)
                _mplt.tight_layout()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as _tmp_p:
                    _palsar_plot_path = _tmp_p.name
                _fig_p.savefig(_palsar_plot_path, dpi=150, bbox_inches="tight")
                _mplt.close(_fig_p)
                st.image(_palsar_plot_path, width="stretch")

                _palsar_csv = _palsar_series.assign(
                    date=_palsar_series["date"].dt.strftime("%Y-%m-%d")
                ).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📋 Download PALSAR Time Series CSV",
                    data=_palsar_csv,
                    file_name="palsar_time_series.csv",
                    mime="text/csv",
                )

    # ── Download ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Downloads")
    col_dl1, col_dl2 = st.columns(2)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    col_dl1.download_button(
        "📋 Download Time Series CSV",
        data=csv_bytes,
        file_name=f"{index}_time_series.csv",
        mime="text/csv",
    )

    if not anomalies.empty:
        anom_csv = anomalies.assign(
            date=anomalies["date"].dt.strftime("%Y-%m-%d")
        ).to_csv(index=False).encode("utf-8")
        col_dl2.download_button(
            "📋 Download Anomalies CSV",
            data=anom_csv,
            file_name=f"{index}_anomalies.csv",
            mime="text/csv",
        )

    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            st.download_button(
                "📊 Download Interactive Chart (HTML)",
                data=f.read(),
                file_name=f"{index}_time_series_interactive.html",
                mime="text/html",
            )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    """Entry point for the Streamlit dashboard."""
    # Header
    st.markdown(
        "<h1 style='color:#c0392b;'>🌪️ Hurricane Vegetation Impact Analyzer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Satellite-based vegetation index change detection for Florida hurricane impacts. "
        "Powered by **Google Earth Engine** + **Sentinel-2 / Landsat** imagery."
    )

    # Sidebar controls
    params = _render_sidebar()

    # Initialize session state for results
    if "results" not in st.session_state:
        st.session_state.results = None

    # Initialize GEE (cached)
    gee_ok = _init_gee(params["gee_project"])
    if not gee_ok:
        st.error(
            "Google Earth Engine could not be initialized. "
            "Please check your GEE credentials (run `earthengine authenticate` in your terminal)."
        )
        st.stop()

    # Run analysis when button is pressed
    if params["run"]:
        # Remove the previous run's temp directory before creating a new one
        prev = st.session_state.get("results")
        if prev is not None:
            _cleanup_temp_dir(prev.get("output_dir"))
        with st.spinner("Running analysis …"):
            results = _run_analysis(params)
        if results is not None:
            st.session_state.results = results
            st.success("Analysis complete!")

    # Render results (if available)
    if st.session_state.results is not None:
        results = st.session_state.results

        # Show ROI area (and land area when water masking is active) in sidebar
        _roi_area_km2  = results.get("roi_area_km2")
        _land_area_km2 = results.get("land_area_km2")
        if _roi_area_km2 is not None:
            st.sidebar.markdown("---")
            st.sidebar.metric("📐 ROI Area", f"{_roi_area_km2:,.1f} km²")
            if _land_area_km2 is not None:
                _land_pct = _land_area_km2 / max(_roi_area_km2, 1) * 100
                st.sidebar.caption(
                    f"🌿 Land: {_land_area_km2:,.1f} km² ({_land_pct:.0f}%) "
                    "— water pixels excluded"
                )

        # Summary banner
        stat = results.get("statistics", {})
        if stat:
            conclusion = stat.get("conclusion", "")
            if stat.get("significant"):
                st.error(f"**{conclusion}**")
            else:
                st.success(f"**{conclusion}**")

        # Image acquisition details expander
        _render_metadata_expander(results)

        # Tabs (add Multi-Sensor tab when SAR or GEDI was requested)
        _has_structural = bool(results.get("structural"))
        _tab_labels = [
            "🗺️ Difference Map", "📊 Statistics", "🎯 Classification",
            "⬇️ Downloads", "⏱️ Time Series",
        ]
        if _has_structural:
            _tab_labels.append("🛰️ Multi-Sensor")

        _tabs = st.tabs(_tab_labels)
        with _tabs[0]:
            _render_map_tab(results)
        with _tabs[1]:
            _render_statistics_tab(results)
        with _tabs[2]:
            _render_classification_tab(results)
        with _tabs[3]:
            _render_downloads_tab(results)
        with _tabs[4]:
            _render_time_series_tab(results)
        if _has_structural:
            with _tabs[5]:
                _render_multisensor_tab(results)

    else:
        # Welcome / instruction state
        st.info(
            "👈 Configure the analysis parameters in the sidebar and click **Run Analysis**.\n\n"
            "**Quick start:** Select a hurricane preset (e.g. *Hurricane Ian*) and click Run."
        )

        # Show preset overview
        st.subheader("Available Florida Hurricane Presets")
        cols = st.columns(3)
        presets = [(k, v) for k, v in HURRICANE_PRESETS.items() if v is not None]
        for i, (name, info) in enumerate(presets):
            cols[i % 3].info(
                f"**{name}**\n\n"
                f"📅 {info['date']}\n\n"
                f"{info['description']}"
            )


if __name__ == "__main__":
    main()
