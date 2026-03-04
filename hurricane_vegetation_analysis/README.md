# Hurricane Vegetation Impact Analyzer

A modular Python tool that detects hurricane-induced storm surge and wind damage
by analyzing vegetation index changes in satellite imagery before and after landfall.

Built on **Google Earth Engine (GEE)** with support for Sentinel-2, Landsat 8/9,
Sentinel-1 SAR, ALOS PALSAR-2, and GEDI lidar, the tool applies paired statistical
tests, classifies impact severity, and produces interactive maps and HTML reports —
all without downloading raw rasters locally.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [GEE Authentication](#gee-authentication)
5. [CLI Usage](#cli-usage)
6. [Multi-Sensor Analysis](#multi-sensor-analysis)
7. [Time Series CLI](#time-series-cli)
8. [Streamlit Dashboard](#streamlit-dashboard)
9. [Scientific Methodology](#scientific-methodology)
10. [Florida Test Cases](#florida-test-cases)
11. [Output Files](#output-files)
12. [Running Tests](#running-tests)
13. [Configuration Reference](#configuration-reference)

---

## Features

| Feature | Details |
|---------|---------|
| **Satellites** | Sentinel-2 SR (10 m) and Landsat 8/9 C2 L2 (30 m) |
| **Indices** | NDVI, EVI, SAVI, NDMI |
| **Cloud masking** | QA60 + SCL (Sentinel-2); QA_PIXEL (Landsat) |
| **Water masking** | Per-image JRC Global Surface Water mask applied before compositing — water pixels are excluded from cloud filtering, index computation, and median reduction; effective land area (km²) reported in sidebar and CLI |
| **Compositing** | Median composite over configurable pre/post windows |
| **Statistics** | Paired t-test, Wilcoxon signed-rank, Cohen's d |
| **Classification** | 4-class severity map (No / Low / Moderate / Severe Impact) |
| **Baseline check** | Historical seasonal variability comparison (2–3 prior years) |
| **Sentinel-1 SAR** | C-band (5.6 cm) Radar Vegetation Index (RVI = 4·VH/(VV+VH)) change; focal-median speckle filter; averaged in linear scale to avoid bias; DESCENDING IW GRD |
| **ALOS PALSAR-2** | L-band (24 cm) HV backscatter change; penetrates foliage to detect trunk and primary-branch loss; annual mosaics 2014–present; JAXA calibration dB = 20·log10(DN) − 83 |
| **GEDI lidar** | Canopy height (rh95, rh50) and tree cover change from the ISS; available April 2019 – March 2023 |
| **Concordance maps** | 4-class (optical + C-band); 8-class extended (optical + C-band + L-band bitmask) |
| **PALSAR damage** | 4-class HV structural damage: No / Light / Moderate / Severe |
| **AGB estimate** | Experimental above-ground biomass proxy from PALSAR-2 HV (allometric formula; regional calibration required) |
| **Visualization** | Interactive HTML folium maps; matplotlib histograms; Jinja2 HTML report; time series; STL decomposition; anomaly timeline; recovery trajectory; multi-point comparison; SAR dual-axis overlay; PALSAR annual timeline |
| **Time Series** | Multi-year temporal monitoring: STL seasonal decomposition, harmonic regression, 3 anomaly detection methods, CUSUM change-point detection, seasonal recovery analysis |
| **Hurricane catalog** | YAML-driven event catalog; vertical markers auto-overlaid on every chart — add future events without touching code |
| **Reporting** | Auto-generated HTML report with embedded figures |
| **CLI** | `click`-based with presets for all major FL hurricanes; `--sensors` flag; `--palsar-pre-year` / `--palsar-post-year` overrides |
| **ROI size advisory** | Automatic tier check before time series analysis: < 100 km² (normal), 100–500 km² (note), 500–2000 km² (auto-scale to 500 m, warn), > 2000 km² (confirmation required). ROI and land area displayed in sidebar |
| **Dashboard** | Streamlit app with GEE live analysis, customizable ROI, PALSAR checkbox, Multi-Sensor tab, map-click point selection |
| **Fallback** | Microsoft Planetary Computer (STAC) if GEE is unavailable |

---

## Project Structure

```
hurricane_vegetation_analysis/
├── README.md
├── requirements.txt
├── config.yaml                 # Tunable thresholds, window defaults, presets
├── cli.py                      # Click CLI entry point
├── app.py                      # Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── utils.py                # Config loading, ROI parsing, GEE init, date math,
│   │                           #   compute_roi_area_km2(), classify_roi_size()
│   ├── data_acquisition.py     # Cloud masking, collection retrieval, compositing,
│   │                           #   build_jrc_land_mask(), compute_land_area_km2()
│   ├── vegetation_indices.py   # NDVI/EVI/SAVI/NDMI on ee.Image
│   ├── analysis.py             # Δ map, stats, classification, baseline check
│   ├── structural_analysis.py  # Sentinel-1 SAR, ALOS PALSAR-2, GEDI lidar,
│   │                           #   4-class and 8-class concordance maps
│   ├── visualization.py        # folium maps, matplotlib plots, Jinja2 HTML report
│   │                           #   + SAR/PALSAR/concordance interactive maps
│   ├── time_series.py          # Temporal monitoring: extraction, STL/harmonic, anomaly
│   │                           #   detection, CUSUM, recovery, SAR + PALSAR time series
│   └── metadata_utils.py       # Sensor metadata tracking and composite provenance
└── tests/
    ├── test_analysis.py        # 43 unit tests — impact analysis (no GEE required)
    └── test_time_series.py     # 25 unit tests — anomaly detection (no GEE required)
```

---

## Installation

### Requirements

- Python ≥ 3.10
- A Google Earth Engine account (free): https://earthengine.google.com/

Two installation paths are supported: **conda (recommended)** and **pip**.

---

### Option A — Conda (recommended, especially on Apple Silicon)

The conda path is preferred because `rasterio` and `fiona` require native GDAL
bindings that the conda-forge builds handle correctly.  On Apple Silicon (M1/M2/M3)
the PyPI wheels for these packages often fail to link GDAL properly.

A hybrid strategy is used: the geospatial stack comes from conda-forge, while
GEE-specific packages (not available or badly lagging on conda-forge) are installed
via pip inside the same environment.

```bash
# 1. Create and activate the environment from the provided file
conda env create -f environment.yml
conda activate hurricane_veg
```

The `environment.yml` file handles both steps automatically — it installs
conda-forge packages first, then calls pip for the remainder:

| Installed via conda-forge | Installed via pip (inside conda env) |
| ------------------------- | ------------------------------------ |
| numpy, pandas, scipy | earthengine-api |
| geopandas, rasterio, fiona, pyproj, shapely | geemap |
| matplotlib, plotly, folium | streamlit, streamlit-folium |
| pyyaml, jinja2, click | pystac-client, planetary-computer, stackstac |
| pytest, pytest-cov | — |

> **Why some packages need pip:** `earthengine-api`, `geemap`, `streamlit-folium`,
> `stackstac`, and `planetary-computer` either have no conda-forge build or their
> conda builds lag significantly behind PyPI releases.

---

### Option B — pip only (virtualenv)

```bash
# 1. Clone or download this project
cd hurricane_vegetation_analysis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Note for Apple Silicon users:** If `rasterio` or `fiona` fail to install via
> pip, switch to Option A (conda) — the conda-forge builds resolve native GDAL
> linking issues automatically.

---

## GEE Authentication

GEE requires a one-time browser-based OAuth flow.

```bash
# Authenticate (opens a browser window)
earthengine authenticate

# Verify it works
python -c "import ee; ee.Initialize(project='vegetation-impact-analysis'); print('GEE OK')"
```

> **Tip:** If you use a different GEE Cloud project, pass `--gee-project YOUR_PROJECT`
> to the CLI or set `gee.project` in `config.yaml`.

---

## CLI Usage

### Quick start — Hurricane Ian (recommended first test)

```bash
python cli.py analyze \
  --roi bbox:-82.2,26.4,-81.7,26.8 \
  --event-date 2022-09-28 \
  --index NDVI \
  --satellite sentinel2 \
  --output-dir ./results/ian \
  --report
```

Expected output:

```
Statistically significant vegetation decline detected
(Wilcoxon p < 0.001, Cohen's d = -1.24, large effect).
Mean NDVI dropped from 0.72 to 0.41 (-43%).

  No Impact          12.34 km²
  Low Impact          8.21 km²
  Moderate Impact     5.67 km²
  Severe Impact      14.92 km²
```

### Full option reference

```
python cli.py analyze --help
```

```
Options:
  --roi TEXT                    bbox:W,S,E,N  or  file:path/to/region.geojson / region.shp
  --event-date TEXT             Hurricane date YYYY-MM-DD  [required]
  --index [NDVI|EVI|SAVI|NDMI]  Vegetation index  [default: NDVI]
  --satellite [sentinel2|landsat]  [default: sentinel2]
  --pre-days INTEGER            Days in pre-event window  [default from config]
  --post-days INTEGER           Days in post-event window
  --buffer-days INTEGER         Days to exclude around event
  --significance-level FLOAT    Statistical α  [default: 0.05]
  --historical-years INTEGER    Prior years for baseline  [default: 3]
  --sample-size INTEGER         Pixels to sample  [default: 500]
  --output-dir TEXT             Output directory  [default: ./results]
  --mask-water                  Exclude ocean/lakes/rivers via JRC water mask
  --sensors TEXT                Comma-separated sensors: optical, sar, gedi, palsar, all
                                  [default: optical]
  --palsar-pre-year INTEGER     Override PALSAR pre-event mosaic year (auto if omitted)
  --palsar-post-year INTEGER    Override PALSAR post-event mosaic year (auto if omitted)
  --report                      Generate HTML report
  --time-series                 Generate monthly time series plot (slow)
  --gee-project TEXT            Override GEE project ID
  -v, --verbose                 Debug logging
```

### Named presets

All 12 built-in Florida hurricane presets plus a control location:

| Key | Event | Date | Default ROI |
| --- | ----- | ---- | ----------- |
| `frances` | Hurricane Frances (Cat 2) | 2004-09-05 | Stuart / Treasure Coast |
| `jeanne` | Hurricane Jeanne (Cat 3) | 2004-09-26 | Stuart / Treasure Coast |
| `hermine` | Hurricane Hermine (Cat 1) | 2016-09-02 | St. Marks / Wakulla County |
| `irma` | Hurricane Irma (Cat 4) | 2017-09-10 | Florida Keys / SW Florida |
| `michael` | Hurricane Michael (Cat 5) | 2018-10-10 | Mexico Beach / Panama City |
| `dorian` | Hurricane Dorian (Cat 1 FL) | 2019-09-03 | NE Florida coast |
| `ian` | Hurricane Ian (Cat 4) | 2022-09-28 | Fort Myers / Lee County |
| `nicole` | Hurricane Nicole (Cat 1) | 2022-11-10 | Vero Beach / Indian River |
| `idalia` | Hurricane Idalia (Cat 3) | 2023-08-30 | Cedar Key / Big Bend |
| `debby` | Hurricane Debby (Cat 1) | 2024-08-05 | Steinhatchee / Taylor County |
| `helene` | Hurricane Helene (Cat 4) | 2024-09-26 | Keaton Beach / Perry / Taylor County |
| `milton` | Hurricane Milton (Cat 3) | 2024-10-09 | Siesta Key / Sarasota |
| `orlando` | Control (no hurricane) | 2022-09-28 | Central Florida (inland) |

```bash
# List all built-in FL hurricane presets
python cli.py list-presets

# Run a preset directly
python cli.py run-preset ian --report
python cli.py run-preset michael --index EVI --satellite landsat --report
python cli.py run-preset idalia --report
python cli.py run-preset milton --mask-water --report

# Water masking removes ocean pixels — useful for coastal ROIs
python cli.py run-preset irma --mask-water --report
```

### Custom date / file ROI

The `--roi` flag accepts a GeoJSON file, a shapefile, or a geopackage.
Multi-feature files are automatically unioned into one geometry, and any
CRS is re-projected to WGS84 before being sent to GEE.

```bash
# Using a GeoJSON file as ROI
python cli.py analyze \
  --roi file:my_region.geojson \
  --event-date 2023-08-30 \
  --index NDMI \
  --output-dir ./results/idalia_custom

# Using a shapefile as ROI
python cli.py analyze \
  --roi file:my_region.shp \
  --event-date 2022-09-28 \
  --output-dir ./results/ian_shp

# Landsat, EVI, strict significance level
python cli.py analyze \
  --roi bbox:-85.8,29.9,-85.3,30.3 \
  --event-date 2018-10-10 \
  --index EVI \
  --satellite landsat \
  --significance-level 0.01 \
  --historical-years 3 \
  --output-dir ./results/michael \
  --report
```

---

## Multi-Sensor Analysis

The `--sensors` flag enables radar and lidar sensors alongside the optical analysis.
All sensors are accessed directly through GEE — no additional downloads required.

### Sensor overview

| Sensor | Flag token | Wavelength | What it detects |
| ------ | ---------- | ---------- | --------------- |
| Optical (Sentinel-2 / Landsat) | `optical` | 400–2500 nm | Chlorophyll, leaf area, canopy moisture |
| Sentinel-1 SAR | `sar` | 5.6 cm (C-band) | Canopy surface structure, small branches, leaves |
| ALOS PALSAR-2 | `palsar` | 24 cm (L-band) | Trunk and primary branches — penetrates foliage |
| GEDI lidar | `gedi` | 1064 nm | Canopy height (rh95, rh50) and tree cover fraction |

Using multiple sensors together reveals damage mechanisms that no single sensor
can resolve alone — see [Concordance Maps](#concordance-maps) below.

### Sentinel-1 C-band SAR

```bash
# Optical + C-band SAR — Hurricane Ian
python cli.py analyze \
  --roi bbox:-82.2,26.4,-81.7,26.8 \
  --event-date 2022-09-28 \
  --sensors optical,sar \
  --mask-water \
  --output-dir ./results/ian_sar \
  --report
```

Outputs `sar_change_map.html` (∆RVI, ∆VV, ∆VH interactive layers) and
`concordance_map.html` (4-class optical + C-band agreement map).

### ALOS PALSAR-2 L-band SAR

```bash
# Optical + C-band + L-band — Hurricane Michael (Cat 5, intense canopy destruction)
python cli.py analyze \
  --roi bbox:-85.8,29.9,-85.3,30.3 \
  --event-date 2018-10-10 \
  --sensors optical,sar,palsar \
  --output-dir ./results/michael_palsar \
  --report
```

Outputs `palsar_change_map.html` (∆HV dB sequential map) and
`concordance_ext_map.html` (8-class extended concordance).

**PALSAR sensor activation** — PALSAR is triggered by including `palsar` in the
`--sensors` flag, either alone (`optical,palsar`) or combined with other sensors.
The PALSAR checkbox in the Streamlit sidebar appends PALSAR regardless of which
optical/SAR radio option is selected.

**PALSAR year selection** — Yearly mosaics cover one calendar year.
The tool auto-selects pre/post years from the event date:

- Event month ≥ 7 (storm season): pre = event year, post = event year + 1
- Event month < 7: pre = event year − 1, post = event year

```
Irma  2017-09-10 → pre=2017, post=2018
Ian   2022-09-28 → pre=2022, post=2023
```

Override with `--palsar-pre-year` / `--palsar-post-year` if needed.

### GEDI lidar

GEDI is available only for events between **April 2019 and March 2023** (ISS
operational window). Outside this range the sensor block is skipped with an
informative message.

```bash
python cli.py analyze \
  --roi bbox:-82.2,26.4,-81.7,26.8 \
  --event-date 2022-09-28 \
  --sensors optical,sar,gedi \
  --output-dir ./results/ian_gedi \
  --report
```

### All sensors at once

```bash
python cli.py analyze \
  --roi bbox:-85.8,29.9,-85.3,30.3 \
  --event-date 2018-10-10 \
  --sensors all \
  --output-dir ./results/michael_all \
  --report
```

`--sensors all` expands to `sar,gedi,palsar`.

### Concordance maps

#### 4-class (optical + C-band)

| Class | Optical | SAR | Interpretation |
| ----- | ------- | --- | -------------- |
| 0 No Change | stable | stable | No damage signal from either sensor |
| 1 Foliar Loss | ↓ | stable | Leaf/chlorophyll loss, structure intact |
| 2 Structural Damage | stable | ↓ | Physical structure loss, canopy still reflecting |
| 3 High-Confidence Damage | ↓ | ↓ | Both sensors confirm damage |

#### 8-class extended (optical + C-band + L-band)

Bitmask: bit 0 = optical loss, bit 1 = C-band loss, bit 2 = L-band loss.

| Class | Sensors | Damage mechanism |
|-------|---------|-----------------|
| 0 | — | No Change |
| 1 | Optical | Foliar loss only — leaf-scale stress |
| 2 | C-band | Canopy surface roughness change |
| 3 | Optical + C-band | Leaf + small-branch loss (classic storm stress) |
| 4 | L-band | Trunk damage beneath an apparently intact canopy |
| 5 | Optical + L-band | Foliar loss + structural damage |
| 6 | C-band + L-band | Dead standing timber (C-band + L-band confirm, canopy partially intact) |
| 7 | All three | Full concordance — highest-confidence severe structural destruction |

---

## Time Series CLI

The `timeseries` subcommand extracts a multi-year vegetation index trajectory from
GEE and runs temporal analysis — independent of the impact analysis.

### Single point

```bash
python cli.py timeseries \
  --point 26.45,-81.95 \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --index NDVI \
  --satellite sentinel2 \
  --composite monthly \
  --event-date 2022-09-28 \
  --anomaly-method zscore \
  --anomaly-method moving_window \
  --detect-changepoints \
  --recovery-analysis \
  --output-dir ./results/ian_ts
```

### Spatial mean over an ROI

```bash
python cli.py timeseries \
  --roi bbox:-82.2,26.4,-81.7,26.8 \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --event-date 2022-09-28 \
  --composite monthly \
  --output-dir ./results/ian_ts_roi
```

### Multi-point comparison from CSV

```bash
# CSV format: label,lat,lon
python cli.py timeseries \
  --multi-point sites.csv \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --event-date 2022-09-28 \
  --output-dir ./results/multipoint
```

### timeseries option reference

```bash
python cli.py timeseries --help
```

```text
Options:
  --point TEXT           Lat,lon point (e.g. 26.45,-81.95)
  --roi TEXT             bbox:W,S,E,N  or  file:path.geojson
  --multi-point TEXT     CSV file with columns label,lat,lon
  --start-date TEXT      Start date YYYY-MM-DD  [default: 4 years before today]
  --end-date TEXT        End date YYYY-MM-DD  [default: today]
  --index TEXT           Vegetation index  [default: NDVI]
  --satellite TEXT       sentinel2 | landsat  [default: sentinel2]
  --composite TEXT       monthly | biweekly | weekly | raw  [default: monthly]
  --anomaly-method TEXT  Detection method (repeatable): zscore, moving_window, climatology
  --anomaly-threshold FLOAT  Z-score threshold  [default: 2.0]
  --detect-changepoints  Run CUSUM change-point detection
  --event-date TEXT      Hurricane date for annotation and recovery analysis
  --recovery-analysis    Estimate post-event recovery time and rate
  --recovery-style TEXT  seasonal | flat | all  [default: seasonal]
  --plot-type TEXT       raw | residual | zscore | departure | cusum | all  [default: all]
  --output-dir TEXT      Output directory  [default: ./results/timeseries]
  --scale INTEGER        Override GEE reduction scale in metres (default: 250 or auto from ROI size)
  --force                Skip the large-ROI confirmation prompt (> 2000 km²)
```

### Time series output files

| File | Description |
| ---- | ----------- |
| `NDVI_time_series.csv` | Full composited time series (date, index_value) |
| `NDVI_anomalies.csv` | Detected anomalies with method, z-score, severity |
| `NDVI_changepoints.csv` | CUSUM change point records (if `--detect-changepoints`) |
| `NDVI_time_series_interactive.html` | Plotly chart with hurricane event markers (open in browser) |
| `NDVI_time_series_static.png` | Publication-quality static chart with hurricane event markers |
| `NDVI_stl_decomposition.png` | 4-panel STL decomposition (trend / seasonal / residual) with markers |
| `NDVI_anomaly_timeline.png` | Z-score bar chart coloured by severity with hurricane markers |
| `NDVI_recovery_trajectory_seasonal.png` | **Default.** 2-panel seasonal recovery chart |
| `NDVI_recovery_trajectory.png` | Flat-baseline recovery plot (with `--recovery-style flat` or `all`) |
| `NDVI_multi_point_comparison.png` | Overlaid trajectories (multi-point mode only) |
| `NDVI_residual.png` | STL residual bar chart with ±2σ band |
| `NDVI_zscore.png` | Standardised z-score plot with ±2σ and ±3σ bands |
| `NDVI_seasonal_departure.png` | Leave-one-year-out DOY departure |
| `NDVI_cusum.png` | Cumulative deviation from the overall mean with changepoint markers |
| `NDVI_combined_panel.png` | 5-row figure with all views aligned on a shared x-axis |

> **Hurricane markers** — All charts automatically overlay dark-red dashed lines with "Name (Cat N)"
> labels for any hurricane in the `hurricane_events` catalog whose date falls within the analysis
> date range.  The catalog lives in `config.yaml` — no code changes are needed to add future events.
>
> **`--recovery-style` control** — The default `--recovery-style seasonal` generates the
> two-panel seasonal envelope plot.  Use `--recovery-style flat` for the legacy flat
> pre-event mean ±1σ baseline.  Use `--recovery-style all` to generate both plots side by side.

---

## Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` and provides:

1. **Sidebar** — Select one of the 12 built-in hurricane presets, or enter a
   custom event date and bounding box. Choose index, satellite, and window settings.
   - **Customize ROI**: When a preset is selected, an **"✏️ Customize ROI
     (optional)"** expander appears to override the preset bounding box.
   - **Upload ROI**: Accepts GeoJSON or a zipped shapefile.
   - **Water masking**: Excludes permanent water pixels via JRC Global Surface Water
     (applied per-image before compositing).
   - **Sensors radio**: Choose optical only, optical + SAR (Sentinel-1), or optical + SAR + GEDI.
   - **PALSAR-2 L-band checkbox**: Appends PALSAR to the sensor list.
     An optional expander lets you override the pre/post mosaic years
     (auto-selected from the event date when left blank).
   - **ROI Area metric**: Always-visible sidebar card showing ROI size in km².
     When water masking is active, a caption shows the effective land area (km² and %).
2. **Run Analysis** button — Triggers the full GEE pipeline. Results are
   cached in `st.session_state`; switching tabs does not re-run GEE.
3. **Map tab** — Interactive folium map with a built-in legend:
   pre/post index layers, diverging delta layer, classification overlay, ROI boundary.
4. **Statistics tab** — Metric cards, overlaid histogram + KDE, test results table.
5. **Classification tab** — Area pie chart by severity class, baseline variability flag.
6. **Downloads tab** — Pre/post/difference GeoTIFFs, distribution plot, HTML report, CSV stats.
7. **Time Series tab** — Independent multi-year temporal analysis:
   - **Location**: ROI spatial mean, manual lat/lon, or map-click point selection.
   - **ROI size advisory**: Before running, the tab checks ROI area against four tiers
     (< 100, 100–500, 500–2000, > 2000 km²). For tier 3 the resolution is automatically
     raised to 500 m. For tier 4 a confirmation checkbox is required to proceed.
   - **Anomaly detection**: z-score, moving window, climatology — one or all three.
   - **Detrended views**: residual, z-score, seasonal departure, CUSUM, combined panel.
   - **Recovery analysis** with seasonal envelope or flat baseline selector.
   - **SAR dual-axis overlay** expander (point mode): extract Sentinel-1 RVI at the
     selected point and compare with the optical index on a dual-axis chart.
   - **PALSAR-2 annual TS** expander: extract annual HV dB spatial means for the ROI
     and plot a long-term structural backscatter timeline (2014 onward).
8. **Multi-Sensor tab** (appears when SAR or GEDI or PALSAR results are available):
   - Sentinel-1 C-band SAR change map (∆RVI, ∆VV, ∆VH layers).
   - PALSAR-2 L-band ∆HV map with a sequential colourmap and statistics table
     (mean pre/post HV, Wilcoxon p, Cohen's d).
   - Auto-selected PALSAR mosaic years shown beneath the map.
   - **8-class extended concordance** map when PALSAR + SAR are both available;
     falls back to 4-class when only SAR is available.
   - GEDI lidar availability note and canopy change information.
   - Experimental AGB note (formula, calibration caveat).
   - Download buttons for all interactive HTML maps.

---

## Scientific Methodology

### Data Sources

| Satellite / Sensor | GEE Collection | Resolution | Revisit / Cadence |
| ------------------ | -------------- | ---------- | ----------------- |
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` | 10 m | ~5 days |
| Landsat 8/9 | `LANDSAT/LC{08,09}/C02/T1_L2` | 30 m | ~16 days |
| Sentinel-1 SAR | `COPERNICUS/S1_GRD` | 10 m | ~6 days |
| ALOS PALSAR-2 | `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH` | 25 m | Annual mosaic |
| GEDI lidar | `LARSE/GEDI/GEDI02_A_002_MONTHLY` + `…02_B…` | ~25 m footprint | Monthly composite |

### Cloud Masking

**Sentinel-2:** Combined two-pass filter:
1. `QA60` bits 10 (opaque cloud) and 11 (cirrus) must both be 0.
2. Scene Classification Layer (SCL) — only classes 4 (vegetation),
   5 (not-vegetated), 6 (water), and 7 (unclassified) are retained.

**Landsat:** `QA_PIXEL` bitmask — cloud (bit 5), cloud shadow (bit 3), and snow/ice (bit 4) are masked.

### Temporal Windows and Compositing

```text
                 PRE window (60 days)          POST window (60 days)
        |─────────────────────────|  buffer  |─────────────────────────|
        pre_start             pre_end  |  post_start             post_end
                                  event date
```

**Default settings** (configurable via `--pre-days`, `--post-days`, `--buffer-days`):

| Parameter | Default | Purpose |
| --------- | ------- | ------- |
| `pre_days` | 60 | Length of the pre-event collection window |
| `post_days` | 60 | Length of the post-event collection window |
| `buffer_days` | 5 | Days excluded on each side of the event date |

### Vegetation Indices

All band values are converted to surface reflectance (0–1) before computation:

| Index | Formula | Key application |
|-------|---------|----------------|
| **NDVI** | (NIR − Red) / (NIR + Red) | General canopy health |
| **EVI**  | 2.5 × (NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1) | Dense forest (avoids saturation) |
| **SAVI** | ((NIR − Red) / (NIR + Red + 0.5)) × 1.5 | Sparse / coastal vegetation |
| **NDMI** | (NIR − SWIR1) / (NIR + SWIR1) | Moisture / salt stress |

### Statistical Tests

Approximately 500 pixels are randomly sampled at paired locations from both composites.
Two tests are applied:

1. **Paired t-test** (parametric): tests whether the mean Δindex ≠ 0.
2. **Wilcoxon signed-rank test** (non-parametric, preferred): robust to
   non-normal distributions typical of vegetation indices.

**Effect size:** Cohen's d quantifies practical significance:
- |d| < 0.2: negligible
- 0.2–0.5: small | 0.5–0.8: medium | ≥ 0.8: large

### Impact Classification

| Class | Δ Index | Interpretation |
|-------|---------|----------------|
| **No Impact** | > −0.05 | Within normal variability |
| **Low Impact** | −0.05 to −0.15 | Minor stress / partial defoliation |
| **Moderate Impact** | −0.15 to −0.30 | Substantial canopy loss |
| **Severe Impact** | < −0.30 | Near-total vegetation destruction |

All thresholds are configurable in `config.yaml`.

### Multi-Sensor Structural Analysis

#### Water / Land Masking

When `mask_water: true` is set in `config.yaml` (or `--mask-water` on the CLI), the
JRC Global Surface Water occurrence layer (`JRC/GSW1_4/GlobalSurfaceWater`) is used
to build a binary land mask at the start of the pipeline.

- **Early application**: The mask is applied per-image — before cloud filtering, index
  computation, or median reduction — so water pixels never enter any downstream GEE
  operation.  Pixels with occurrence ≥ `mask_water_threshold` (default 80 %) are
  excluded; pixels with no occurrence data (`unmask(0)`) are treated as land.
- **All sensors**: The same land mask is applied to optical (Sentinel-2, Landsat),
  C-band SAR (before the focal-median speckle filter), and PALSAR-2 pipelines.
- **Land area reporting**: After masking, the effective land area is computed via
  `ee.Image.pixelArea()` and reported in the sidebar (km² and % of ROI) and in
  the CLI output line.
- **Why per-image vs post-composite?**: For median compositing, masking per-image is
  semantically correct — ocean pixels are excluded from the pixel stack used to
  compute the median.  For SAR, masking before the focal-median speckle filter also
  prevents low-backscatter ocean pixels from bleeding through the 50 m kernel into
  adjacent coastal land pixels.

#### Sentinel-1 SAR (C-band)

- **Collection**: `COPERNICUS/S1_GRD`, IW mode, DESCENDING orbit
- **Speckle filtering**: circular focal-median kernel (50 m radius), applied after
  per-image land masking when `mask_water` is enabled
- **Compositing**: scenes converted from dB to linear scale (10^(dB/10)) before averaging,
  then converted back to dB — avoids the non-linear bias of averaging in dB space
- **RVI**: RVI = 4·VH_linear / (VV_linear + VH_linear); sensitive to vegetation volume scattering;
  decreases as structural biomass is lost

#### ALOS PALSAR-2 (L-band)

- **Collection**: `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH` — one annual mosaic per year
- **Calibration**: dB = 20·log10(DN) − 83.0 (JAXA standard offset)
- **HV polarisation** is used for vegetation analysis (cross-pol is more sensitive to
  volume scattering from woody stems); HH is retained as an ancillary band
- **Year selection**: automatic based on event month (month ≥ 7 → pre = event year,
  post = event year + 1); overridable with `--palsar-pre-year` / `--palsar-post-year`
- **Damage classes** (∆HV):

| Class | ∆HV | Interpretation |
| ----- | --- | -------------- |
| No Damage | > −1.0 dB | Backscatter within normal inter-annual variability |
| Light Damage | −1.0 to −2.0 dB | Minor structural loss |
| Moderate Damage | −2.0 to −4.0 dB | Significant trunk/branch scatterer loss |
| Severe Damage | < −4.0 dB | Near-complete structural destruction |

- **AGB estimate** (experimental): AGB (Mg/ha) ≈ 10^((HV_dB + 83) × A + B).
  Default coefficients A = 0.04, B = −2.0 are illustrative placeholders.
  Regional calibration with field plots is required before use in publications.

#### GEDI Lidar

- **Collections**: `LARSE/GEDI/GEDI02_A_002_MONTHLY` (rh95, rh50) and
  `LARSE/GEDI/GEDI02_B_002_MONTHLY` (cover fraction)
- **Quality filters**: quality_flag == 1, degrade_flag == 0, sensitivity ≥ 0.9
- **Operational window**: April 2019 – March 2023 (ISS science mission); events
  outside this range are skipped automatically

#### Concordance Classification

Combining signals from multiple sensors reveals damage mechanisms invisible to any
single sensor:

- **Optical ↓ only** → leaf-level physiological stress (chlorophyll loss,
  defoliation); structural wood may be intact
- **C-band ↓ only** → loss of small canopy scatterers (bark, small branches);
  reflective enough to suppress volume scattering without visible foliar change
- **L-band ↓ only** → trunk and primary-branch damage beneath an apparently
  intact understory (L-band penetrates leaves; rarely triggered without visible damage)
- **C-band + L-band ↓** → dead standing timber (structural damage confirmed by both
  radars, but leafy understory or epiphytes maintain some optical reflectance)
- **All three ↓** → full concordance; highest-confidence, most severe structural damage

### Baseline Variability Check

To reduce false positives, the same index difference is computed for the
identical calendar window in 2–3 prior hurricane-free years.  If the observed
delta falls within ±2 SD of this historical distribution (z-score < 2), the
result is flagged as potentially within normal seasonal variation.

---

## Florida Test Cases

All 12 hurricane presets can be run with `python cli.py run-preset <key> --report`.
The cases below illustrate different damage types and recommended settings.

### 1. Hurricane Ian (2022-09-28) — Fort Myers / Lee County

```bash
python cli.py run-preset ian --mask-water --report

# With full multi-sensor stack (C-band + L-band + optical)
python cli.py analyze \
  --roi bbox:-82.2,26.4,-81.7,26.8 \
  --event-date 2022-09-28 \
  --sensors optical,sar,palsar \
  --mask-water \
  --output-dir ./results/ian_multisensor --report
```

**Expected result:** Severe vegetation loss (NDVI −0.3 to −0.5) in the coastal
strip — catastrophic storm surge inundated mangroves and urban vegetation.
PALSAR L-band should show strong ∆HV decrease in mangrove zones
(trunk/root structural damage from surge inundation).

---

### 2. Hurricane Michael (2018-10-10) — Mexico Beach / Panama City

```bash
python cli.py run-preset michael --index EVI --satellite landsat --report

# With C-band + L-band (no GEDI — pre-2019)
python cli.py analyze \
  --roi bbox:-85.8,29.9,-85.3,30.3 \
  --event-date 2018-10-10 \
  --sensors optical,sar,palsar \
  --output-dir ./results/michael_multisensor --report
```

**Expected result:** Near-total canopy destruction from Category 5 winds.
PALSAR expected to show severe (< −4 dB) HV loss across longleaf pine forest.
8-class concordance should be dominated by class 7 (all three sensors agree).

---

### 3. Hurricane Idalia (2023-08-30) — Cedar Key / Big Bend

```bash
python cli.py run-preset idalia --index NDMI --mask-water --report
```

**Expected result:** Surge-driven saltwater inundation in low-lying coastal
marshes. NDMI highlights moisture/salinity stress in marsh vegetation.

---

### 4. Hurricane Irma (2017-09-10) — Florida Keys / SW Florida

```bash
python cli.py run-preset irma --mask-water --report
```

**Expected result:** Widespread mangrove damage along the Keys coastline.
Water masking is especially important here given the island geography.

> **Note:** PALSAR is available for Irma (2017). GEDI is not (pre-2019 event).

---

### 5. Control — Central Florida (Orlando, no hurricane impact)

```bash
python cli.py run-preset orlando --report
```

**Expected result:** No statistically significant change (p > 0.05). This
validates that the tool does not generate false positives for inland areas
unaffected by storm surge.

---

## Output Files

After running the CLI, the output directory contains:

| File | Bands / Content | Description |
| ---- | --------------- | ----------- |
| `NDVI_pre.tif` | 1 band: `NDVI` | Pre-event median composite |
| `NDVI_post.tif` | 1 band: `NDVI` | Post-event median composite |
| `NDVI_difference.tif` | 2 bands: `delta`, `pct_change` | Post − pre change raster |
| `NDVI_distribution.png` | — | Pre/post histogram + KDE overlay |
| `difference_map.html` | — | Interactive folium map (optical) |
| `sar_change_map.html` | — | Interactive SAR ∆RVI / ∆VV / ∆VH map |
| `concordance_map.html` | — | 4-class optical + C-band concordance map |
| `palsar_change_map.html` | — | Interactive PALSAR-2 ∆HV map |
| `concordance_ext_map.html` | — | 8-class extended concordance (optical + C + L) |
| `NDVI_time_series.png` | — | Monthly index trajectory (if `--time-series`) |
| `impact_report.html` | — | Full HTML report with all figures and statistics |

> **Note:** `_pre.tif` and `_post.tif` are the median-composited vegetation
> index values for their respective time windows (not raw reflectance).
> SAR/PALSAR/concordance HTML maps are only generated when the corresponding
> sensors are enabled via `--sensors`.

---

## Running Tests

No GEE connection required for any of the unit tests — all use synthetic
numpy/pandas data.

```bash
# Run all 68 tests with verbose output (no GEE connection required)
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file or class
python -m pytest tests/test_analysis.py -v
python -m pytest tests/test_time_series.py -v
python -m pytest tests/test_analysis.py::TestCohensD -v
python -m pytest tests/test_time_series.py::TestAnomalyDetectionZscore -v
```

### `test_analysis.py` — 43 tests (impact analysis)

- Cohen's d effect size (known values, sign, magnitude)
- Paired t-test / Wilcoxon (significant and non-significant cases)
- Baseline variability check (within / anomalous / empty)
- Impact classification thresholds (boundary tests for all 4 classes)
- Date window arithmetic (buffer, window lengths, historical seasons)
- Vegetation index formulas (NDVI range, SAVI/NDVI equivalence at L=0, EVI desaturation)
- Config loading (structure, hurricane presets)

### `test_time_series.py` — 25 tests (temporal analysis)

| Class | Tests | Covers |
| ----- | ----- | ------ |
| `TestSeverityLabel` | 6 | All asymmetric z-score severity boundaries |
| `TestAnomalyDetectionZscore` | 7 | Flat series, spike/dip detection, threshold gate, columns, STL residuals |
| `TestAnomalyDetectionMovingWindow` | 4 | Stable series, step-change detection, short-series guard, k sensitivity |
| `TestAnomalyDetectionClimatology` | 4 | Single-year guard, extreme anomaly, false-positive rate, method label |
| `TestDetectAllAnomalies` | 4 | Method combination, filter, empty list, `ANOMALY_METHODS` constant |

The synthetic series helper (`_make_synthetic_series`) generates realistic 3–4 year
monthly NDVI trajectories (seasonal sine + noise + optional drift) with
`_inject_anomaly` for known ground-truth insertions.

---

## Configuration Reference

All defaults are in `config.yaml`. Key sections:

```yaml
gee:
  project: "vegetation-impact-analysis"  # Your GEE Cloud project ID

windows:
  pre_days: 60         # Days before event (minus buffer)
  post_days: 60        # Days after event (after buffer)
  buffer_days: 5       # Exclusion zone around the event date

satellite: "sentinel2"  # sentinel2 | landsat
index: "NDVI"           # NDVI | EVI | SAVI | NDMI

statistics:
  significance_level: 0.05
  sample_size: 500     # Pixels sampled for local statistics
  min_sample_size: 30  # Below this, skip inferential tests with a warning
  historical_years: 3  # Prior years for baseline comparison

thresholds:
  no_impact: -0.05
  low_impact: -0.15
  moderate_impact: -0.30   # < -0.30 → Severe Impact

processing:
  mask_water: false          # Set to true to exclude permanent water bodies (per-image)
  mask_water_threshold: 80   # JRC occurrence % above which a pixel is water
                             # (80 = open water excluded, mangroves / tidal flats retained)

# ── ROI Size Tiers (time series) ─────────────────────────────────────────────
# Tier 1: < 100 km²    → 250 m scale, proceed normally
# Tier 2: 100–500 km²  → 250 m scale, note that computation may take a few minutes
# Tier 3: 500–2000 km² → auto-scales to 500 m, warns user
# Tier 4: > 2000 km²   → 500 m scale, requires --force (CLI) or checkbox (Streamlit)

sensors_default: "optical"   # optical | optical,sar | optical,sar,gedi | optical,sar,gedi,palsar | all

# ── Sentinel-1 SAR ───────────────────────────────────────────────────────────
sar:
  orbit: "DESCENDING"           # IW mode; DESCENDING preferred for FL coastline
  speckle_filter_radius: 50     # focal_median radius in metres
  thresholds:
    rvi_change: -0.10           # ∆RVI threshold to flag structural change
    vv_change_db: -1.5          # VV dB-change threshold (negative = loss)
    vh_change_db: -1.5          # VH dB-change threshold (negative = loss)

# ── GEDI Lidar ───────────────────────────────────────────────────────────────
gedi:
  height_band: "rh95"           # canopy-top height (95th-percentile relative height)
  cover_band: "cover"           # tree cover fraction (0–1)
  sensitivity_threshold: 0.9   # minimum beam sensitivity for quality filtering
  operational_start: "2019-04-01"
  operational_end: "2023-03-31"

# ── ALOS PALSAR-2 ────────────────────────────────────────────────────────────
palsar_settings:
  calibration_offset: -83.0    # JAXA standard: dB = 20*log10(DN) - 83.0
  forest_threshold_hv_db: -15.0 # HV dB below which pixel is non-forest
  agb_A: 0.04                  # experimental AGB coefficient A
  agb_B: -2.0                  # experimental AGB coefficient B
                               # AGB (Mg/ha) ≈ 10^((HV_dB + 83) * A + B)
                               # !! needs regional calibration before use !!

palsar_thresholds:
  no_damage: -1.0              # ∆HV > -1.0 dB   → No Damage
  light_damage: -2.0           # -1.0 to -2.0 dB → Light Damage
  moderate_damage: -4.0        # -2.0 to -4.0 dB → Moderate Damage
                               # < -4.0 dB        → Severe Damage

export:
  scale_sentinel2: 10  # Native Sentinel-2 resolution (m)
  scale_landsat: 30    # Native Landsat resolution (m)
  crs: "EPSG:4326"

# Hurricane event catalog for time series chart annotations.
# Any event whose date falls within the analysis date range is automatically
# drawn as a dark-red dashed vertical line on every time series chart.
# Add future events here — no code changes required.
hurricane_events:
  - name: Irma
    date: "2017-09-10"
    category: 4
  - name: Michael
    date: "2018-10-10"
    category: 5
  - name: Ian
    date: "2022-09-28"
    category: 4
  - name: Idalia
    date: "2023-08-30"
    category: 3
  - name: Milton
    date: "2024-10-09"
    category: 4
```

---

## Dependencies

See `requirements.txt` for pinned versions. Core dependencies:

- `earthengine-api` — GEE Python client (also provides Sentinel-1, PALSAR, GEDI access)
- `geemap` — GEE interop (ee_export_image, folium maps)
- `geopandas`, `rasterio`, `shapely` — local vector/raster handling
- `scipy` — t-test, Wilcoxon, KDE
- `matplotlib`, `folium` — plots and maps
- `click` — CLI framework
- `streamlit`, `streamlit-folium` — dashboard
- `jinja2` — HTML report templating
- `pystac-client`, `stackstac` — Planetary Computer fallback

> **No additional dependencies** are required for SAR, PALSAR, or GEDI analysis —
> all three sensors are accessed through the standard `earthengine-api` client,
> the same as Sentinel-2 and Landsat.

---

## License

This project source code is licensed under the **MIT License**.

Usage of Google Earth Engine and all remote datasets is governed separately by
their own terms and licenses:

- Google Earth Engine platform access and usage terms
- Copernicus Sentinel-1 / Sentinel-2 data terms
- USGS Landsat Collection 2 terms
- JAXA ALOS PALSAR-2 terms
- NASA GEDI terms
