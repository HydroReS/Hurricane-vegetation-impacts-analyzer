"""
cli.py
======
Command-line interface for the Hurricane Vegetation Impact Analysis tool.

Usage
-----
::

    python cli.py analyze \\
        --roi bbox:-82.2,26.4,-81.7,26.8 \\
        --event-date 2022-09-28 \\
        --index NDVI \\
        --satellite sentinel2 \\
        --pre-days 60 \\
        --post-days 60 \\
        --buffer-days 5 \\
        --output-dir ./results/ian_fort_myers \\
        --report

    python cli.py analyze \\
        --roi file:region.geojson \\
        --event-date 2023-08-30 \\
        --index EVI \\
        --satellite sentinel2 \\
        --significance-level 0.01 \\
        --historical-years 3 \\
        --output-dir ./results/idalia \\
        --report

    python cli.py list-presets

Run ``python cli.py analyze --help`` for the full option list.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG_DEFAULT = Path(__file__).parent / "config.yaml"


def _load_and_merge_config(config_path: str, overrides: dict) -> dict:
    """Load YAML config then apply CLI overrides."""
    from src.utils import load_config

    cfg = load_config(config_path)

    # Apply overrides into nested sections
    if overrides.get("satellite"):
        cfg["satellite"] = overrides["satellite"]
    if overrides.get("index"):
        cfg["index"] = overrides["index"]

    windows = cfg.setdefault("windows", {})
    if overrides.get("pre_days") is not None:
        windows["pre_days"] = overrides["pre_days"]
    if overrides.get("post_days") is not None:
        windows["post_days"] = overrides["post_days"]
    if overrides.get("buffer_days") is not None:
        windows["buffer_days"] = overrides["buffer_days"]

    stats = cfg.setdefault("statistics", {})
    if overrides.get("significance_level") is not None:
        stats["significance_level"] = overrides["significance_level"]
    if overrides.get("historical_years") is not None:
        stats["historical_years"] = overrides["historical_years"]
    if overrides.get("sample_size") is not None:
        stats["sample_size"] = overrides["sample_size"]

    return cfg


def _print_results(results: dict) -> None:
    """Pretty-print the statistical results and impact classification to stdout."""
    click.echo()
    click.secho("=" * 60, fg="cyan")
    click.secho("  VEGETATION IMPACT ANALYSIS RESULTS", fg="cyan", bold=True)
    click.secho("=" * 60, fg="cyan")

    stat = results.get("statistics", {})
    if stat:
        click.echo(f"  Pixels sampled  : {stat.get('n', 'N/A')}")
        click.echo(f"  Pre mean {results['index']:5s}  : {stat.get('pre_mean', float('nan')):.4f}")
        click.echo(f"  Post mean {results['index']:5s} : {stat.get('post_mean', float('nan')):.4f}")
        delta_pct = stat.get("delta_pct")
        pct_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "N/A"
        click.echo(
            f"  Mean delta      : {stat.get('delta_mean', float('nan')):+.4f}  ({pct_str})"
        )
        click.echo()
        click.echo(f"  Paired t-test   : t = {stat.get('ttest_stat', 0):.4f}, "
                   f"p = {stat.get('ttest_pvalue', 1):.4f}")
        click.echo(f"  Wilcoxon test   : W = {stat.get('wilcoxon_stat', 0):.4f}, "
                   f"p = {stat.get('wilcoxon_pvalue', 1):.4f}")
        click.echo(f"  Cohen's d       : {stat.get('cohens_d', 0):.4f} "
                   f"({stat.get('effect_label', 'N/A')} effect)")
        ci = stat.get("ttest_ci", (float("nan"), float("nan")))
        click.echo(f"  95% CI (delta)  : [{ci[0]:.4f}, {ci[1]:.4f}]")

        click.echo()
        color = "red" if stat.get("significant") else "green"
        click.secho(f"  CONCLUSION: {stat.get('conclusion', '')}", fg=color, bold=True)

    click.echo()
    area = results.get("area_by_class", {})
    if area:
        click.secho("  Impact Classification (area):", bold=True)
        colors = {"No Impact": "green", "Low Impact": "yellow",
                  "Moderate Impact": "bright_red", "Severe Impact": "red"}
        for cls, km2 in area.items():
            click.secho(f"    {cls:<22} {km2:>8.2f} km²", fg=colors.get(cls, "white"))

    baseline = results.get("baseline", {})
    if baseline.get("interpretation"):
        click.echo()
        click.secho("  Baseline Variability Check:", bold=True)
        within = baseline.get("within_normal_range")
        color = "yellow" if within else "magenta"
        click.secho(f"  {baseline['interpretation']}", fg=color)

    # Multi-sensor structural results
    structural = results.get("structural", {})
    if structural:
        click.echo()
        click.secho("  Multi-Sensor Structural Analysis:", bold=True)
        if structural.get("sar_available"):
            click.secho("  ✓ Sentinel-1 SAR: composites and ∆RVI computed.", fg="cyan")
            if structural.get("concordance_img") is not None:
                click.secho("  ✓ Concordance map (optical + SAR) generated.", fg="cyan")
        elif "sar_error" in structural:
            click.secho(f"  ✗ SAR failed: {structural['sar_error']}", fg="yellow")
        if structural.get("gedi_available"):
            click.secho("  ✓ GEDI lidar: height and cover composites computed.", fg="cyan")
        elif "gedi_info" in structural:
            click.echo(f"  ℹ GEDI: {structural['gedi_info']}")
        elif "gedi_error" in structural:
            click.secho(f"  ✗ GEDI failed: {structural['gedi_error']}", fg="yellow")
        if structural.get("palsar_available"):
            ps = structural.get("palsar_stats", {})
            hv_delta = ps.get("hv_delta_mean")
            delta_str = f" (mean ΔHV = {hv_delta:+.2f} dB)" if hv_delta is not None else ""
            click.secho(f"  ✓ PALSAR L-band: pre/post mosaics and ΔHV computed{delta_str}.", fg="cyan")
            if structural.get("concordance_ext") is not None:
                click.secho("  ✓ Extended concordance (8-class optical+C+L-band) generated.", fg="cyan")
        elif "palsar_error" in structural:
            click.secho(f"  ✗ PALSAR failed: {structural['palsar_error']}", fg="yellow")

    # Scene-level acquisition summary
    scene_meta = results.get("scene_metadata", {})
    if scene_meta:
        click.echo()
        click.secho("  Image Acquisition Summary:", bold=True)
        sensor_labels = {
            "optical": "Optical",
            "sar":     "Sentinel-1 SAR",
            "palsar":  "PALSAR",
            "gedi":    "GEDI",
        }
        for key, label in sensor_labels.items():
            smeta = scene_meta.get(key)
            if not smeta:
                continue
            pre  = smeta.get("pre",  {})
            post = smeta.get("post", {})

            def _fmt_count(m):
                c = m.get("count")
                return str(c) if c is not None else "?"

            def _fmt_window(m, skey=key):
                if skey == "palsar":
                    yr = m.get("year")
                    return f"year {yr}" if yr else "—"
                if skey == "gedi":
                    ds = m.get("date_start")
                    de = m.get("date_end")
                    return f"{ds}→{de}" if ds else "—"
                return f"{m.get('window_start', '?')}→{m.get('window_end', '?')}"

            pre_str  = f"{_fmt_count(pre)} scene(s) [{_fmt_window(pre)}]"
            post_str = f"{_fmt_count(post)} scene(s) [{_fmt_window(post)}]"
            click.echo(f"    {label:<18}  pre: {pre_str}  |  post: {post_str}")

        warnings = scene_meta.get("warnings", [])
        if warnings:
            click.echo()
            click.secho("  Acquisition Warnings:", bold=True, fg="yellow")
            for w in warnings:
                click.secho(f"    ⚠  {w}", fg="yellow")

    click.echo()
    click.secho("=" * 60, fg="cyan")

    if results.get("pre_geotiff_path"):
        click.echo(f"  Pre GeoTIFF : {results['pre_geotiff_path']}")
    if results.get("post_geotiff_path"):
        click.echo(f"  Post GeoTIFF: {results['post_geotiff_path']}")
    if results.get("geotiff_path"):
        click.echo(f"  Diff GeoTIFF: {results['geotiff_path']}")
    click.echo(f"  Output      : {results.get('output_dir', 'N/A')}")
    click.echo()


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

@click.group()
@click.version_option("1.0.0", prog_name="hurricane-veg-analysis")
def cli():
    """
    Hurricane Vegetation Impact Analysis Tool

    Analyzes satellite-derived vegetation index changes before and after
    Florida hurricane events to detect storm surge and wind damage.
    """


@cli.command("analyze")
@click.option(
    "--roi", required=True,
    help=(
        "Region of interest. "
        "Format: 'bbox:W,S,E,N' (e.g. bbox:-82.2,26.4,-81.7,26.8) "
        "or 'file:path/to/region.geojson'."
    ),
)
@click.option(
    "--event-date", required=True,
    help="Hurricane landfall date (YYYY-MM-DD). E.g. 2022-09-28 for Hurricane Ian.",
)
@click.option(
    "--index", default="NDVI", show_default=True,
    type=click.Choice(["NDVI", "EVI", "SAVI", "NDMI"], case_sensitive=False),
    help="Vegetation index to compute.",
)
@click.option(
    "--satellite", default="sentinel2", show_default=True,
    type=click.Choice(["sentinel2", "landsat"], case_sensitive=False),
    help="Satellite dataset to use.",
)
@click.option("--pre-days", type=int, default=None, help="Days in the pre-event window (default from config).")
@click.option("--post-days", type=int, default=None, help="Days in the post-event window (default from config).")
@click.option("--buffer-days", type=int, default=None, help="Days to exclude around the event (default from config).")
@click.option(
    "--significance-level", type=float, default=None,
    help="Statistical significance threshold α (default 0.05).",
)
@click.option(
    "--historical-years", type=int, default=None,
    help="Years of historical baseline data to compare against (default 3).",
)
@click.option(
    "--sample-size", type=int, default=None,
    help="Number of pixels to sample for statistical tests (default 500).",
)
@click.option(
    "--output-dir", default="./results", show_default=True,
    help="Directory for output files (GeoTIFF, maps, report).",
)
@click.option(
    "--config", "config_path",
    default=str(_CONFIG_DEFAULT), show_default=True,
    help="Path to config.yaml.",
)
@click.option("--report", is_flag=True, default=False, help="Generate an HTML report.")
@click.option("--time-series", is_flag=True, default=False, help="Generate a time series plot (slower).")
@click.option("--mask-water", is_flag=True, default=False,
              help="Mask permanent water bodies (JRC GSW). Threshold set by processing.mask_water_threshold in config (default 80%). Recommended for coastal ROIs.")
@click.option("--sensors", default="optical", show_default=True,
              help=(
                  "Comma-separated list of sensors to use. "
                  "Choices: optical, sar, gedi, palsar, all. "
                  "E.g. 'optical,sar' adds Sentinel-1 C-band SAR analysis. "
                  "'optical,sar,palsar' adds both C- and L-band radar. "
                  "'all' enables SAR + GEDI + PALSAR (GEDI only available "
                  "2019-04-01 – 2023-03-31; PALSAR from 2014 onward)."
              ))
@click.option("--palsar-pre-year", type=int, default=None,
              help="Override PALSAR pre-event mosaic year (auto-selected when omitted).")
@click.option("--palsar-post-year", type=int, default=None,
              help="Override PALSAR post-event mosaic year (auto-selected when omitted).")
@click.option("--gee-project", default=None, help="GEE Cloud project ID (overrides config).")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose (DEBUG) logging.")
def analyze(
    roi, event_date, index, satellite, pre_days, post_days, buffer_days,
    significance_level, historical_years, sample_size, output_dir, config_path,
    report, time_series, mask_water, sensors, palsar_pre_year, palsar_post_year,
    gee_project, verbose,
):
    """
    Run the full vegetation impact analysis pipeline.

    Downloads satellite composites, computes a vegetation index, performs
    before/after comparison with statistical testing, classifies impact
    severity, and saves outputs to OUTPUT_DIR.

    \b
    Examples:
      # Hurricane Ian — Fort Myers, NDVI, Sentinel-2
      python cli.py analyze \\
        --roi bbox:-82.2,26.4,-81.7,26.8 \\
        --event-date 2022-09-28 \\
        --output-dir ./results/ian --report

      # Hurricane Michael — Landsat, EVI
      python cli.py analyze \\
        --roi bbox:-85.8,29.9,-85.3,30.3 \\
        --event-date 2018-10-10 \\
        --index EVI --satellite landsat \\
        --output-dir ./results/michael --report
    """
    from src.utils import setup_logging, ee_init, parse_roi, ensure_dir
    from src.analysis import run_analysis
    from src.visualization import (
        plot_distributions,
        create_difference_map,
        plot_time_series,
        generate_report,
    )

    setup_logging(verbose=verbose)
    logger = logging.getLogger("cli")

    # Load and merge config
    overrides = {
        "satellite": satellite,
        "index": index.upper(),
        "pre_days": pre_days,
        "post_days": post_days,
        "buffer_days": buffer_days,
        "significance_level": significance_level,
        "historical_years": historical_years,
        "sample_size": sample_size,
    }
    try:
        cfg = _load_and_merge_config(config_path, overrides)
    except Exception as exc:
        click.secho(f"ERROR loading config: {exc}", fg="red", err=True)
        sys.exit(1)

    # Apply processing overrides
    if mask_water:
        cfg.setdefault("processing", {})["mask_water"] = True

    # Override GEE project if provided
    if gee_project:
        cfg.setdefault("gee", {})["project"] = gee_project

    project = cfg.get("gee", {}).get("project", "vegetation-impact-analysis")

    # GEE initialization
    click.echo(f"Initializing Google Earth Engine (project: {project}) …")
    try:
        ee_init(project=project)
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(1)

    # Parse ROI
    click.echo(f"Parsing ROI: {roi}")
    try:
        roi_geom = parse_roi(roi)
    except Exception as exc:
        click.secho(f"ERROR parsing ROI: {exc}", fg="red", err=True)
        sys.exit(1)

    # Ensure output directory exists
    out_dir = ensure_dir(output_dir)
    click.echo(f"Output directory: {out_dir}")

    # Run the analysis
    click.echo(f"\nStarting analysis: {index} | {satellite} | event {event_date}\n")
    try:
        results = run_analysis(
            roi=roi_geom,
            event_date=event_date,
            satellite=satellite.lower(),
            index=index.upper(),
            output_dir=str(out_dir),
            config=cfg,
            sensors=sensors,
            palsar_pre_year=palsar_pre_year,
            palsar_post_year=palsar_post_year,
        )
        results["config"] = cfg  # pass config for report context
    except Exception as exc:
        click.secho(f"\nERROR during analysis: {exc}", fg="red", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print results
    _print_results(results)

    # Distribution plot
    dist_path = str(out_dir / f"{index}_distribution.png")
    try:
        plot_distributions(
            results["pre_vals"], results["post_vals"],
            index=index.upper(),
            output_path=dist_path,
            event_date=event_date,
        )
        click.echo(f"Distribution plot: {dist_path}")
    except Exception as exc:
        logger.warning("Distribution plot failed: %s", exc)
        dist_path = ""

    # Interactive map
    map_path = str(out_dir / "difference_map.html")
    try:
        create_difference_map(
            pre_img=results["pre_img"],
            post_img=results["post_img"],
            diff_img=results["diff_img"],
            classified_img=results["classified_img"],
            roi=roi_geom,
            index=index.upper(),
            output_path=map_path,
            thresholds=cfg.get("thresholds"),
        )
        click.echo(f"Interactive map : {map_path}")
    except Exception as exc:
        logger.warning("Map generation failed: %s", exc)
        map_path = ""

    # Multi-sensor maps (SAR change + concordance)
    structural = results.get("structural", {})
    if structural.get("sar_available"):
        from src.visualization import create_sar_change_map, create_concordance_map

        sar_map_path = str(out_dir / "sar_change_map.html")
        try:
            create_sar_change_map(
                structural["pre_sar"], structural["post_sar"],
                structural["diff_sar"], roi_geom, sar_map_path,
            )
            click.echo(f"SAR change map  : {sar_map_path}")
        except Exception as exc:
            logger.warning("SAR map generation failed: %s", exc)

        if structural.get("concordance_img") is not None:
            conc_map_path = str(out_dir / "concordance_map.html")
            try:
                create_concordance_map(
                    structural["concordance_img"], results["diff_img"],
                    roi_geom, conc_map_path,
                    event_date=event_date, index=index.upper(),
                )
                click.echo(f"Concordance map : {conc_map_path}")
            except Exception as exc:
                logger.warning("Concordance map generation failed: %s", exc)

    if structural.get("palsar_available"):
        from src.visualization import (
            create_palsar_change_map,
            create_extended_concordance_map,
        )
        palsar_map_path = str(out_dir / "palsar_change_map.html")
        try:
            create_palsar_change_map(
                structural["pre_palsar"], structural["post_palsar"],
                structural["diff_palsar"], roi_geom, palsar_map_path,
                event_date=event_date,
            )
            click.echo(f"PALSAR HV map   : {palsar_map_path}")
        except Exception as exc:
            logger.warning("PALSAR map generation failed: %s", exc)

        if structural.get("concordance_ext") is not None:
            ext_conc_path = str(out_dir / "concordance_ext_map.html")
            try:
                create_extended_concordance_map(
                    structural["concordance_ext"], roi_geom, ext_conc_path,
                    event_date=event_date, index=index.upper(),
                )
                click.echo(f"Extended concordance map: {ext_conc_path}")
            except Exception as exc:
                logger.warning("Extended concordance map generation failed: %s", exc)

    # Time series (optional, slow)
    ts_path = ""
    if time_series:
        ts_path = str(out_dir / f"{index}_time_series.png")
        try:
            plot_time_series(
                roi=roi_geom,
                event_date=event_date,
                satellite=satellite.lower(),
                index=index.upper(),
                output_path=ts_path,
            )
            click.echo(f"Time series plot: {ts_path}")
        except Exception as exc:
            logger.warning("Time series plot failed: %s", exc)
            ts_path = ""

    # HTML report
    if report:
        try:
            report_path = generate_report(
                results=results,
                output_dir=str(out_dir),
                dist_plot_path=dist_path,
                ts_plot_path=ts_path,
                map_html_path=map_path,
            )
            click.secho(f"\nHTML Report     : {report_path}", fg="green", bold=True)
        except Exception as exc:
            logger.warning("Report generation failed: %s", exc)

    click.secho("\nAnalysis complete!", fg="green", bold=True)


@cli.command("list-presets")
@click.option(
    "--config", "config_path",
    default=str(_CONFIG_DEFAULT), show_default=True,
    help="Path to config.yaml.",
)
def list_presets(config_path):
    """List available Florida hurricane presets from the configuration."""
    from src.utils import load_config

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(1)

    hurricanes = cfg.get("hurricanes", {})
    if not hurricanes:
        click.echo("No presets found in config.yaml.")
        return

    click.secho("\nAvailable Florida Hurricane Presets:", bold=True)
    click.secho("-" * 60)
    for key, info in hurricanes.items():
        click.secho(f"  {key.upper():12s}", fg="cyan", bold=True, nl=False)
        click.echo(f" {info.get('date', 'N/A')}  {info.get('description', '')}")
        bbox = info.get("bbox", [])
        if bbox:
            bbox_str = ",".join(str(v) for v in bbox)
            click.echo(f"    ROI bbox: {bbox_str}")
        notes = info.get("notes", "")
        if notes:
            click.echo(f"    Note: {notes}")
        click.echo()

    click.secho("\nExample usage:", bold=True)
    first_key = next(iter(hurricanes))
    first = hurricanes[first_key]
    bbox_str = ",".join(str(v) for v in first.get("bbox", []))
    click.echo(
        f"  python cli.py analyze \\\n"
        f"    --roi bbox:{bbox_str} \\\n"
        f"    --event-date {first.get('date', 'YYYY-MM-DD')} \\\n"
        f"    --index NDVI --satellite sentinel2 \\\n"
        f"    --output-dir ./results/{first_key} --report\n"
    )


@cli.command("run-preset")
@click.argument("preset_name", type=str)
@click.option("--index", default="NDVI", show_default=True,
              type=click.Choice(["NDVI", "EVI", "SAVI", "NDMI"], case_sensitive=False))
@click.option("--satellite", default="sentinel2", show_default=True,
              type=click.Choice(["sentinel2", "landsat"], case_sensitive=False))
@click.option("--output-dir", default="./results", show_default=True)
@click.option("--config", "config_path", default=str(_CONFIG_DEFAULT))
@click.option("--report", is_flag=True, default=False)
@click.option("--time-series", is_flag=True, default=False)
@click.option("--mask-water", is_flag=True, default=False,
              help="Mask permanent water bodies (JRC GSW).")
@click.option("--sensors", default="optical", show_default=True,
              help="Comma-separated sensors: optical, sar, gedi, palsar, all.")
@click.option("--palsar-pre-year", type=int, default=None,
              help="Override PALSAR pre-event mosaic year.")
@click.option("--palsar-post-year", type=int, default=None,
              help="Override PALSAR post-event mosaic year.")
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.pass_context
def run_preset(ctx, preset_name, index, satellite, output_dir, config_path,
               report, time_series, mask_water, sensors, palsar_pre_year,
               palsar_post_year, verbose):
    """
    Run the analysis for a named Florida hurricane preset.

    PRESET_NAME is one of: ian, michael, idalia, irma, helene, milton
    (or any entry in config.yaml hurricanes section).

    \b
    Examples:
      python cli.py run-preset ian --report
      python cli.py run-preset michael --index EVI --satellite landsat
      python cli.py run-preset irma --sensors optical,sar,palsar --mask-water --report
      python cli.py run-preset helene --sensors all --mask-water --report
    """
    from src.utils import load_config

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(1)

    preset = cfg.get("hurricanes", {}).get(preset_name.lower())
    if not preset:
        available = list(cfg.get("hurricanes", {}).keys())
        click.secho(
            f"ERROR: Preset '{preset_name}' not found. "
            f"Available: {available}", fg="red", err=True
        )
        sys.exit(1)

    bbox = preset["bbox"]
    bbox_str = ",".join(str(v) for v in bbox)
    event_date = preset["date"]
    out_dir = os.path.join(output_dir, preset_name.lower())

    click.secho(f"Running preset: {preset_name.upper()}", bold=True)
    click.echo(f"  {preset.get('description', '')}")
    click.echo(f"  Date: {event_date}  |  ROI: [{bbox_str}]")

    # Delegate to the 'analyze' command
    ctx.invoke(
        analyze,
        roi=f"bbox:{bbox_str}",
        event_date=event_date,
        index=index,
        satellite=satellite,
        pre_days=None,
        post_days=None,
        buffer_days=None,
        significance_level=None,
        historical_years=None,
        sample_size=None,
        output_dir=out_dir,
        config_path=config_path,
        report=report,
        time_series=time_series,
        mask_water=mask_water,
        sensors=sensors,
        palsar_pre_year=palsar_pre_year,
        palsar_post_year=palsar_post_year,
        gee_project=None,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# timeseries command
# ---------------------------------------------------------------------------

@cli.command("timeseries")
@click.option("--point", "point_str", default=None,
              help="Point location as LAT,LON (e.g. 26.45,-81.95).")
@click.option("--roi", "roi_str", default=None,
              help="ROI as bbox:W,S,E,N or file:path.geojson (spatial mean).")
@click.option("--multi-point", "multi_point_csv", default=None,
              help="CSV with lat,lon,label columns for multi-point analysis.")
@click.option("--start-date", required=True, help="Start date YYYY-MM-DD.")
@click.option("--end-date", required=True, help="End date YYYY-MM-DD.")
@click.option("--index", default="NDVI", show_default=True,
              type=click.Choice(["NDVI", "EVI", "SAVI", "NDMI"], case_sensitive=False))
@click.option("--satellite", default="sentinel2", show_default=True,
              type=click.Choice(["sentinel2", "landsat"], case_sensitive=False))
@click.option("--composite", default="monthly", show_default=True,
              type=click.Choice(["raw", "weekly", "biweekly", "monthly"]),
              help="Temporal compositing interval.")
@click.option("--anomaly-method", "anomaly_method", default="all", show_default=True,
              type=click.Choice(["zscore", "moving_window", "climatology", "all"]),
              help="Anomaly detection method.")
@click.option("--anomaly-threshold", type=float, default=2.0, show_default=True,
              help="Z-score threshold for anomaly flagging.")
@click.option("--detect-changepoints", is_flag=True, default=False,
              help="Run CUSUM change point detection.")
@click.option("--event-date", default=None,
              help="Hurricane/event date YYYY-MM-DD (for annotation and recovery).")
@click.option("--recovery-analysis", is_flag=True, default=False,
              help="Run recovery analysis (requires --event-date).")
@click.option("--recovery-style", "recovery_style", default="seasonal", show_default=True,
              type=click.Choice(["seasonal", "flat", "all"], case_sensitive=False),
              help=(
                  "Recovery baseline: 'seasonal' (default) = monthly climatology ±1σ envelope; "
                  "'flat' = legacy flat pre-event mean ±1σ; "
                  "'all' = generate both plots."
              ))
@click.option("--output-dir", default="./results/timeseries", show_default=True,
              help="Directory for output files.")
@click.option("--plot-type", "plot_type", default="all", show_default=True,
              type=click.Choice(["raw", "residual", "zscore", "departure", "cusum", "all"],
                                case_sensitive=False),
              help=(
                  "Which plot families to generate. "
                  "'raw' = existing plots only; "
                  "'residual'/'zscore'/'departure'/'cusum' = one detrended view; "
                  "'all' = raw + all four detrended types + combined panel."
              ))
@click.option("--config", "config_path", default=str(_CONFIG_DEFAULT), show_default=True)
@click.option("--gee-project", default=None, help="GEE Cloud project ID.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def timeseries(
    point_str, roi_str, multi_point_csv,
    start_date, end_date, index, satellite, composite,
    anomaly_method, anomaly_threshold, detect_changepoints,
    event_date, recovery_analysis, recovery_style, output_dir, plot_type,
    config_path, gee_project, verbose,
):
    """
    Extract and analyse a vegetation index time series for a point or ROI.

    Applies seasonal decomposition, anomaly detection, and optional change
    point / recovery analysis over a user-defined date range.

    \b
    Examples:
      # Monthly NDVI time series at a point near Fort Myers
      python cli.py timeseries \\
        --point 26.45,-81.95 \\
        --start-date 2020-01-01 --end-date 2023-12-31 \\
        --event-date 2022-09-28 --recovery-analysis

      # Spatial mean over an ROI with change point detection
      python cli.py timeseries \\
        --roi bbox:-82.2,26.4,-81.7,26.8 \\
        --start-date 2019-01-01 --end-date 2024-12-31 \\
        --composite monthly --detect-changepoints \\
        --event-date 2022-09-28

      # Multi-point comparison from a CSV
      python cli.py timeseries \\
        --multi-point points.csv \\
        --start-date 2021-01-01 --end-date 2024-12-31
    """
    from src.utils import setup_logging, ee_init, parse_roi, ensure_dir, load_config
    from src.time_series import (
        run_time_series_analysis,
        extract_multi_point_time_series,
        plot_multi_point_comparison,
        points_from_csv,
    )

    setup_logging(verbose=verbose)
    logger = logging.getLogger("cli.timeseries")

    # Load config
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        click.secho(f"ERROR loading config: {exc}", fg="red", err=True)
        sys.exit(1)

    project = gee_project or cfg.get("gee", {}).get("project", "vegetation-impact-analysis")
    click.echo(f"Initializing GEE (project: {project}) …")
    try:
        ee_init(project=project)
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(1)

    out_dir = ensure_dir(output_dir)
    methods = [anomaly_method] if anomaly_method != "all" else ["all"]

    # ── Multi-point mode ──────────────────────────────────────────────────
    if multi_point_csv:
        try:
            points = points_from_csv(multi_point_csv)
        except Exception as exc:
            click.secho(f"ERROR reading CSV: {exc}", fg="red", err=True)
            sys.exit(1)

        click.echo(f"Running multi-point analysis for {len(points)} locations …")
        point_data = extract_multi_point_time_series(
            points, start_date, end_date,
            satellite=satellite.lower(), index=index.upper(),
            composite=composite,
        )

        mp_path = str(out_dir / f"{index}_multi_point.png")
        try:
            plot_multi_point_comparison(point_data, index.upper(), mp_path, event_date)
            click.secho(f"Multi-point plot: {mp_path}", fg="green")
        except Exception as exc:
            logger.warning("Multi-point plot failed: %s", exc)

        # Summary table
        click.echo()
        click.secho("  Mean index value per location:", bold=True)
        for label, df in point_data.items():
            if not df.empty:
                click.echo(f"  {label:<35} mean={df['index_value'].mean():.4f}")
        return

    # ── Single point or ROI mode ──────────────────────────────────────────
    if point_str is None and roi_str is None:
        click.secho(
            "ERROR: Provide --point LAT,LON, --roi bbox:..., or --multi-point CSV.",
            fg="red", err=True,
        )
        sys.exit(1)

    if point_str:
        try:
            lat_s, lon_s = point_str.split(",")
            location = (float(lat_s), float(lon_s))
        except ValueError:
            click.secho("ERROR: --point must be LAT,LON (e.g. 26.45,-81.95).",
                        fg="red", err=True)
            sys.exit(1)
    else:
        try:
            location = parse_roi(roi_str)
        except Exception as exc:
            click.secho(f"ERROR parsing ROI: {exc}", fg="red", err=True)
            sys.exit(1)

    # Filter hurricane catalog to analysis date range
    all_hurricanes = cfg.get("hurricane_events", [])
    visible_hurricanes = [
        he for he in all_hurricanes
        if start_date <= str(he.get("date", "")) <= end_date
    ]

    click.echo(f"\nRunning time series analysis: {index} | {satellite} | {composite}\n")
    try:
        ts_results = run_time_series_analysis(
            location=location,
            start_date=start_date,
            end_date=end_date,
            satellite=satellite.lower(),
            index=index.upper(),
            composite=composite,
            anomaly_methods=methods,
            anomaly_threshold=anomaly_threshold,
            detect_changepoints=detect_changepoints,
            event_date=event_date,
            recovery_analysis=recovery_analysis,
            recovery_style=recovery_style,
            output_dir=str(out_dir),
            hurricane_events=visible_hurricanes,
            plot_type=plot_type,
        )
    except Exception as exc:
        click.secho(f"\nERROR: {exc}", fg="red", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print summary
    df = ts_results["df"]
    anomalies = ts_results["anomalies"]
    click.secho("=" * 60, fg="cyan")
    click.secho("  TIME SERIES ANALYSIS RESULTS", fg="cyan", bold=True)
    click.secho("=" * 60, fg="cyan")
    click.echo(f"  Location  : {ts_results['location_label']}")
    click.echo(f"  Period    : {start_date} → {end_date}")
    click.echo(f"  Index     : {index.upper()} | {satellite} | {composite}")
    click.echo(f"  Obs count : {len(df)}")
    click.echo(f"  Mean      : {df['index_value'].mean():.4f}")
    click.echo(f"  Std dev   : {df['index_value'].std():.4f}")
    click.echo()

    if not anomalies.empty:
        click.secho(f"  Anomalies detected: {len(anomalies)}", fg="yellow", bold=True)
        for method in anomalies["method"].unique():
            sub = anomalies[anomalies["method"] == method]
            click.echo(f"    {method}: {len(sub)} anomalies")
            for _, row in sub.iterrows():
                click.secho(
                    f"      {str(row['date'])[:10]}  "
                    f"obs={row['observed_value']:.4f}  "
                    f"z={row['z_score']:+.2f}  [{row['severity']}]",
                    fg="red" if "extreme" in str(row["severity"]) else "yellow",
                )
    else:
        click.secho("  No anomalies detected.", fg="green")

    changepoints = ts_results.get("changepoints")
    if changepoints is not None and not changepoints.empty:
        click.echo()
        click.secho(f"  Change points: {len(changepoints)}", bold=True)
        for _, row in changepoints.iterrows():
            click.echo(
                f"    {str(row['date'])[:10]}  direction={row['direction']}  "
                f"magnitude={row['magnitude']:.4f}"
            )

    recovery = ts_results.get("recovery", {})
    if recovery and recovery.get("interpretation"):
        click.echo()
        click.secho("  Recovery Analysis:", bold=True)
        within = recovery.get("recovery_status", "")
        color = "green" if within == "full_recovery" else "yellow"
        click.secho(f"  {recovery['interpretation']}", fg=color)

    click.echo()
    click.secho("=" * 60, fg="cyan")
    for key, path in ts_results.get("plot_paths", {}).items():
        click.echo(f"  {key:<25}: {path}")
    click.secho("\nTime series analysis complete!", fg="green", bold=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
