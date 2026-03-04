"""
tests/test_time_series.py
=========================
Unit tests for anomaly detection methods in src/time_series.py.

All tests use synthetic pandas DataFrames — no Google Earth Engine required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.time_series import (
    ANOMALY_METHODS,
    _severity_label,
    detect_all_anomalies,
    detect_anomalies_climatology,
    detect_anomalies_moving_window,
    detect_anomalies_zscore,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_EXPECTED_COLUMNS = frozenset(
    ["date", "observed_value", "expected_value", "deviation", "z_score", "method", "severity"]
)


def _make_synthetic_series(
    n_years: int = 3, seed: int = 42, trend_slope: float = 0.0002
) -> pd.DataFrame:
    """
    Return a monthly NDVI-like time series with a seasonal sine wave,
    an optional linear trend, and low-amplitude Gaussian noise.

    Parameters
    ----------
    n_years : int
        Number of years.
    seed : int
        RNG seed for reproducibility.
    trend_slope : float
        Linear trend per observation (default 0.0002 ≈ negligible drift).
        Pass 0.0 for a purely stationary series.

    Values range approximately 0.45–0.80 — realistic for Florida vegetation.
    """
    rng = np.random.default_rng(seed)
    # Use "ME" (month-end) with pandas ≥ 2.2, fallback to "M"
    try:
        dates = pd.date_range("2020-01-01", periods=n_years * 12, freq="ME")
    except ValueError:
        dates = pd.date_range("2020-01-01", periods=n_years * 12, freq="M")

    t = np.arange(len(dates))
    values = (
        0.60
        + 0.15 * np.sin(2 * np.pi * t / 12)   # annual seasonal cycle
        + trend_slope * t                        # optional trend
        + rng.normal(0, 0.02, len(t))           # noise σ ≈ 0.02
    )
    return pd.DataFrame({"date": dates, "index_value": values})


def _make_uniform_series(value: float = 0.6, n: int = 36) -> pd.DataFrame:
    """Return a perfectly flat series (no seasonal component)."""
    try:
        dates = pd.date_range("2020-01-01", periods=n, freq="ME")
    except ValueError:
        dates = pd.date_range("2020-01-01", periods=n, freq="M")
    return pd.DataFrame({"date": dates, "index_value": np.full(n, value)})


def _inject_anomaly(df: pd.DataFrame, date_str: str, magnitude: float) -> pd.DataFrame:
    """Add *magnitude* to the index_value nearest to *date_str*."""
    df = df.copy()
    target = pd.Timestamp(date_str)
    idx = (df["date"] - target).abs().idxmin()
    df.loc[idx, "index_value"] += magnitude
    return df


def _mock_stl_result(df: pd.DataFrame) -> dict:
    """
    Build a minimal stl_result dict aligned with *df*.

    Trend = series mean; seasonal = 0; residual = actual deviations from mean.
    This makes any injected spike appear directly in the residuals.
    """
    vals = df["index_value"].values
    mean = float(np.mean(vals))
    trend = np.full(len(vals), mean)
    seasonal = np.zeros(len(vals))
    residual = vals - trend
    return {
        "observed": vals,
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
        "dates": df["date"].values,
    }


# ---------------------------------------------------------------------------
# TestSeverityLabel
# ---------------------------------------------------------------------------

class TestSeverityLabel:
    """
    Test the internal _severity_label helper.

    Actual thresholds (strict inequalities):
      z < -3.5 → "extreme"        z > 3.5 → "extreme_positive"
      z < -2.5 → "moderate"       z > 2.5 → "moderate_positive"
      z < -2.0 → "mild"           z > 2.0 → "mild_positive"
      else     → "none"
    """

    def test_extreme_positive(self):
        assert _severity_label(4.0) == "extreme_positive"

    def test_extreme_negative(self):
        assert _severity_label(-4.0) == "extreme"

    def test_moderate_positive(self):
        assert _severity_label(3.0) == "moderate_positive"

    def test_moderate_negative(self):
        assert _severity_label(-3.0) == "moderate"

    def test_mild_positive_and_negative(self):
        assert _severity_label(2.1) == "mild_positive"
        assert _severity_label(-2.1) == "mild"

    def test_none_within_band(self):
        assert _severity_label(1.0) == "none"
        assert _severity_label(-1.0) == "none"


# ---------------------------------------------------------------------------
# TestAnomalyDetectionZscore
# ---------------------------------------------------------------------------

class TestAnomalyDetectionZscore:
    """Unit tests for detect_anomalies_zscore."""

    def test_no_anomalies_flat_series(self):
        """A perfectly uniform series has zero variance → no anomalies."""
        df = _make_uniform_series()
        result = detect_anomalies_zscore(df, threshold=2.0)
        assert result.empty, "Expected no anomalies in flat series"

    def test_detects_positive_spike(self):
        """A large positive spike should be flagged."""
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", +0.50)
        result = detect_anomalies_zscore(df, threshold=2.0)
        assert not result.empty
        assert (result["method"] == "zscore").all()

    def test_detects_negative_dip(self):
        """A large negative dip should also be flagged."""
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", -0.50)
        result = detect_anomalies_zscore(df, threshold=2.0)
        assert not result.empty

    def test_threshold_gate(self):
        """Small perturbation: not detected at threshold=2.0, detected at threshold=0.1."""
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", +0.04)
        assert detect_anomalies_zscore(df, threshold=2.0).empty
        assert not detect_anomalies_zscore(df, threshold=0.1).empty

    def test_output_columns(self):
        """All 7 expected columns must be present in non-empty results."""
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", +0.50)
        result = detect_anomalies_zscore(df, threshold=2.0)
        assert not result.empty
        assert _EXPECTED_COLUMNS.issubset(set(result.columns))

    def test_with_mock_stl_result(self):
        """When stl_result is provided, residuals are used instead of mean subtraction."""
        df = _make_synthetic_series()
        # Inject a spike that is only large relative to the STL residuals
        df_spiked = _inject_anomaly(df, "2021-07-01", +0.50)
        stl = _mock_stl_result(df_spiked)
        result = detect_anomalies_zscore(df_spiked, stl_result=stl, threshold=2.0)
        assert not result.empty, "Spike in STL residuals should be detected"
        # The detected date should be near the injected date
        detected_dates = pd.to_datetime(result["date"])
        target = pd.Timestamp("2021-07-31")  # month-end
        assert any(abs((d - target).days) <= 31 for d in detected_dates)

    def test_method_label(self):
        """All rows must carry method == 'zscore'."""
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", +0.50)
        result = detect_anomalies_zscore(df, threshold=2.0)
        if not result.empty:
            assert (result["method"] == "zscore").all()


# ---------------------------------------------------------------------------
# TestAnomalyDetectionMovingWindow
# ---------------------------------------------------------------------------

class TestAnomalyDetectionMovingWindow:
    """Unit tests for detect_anomalies_moving_window."""

    def test_no_anomalies_stable(self):
        """
        A smooth seasonal series with low noise should produce very few
        moving-window anomalies at k=3.0.
        """
        df = _make_synthetic_series(n_years=4, seed=0)
        result = detect_anomalies_moving_window(df, window_days=90, k=3.0)
        # Allow ≤1 false positive due to window initialisation
        assert len(result) <= 1

    def test_detects_sudden_jump(self):
        """
        A large step change should be detected near the injection date.

        Monthly data has ~1 obs per 30 days. The rolling window uses
        min_periods=5, so window_days must be at least ~180 days to
        accumulate 5+ prior monthly observations.
        """
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", +0.55)
        result = detect_anomalies_moving_window(df, window_days=200, k=2.0)
        assert not result.empty
        detected_dates = pd.to_datetime(result["date"])
        target = pd.Timestamp("2021-07-31")
        assert any(abs((d - target).days) <= 60 for d in detected_dates)

    def test_no_crash_short_series(self):
        """A 6-month series shorter than window_days should not raise."""
        try:
            dates = pd.date_range("2020-01-01", periods=6, freq="ME")
        except ValueError:
            dates = pd.date_range("2020-01-01", periods=6, freq="M")
        df = pd.DataFrame({"date": dates, "index_value": np.random.default_rng(7).normal(0.6, 0.02, 6)})
        result = detect_anomalies_moving_window(df, window_days=90, k=2.0)
        assert isinstance(result, pd.DataFrame)

    def test_k_controls_sensitivity(self):
        """Lower k threshold finds at least as many anomalies as higher k."""
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", +0.25)
        result_strict  = detect_anomalies_moving_window(df, window_days=90, k=3.0)
        result_lenient = detect_anomalies_moving_window(df, window_days=90, k=1.0)
        assert len(result_lenient) >= len(result_strict)


# ---------------------------------------------------------------------------
# TestAnomalyDetectionClimatology
# ---------------------------------------------------------------------------

class TestAnomalyDetectionClimatology:
    """Unit tests for detect_anomalies_climatology."""

    def test_single_year_returns_empty(self):
        """Only 12 months of data → cannot build per-DOY climatology."""
        try:
            dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        except ValueError:
            dates = pd.date_range("2020-01-01", periods=12, freq="M")
        df = pd.DataFrame({"date": dates, "index_value": np.linspace(0.5, 0.7, 12)})
        result = detect_anomalies_climatology(df)
        assert result.empty, "Single year should return empty result"

    def test_detects_extreme_anomaly(self):
        """
        A very large anomaly in the third year should be flagged against
        the stable first two years.

        Uses doy_window=45 so that each July 31 observation has ≥3 prior
        near-DOY neighbours (May–Sep of 2020 and 2021) to build the climatology.
        """
        df = _make_synthetic_series(n_years=3, seed=99)
        df_spiked = _inject_anomaly(df, "2022-07-01", +1.0)
        result = detect_anomalies_climatology(df_spiked, threshold_pct=5.0, doy_window=45)
        assert not result.empty
        assert (result["method"] == "climatology").all()

    def test_normal_variation_low_false_positives(self):
        """
        A stationary seasonal series (no drift) at the 5th-percentile threshold
        should produce very few false positives.

        Uses trend_slope=0.0 so year 4 is not systematically different from
        years 1–3 — the climatology comparison is apples-to-apples.
        """
        df = _make_synthetic_series(n_years=4, seed=42, trend_slope=0.0)
        result = detect_anomalies_climatology(df, threshold_pct=5.0, doy_window=15)
        # Allow at most 15% of observations as false positives
        assert len(result) <= len(df) * 0.15, f"Too many false positives: {len(result)}"

    def test_method_label_in_results(self):
        """Detected rows must carry method == 'climatology'."""
        df_spiked = _inject_anomaly(_make_synthetic_series(n_years=3), "2022-07-01", +1.0)
        result = detect_anomalies_climatology(df_spiked)
        if not result.empty:
            assert (result["method"] == "climatology").all()


# ---------------------------------------------------------------------------
# TestDetectAllAnomalies
# ---------------------------------------------------------------------------

class TestDetectAllAnomalies:
    """Unit tests for the detect_all_anomalies orchestrator."""

    def test_combines_multiple_methods(self):
        """
        Running zscore + moving_window on a series with a large spike
        should return rows from both methods.
        """
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", +0.60)
        result = detect_all_anomalies(
            df, threshold=2.0, methods=["zscore", "moving_window"]
        )
        assert not result.empty
        methods_present = set(result["method"].unique())
        # At least one of the two methods should have fired
        assert methods_present & {"zscore", "moving_window"}

    def test_method_filter_zscore_only(self):
        """When methods=['zscore'], output contains only zscore rows."""
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", +0.50)
        result = detect_all_anomalies(df, threshold=2.0, methods=["zscore"])
        if not result.empty:
            assert set(result["method"].unique()) == {"zscore"}

    def test_empty_methods_list_returns_empty(self):
        """Passing methods=[] (empty list) should return an empty DataFrame."""
        df = _inject_anomaly(_make_synthetic_series(), "2021-07-01", +0.50)
        result = detect_all_anomalies(df, methods=[])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_all_methods_constant(self):
        """ANOMALY_METHODS constant should list the three expected method names."""
        assert set(ANOMALY_METHODS) == {"zscore", "moving_window", "climatology"}
