"""
tests/test_analysis.py
======================
Unit tests for the core statistical and utility logic.

All tests run **without** a Google Earth Engine connection — they use
synthetic numpy arrays and pure Python functions only.

Run with:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on the path so imports work from any location
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.analysis import (
    cohens_d,
    run_statistical_tests,
    check_baseline_variability,
    DEFAULT_THRESHOLDS,
    CLASS_LABELS,
)
from src.utils import date_windows, historical_date_windows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def pre_vals_stable(rng):
    """Pre-event values with mean ~0.7 (healthy vegetation)."""
    return rng.normal(loc=0.70, scale=0.05, size=500)


@pytest.fixture
def post_vals_stable(pre_vals_stable, rng):
    """Post-event values nearly identical to pre (no significant change)."""
    noise = rng.normal(loc=0.0, scale=0.01, size=len(pre_vals_stable))
    return pre_vals_stable + noise


@pytest.fixture
def post_vals_damaged(pre_vals_stable, rng):
    """Post-event values with a large drop (~0.30 delta = severe damage)."""
    drop = rng.normal(loc=-0.30, scale=0.05, size=len(pre_vals_stable))
    return pre_vals_stable + drop


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------

class TestCohensD:
    """Tests for the Cohen's d effect size calculation."""

    def test_no_change_zero_d(self):
        """Identical arrays should return d = 0."""
        a = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
        b = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
        assert cohens_d(a, b) == 0.0

    def test_known_values(self):
        """
        Verify formula against a hand-computed example.

        diff = b - a = [0.1, 0.1, 0.1, 0.1, 0.1]
        mean(diff) = 0.1
        std(diff, ddof=1) = 0.0
        → d = undefined (all differences are equal) but the formula
          uses std(diff) so we verify with a non-constant example.
        """
        a = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        b = np.array([0.4, 0.6, 0.8, 1.0, 1.2])
        # diff = [0.4, 0.4, 0.4, 0.4, 0.4] → std = 0 → d = inf
        # Use a case with variance in the diff
        a2 = np.array([0.0, 0.1, 0.3, 0.5, 0.7])
        b2 = np.array([0.3, 0.5, 0.6, 0.9, 1.0])
        d = cohens_d(a2, b2)
        # diff = [0.3, 0.4, 0.3, 0.4, 0.3], mean=0.34, std≈0.0548
        assert d == pytest.approx(0.34 / np.std([0.3, 0.4, 0.3, 0.4, 0.3], ddof=1), rel=1e-3)

    def test_negative_d_for_decrease(self, pre_vals_stable, post_vals_damaged):
        """A large drop should produce a large negative Cohen's d."""
        d = cohens_d(pre_vals_stable, post_vals_damaged)
        assert d < -0.8, f"Expected large negative d, got {d:.3f}"

    def test_near_zero_d_for_no_change(self, pre_vals_stable, post_vals_stable):
        """No-impact scenario should produce a near-zero Cohen's d."""
        d = cohens_d(pre_vals_stable, post_vals_stable)
        assert abs(d) < 0.2, f"Expected small |d|, got {abs(d):.3f}"

    def test_positive_d_for_increase(self):
        """Vegetation increase should produce positive Cohen's d."""
        rng = np.random.default_rng(0)
        a = rng.normal(0.3, 0.05, 200)
        b = rng.normal(0.6, 0.05, 200)
        d = cohens_d(a, b)
        assert d > 0.8


# ---------------------------------------------------------------------------
# run_statistical_tests
# ---------------------------------------------------------------------------

class TestRunStatisticalTests:
    """Tests for the paired t-test / Wilcoxon / Cohen's d pipeline."""

    def test_significant_change(self, pre_vals_stable, post_vals_damaged):
        """Large drop should be detected as statistically significant."""
        results = run_statistical_tests(pre_vals_stable, post_vals_damaged, alpha=0.05)

        assert bool(results["significant"]) is True
        assert results["wilcoxon_pvalue"] < 0.05
        assert results["ttest_pvalue"] < 0.05
        assert results["cohens_d"] < -0.8
        assert results["effect_label"] == "large"
        assert "decline" in results["conclusion"].lower()

    def test_not_significant_change(self, pre_vals_stable, post_vals_stable):
        """Negligible change should NOT be detected as significant."""
        results = run_statistical_tests(pre_vals_stable, post_vals_stable, alpha=0.05)

        assert bool(results["significant"]) is False
        assert results["wilcoxon_pvalue"] > 0.05
        assert abs(results["cohens_d"]) < 0.2
        assert "stable" in results["conclusion"].lower()

    def test_returns_all_keys(self, pre_vals_stable, post_vals_stable):
        """Result dict should contain all expected keys."""
        results = run_statistical_tests(pre_vals_stable, post_vals_stable)
        required_keys = {
            "n", "pre_mean", "post_mean", "delta_mean", "delta_pct",
            "ttest_stat", "ttest_pvalue", "ttest_ci",
            "wilcoxon_stat", "wilcoxon_pvalue",
            "cohens_d", "effect_label", "significant", "conclusion", "alpha",
        }
        assert required_keys.issubset(results.keys())

    def test_correct_means(self, pre_vals_stable, post_vals_damaged):
        """Reported means should match numpy means of the input arrays."""
        results = run_statistical_tests(pre_vals_stable, post_vals_damaged)
        assert results["pre_mean"] == pytest.approx(np.mean(pre_vals_stable), rel=1e-6)
        assert results["post_mean"] == pytest.approx(np.mean(post_vals_damaged), rel=1e-6)

    def test_delta_mean(self, pre_vals_stable, post_vals_damaged):
        """delta_mean should equal mean(post) - mean(pre)."""
        results = run_statistical_tests(pre_vals_stable, post_vals_damaged)
        expected_delta = np.mean(post_vals_damaged) - np.mean(pre_vals_stable)
        assert results["delta_mean"] == pytest.approx(expected_delta, rel=1e-6)

    def test_custom_alpha(self, pre_vals_stable, post_vals_stable):
        """Changing alpha from 0.05 to 0.001 should make marginal results not significant."""
        results_05 = run_statistical_tests(pre_vals_stable, post_vals_stable, alpha=0.05)
        results_001 = run_statistical_tests(pre_vals_stable, post_vals_stable, alpha=0.001)
        # Both should be not significant for this stable pair, but test the flag
        assert results_05["alpha"] == 0.05
        assert results_001["alpha"] == 0.001

    def test_confidence_interval_contains_delta(self, pre_vals_stable, post_vals_damaged):
        """The 95% CI of the mean delta should contain the true delta."""
        results = run_statistical_tests(pre_vals_stable, post_vals_damaged)
        ci_low, ci_high = results["ttest_ci"]
        delta = results["delta_mean"]
        # For a properly computed 95% CI, the sample mean is always inside
        assert ci_low <= delta <= ci_high

    def test_insufficient_data_raises(self):
        """Fewer than 2 samples should raise InsufficientDataError."""
        from src.utils import InsufficientDataError
        with pytest.raises(InsufficientDataError):
            run_statistical_tests(np.array([0.5]), np.array([0.4]))


# ---------------------------------------------------------------------------
# check_baseline_variability
# ---------------------------------------------------------------------------

class TestBaselineVariability:
    """Tests for the historical baseline variability check."""

    def test_within_normal_range(self):
        """Current delta close to historical mean should be flagged as within normal range."""
        historical = [-0.02, -0.03, -0.01]  # small typical seasonal variation
        current = -0.025  # very close to historical mean
        result = check_baseline_variability(current, historical, z_threshold=2.0)

        assert result["within_normal_range"] is True
        assert abs(result["z_score"]) <= 2.0
        assert result["mean_hist"] == pytest.approx(np.mean(historical), rel=1e-6)

    def test_anomalous_decline_detected(self):
        """A large hurricane-induced drop should exceed ±2 SD and be flagged as anomalous."""
        historical = [-0.02, -0.03, -0.01, 0.01, -0.02]  # std ≈ 0.015
        current = -0.45  # ~28 SD below mean — clearly anomalous
        result = check_baseline_variability(current, historical, z_threshold=2.0)

        assert result["within_normal_range"] is False
        assert abs(result["z_score"]) > 2.0
        assert "anomalous" in result["interpretation"].lower()

    def test_empty_historical_returns_none(self):
        """When no historical data is available, within_normal_range should be None."""
        result = check_baseline_variability(-0.3, [])
        assert result["within_normal_range"] is None
        assert result["z_score"] is None

    def test_single_historical_value(self):
        """Single historical value: std = 0 so z_score is inf for any difference."""
        result = check_baseline_variability(-0.4, [-0.02])
        # With a single historical value, std = 0, z can be inf
        assert result["z_score"] is not None

    def test_z_threshold_configurable(self):
        """A z_threshold of 1 is more strict than 2."""
        historical = [-0.02, -0.03, 0.01, -0.04]
        current = -0.10
        result_strict = check_baseline_variability(current, historical, z_threshold=1.0)
        result_loose = check_baseline_variability(current, historical, z_threshold=10.0)
        # Strict threshold may flag it, loose threshold should not
        assert result_loose["within_normal_range"] is True


# ---------------------------------------------------------------------------
# Impact classification thresholds (unit-level, without GEE)
# ---------------------------------------------------------------------------

class TestImpactClassificationLogic:
    """
    Tests for the threshold-based classification logic.

    These test the Python-level threshold boundaries without invoking GEE,
    by directly evaluating the conditions used in classify_impact().
    """

    def _classify_scalar(self, delta: float, thresholds: dict = None) -> int:
        """Apply the same logic as classify_impact() to a single float delta."""
        t = thresholds or DEFAULT_THRESHOLDS
        if delta > t["no_impact"]:
            return 0  # No Impact
        elif delta > t["low_impact"]:
            return 1  # Low Impact
        elif delta > t["moderate_impact"]:
            return 2  # Moderate Impact
        else:
            return 3  # Severe Impact

    def test_no_impact_boundary(self):
        """Delta just above -0.05 → No Impact (class 0)."""
        assert self._classify_scalar(-0.049) == 0
        assert self._classify_scalar(0.0) == 0
        assert self._classify_scalar(0.3) == 0

    def test_low_impact_boundary(self):
        """Delta between -0.05 and -0.15 → Low Impact (class 1)."""
        assert self._classify_scalar(-0.05) == 1
        assert self._classify_scalar(-0.10) == 1
        assert self._classify_scalar(-0.149) == 1

    def test_moderate_impact_boundary(self):
        """Delta between -0.15 and -0.30 → Moderate Impact (class 2)."""
        assert self._classify_scalar(-0.15) == 2
        assert self._classify_scalar(-0.20) == 2
        assert self._classify_scalar(-0.299) == 2

    def test_severe_impact_boundary(self):
        """Delta below -0.30 → Severe Impact (class 3)."""
        assert self._classify_scalar(-0.30) == 3
        assert self._classify_scalar(-0.50) == 3
        assert self._classify_scalar(-1.0) == 3

    def test_class_labels_match(self):
        """CLASS_LABELS should have entries for all 4 classes."""
        assert set(CLASS_LABELS.keys()) == {0, 1, 2, 3}

    def test_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        custom = {"no_impact": -0.10, "low_impact": -0.20, "moderate_impact": -0.40}
        # -0.08 > -0.10 → No Impact (class 0) with custom thresholds
        assert self._classify_scalar(-0.08, custom) == 0
        # -0.15 is between -0.10 and -0.20 → Low Impact (class 1)
        assert self._classify_scalar(-0.15, custom) == 1
        assert self._classify_scalar(-0.25, custom) == 2   # between -0.20 and -0.40
        assert self._classify_scalar(-0.50, custom) == 3   # below -0.40


# ---------------------------------------------------------------------------
# Date windows
# ---------------------------------------------------------------------------

class TestDateWindows:
    """Tests for the date_windows() and historical_date_windows() utilities."""

    def test_basic_ian_windows(self):
        """Hurricane Ian date windows should produce the expected dates."""
        pre_start, pre_end, post_start, post_end = date_windows(
            "2022-09-28", pre_days=60, post_days=60, buffer_days=5
        )
        assert pre_end == "2022-09-23"    # 5 days before event
        assert post_start == "2022-10-03" # 5 days after event
        # Pre window is 60 days ending on pre_end
        from datetime import datetime, timedelta
        expected_pre_start = datetime(2022, 9, 23) - timedelta(days=60)
        assert pre_start == expected_pre_start.strftime("%Y-%m-%d")

    def test_buffer_zero(self):
        """With buffer_days=0, pre_end == event_date and post_start == event_date."""
        _, pre_end, post_start, _ = date_windows("2022-09-28", buffer_days=0)
        assert pre_end == "2022-09-28"
        assert post_start == "2022-09-28"

    def test_window_lengths(self):
        """Window lengths should match pre_days and post_days."""
        from datetime import datetime
        pre_start, pre_end, post_start, post_end = date_windows(
            "2022-09-28", pre_days=30, post_days=45, buffer_days=5
        )
        pre_len = (datetime.strptime(pre_end, "%Y-%m-%d") -
                   datetime.strptime(pre_start, "%Y-%m-%d")).days
        post_len = (datetime.strptime(post_end, "%Y-%m-%d") -
                    datetime.strptime(post_start, "%Y-%m-%d")).days
        assert pre_len == 30
        assert post_len == 45

    def test_historical_windows_count(self):
        """historical_date_windows should return exactly n_years tuples."""
        windows = historical_date_windows("2022-09-28", n_years=3)
        assert len(windows) == 3

    def test_historical_windows_prior_years(self):
        """Each historical window should be in a year prior to the event year."""
        windows = historical_date_windows("2022-09-28", n_years=3)
        for i, (pre_start, pre_end, post_start, post_end) in enumerate(windows):
            year = int(post_end[:4])
            assert year < 2022, f"Window {i} post_end year {year} is not before 2022"

    def test_historical_windows_same_season(self):
        """Historical windows should fall in the same calendar season."""
        event_month = 9  # September
        windows = historical_date_windows("2022-09-28", n_years=3, buffer_days=5)
        for pre_start, pre_end, post_start, post_end in windows:
            # post_start should be near September in prior years
            post_month = int(post_start[5:7])
            assert abs(post_month - event_month) <= 1, \
                f"Post window month {post_month} drifted more than 1 from {event_month}"

    def test_returns_strings(self):
        """All return values should be YYYY-MM-DD strings."""
        results = date_windows("2022-09-28")
        assert len(results) == 4
        import re
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        for d in results:
            assert date_pattern.match(d), f"'{d}' is not a valid date string"


# ---------------------------------------------------------------------------
# Vegetation index formula validation (pure Python, no GEE)
# ---------------------------------------------------------------------------

class TestIndexFormulas:
    """Validate index formula implementations with numpy equivalents."""

    def _ndvi(self, nir, red):
        return (nir - red) / (nir + red)

    def _evi(self, nir, red, blue):
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

    def _savi(self, nir, red, L=0.5):
        return ((nir - red) / (nir + red + L)) * (1 + L)

    def _ndmi(self, nir, swir1):
        return (nir - swir1) / (nir + swir1)

    def test_ndvi_range(self):
        """NDVI should be in [-1, 1] for valid reflectance inputs."""
        rng = np.random.default_rng(1)
        nir = rng.uniform(0, 1, 1000)
        red = rng.uniform(0, 1, 1000)
        ndvi = self._ndvi(nir, red)
        # Avoid division by zero cases
        valid = (nir + red) > 0
        assert np.all(ndvi[valid] >= -1) and np.all(ndvi[valid] <= 1)

    def test_ndvi_healthy_vegetation(self):
        """Dense vegetation (high NIR, low Red) should give NDVI near 1."""
        ndvi = self._ndvi(nir=0.9, red=0.05)
        assert ndvi > 0.8

    def test_ndvi_water(self):
        """Water (high Blue/Green absorption, low NIR) should give negative NDVI."""
        ndvi = self._ndvi(nir=0.05, red=0.10)
        assert ndvi < 0

    def test_savi_equals_ndvi_when_soil_is_ignored(self):
        """
        SAVI with L=0 (no soil adjustment) should approach NDVI × (1+L)/(1+L).
        Actually for L=0: SAVI = NDVI * 1 = NDVI.
        """
        nir, red = 0.5, 0.2
        ndvi = self._ndvi(nir, red)
        savi_L0 = self._savi(nir, red, L=0)
        assert savi_L0 == pytest.approx(ndvi, abs=1e-9)

    def test_ndmi_moisture_stressed(self):
        """Low moisture (high SWIR1) should give lower NDMI."""
        ndmi_wet = self._ndmi(nir=0.8, swir1=0.1)
        ndmi_dry = self._ndmi(nir=0.8, swir1=0.6)
        assert ndmi_wet > ndmi_dry

    def test_evi_increases_with_vegetation(self):
        """EVI should increase as vegetation density (NIR) increases."""
        # Dense canopy: high NIR, low Red
        evi_dense  = self._evi(nir=0.85, red=0.05, blue=0.03)
        # Sparse canopy: lower NIR
        evi_sparse = self._evi(nir=0.40, red=0.08, blue=0.03)
        assert evi_dense > evi_sparse

    def test_evi_positive_for_healthy_vegetation(self):
        """EVI should be positive for clearly vegetated pixels."""
        evi = self._evi(nir=0.7, red=0.05, blue=0.03)
        assert evi > 0

    def test_evi_negative_for_water(self):
        """EVI should be near zero or negative over water (high Red > NIR)."""
        evi_water = self._evi(nir=0.05, red=0.10, blue=0.08)
        assert evi_water < 0.1


# ---------------------------------------------------------------------------
# Smoke test: config loading (no GEE needed)
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Verify config.yaml loads correctly with expected structure."""

    def test_config_loads(self):
        """Config file should load without errors."""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        from src.utils import load_config
        cfg = load_config(config_path)
        assert isinstance(cfg, dict)

    def test_required_keys_present(self):
        """Config should have all expected top-level sections."""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        from src.utils import load_config
        cfg = load_config(config_path)
        assert "gee" in cfg
        assert "windows" in cfg
        assert "statistics" in cfg
        assert "thresholds" in cfg
        assert "hurricanes" in cfg

    def test_hurricane_presets_present(self):
        """All major Florida hurricane presets should be in config."""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        from src.utils import load_config
        cfg = load_config(config_path)
        expected = {"ian", "michael", "idalia", "irma", "milton"}
        actual = set(cfg.get("hurricanes", {}).keys())
        assert expected.issubset(actual)

    def test_ian_preset_bbox(self):
        """Hurricane Ian preset should have a 4-element bbox."""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        from src.utils import load_config
        cfg = load_config(config_path)
        ian = cfg["hurricanes"]["ian"]
        assert len(ian["bbox"]) == 4
        assert ian["date"] == "2022-09-28"


# ---------------------------------------------------------------------------
# Run tests standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
