"""Tests for z-score computation."""
import pandas as pd
import numpy as np
from app.causal_discovery.engine.zscore import compute_rolling_zscore


def test_zscore_basic():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    df = pd.DataFrame({"A": [100.0] * 100}, index=dates)
    zscores = compute_rolling_zscore(df, window=20)
    assert zscores["A"].dropna().abs().max() < 0.01 or zscores["A"].isna().all()


def test_zscore_known_values():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    df = pd.DataFrame({"A": list(range(100))}, index=dates)
    zscores = compute_rolling_zscore(df, window=20)
    assert zscores["A"].iloc[-1] > 0


def test_zscore_multiple_columns():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    df = pd.DataFrame({"A": np.random.randn(100).cumsum(), "B": np.random.randn(100).cumsum()}, index=dates)
    zscores = compute_rolling_zscore(df, window=20)
    assert "A" in zscores.columns and "B" in zscores.columns


def test_zscore_clamped():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    df = pd.DataFrame({"A": [100.0] * 99 + [10000.0]}, index=dates)
    zscores = compute_rolling_zscore(df, window=20, clamp=3.0)
    assert zscores["A"].iloc[-1] <= 3.0
