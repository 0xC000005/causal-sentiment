"""Rolling z-score computation for daily matrix columns."""
from __future__ import annotations

import pandas as pd


def compute_rolling_zscore(
    df: pd.DataFrame,
    window: int = 90,
    clamp: float | None = 3.0,
) -> pd.DataFrame:
    """Compute rolling z-scores for each column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Daily matrix (index=date, columns=node_id).
    window : int
        Rolling window size (business days).
    clamp : float | None
        If set, clip z-scores to [-clamp, +clamp].

    Returns
    -------
    pd.DataFrame
        Z-scores with the same shape as *df*.
    """
    min_periods = max(1, window // 4)
    rolling_mean = df.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = df.rolling(window=window, min_periods=min_periods).std()
    # Avoid division by zero — replace zero std with NaN
    rolling_std = rolling_std.replace(0, float("nan"))
    zscores = (df - rolling_mean) / rolling_std
    zscores = zscores.fillna(0.0)
    if clamp is not None:
        zscores = zscores.clip(lower=-clamp, upper=clamp)
    return zscores
