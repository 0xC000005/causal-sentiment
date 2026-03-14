"""Causal discovery algorithms — PCMCI+ (tigramite) and VARLiNGAM (lingam).

Both functions use lazy imports so the module can be imported even when
tigramite or lingam are not installed.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def discover_edges_pcmci(
    df: pd.DataFrame,
    max_lag: int = 5,
    significance_level: float = 0.01,
) -> list[dict[str, Any]]:
    """Run PCMCI+ with ParCorr on a daily matrix and return discovered edges.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned daily matrix (index=date, columns=node_id). NaNs should be
        forward-filled before calling.
    max_lag : int
        Maximum time lag to test.
    significance_level : float
        P-value threshold for significant links.

    Returns
    -------
    list[dict]
        Each dict has keys: source, target, weight (abs val_matrix entry),
        lag, direction ("positive" | "negative").
    """
    import tigramite
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    columns = list(df.columns)
    data = df.values.astype(np.float64)

    # Build tigramite dataframe
    dataframe = pp.DataFrame(data, var_names=columns)

    # Run PCMCI+
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=max_lag, pc_alpha=significance_level)

    # Extract significant edges
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    n_vars = len(columns)

    edges: list[dict[str, Any]] = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            for lag in range(1, max_lag + 1):
                p_val = p_matrix[i, j, lag]
                val = val_matrix[i, j, lag]
                if p_val < significance_level and abs(val) > 0.0:
                    edges.append({
                        "source": columns[i],
                        "target": columns[j],
                        "weight": float(abs(val)),
                        "lag": int(lag),
                        "direction": "positive" if val > 0 else "negative",
                    })

    # Also check contemporaneous (lag=0) links — directed by PCMCI+
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            p_val = p_matrix[i, j, 0]
            val = val_matrix[i, j, 0]
            if p_val < significance_level and abs(val) > 0.0:
                edges.append({
                    "source": columns[i],
                    "target": columns[j],
                    "weight": float(abs(val)),
                    "lag": 0,
                    "direction": "positive" if val > 0 else "negative",
                })

    logger.info("PCMCI+ discovered %d edges from %d variables", len(edges), n_vars)
    return edges


def discover_edges_varlingam(
    df: pd.DataFrame,
    max_lag: int = 3,
    min_weight: float = 0.1,
) -> list[dict[str, Any]]:
    """Run VARLiNGAM on a daily matrix and return discovered edges.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned daily matrix (index=date, columns=node_id).
    max_lag : int
        Maximum time lag for the VAR model.
    min_weight : float
        Minimum absolute weight to include an edge.

    Returns
    -------
    list[dict]
        Each dict has keys: source, target, weight, lag, direction.
    """
    import lingam

    columns = list(df.columns)
    data = df.values.astype(np.float64)

    model = lingam.VARLiNGAM(lags=max_lag)
    model.fit(data)

    edges: list[dict[str, Any]] = []
    n_vars = len(columns)

    # model.causal_order_ gives the causal ordering
    # model.adjacency_matrices_ is a list of (n_vars, n_vars) arrays for lag 0..max_lag
    for lag_idx, adj_matrix in enumerate(model.adjacency_matrices_):
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j and lag_idx == 0:
                    continue
                val = adj_matrix[i, j]
                if abs(val) >= min_weight:
                    edges.append({
                        "source": columns[j],
                        "target": columns[i],
                        "weight": float(abs(val)),
                        "lag": int(lag_idx),
                        "direction": "positive" if val > 0 else "negative",
                    })

    logger.info("VARLiNGAM discovered %d edges from %d variables", len(edges), n_vars)
    return edges
