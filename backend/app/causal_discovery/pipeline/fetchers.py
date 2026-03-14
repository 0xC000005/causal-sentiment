"""Bulk history fetchers for yfinance and FRED data sources.

These fetchers retrieve multi-year daily price/value histories that feed into
the causal discovery pipeline's matrix builder and statistical tests.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx
import pandas as pd
import yfinance as yf

from app.config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------

def _yfinance_download_sync(
    tickers: list[str],
    period: str = "5y",
) -> pd.DataFrame:
    """Synchronous yfinance download returning a clean DataFrame.

    Returns a DataFrame with columns = tickers, index = dates (DatetimeIndex),
    values = Close prices.
    """
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        return pd.DataFrame()

    # yfinance returns multi-level columns (Price, Ticker) for multiple tickers,
    # but a single-level column for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        # Extract the "Close" price level
        df = raw["Close"]
    else:
        # Single ticker — raw columns are just price types like "Close", "Open", etc.
        df = raw[["Close"]].copy()
        df.columns = [tickers[0]]

    # Ensure column names are plain strings
    df.columns = [str(c) for c in df.columns]

    return df


async def fetch_yfinance_history(
    tickers: list[str],
    period: str = "5y",
) -> pd.DataFrame:
    """Async wrapper around yfinance download.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols (e.g. ``["SPY", "QQQ"]``).
    period : str
        yfinance period string (e.g. ``"5y"``, ``"1mo"``).

    Returns
    -------
    pd.DataFrame
        Columns = tickers, index = DatetimeIndex, values = Close prices.
    """
    return await asyncio.to_thread(_yfinance_download_sync, tickers, period)


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

def parse_fred_observations(
    raw_observations: list[dict],
) -> list[tuple[str, float]]:
    """Parse raw FRED API observations into (date_str, value) tuples.

    Filters out entries where the value is ``"."`` (FRED's missing-data
    sentinel).

    Parameters
    ----------
    raw_observations : list[dict]
        List of ``{"date": "YYYY-MM-DD", "value": "..."}`` dicts as returned
        by the FRED ``series/observations`` endpoint.

    Returns
    -------
    list[tuple[str, float]]
        Parsed ``(date_string, numeric_value)`` pairs.
    """
    results: list[tuple[str, float]] = []
    for obs in raw_observations:
        val = obs.get("value", ".")
        if val == ".":
            continue
        try:
            results.append((obs["date"], float(val)))
        except (ValueError, KeyError):
            continue
    return results


async def fetch_fred_series_history(
    series_id: str,
    observation_start: str = "2020-01-01",
) -> list[tuple[str, float]]:
    """Fetch a full observation history for a single FRED series.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. ``"FEDFUNDS"``).
    observation_start : str
        ISO date string for the start of the observation window.

    Returns
    -------
    list[tuple[str, float]]
        Parsed ``(date_string, value)`` pairs, or an empty list if the FRED
        API key is not configured or the request fails.
    """
    if not settings.fred_api_key:
        logger.warning(
            "FRED API key not configured — skipping fetch for %s", series_id,
        )
        return []

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "sort_order": "asc",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return parse_fred_observations(data.get("observations", []))
    except httpx.HTTPError as exc:
        logger.error("FRED API error for %s: %s", series_id, exc)
        return []
