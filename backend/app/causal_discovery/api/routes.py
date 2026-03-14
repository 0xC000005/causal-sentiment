"""API routes for causal discovery — backfill, sources, stats, discover, graph."""
from __future__ import annotations

import logging
import time
from typing import Any

import networkx as nx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.causal_discovery.engine.anchors import propagate_polarity
from app.causal_discovery.engine.causal import discover_edges_pcmci
from app.causal_discovery.engine.importance import rank_nodes_by_importance
from app.causal_discovery.engine.matrix import get_daily_matrix
from app.causal_discovery.engine.zscore import compute_rolling_zscore
from app.causal_discovery.pipeline.backfill import run_backfill
from app.causal_discovery.pipeline.sources import get_all_sources
from app.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/causal", tags=["causal-discovery"])

# ---------------------------------------------------------------------------
# Module-level backfill status tracker
# ---------------------------------------------------------------------------
_backfill_status: dict[str, Any] = {
    "state": "idle",       # idle | running | completed | failed
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}


def _reset_status() -> None:
    _backfill_status.update({
        "state": "idle",
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    })


async def _run_backfill_task(
    yfinance_period: str = "5y",
    fred_start: str = "2020-01-01",
) -> None:
    """Background task wrapper that updates ``_backfill_status``."""
    from app.db.connection import async_session

    _backfill_status["state"] = "running"
    _backfill_status["started_at"] = time.time()
    _backfill_status["finished_at"] = None
    _backfill_status["result"] = None
    _backfill_status["error"] = None

    try:
        async with async_session() as session:
            result = await run_backfill(
                session,
                yfinance_period=yfinance_period,
                fred_start=fred_start,
            )
        _backfill_status["state"] = "completed"
        _backfill_status["result"] = result
    except Exception as exc:
        logger.exception("Backfill task failed")
        _backfill_status["state"] = "failed"
        _backfill_status["error"] = str(exc)
    finally:
        _backfill_status["finished_at"] = time.time()


# ---------------------------------------------------------------------------
# Module-level discovery status tracker
# ---------------------------------------------------------------------------
_discovery_status: dict[str, Any] = {
    "state": "idle",       # idle | running | completed | failed
    "started_at": None,
    "finished_at": None,
    "last_result": None,
    "error": None,
}

# Default anchors for polarity propagation
_DEFAULT_ANCHORS: dict[str, int] = {
    "sp500": +1,
    "nasdaq": +1,
    "us_gdp_growth": +1,
    "unemployment_rate": -1,
}


async def _run_discovery_task() -> None:
    """Background task: daily matrix -> z-scores -> PCMCI+ -> store result."""
    from app.db.connection import async_session

    _discovery_status["state"] = "running"
    _discovery_status["started_at"] = time.time()
    _discovery_status["finished_at"] = None
    _discovery_status["last_result"] = None
    _discovery_status["error"] = None

    try:
        async with async_session() as session:
            df = await get_daily_matrix(session, days=252)
        if df.empty or len(df.columns) < 2:
            raise ValueError(
                f"Insufficient data for discovery: {len(df)} rows, "
                f"{len(df.columns)} columns"
            )
        zscores = compute_rolling_zscore(df, window=90)
        edges = discover_edges_pcmci(zscores, max_lag=5)
        _discovery_status["state"] = "completed"
        _discovery_status["last_result"] = {
            "edges_found": len(edges),
            "nodes": list(df.columns),
            "edges": edges,
        }
    except Exception as exc:
        logger.exception("Discovery task failed")
        _discovery_status["state"] = "failed"
        _discovery_status["error"] = str(exc)
    finally:
        _discovery_status["finished_at"] = time.time()


def _build_graph_from_edges(
    edges: list[dict[str, Any]],
) -> nx.DiGraph:
    """Build a NetworkX DiGraph from discovered edge dicts."""
    g = nx.DiGraph()
    for edge in edges:
        g.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["weight"],
            lag=edge["lag"],
            direction=edge["direction"],
        )
    return g


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/backfill")
async def trigger_backfill(
    background_tasks: BackgroundTasks,
    yfinance_period: str = "5y",
    fred_start: str = "2020-01-01",
) -> dict:
    """Trigger a backfill job in the background.

    Returns immediately with the current status. Poll
    ``GET /api/causal/backfill/status`` for progress.
    """
    if _backfill_status["state"] == "running":
        return {"message": "Backfill already running", "status": _backfill_status}

    background_tasks.add_task(
        _run_backfill_task,
        yfinance_period=yfinance_period,
        fred_start=fred_start,
    )
    return {"message": "Backfill started", "status": {"state": "running"}}


@router.get("/backfill/status")
async def backfill_status() -> dict:
    """Return the current backfill job status."""
    return dict(_backfill_status)


@router.get("/sources")
async def list_sources() -> list[dict]:
    """Return all registered data source definitions."""
    return get_all_sources()


@router.get("/stats")
async def data_stats(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Return row counts per node from the node_values table."""
    result = await session.execute(text(
        "SELECT node_id, COUNT(*) as row_count, "
        "MIN(ts) as first_ts, MAX(ts) as last_ts "
        "FROM node_values "
        "GROUP BY node_id "
        "ORDER BY node_id"
    ))
    rows = result.fetchall()
    return [
        {
            "node_id": row.node_id,
            "row_count": row.row_count,
            "first_ts": row.first_ts.isoformat() if row.first_ts else None,
            "last_ts": row.last_ts.isoformat() if row.last_ts else None,
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------

@router.post("/discover")
async def trigger_discovery(background_tasks: BackgroundTasks) -> dict:
    """Trigger causal discovery (PCMCI+) in the background.

    Returns immediately. Poll ``GET /api/causal/discover/status`` for progress.
    """
    if _discovery_status["state"] == "running":
        return {"message": "Discovery already running", "status": _discovery_status}

    background_tasks.add_task(_run_discovery_task)
    return {"message": "Discovery started", "status": {"state": "running"}}


@router.get("/discover/status")
async def discovery_status() -> dict:
    """Return the current discovery job status."""
    return dict(_discovery_status)


@router.get("/graph")
async def get_discovered_graph(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Run the full causal-discovery pipeline synchronously and return the graph.

    Pipeline: daily matrix -> z-scores -> PCMCI+ -> build graph ->
    anchor propagation -> importance ranking -> JSON response.
    """
    # 1. Get daily matrix from DB
    df = await get_daily_matrix(session, days=252)
    if df.empty or len(df.columns) < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient data for causal discovery: "
                f"{len(df)} rows, {len(df.columns)} columns. "
                f"Run POST /api/causal/backfill first."
            ),
        )

    # 2. Compute z-scores
    zscores = compute_rolling_zscore(df, window=90)

    # 3. Run PCMCI+
    edges = discover_edges_pcmci(zscores, max_lag=5)

    # 4. Build NetworkX graph from discovered edges
    g = _build_graph_from_edges(edges)

    # 5. Anchor propagation
    polarity = propagate_polarity(g, _DEFAULT_ANCHORS)

    # 6. Rank nodes by importance (top 100)
    ranking = rank_nodes_by_importance(g, top_n=100)
    importance_map = {r["node_id"]: r for r in ranking}

    # 7. Build response — latest z-score per node from the last row
    latest_zscores: dict[str, float] = {}
    if len(zscores) > 0:
        last_row = zscores.iloc[-1]
        latest_zscores = {col: float(last_row[col]) for col in zscores.columns}

    nodes = []
    for node_id in g.nodes():
        zscore_val = latest_zscores.get(node_id, 0.0)
        pol = polarity.get(node_id, 0)
        # display_sentiment: polarity * abs(zscore), clamped to [-1, 1]
        display_sentiment = max(-1.0, min(1.0, pol * min(abs(zscore_val) / 3.0, 1.0)))
        imp = importance_map.get(node_id, {})
        nodes.append({
            "id": node_id,
            "zscore": round(zscore_val, 4),
            "polarity": pol,
            "display_sentiment": round(display_sentiment, 4),
            "importance": imp.get("score", 0.0),
        })

    edge_list = [
        {
            "source": e["source"],
            "target": e["target"],
            "weight": round(e["weight"], 4),
            "lag": e["lag"],
            "direction": e["direction"],
        }
        for e in edges
    ]

    return {
        "nodes": nodes,
        "edges": edge_list,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edge_list),
            "data_columns": len(df.columns),
            "data_rows": len(df),
        },
    }
