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
from app.causal_discovery.engine.scoring import compute_rolling_zscore
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


async def _run_discovery_task(
    run_name: str = "default",
    algorithm: str = "pcmci",
    scoring: str = "zscore",
    max_lag: int = 5,
    significance_level: float = 0.01,
    days: int = 252,
    zscore_window: int = 90,
) -> None:
    """Background task: daily matrix -> scoring -> causal discovery -> persist to DB."""
    from app.db.connection import async_session
    from app.causal_discovery.models import DiscoveredGraph

    _discovery_status["state"] = "running"
    _discovery_status["started_at"] = time.time()
    _discovery_status["finished_at"] = None
    _discovery_status["last_result"] = None
    _discovery_status["error"] = None

    try:
        async with async_session() as session:
            df = await get_daily_matrix(session, days=days)

        if df.empty or len(df.columns) < 2:
            raise ValueError(
                f"Insufficient data for discovery: {len(df)} rows, "
                f"{len(df.columns)} columns"
            )

        from app.causal_discovery.engine.scoring import compute_scores
        if scoring == "zscore":
            zscores = compute_scores(df, method="zscore", window=zscore_window)
        elif scoring == "returns":
            zscores = compute_scores(df, method="returns")
        elif scoring == "volatility":
            zscores = compute_scores(df, method="volatility", window=20)
        else:
            zscores = compute_scores(df, method=scoring)

        # Run the selected algorithm in a thread pool to avoid blocking the event loop.
        # All causal discovery algorithms are CPU-bound (numpy matrix ops) and would
        # block async endpoints (health, status, WebSocket) if run in the main thread.
        import asyncio

        if algorithm == "varlingam":
            from app.causal_discovery.engine.causal import discover_edges_varlingam
            edges = await asyncio.to_thread(discover_edges_varlingam, zscores, max_lag)
        elif algorithm == "pc":
            from app.causal_discovery.engine.causal import discover_edges_pc
            result = await asyncio.to_thread(
                discover_edges_pc, zscores, significance_level, True,
            )
            edges = result["final"]
            extra_metadata = {"steps": result["steps"]}
        elif algorithm == "ges":
            from app.causal_discovery.engine.causal import discover_edges_ges
            edges = await asyncio.to_thread(discover_edges_ges, zscores)
        elif algorithm == "granger":
            from app.causal_discovery.engine.causal import discover_edges_granger
            edges = await asyncio.to_thread(
                discover_edges_granger, zscores, max_lag, significance_level,
            )
        else:
            edges = await asyncio.to_thread(
                discover_edges_pcmci, zscores, max_lag, significance_level,
            )

        extra_metadata = locals().get("extra_metadata", {})

        # Build graph for polarity + importance
        g = _build_graph_from_edges(edges)
        polarity = propagate_polarity(g, _DEFAULT_ANCHORS)
        ranking = rank_nodes_by_importance(g, top_n=100)
        importance_map = {r["node_id"]: r for r in ranking}

        latest_zscores: dict[str, float] = {}
        if len(zscores) > 0:
            last_row = zscores.iloc[-1]
            latest_zscores = {col: float(last_row[col]) for col in zscores.columns}

        nodes_json = []
        for node_id in g.nodes():
            z = latest_zscores.get(node_id, 0.0)
            p = polarity.get(node_id, 0)
            imp = importance_map.get(node_id, {})
            # Display sentiment: use polarity if available, otherwise use raw z-score sign
            # This ensures nodes unreachable from anchors still show color
            if p != 0:
                display = p * min(abs(z) / 3.0, 1.0)
            else:
                display = z / 3.0  # No polarity info — use z-score directly
            nodes_json.append({
                "id": node_id,
                "zscore": round(z, 4),
                "polarity": p,
                "display_sentiment": round(max(-1.0, min(1.0, display)), 4),
                "importance": round(imp.get("score", 0.0), 4),
            })

        edges_json = [
            {"source": e["source"], "target": e["target"],
             "weight": round(e["weight"], 4), "lag": e["lag"], "direction": e["direction"]}
            for e in edges
        ]

        # Persist to DB
        data_start = df.index.min() if len(df) > 0 else None
        data_end = df.index.max() if len(df) > 0 else None
        # Convert pandas Timestamps to Python datetimes for SQLAlchemy
        if data_start is not None:
            data_start = data_start.to_pydatetime()
        if data_end is not None:
            data_end = data_end.to_pydatetime()

        async with async_session() as session:
            snapshot = DiscoveredGraph(
                run_name=run_name,
                algorithm=algorithm,
                data_start=data_start,
                data_end=data_end,
                node_count=len(nodes_json),
                edge_count=len(edges_json),
                parameters={
                    "scoring": scoring,
                    "max_lag": max_lag,
                    "significance_level": significance_level,
                    "days": days,
                    "zscore_window": zscore_window,
                },
                nodes=nodes_json,
                edges=edges_json,
                metadata_=extra_metadata,
            )
            session.add(snapshot)
            await session.commit()
            snapshot_id = snapshot.id

        _discovery_status["state"] = "completed"
        _discovery_status["last_result"] = {
            "snapshot_id": snapshot_id,
            "run_name": run_name,
            "algorithm": algorithm,
            "edges_found": len(edges_json),
            "nodes": [n["id"] for n in nodes_json],
            "edges": edges_json,
        }
        logger.info("Discovery complete: %d nodes, %d edges, saved as snapshot #%d",
                     len(nodes_json), len(edges_json), snapshot_id)
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
async def trigger_discovery(
    background_tasks: BackgroundTasks,
    run_name: str = "default",
    algorithm: str = "pcmci",
    scoring: str = "zscore",
    max_lag: int = 5,
    significance_level: float = 0.01,
    days: int = 252,
    zscore_window: int = 90,
) -> dict:
    """Trigger causal discovery in the background. Result is persisted to DB.

    Parameters:
        run_name: Label for this series of snapshots (e.g. 'weekly_full')
        algorithm: 'pcmci', 'granger', 'pc', 'ges', 'varlingam'
        scoring: 'zscore', 'returns', 'volatility' — how to transform raw data before discovery
        max_lag: Maximum causal lag in days
        significance_level: p-value threshold (pcmci only)
        days: How many days of history to use
        zscore_window: Rolling z-score window size (zscore scoring only)

    Returns immediately. Poll ``GET /api/causal/discover/status`` for progress.
    """
    if _discovery_status["state"] == "running":
        return {"message": "Discovery already running", "status": _discovery_status}

    background_tasks.add_task(
        _run_discovery_task,
        run_name=run_name,
        algorithm=algorithm,
        scoring=scoring,
        max_lag=max_lag,
        significance_level=significance_level,
        days=days,
        zscore_window=zscore_window,
    )
    return {"message": "Discovery started", "status": {"state": "running"}}


@router.get("/discover/status")
async def discovery_status() -> dict:
    """Return the current discovery job status."""
    return dict(_discovery_status)


# ---------------------------------------------------------------------------
# Stored graph endpoints
# ---------------------------------------------------------------------------

@router.get("/graphs")
async def list_graphs(
    run_name: str | None = None,
    algorithm: str | None = None,
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """List stored graph snapshots. Filter by run_name or algorithm."""
    from app.causal_discovery.models import DiscoveredGraph
    from sqlalchemy import select

    query = select(DiscoveredGraph).order_by(DiscoveredGraph.created_at.desc()).limit(limit)
    if run_name:
        query = query.where(DiscoveredGraph.run_name == run_name)
    if algorithm:
        query = query.where(DiscoveredGraph.algorithm == algorithm)

    result = await session.execute(query)
    rows = result.scalars().all()
    return [
        {
            "id": r.id,
            "run_name": r.run_name,
            "algorithm": r.algorithm,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "data_start": r.data_start.isoformat() if r.data_start else None,
            "data_end": r.data_end.isoformat() if r.data_end else None,
            "node_count": r.node_count,
            "edge_count": r.edge_count,
            "parameters": r.parameters,
        }
        for r in rows
    ]


@router.get("/graph/animate")
async def get_graph_animate(
    id: int | None = None,
    run_name: str | None = None,
    n_frames: int = 30,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Generate animation frames: complete correlation graph -> discovered graph.

    Builds all pairwise correlations, marks which edges survive in the discovered
    graph, then produces ``n_frames`` by progressively removing the weakest
    non-surviving edges (weakest first).

    Parameters:
        id: Discovered graph snapshot id.
        run_name: Lookup latest snapshot by run_name.
        n_frames: Number of animation frames to generate (default 30).
    """
    import numpy as np
    from app.causal_discovery.models import DiscoveredGraph
    from sqlalchemy import select

    # --- load discovered graph ---
    if id is not None:
        query = select(DiscoveredGraph).where(DiscoveredGraph.id == id)
    elif run_name is not None:
        query = (select(DiscoveredGraph)
                 .where(DiscoveredGraph.run_name == run_name)
                 .order_by(DiscoveredGraph.created_at.desc())
                 .limit(1))
    else:
        query = select(DiscoveredGraph).order_by(DiscoveredGraph.created_at.desc()).limit(1)

    result = await session.execute(query)
    graph = result.scalar_one_or_none()
    if graph is None:
        raise HTTPException(status_code=404, detail="No discovered graph found.")

    # --- load daily matrix and compute correlation ---
    days = graph.parameters.get("days", 252) if graph.parameters else 252
    df = await get_daily_matrix(session, days=days)
    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Insufficient data for correlation matrix.")

    corr = df.corr()

    # Set of discovered edges (unordered pairs for lookup)
    discovered_edges = set()
    for e in graph.edges:
        discovered_edges.add((e["source"], e["target"]))

    # Build complete edge list from correlation matrix
    columns = list(corr.columns)
    all_edges: list[dict] = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            c = corr.iloc[i, j]
            if np.isnan(c):
                continue
            abs_c = abs(float(c))
            if abs_c <= 0:
                continue
            src, tgt = columns[i], columns[j]
            survives = (src, tgt) in discovered_edges or (tgt, src) in discovered_edges
            all_edges.append({
                "source": src,
                "target": tgt,
                "correlation": round(float(c), 4),
                "abs_correlation": round(abs_c, 4),
                "survives": survives,
            })

    # Separate surviving vs non-surviving
    surviving = [e for e in all_edges if e["survives"]]
    non_surviving = [e for e in all_edges if not e["survives"]]

    # Sort non-surviving by absolute correlation ascending (weakest first)
    non_surviving.sort(key=lambda e: e["abs_correlation"])

    # Generate frames by removing non-surviving edges in batches
    total_non_surviving = len(non_surviving)
    frames: list[dict] = []

    if total_non_surviving == 0 or n_frames <= 1:
        # Frame 0: all edges; Frame 1 (or only): just surviving
        frames.append({
            "frame": 0,
            "edge_count": len(all_edges),
            "edges": all_edges,
        })
        if n_frames > 1:
            frames.append({
                "frame": 1,
                "edge_count": len(surviving),
                "edges": surviving,
            })
    else:
        for frame_idx in range(n_frames):
            # How many non-surviving edges to remove by this frame
            remove_count = int(total_non_surviving * frame_idx / (n_frames - 1))
            remaining_non_surviving = non_surviving[remove_count:]
            frame_edges = surviving + remaining_non_surviving
            frames.append({
                "frame": frame_idx,
                "edge_count": len(frame_edges),
                "edges": frame_edges,
            })

    return {
        "graph_id": graph.id,
        "n_frames": len(frames),
        "total_edges": len(all_edges),
        "surviving_edges": len(surviving),
        "frames": frames,
    }


@router.get("/graph/compare")
async def get_graph_compare(
    id: int | None = None,
    run_name: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Compare a discovered graph against the expert graph from topology.py.

    Returns overlap metrics: shared edges, expert-only edges, discovered-only
    edges, Jaccard similarity, and the lists of shared and novel edges.
    """
    from app.causal_discovery.models import DiscoveredGraph
    from app.graph_engine.topology import MVP_NODES, MVP_EDGES
    from sqlalchemy import select

    # --- load discovered graph ---
    if id is not None:
        query = select(DiscoveredGraph).where(DiscoveredGraph.id == id)
    elif run_name is not None:
        query = (select(DiscoveredGraph)
                 .where(DiscoveredGraph.run_name == run_name)
                 .order_by(DiscoveredGraph.created_at.desc())
                 .limit(1))
    else:
        query = select(DiscoveredGraph).order_by(DiscoveredGraph.created_at.desc()).limit(1)

    result = await session.execute(query)
    graph = result.scalar_one_or_none()
    if graph is None:
        raise HTTPException(status_code=404, detail="No discovered graph found.")

    # Expert edges as (source, target) pairs
    expert_edge_set = {(e["source_id"], e["target_id"]) for e in MVP_EDGES}
    expert_node_set = {n["id"] for n in MVP_NODES}

    # Discovered edges as (source, target) pairs
    discovered_edge_set = {(e["source"], e["target"]) for e in graph.edges}
    discovered_node_set = {n["id"] for n in graph.nodes}

    shared_edges = expert_edge_set & discovered_edge_set
    expert_only = expert_edge_set - discovered_edge_set
    discovered_only = discovered_edge_set - expert_edge_set

    union_size = len(expert_edge_set | discovered_edge_set)
    jaccard = len(shared_edges) / union_size if union_size > 0 else 0.0

    return {
        "expert": {
            "node_count": len(expert_node_set),
            "edge_count": len(expert_edge_set),
        },
        "discovered": {
            "node_count": len(discovered_node_set),
            "edge_count": len(discovered_edge_set),
        },
        "shared_edge_count": len(shared_edges),
        "expert_only_edge_count": len(expert_only),
        "discovered_only_edge_count": len(discovered_only),
        "jaccard_similarity": round(jaccard, 4),
        "shared_edges": sorted([{"source": s, "target": t} for s, t in shared_edges],
                                key=lambda e: (e["source"], e["target"])),
        "novel_edges": sorted([{"source": s, "target": t} for s, t in discovered_only],
                               key=lambda e: (e["source"], e["target"])),
    }


@router.get("/graph")
async def get_graph(
    id: int | None = None,
    run_name: str | None = None,
    top_n: int | None = None,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get a stored discovered graph.

    - By id: ``GET /api/causal/graph?id=42``
    - Latest for a run_name: ``GET /api/causal/graph?run_name=default``
    - Latest overall: ``GET /api/causal/graph``
    - With top_n: ``GET /api/causal/graph?top_n=10`` — keep only the top N
      most important nodes and edges between them.

    Returns nodes + edges in a format compatible with the frontend's GraphData.
    """
    from app.causal_discovery.models import DiscoveredGraph
    from sqlalchemy import select

    if id is not None:
        query = select(DiscoveredGraph).where(DiscoveredGraph.id == id)
    elif run_name is not None:
        query = (select(DiscoveredGraph)
                 .where(DiscoveredGraph.run_name == run_name)
                 .order_by(DiscoveredGraph.created_at.desc())
                 .limit(1))
    else:
        query = select(DiscoveredGraph).order_by(DiscoveredGraph.created_at.desc()).limit(1)

    result = await session.execute(query)
    graph = result.scalar_one_or_none()

    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="No discovered graph found. Run POST /api/causal/discover first.",
        )

    nodes = graph.nodes
    edges = graph.edges

    if top_n is not None and top_n > 0:
        # Sort nodes by importance (descending), keep top_n
        sorted_nodes = sorted(nodes, key=lambda n: n.get("importance", 0.0), reverse=True)
        nodes = sorted_nodes[:top_n]
        surviving_ids = {n["id"] for n in nodes}
        # Filter edges to only include those between surviving nodes
        edges = [
            e for e in edges
            if e["source"] in surviving_ids and e["target"] in surviving_ids
        ]

    return {
        "id": graph.id,
        "run_name": graph.run_name,
        "algorithm": graph.algorithm,
        "created_at": graph.created_at.isoformat() if graph.created_at else None,
        "data_start": graph.data_start.isoformat() if graph.data_start else None,
        "data_end": graph.data_end.isoformat() if graph.data_end else None,
        "parameters": graph.parameters,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }


@router.get("/graph/history")
async def get_graph_history(
    run_name: str = "default",
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get all snapshots for a run_name, ordered chronologically.

    For comparing how the network evolved over time.
    """
    from app.causal_discovery.models import DiscoveredGraph
    from sqlalchemy import select

    query = (select(DiscoveredGraph)
             .where(DiscoveredGraph.run_name == run_name)
             .order_by(DiscoveredGraph.created_at.asc()))

    result = await session.execute(query)
    rows = result.scalars().all()

    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "data_end": r.data_end.isoformat() if r.data_end else None,
            "node_count": r.node_count,
            "edge_count": r.edge_count,
            "nodes": r.nodes,
            "edges": r.edges,
        }
        for r in rows
    ]
