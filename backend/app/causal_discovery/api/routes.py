"""API routes for causal discovery — backfill, sources, and stats."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

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
