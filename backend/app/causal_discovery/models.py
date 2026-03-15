"""SQLAlchemy models for causal discovery — TimescaleDB hypertable + graph snapshots."""
from __future__ import annotations
import logging
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncConnection
from app.models.graph import Base

logger = logging.getLogger(__name__)


class NodeValue(Base):
    __tablename__ = "node_values"
    node_id = Column(String(64), nullable=False, primary_key=True)
    ts = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    value = Column(Float(precision=53), nullable=False)
    source = Column(String(64), nullable=False)


class DiscoveredGraph(Base):
    """A snapshot of a discovered causal graph.

    Each row is one complete graph produced by a causal discovery algorithm at a
    point in time. Stores the full node list and edge list as JSONB so the entire
    graph can be loaded in one query. Multiple snapshots with the same `run_name`
    form a time-series of evolving graphs for comparison.
    """
    __tablename__ = "discovered_graphs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String(128), nullable=False, index=True)  # groups snapshots into a series
    algorithm = Column(String(64), nullable=False)               # 'pcmci', 'varlingam', etc.
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    data_start = Column(DateTime(timezone=True), nullable=True)  # earliest data used
    data_end = Column(DateTime(timezone=True), nullable=True)    # latest data used
    node_count = Column(Integer, default=0)
    edge_count = Column(Integer, default=0)
    parameters = Column(JSONB, default=dict)                     # algorithm params
    nodes = Column(JSONB, nullable=False, default=list)          # [{id, zscore, polarity, display_sentiment}]
    edges = Column(JSONB, nullable=False, default=list)          # [{source, target, weight, lag, direction}]
    metadata_ = Column("metadata", JSONB, default=dict)          # run time, notes, etc.


async def create_hypertable_if_needed(conn: AsyncConnection) -> None:
    try:
        await conn.execute(text("""
            SELECT create_hypertable('node_values', 'ts', if_not_exists => TRUE, migrate_data => TRUE);
        """))
        logger.info("node_values hypertable ready")
    except Exception as e:
        logger.warning("Could not create hypertable (TimescaleDB may not be installed): %s", e)


async def create_node_values_index(conn: AsyncConnection) -> None:
    try:
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_node_values_node_ts ON node_values (node_id, ts DESC);
        """))
    except Exception as e:
        logger.warning("Could not create index: %s", e)
