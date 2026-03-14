"""SQLAlchemy models for causal discovery — TimescaleDB hypertable."""
from __future__ import annotations
import logging
from sqlalchemy import Column, DateTime, Float, String, text
from sqlalchemy.ext.asyncio import AsyncConnection
from app.models.graph import Base

logger = logging.getLogger(__name__)


class NodeValue(Base):
    __tablename__ = "node_values"
    node_id = Column(String(64), nullable=False, primary_key=True)
    ts = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    value = Column(Float(precision=53), nullable=False)
    source = Column(String(64), nullable=False)


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
