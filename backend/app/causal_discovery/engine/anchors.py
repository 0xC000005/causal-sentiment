"""Anchor polarity propagation via BFS through a causal graph.

Anchors are nodes with known polarity (e.g. sp500 = +1 means "positive is
risk-on").  Polarity propagates through edges: positive edges preserve the
sign, negative edges flip it.  When multiple paths reach a node the
accumulated signals are summed and the final polarity is the sign of the sum.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Dict

import networkx as nx

logger = logging.getLogger(__name__)


def propagate_polarity(
    g: nx.DiGraph,
    anchors: Dict[str, int],
) -> Dict[str, int]:
    """BFS polarity propagation from anchor nodes.

    Parameters
    ----------
    g : nx.DiGraph
        Causal graph.  Edges should have a ``direction`` attribute
        ("positive" or "negative") and optionally a ``weight``.
    anchors : dict[str, int]
        Mapping of anchor node_id to polarity (+1 or -1).

    Returns
    -------
    dict[str, int]
        Mapping of every reachable node_id to its inferred polarity
        (+1, -1, or 0 if signals cancel out).
    """
    # Accumulate raw signal per node (float) before converting to sign
    signal: Dict[str, float] = {}

    for anchor_id, anchor_pol in anchors.items():
        if anchor_id not in g:
            continue
        signal.setdefault(anchor_id, 0.0)
        signal[anchor_id] += float(anchor_pol)

        # BFS from this anchor
        queue: deque[tuple[str, float]] = deque()
        queue.append((anchor_id, float(anchor_pol)))
        visited: set[str] = {anchor_id}

        while queue:
            node, current_pol = queue.popleft()
            for _, successor, edge_data in g.out_edges(node, data=True):
                direction = edge_data.get("direction", "positive")
                multiplier = -1.0 if direction == "negative" else 1.0
                next_pol = current_pol * multiplier

                signal.setdefault(successor, 0.0)
                signal[successor] += next_pol

                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, next_pol))

    # Convert accumulated signal to discrete polarity
    polarity: Dict[str, int] = {}
    for node_id, val in signal.items():
        if val > 0:
            polarity[node_id] = 1
        elif val < 0:
            polarity[node_id] = -1
        else:
            polarity[node_id] = 0

    logger.info(
        "Propagated polarity from %d anchor(s) to %d node(s)",
        len(anchors),
        len(polarity),
    )
    return polarity
