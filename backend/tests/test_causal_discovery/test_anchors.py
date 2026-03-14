"""Tests for anchor polarity propagation."""
import networkx as nx
from app.causal_discovery.engine.anchors import propagate_polarity


def test_positive_edge_preserves_polarity():
    g = nx.DiGraph()
    g.add_edge("sp500", "tech", direction="positive", weight=0.8)
    assert propagate_polarity(g, {"sp500": 1})["tech"] == 1


def test_negative_edge_flips_polarity():
    g = nx.DiGraph()
    g.add_edge("sp500", "vix", direction="negative", weight=0.8)
    assert propagate_polarity(g, {"sp500": 1})["vix"] == -1


def test_transitivity():
    g = nx.DiGraph()
    g.add_edge("sp500", "vix", direction="negative", weight=0.8)
    g.add_edge("vix", "put_call", direction="positive", weight=0.6)
    polarity = propagate_polarity(g, {"sp500": 1})
    assert polarity["vix"] == -1
    assert polarity["put_call"] == -1


def test_unconnected_node_gets_zero():
    g = nx.DiGraph()
    g.add_node("sp500")
    g.add_node("isolated")
    assert propagate_polarity(g, {"sp500": 1}).get("isolated", 0) == 0
