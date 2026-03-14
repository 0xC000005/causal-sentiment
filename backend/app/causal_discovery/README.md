# Causal Discovery

A data-driven alternative to the main sentiment analysis. Instead of hand-crafting nodes, edges, and LLM-assigned sentiment scores, this module discovers the causal graph structure computationally from historical time-series data.

## How It Differs From the Main System

| Aspect | Main system (sentiment analysis) | This module (causal discovery) |
|--------|----------------------------------|-------------------------------|
| Nodes | 52 hand-picked in `topology.py` | All available from yfinance/FRED, filtered by importance |
| Edges | 117 hand-drawn with expert weights | Discovered by PCMCI/VARLiNGAM from data |
| Edge weights | Expert-defined base + Pearson dynamic | Fully learned from causal discovery algorithms |
| Edge direction | Expert-defined (positive/negative) | Learned from data |
| Node scores | LLM API call per node (~$0.04 each, ~30s) | Z-score computation (free, instant) |
| Display polarity | Implicit in LLM reasoning | Anchor propagation (5-10 anchors, rest inferred) |
| Scalability | ~52 nodes max (LLM cost/speed bottleneck) | Hundreds of nodes (no API calls for scoring) |

## Architecture

```
Data sources (yfinance, FRED, CBOE, ECB, etc.)
        |
        v
node_values table (TimescaleDB hypertable)
        |
        v
Aligned daily matrix (time_bucket + last)
        |
        +---> Z-score computation (rolling 90-day window)
        |           |
        |           v
        |     Display scores: z_score x polarity -> [-1, +1]
        |
        +---> Causal discovery (PCMCI / VARLiNGAM)
        |           |
        |           v
        |     Discovered edges with direction + weight
        |
        +---> Node importance (centrality, variance)
                    |
                    v
              Filtered graph (top N nodes)
                    |
                    v
              Anchor propagation (infer display polarity)
                    |
                    v
              3D visualization (reuse existing Graph3D)
```

## Pipeline

### Phase 1: Data Collection

Fetch historical daily values from free sources into the `node_values` table.

**Data sources:**

| Source | What | Nodes covered | API key needed |
|--------|------|--------------|----------------|
| yfinance | Equities, ETFs, futures, forex, volatility indices | SPY, QQQ, EURUSD=X, ^MOVE, ^SKEW, etc. | No |
| FRED | Macro indicators, rates, credit spreads | FEDFUNDS, CPIAUCSL, DGS10, WALCL, UMCSENT, etc. | Yes (free) |
| CBOE | Put/call ratio | Daily equity P/C ratio | No (CSV download) |
| ECB | EU HICP, ECB policy rate | eu_hicp, ecb_policy_rate | No |
| BOJ | BOJ policy rate | japan_boj_policy | No |
| BIS | Central bank policy rates (PBOC) | usdcny proxy | No |
| CFTC | Commitments of Traders (weekly) | institutional_positioning | No |
| AAII | Investor sentiment survey (weekly) | retail_sentiment | No |
| Shiller | CAPE / P/E ratio (monthly, since 1871) | pe_valuations | No |
| GDELT | Global event monitoring, tone/sentiment (15-min updates) | geopolitical_risk, trade_policy, sanctions | No |
| GPR Index | Caldara-Iacoviello Geopolitical Risk Index (monthly CSV) | geopolitical_risk_index | No |
| EPU Index | Economic Policy Uncertainty (daily, on FRED as USEPUINDXD) | us_political_risk, trade_policy | No (via FRED) |

**Key ticker/series mappings (for `sources.py`):**

| Node | Source | Ticker/Series ID |
|------|--------|-----------------|
| eurusd | yfinance | `EURUSD=X` |
| usdjpy | yfinance | `USDJPY=X` |
| usdcny | yfinance | `USDCNY=X` |
| move_index | yfinance | `^MOVE` |
| skew_index | yfinance | `^SKEW` |
| brent_crude | yfinance | `BZ=F` |
| consumer_confidence | FRED | `UMCSENT` |
| pce_deflator | FRED | `PCEPILFE` |
| wage_growth | FRED | `CES0500000003` |
| fed_balance_sheet | FRED | `WALCL` |
| us_political_risk | FRED | `USEPUINDXD` |
| eurusd (alt) | FRED | `DEXUSEU` |
| usdjpy (alt) | FRED | `DEXJPUS` |
| usdcny (alt) | FRED | `DEXCHUS` |

**Backfill:** One-time job pulls 5 years of history. Then incremental daily updates via scheduler.

**Storage:**

```sql
CREATE TABLE node_values (
    node_id     TEXT             NOT NULL,
    ts          TIMESTAMPTZ      NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    source      TEXT             NOT NULL,
    PRIMARY KEY (node_id, ts)
);

SELECT create_hypertable('node_values', 'ts');
CREATE INDEX idx_node_values_node_ts ON node_values (node_id, ts DESC);
```

One row = one observation. TIMESTAMPTZ (not DATE) for future-proofing to sub-daily resolution.

**Mixed-frequency handling:** Stored at native frequency. Aligned at query time using TimescaleDB `time_bucket()` + `last()`:

```sql
SELECT node_id,
       time_bucket('1 day', ts) AS day,
       last(value, ts) AS value
FROM node_values
GROUP BY node_id, day;
```

`last(value, ts)` forward-fills sparse series (e.g., monthly CPI carried to daily).

### Phase 2: Scoring

**Z-score** as the universal score for every node:

```
z_score = (current_value - rolling_mean) / rolling_std_dev
```

Rolling window: 90 days (configurable).

Why z-score works:
- Normalizes each factor to its own history (no cross-factor scale issues)
- A z-score of +2 means "unusually high" for ANY factor, regardless of units
- Free, instant, deterministic (no API calls, no LLM inconsistency)
- The causal discovery algorithms don't care about absolute scale — they use statistical tests on the z-score time-series

Why not LLM sentiment:
- Cost: 500 nodes x $0.04 = $20/run, $80/day, $30K/year
- Speed: 500 nodes x 30s = 4+ hours per run (data becomes stale)
- Inconsistency: same input can produce different scores on different days
- Unnecessary: numerical data has no "sentiment" — direction is computable

### Phase 3: Edge Discovery

Feed the z-score time-series matrix into causal discovery algorithms.

**Primary algorithms:**

| Algorithm | Library | What it discovers | Why |
|-----------|---------|------------------|-----|
| PCMCI+ | `tigramite` | Lagged + contemporaneous causal edges | Purpose-built for time-series; handles autocorrelation |
| VARLiNGAM | `lingam` | Causal ordering + directed edges | Exploits non-Gaussianity of financial returns (fat tails are a feature) |
| DynamicsBlockState | `graph-tool` | Graph structure from dynamics + posterior edge probabilities | Bayesian inference; MDL prevents overfitting; also discovers hierarchical communities |

**Alternative algorithms** (evaluated but not primary — see research log for details):
- **NOTEARS/GOLEM** (`gCastle`) — score-based continuous optimization, scales well but assumes i.i.d. data
- **PC/FCI** (`causal-learn`) — constraint-based iterative edge elimination; FCI handles latent confounders
- **DoWhy** — not for discovery but for validation (`arrow_strength()`, `refute_causal_structure()`)

**Output:** Adjacency matrix with edge weights and directions (positive/negative).

**Per-regime discovery:** RPCMCI (tigramite) discovers different causal structures per market regime (risk-on vs risk-off). This maps to the existing `regimes.py`. Different regimes may have fundamentally different causal relationships — e.g., correlations spike during crises.

**Validation step:** After discovering edges, use DoWhy `refute_causal_structure()` to test whether discovered edges hold under refutation tests. Use `arrow_strength()` to quantify edge weights more rigorously. Optionally compare discovered edges against the existing expert graph in `topology.py` to produce a consensus:
- Confirmed edges (data + expert agree)
- Flagged edges (expert only, no data support)
- Suggested edges (data finds, expert missed)

**Challenges addressed:**
- Non-stationarity: use returns or z-scores, not raw levels
- Latent confounders: PCMCI+ handles contemporaneous effects; FCI variant handles latent variables
- Fat tails: LiNGAM leverages non-Gaussianity; tigramite offers kernel-based independence tests
- Short samples: sparsity priors in the algorithms; domain knowledge as prior
- Feedback loops: time-indexed variables unroll cycles into a DAG

### Phase 4: Node Importance

Not all discovered nodes are equally important. Filter to the most relevant using:

- **Betweenness centrality** — nodes that sit on many causal paths
- **Eigenvector centrality** — nodes connected to other important nodes
- **Variance** — nodes with high z-score variability (active factors)
- **Degree** — nodes with many causal connections

Display the top N most important nodes (configurable).

### Phase 5: Display Polarity (Anchor Propagation)

Z-scores tell you "above/below average" but not "good/bad." For red/green coloring, we need to know: does "up" mean positive or negative?

**Approach:** Define 5-10 anchor nodes with unambiguous polarity (the ONLY manual input in the system):

```
sp500:        up = positive (+1)
nasdaq:       up = positive (+1)
us_gdp:       up = positive (+1)
unemployment: up = negative (-1)
```

Propagate polarity through the discovered causal edges using BFS:

```
Anchor: S&P 500 = +1
    --(positive edge)--> Tech Sector = +1
    --(negative edge)--> VIX = -1
        --(positive edge)--> Put/Call Ratio = -1
            --(negative edge)--> Consumer Confidence = +1
```

Sign flips on negative edges, preserves on positive. Multiple paths sum with weights (stronger causal paths dominate). This is functionally identical to the existing `propagation.py` code — seed anchors, run BFS, read off signs.

**Display score:** `z_score x inferred_polarity`, clamped to [-1, +1].

**Alternatives considered and rejected:**
- **Option A (no judgment):** Blue/orange instead of green/red. Too unintuitive for users.
- **Option B (manual flag per node):** `equity_aligned = +1/-1` for every node. Defeats the purpose of a computational network.
- **Option C (correlation with S&P 500):** Naive — a factor can correlate +0.5 with SPY but -0.3 with GDP, giving conflicting signs. First-order only, no transitivity.

**Why anchor propagation (Option D) is better:**
- Respects causal direction, not just co-movement
- Handles transitivity through arbitrarily long causal chains
- Resolves conflicting signals via edge-weighted path summation
- Uses the graph structure we already discovered (no extra computation)

### Phase 6: Propagation + Visualization

Reuse the existing 3D force-directed graph (Graph3D component) with the discovered graph. The propagation engine (`propagation.py`) works with any NetworkX DiGraph — it doesn't care whether edges were hand-drawn or discovered.

Signal propagation formula (same as main system):

```
propagated = parent_signal x edge_weight x direction_sign x (1 - decay_rate)
```

- Decay: 30% per hop (configurable)
- Max depth: 4 hops
- Blending: 70% existing + 30% propagated (configurable)
- Clamped to [-1, +1]

## File Structure

```
backend/app/causal_discovery/
|-- __init__.py
|-- README.md                  <-- This file
|
|-- models.py                  <-- NodeValue SQLAlchemy model + hypertable setup
|
|-- pipeline/
|   |-- __init__.py
|   |-- sources.py             <-- Ticker/series registry (what to fetch, node_id mappings)
|   |-- fetchers.py            <-- yfinance + FRED bulk fetch with rate-limit handling
|   |-- backfill.py            <-- One-time historical backfill job (5 years)
|   |-- scheduler.py           <-- Incremental daily update jobs (planned)
|
|-- engine/
|   |-- __init__.py
|   |-- matrix.py              <-- Build aligned daily matrix from node_values
|   |-- zscore.py              <-- Rolling z-score computation per node
|   |-- causal.py              <-- PCMCI/VARLiNGAM wrapper -> edges + directions
|   |-- importance.py          <-- Node filtering (centrality, variance)
|   |-- anchors.py             <-- Anchor propagation for display polarity
|
|-- api/
    |-- __init__.py
    |-- routes.py              <-- REST API endpoints

frontend/src/components/causal-discovery/   (planned — not yet implemented)
|-- CausalGraph3D.tsx          <-- 3D visualization (or mode switch on existing Graph3D)
|-- CausalPanel.tsx            <-- Controls, stats, configuration
```

## API Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| POST | `/api/causal/backfill` | Trigger historical data backfill (background) | Implemented |
| GET | `/api/causal/backfill/status` | Check backfill progress | Implemented |
| POST | `/api/causal/discover` | Run causal discovery on stored data (background) | Implemented |
| GET | `/api/causal/discover/status` | Check discovery progress | Implemented |
| GET | `/api/causal/graph` | Get discovered graph (nodes + edges + scores) | Implemented |
| GET | `/api/causal/sources` | List all registered data sources | Implemented |
| GET | `/api/causal/stats` | Row counts and date ranges per node | Implemented |
| GET | `/api/causal/matrix?days=252` | Get aligned daily matrix | Planned |
| GET | `/api/causal/node/{id}` | Get single node detail (z-score history, edges) | Planned |
| POST | `/api/causal/anchors` | Set/update anchor nodes for polarity | Planned |

## Integration With Main Codebase

Minimal touchpoints (~3 lines in existing code):

```python
# backend/app/main.py
from app.causal_discovery.api.routes import router as causal_router
app.include_router(causal_router)
# In lifespan: await create_hypertable_if_needed()
```

Everything else is self-contained within `backend/app/causal_discovery/`.

## Dependencies

New Python packages (in `requirements.txt`):

```
pandas>=2.2.0        # Matrix operations, DataFrame alignment (MIT)
tigramite>=5.2       # PCMCI/PCMCI+ for time-series causal discovery (GPL-3.0)
lingam>=1.12         # VARLiNGAM for causal ordering (MIT)
```

Planned (not yet added):
```
dowhy>=0.14          # Edge validation: arrow_strength, refute_causal_structure (MIT)
graph-tool           # Bayesian network reconstruction (LGPL-3.0, conda only)
```

Already available in the project:
- `networkx` (graph operations)
- `numpy`, `scipy` (matrix operations, statistics)
- `sqlalchemy[asyncio]`, `asyncpg` (database)
- `yfinance` (market data)
- `httpx` (FRED API calls)

## Configuration

New settings (added to `config.py` or own config):

```python
# Causal Discovery
causal_zscore_window: int = 90           # Rolling z-score window in days
causal_backfill_years: int = 5           # How far back to fetch history
causal_max_display_nodes: int = 100      # Max nodes to show in visualization
causal_discovery_algorithm: str = "pcmci" # "pcmci" or "varlingam"
causal_min_edge_weight: float = 0.1      # Minimum weight to display an edge
causal_anchor_nodes: str = "sp500,nasdaq,us_gdp_growth,unemployment_rate"  # comma-separated (env vars are strings)
```

## What's Manual vs. Computational

| Component | Manual | Computational |
|-----------|--------|--------------|
| Data source registry | Which tickers/series to fetch (sources.py) | -- |
| Anchor nodes | 5-10 nodes with known polarity | -- |
| Everything else | -- | Nodes, edges, weights, directions, scores, display polarity |

## Decisions Made

1. **Edge storage:** Write discovered edges to the existing `edges` table (same schema: source_id, target_id, direction, weight). This allows reusing the existing propagation engine and frontend without changes.
2. **LLM role:** Repositioned as interpreter/deep-dive tool, not primary scorer. LLM can explain anomalies, validate surprising edges, and analyze the top N most important nodes selectively.

## Open Questions

1. How often to re-run causal discovery? Daily? Weekly? On-demand only?
2. Should the frontend have a toggle to switch between "expert graph" and "discovered graph"?
3. Maximum number of nodes before the 3D visualization becomes unusable?
4. Should we run graph-tool's `UncertainBlockState` on the existing expert graph as a first validation step before full causal discovery?

## References

Key papers informing this design (see research log for full details):

| Paper | Year | Relevance |
|-------|------|-----------|
| DYNOTEARS (Pamfil et al.) | 2020 | NOTEARS for time-series, validated on financial data |
| CausalDynamics (Herdeanu et al.) | 2025 | Benchmark: PCMCI+ and VARLiNGAM outperform deep learning for causal discovery |
| CauScientist (Peng et al.) | 2026 | LLMs as hypothesis generators + statistical verifiers |
| Causal-Copilot (Wang et al.) | 2025 | Autonomous agent integrating 20+ causal algorithms |
| Peixoto network reconstruction | 2019 | Infer graph from dynamics using Bayesian SBMs |
| TNCM-VAE (Thumm et al.) | 2025 | DAG-constrained VAE for counterfactual financial time-series |
