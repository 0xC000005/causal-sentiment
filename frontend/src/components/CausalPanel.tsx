"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useCausalStore } from "@/hooks/useCausalStore";
import { useGraphStore } from "@/hooks/useGraphData";

const ALGORITHM_EXPLANATIONS: Record<string, string> = {
  pcmci: "PCMCI+ discovers directed causal edges by testing if past values of A predict future B, after controlling for all other variables. Only statistically significant relationships (p < 0.01) survive.",
  granger: "Granger causality tests each pair independently: does past A help predict B? Simpler than PCMCI+ but doesn't control for confounders, so more edges may be indirect.",
};

const SCORING_EXPLANATIONS: Record<string, string> = {
  zscore: "Z-Score measures how unusual each factor's current value is relative to its 90-day rolling average. The network shows which factors' deviations from normal predict each other.",
  returns: "Log returns capture day-to-day price changes. The network shows which factors' daily moves predict other factors' moves the next day — a trader's perspective.",
  volatility: "Rolling volatility (20-day) measures how choppy each factor is. The network shows how fear and uncertainty spread between factors — a risk manager's perspective.",
};

export default function CausalPanel() {
  const [collapsed, setCollapsed] = useState(true);
  const [algorithm, setAlgorithm] = useState("pcmci");
  const [scoring, setScoring] = useState("zscore");
  const [sliderValue, setSliderValue] = useState(20);
  const [useTopN, setUseTopN] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const graphSource = useCausalStore((s) => s.graphSource);
  const setGraphSource = useCausalStore((s) => s.setGraphSource);
  const snapshots = useCausalStore((s) => s.snapshots);
  const currentGraph = useCausalStore((s) => s.currentGraph);
  const loading = useCausalStore((s) => s.loading);
  const discovering = useCausalStore((s) => s.discovering);
  const error = useCausalStore((s) => s.error);
  const fetchSnapshots = useCausalStore((s) => s.fetchSnapshots);
  const loadGraph = useCausalStore((s) => s.loadGraph);
  const triggerDiscovery = useCausalStore((s) => s.triggerDiscovery);
  const pollDiscoveryStatus = useCausalStore((s) => s.pollDiscoveryStatus);
  const setTopN = useCausalStore((s) => s.setTopN);
  const setError = useCausalStore((s) => s.setError);
  const getForceGraphData = useCausalStore((s) => s.getForceGraphData);

  const fetchExpertGraph = useGraphStore((s) => s.fetchGraph);
  const expertNodes = useGraphStore((s) => s.nodes);

  // Fetch snapshots on mount
  useEffect(() => {
    fetchSnapshots();
  }, [fetchSnapshots]);

  // Inject discovered graph data into the main graph store when source is "discovered"
  useEffect(() => {
    if (graphSource === "discovered" && currentGraph) {
      const { nodes, links } = getForceGraphData();
      useGraphStore.setState({ nodes, links });
    }
  }, [graphSource, currentGraph, getForceGraphData]);

  // Reload expert graph when switching back
  const handleSourceChange = useCallback(
    (source: "expert" | "discovered") => {
      setGraphSource(source);
      if (source === "expert") {
        fetchExpertGraph();
      } else if (source === "discovered" && !currentGraph && snapshots.length > 0) {
        loadGraph(snapshots[0].id);
      }
    },
    [setGraphSource, fetchExpertGraph, currentGraph, snapshots, loadGraph],
  );

  // Poll discovery status
  useEffect(() => {
    if (discovering && !pollRef.current) {
      pollRef.current = setInterval(async () => {
        const done = await pollDiscoveryStatus();
        if (done && pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      }, 2000);
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [discovering, pollDiscoveryStatus]);

  // Handle top_n changes
  useEffect(() => {
    if (useTopN) {
      setTopN(sliderValue);
    } else {
      setTopN(null);
    }
  }, [useTopN, sliderValue, setTopN]);

  // Auto-load matching snapshot when algorithm or scoring changes in discovered mode
  useEffect(() => {
    if (graphSource !== "discovered") return;
    const runName = `${algorithm}_${scoring}`;
    const match = snapshots.find((s) => s.run_name === runName);
    if (match) {
      loadGraph(match.id);
    } else {
      useCausalStore.setState({ currentGraph: null });
    }
  }, [algorithm, scoring, graphSource, snapshots, loadGraph]);

  const matchingRunName = `${algorithm}_${scoring}`;
  const hasExistingSnapshot = snapshots.some((s) => s.run_name === matchingRunName);

  const handleRunDiscovery = useCallback(async () => {
    const existing = snapshots.find((s) => s.run_name === matchingRunName);
    if (existing) {
      await loadGraph(existing.id);
    } else {
      await triggerDiscovery(algorithm, scoring);
    }
  }, [snapshots, matchingRunName, loadGraph, triggerDiscovery, algorithm, scoring]);

  const handleReloadWithTopN = useCallback(() => {
    if (currentGraph) {
      loadGraph(currentGraph.id);
    }
  }, [currentGraph, loadGraph]);

  return (
    <div className="absolute top-4 right-4 z-10">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="bg-gray-900/95 backdrop-blur border border-gray-700 rounded-lg px-3 py-2 text-gray-300 text-xs font-semibold hover:bg-gray-800/95 transition-colors flex items-center gap-2"
      >
        <span>Causal Discovery</span>
        <span className="text-[10px] text-gray-500">
          {collapsed ? "\u25BC" : "\u25B2"}
        </span>
      </button>

      {!collapsed && (
        <div className="mt-2 bg-gray-900/95 backdrop-blur border border-gray-700 rounded-lg p-3 w-64">
          {/* Graph Source */}
          <div className="mb-3">
            <label className="text-[10px] text-gray-500 uppercase block mb-1">
              Graph Source
            </label>
            <select
              value={graphSource}
              onChange={(e) =>
                handleSourceChange(e.target.value as "expert" | "discovered")
              }
              className="w-full bg-gray-800 border border-gray-600 text-gray-200 text-xs rounded px-2 py-1.5 focus:outline-none focus:border-gray-500"
            >
              <option value="expert">
                Expert ({expertNodes.length} nodes)
              </option>
              <option value="discovered">Discovered</option>
            </select>
          </div>

          {graphSource === "discovered" && (
            <>
              {/* Algorithm */}
              <div className="mb-2">
                <label className="text-[10px] text-gray-500 uppercase block mb-1">
                  Algorithm
                </label>
                <select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-600 text-gray-200 text-xs rounded px-2 py-1.5 focus:outline-none focus:border-gray-500"
                >
                  <option value="pcmci">PCMCI+</option>
                  <option value="granger">Granger</option>
                </select>
              </div>

              {/* Scoring */}
              <div className="mb-3">
                <label className="text-[10px] text-gray-500 uppercase block mb-1">
                  Scoring
                </label>
                <select
                  value={scoring}
                  onChange={(e) => setScoring(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-600 text-gray-200 text-xs rounded px-2 py-1.5 focus:outline-none focus:border-gray-500"
                >
                  <option value="zscore">Z-Score</option>
                  <option value="returns">Returns</option>
                  <option value="volatility">Volatility</option>
                </select>
              </div>

              {/* Run Discovery */}
              <button
                onClick={handleRunDiscovery}
                disabled={discovering}
                className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-xs py-2 px-3 rounded transition-colors mb-3"
              >
                {discovering ? "Discovering..." : hasExistingSnapshot ? "Load Result" : "Run Discovery"}
              </button>

              {discovering && (
                <div className="mb-3">
                  <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-purple-500 rounded-full"
                      style={{
                        animation:
                          "causal-progress 1.5s ease-in-out infinite",
                        width: "40%",
                      }}
                    />
                  </div>
                  <div className="text-[10px] text-gray-400 mt-0.5">
                    Running {algorithm} with {scoring} scoring...
                  </div>
                  <style jsx>{`
                    @keyframes causal-progress {
                      0% {
                        margin-left: 0%;
                      }
                      50% {
                        margin-left: 60%;
                      }
                      100% {
                        margin-left: 0%;
                      }
                    }
                  `}</style>
                </div>
              )}

              {/* Node importance slider */}
              <div className="mb-3">
                <div className="flex items-center justify-between mb-1">
                  <label className="text-[10px] text-gray-500 uppercase">
                    Node Filter
                  </label>
                  <button
                    onClick={() => {
                      setUseTopN(!useTopN);
                    }}
                    className={`text-[10px] px-1.5 py-0.5 rounded transition-colors ${
                      useTopN
                        ? "bg-purple-700 text-white"
                        : "bg-gray-700 text-gray-400"
                    }`}
                  >
                    {useTopN ? `Top ${sliderValue}` : "All"}
                  </button>
                </div>
                {useTopN && (
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min={5}
                      max={35}
                      value={sliderValue}
                      onChange={(e) => setSliderValue(Number(e.target.value))}
                      onMouseUp={handleReloadWithTopN}
                      onTouchEnd={handleReloadWithTopN}
                      className="flex-1 h-1 accent-purple-500"
                    />
                    <span className="text-[10px] text-gray-400 w-5 text-right">
                      {sliderValue}
                    </span>
                  </div>
                )}
              </div>

              {/* Current graph info */}
              {currentGraph && (
                <div className="mb-3 bg-gray-800/80 rounded p-2">
                  <div className="text-[10px] text-gray-500 uppercase mb-1">
                    Current Graph
                  </div>
                  <div className="text-[11px] text-gray-300 space-y-0.5">
                    <div>
                      Algorithm:{" "}
                      <span className="text-purple-400">
                        {currentGraph.algorithm}
                      </span>
                    </div>
                    <div>
                      Scoring:{" "}
                      <span className="text-purple-400">
                        {currentGraph.parameters.scoring}
                      </span>
                    </div>
                    <div>
                      Nodes:{" "}
                      <span className="text-gray-200">
                        {currentGraph.summary.node_count}
                      </span>{" "}
                      · Edges:{" "}
                      <span className="text-gray-200">
                        {currentGraph.summary.edge_count}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Explanation */}
              {currentGraph && (
                <div className="mb-3 bg-gray-800/50 rounded p-2 text-[10px] text-gray-400 space-y-1.5">
                  <div className="text-[10px] text-gray-500 uppercase font-semibold">How to read this graph</div>
                  <div>{ALGORITHM_EXPLANATIONS[currentGraph.algorithm] || "Causal edges discovered from historical data."}</div>
                  <div>{SCORING_EXPLANATIONS[currentGraph.parameters?.scoring] || ""}</div>
                  <div>
                    <span className="text-green-400">Green</span> = above average (positive polarity) ·
                    <span className="text-red-400"> Red</span> = below average or negative polarity ·
                    Node size = importance (centrality in the discovered network)
                  </div>
                </div>
              )}

              {/* Saved snapshots */}
              {snapshots.length > 0 && (
                <div>
                  <div className="text-[10px] text-gray-500 uppercase mb-1">
                    Saved Snapshots
                  </div>
                  <div className="max-h-32 overflow-y-auto space-y-1">
                    {snapshots.map((s) => (
                      <button
                        key={s.id}
                        onClick={() => loadGraph(s.id)}
                        className={`w-full text-left px-2 py-1.5 rounded text-[11px] transition-colors ${
                          currentGraph?.id === s.id
                            ? "bg-purple-900/50 text-purple-300 border border-purple-700/50"
                            : "bg-gray-800/60 text-gray-400 hover:bg-gray-700/60 hover:text-gray-300"
                        }`}
                      >
                        <div className="flex justify-between items-center">
                          <span className="font-medium truncate">
                            {s.algorithm}_{String((s.parameters as Record<string, unknown>)?.scoring ?? "zscore")}
                          </span>
                          <span className="text-[9px] text-gray-500 ml-1 flex-shrink-0">
                            {s.node_count}n · {s.edge_count}e
                          </span>
                        </div>
                        <div className="text-[9px] text-gray-500">
                          {new Date(s.created_at).toLocaleString()}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="mt-2 bg-red-900/50 border border-red-700/50 rounded p-2">
                  <div className="text-[10px] text-red-300 flex justify-between items-start">
                    <span>{error}</span>
                    <button
                      onClick={() => setError(null)}
                      className="text-red-500 hover:text-red-300 ml-1 flex-shrink-0"
                    >
                      x
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
