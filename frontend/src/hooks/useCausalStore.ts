"use client";

import { create } from "zustand";
import type {
  CausalGraphListItem,
  CausalGraphSnapshot,
  CausalNode,
  CausalEdge,
  ForceGraphNode,
  ForceGraphLink,
} from "@/types/graph";

import { API_URL } from "@/lib/config";

interface DiscoveryStatus {
  state: "idle" | "running" | "completed" | "failed";
  algorithm?: string;
  scoring?: string;
  error?: string;
  last_result?: { snapshot_id?: number } | null;
}

interface CausalStore {
  snapshots: CausalGraphListItem[];
  currentGraph: CausalGraphSnapshot | null;
  graphSource: "expert" | "discovered";
  topN: number | null;
  loading: boolean;
  discovering: boolean;
  error: string | null;

  fetchSnapshots: () => Promise<void>;
  loadGraph: (id?: number, runName?: string) => Promise<void>;
  triggerDiscovery: (algorithm: string, scoring: string) => Promise<void>;
  pollDiscoveryStatus: () => Promise<boolean>;
  setGraphSource: (source: "expert" | "discovered") => void;
  setTopN: (topN: number | null) => void;
  setError: (error: string | null) => void;
  getForceGraphData: () => { nodes: ForceGraphNode[]; links: ForceGraphLink[] };
}

function transformCausalNode(node: CausalNode): ForceGraphNode {
  return {
    id: node.id,
    label: node.id,
    nodeType: "discovered",
    sentiment: node.display_sentiment,
    confidence: node.importance,
    centrality: node.importance,
    description: `Discovered node (z-score: ${node.zscore.toFixed(2)}, polarity: ${node.polarity.toFixed(2)})`,
    evidence: [],
  };
}

function transformCausalEdge(edge: CausalEdge): ForceGraphLink {
  return {
    source: edge.source,
    target: edge.target,
    direction: edge.direction,
    weight: edge.weight,
    baseWeight: edge.weight,
    dynamicWeight: edge.weight,
    description: `Lag: ${edge.lag}, Weight: ${edge.weight.toFixed(3)}`,
  };
}

export const useCausalStore = create<CausalStore>((set, get) => ({
  snapshots: [],
  currentGraph: null,
  graphSource: "expert",
  topN: null,
  loading: false,
  discovering: false,
  error: null,

  fetchSnapshots: async () => {
    try {
      const res = await fetch(`${API_URL}/api/causal/graphs`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: CausalGraphListItem[] = await res.json();
      set({ snapshots: data });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  loadGraph: async (id?: number, runName?: string) => {
    set({ loading: true, error: null });
    try {
      const params = new URLSearchParams();
      if (id != null) params.set("id", String(id));
      if (runName) params.set("run_name", runName);
      const { topN } = get();
      if (topN != null) params.set("top_n", String(topN));

      const res = await fetch(`${API_URL}/api/causal/graph?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: CausalGraphSnapshot = await res.json();
      set({ currentGraph: data, loading: false, graphSource: "discovered" });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  triggerDiscovery: async (algorithm: string, scoring: string) => {
    set({ discovering: true, error: null });
    try {
      const runName = `${algorithm}_${scoring}`;
      const params = new URLSearchParams({
        algorithm,
        scoring,
        run_name: runName,
      });
      const res = await fetch(`${API_URL}/api/causal/discover?${params.toString()}`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Discovery started — poll for completion
    } catch (e) {
      set({ error: (e as Error).message, discovering: false });
    }
  },

  pollDiscoveryStatus: async () => {
    try {
      const res = await fetch(`${API_URL}/api/causal/discover/status`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: DiscoveryStatus = await res.json();

      if (data.state === "completed") {
        set({ discovering: false });
        // Refresh snapshots and load the latest graph
        await get().fetchSnapshots();
        // Load the specific snapshot that was just created
        if (data.last_result?.snapshot_id) {
          await get().loadGraph(data.last_result.snapshot_id);
        } else {
          await get().loadGraph();
        }
        return true;
      } else if (data.state === "failed") {
        set({
          discovering: false,
          error: data.error || "Discovery failed",
        });
        return true;
      }
      // Still running
      return false;
    } catch (e) {
      set({ error: (e as Error).message, discovering: false });
      return true;
    }
  },

  setGraphSource: (source: "expert" | "discovered") => {
    set({ graphSource: source });
  },

  setTopN: (topN: number | null) => {
    set({ topN });
  },

  setError: (error: string | null) => {
    set({ error });
  },

  getForceGraphData: () => {
    const { currentGraph } = get();
    if (!currentGraph) return { nodes: [], links: [] };

    const nodes = currentGraph.nodes.map(transformCausalNode);
    const links = currentGraph.edges.map(transformCausalEdge);
    return { nodes, links };
  },
}));
