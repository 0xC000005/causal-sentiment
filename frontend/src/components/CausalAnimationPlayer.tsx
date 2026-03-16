"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useGraphStore } from "@/hooks/useGraphData";
import { useCausalStore } from "@/hooks/useCausalStore";
import { API_URL } from "@/lib/config";
import type { ForceGraphLink } from "@/types/graph";

interface AnimationFrameEdge {
  source: string;
  target: string;
  correlation: number;
  survives: boolean;
}

interface AnimationFrame {
  frame: number;
  edge_count: number;
  edges: AnimationFrameEdge[];
}

interface AnimationData {
  snapshot_id: number;
  algorithm: string;
  n_frames: number;
  total_edges: number;
  surviving_edges: number;
  frames: AnimationFrame[];
}

type SpeedLabel = "Slow" | "Normal" | "Fast";
const SPEED_OPTIONS: Record<SpeedLabel, number> = {
  Slow: 300,
  Normal: 150,
  Fast: 50,
};

function transformFrameEdges(edges: AnimationFrameEdge[]): ForceGraphLink[] {
  return edges.map((e) => ({
    source: e.source,
    target: e.target,
    direction: "positive",
    weight: e.correlation,
    baseWeight: e.correlation,
    dynamicWeight: e.correlation,
    description: e.survives ? "Causal edge" : "Correlation only",
  }));
}

export default function CausalAnimationPlayer() {
  const [animationData, setAnimationData] = useState<AnimationData | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<SpeedLabel>("Normal");
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const setIsAnimating = useCausalStore((s) => s.setIsAnimating);
  const graphSource = useCausalStore((s) => s.graphSource);
  const currentGraph = useCausalStore((s) => s.currentGraph);
  const getForceGraphData = useCausalStore((s) => s.getForceGraphData);

  // Only show in discovered mode
  const isDiscovered = graphSource === "discovered";

  // Fetch animation frames
  const fetchFrames = useCallback(async () => {
    setLoading(true);
    setFetchError(null);
    try {
      const res = await fetch(`${API_URL}/api/causal/graph/animate?n_frames=30`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: AnimationData = await res.json();
      setAnimationData(data);
      setCurrentFrame(0);
    } catch (e) {
      setFetchError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Inject a frame's edges into the graph store (nodes untouched)
  const injectFrame = useCallback(
    (frameIndex: number) => {
      if (!animationData) return;
      const frame = animationData.frames[frameIndex];
      if (!frame) return;
      const links = transformFrameEdges(frame.edges);
      useGraphStore.setState({ links });
    },
    [animationData],
  );

  // Fetch frames when player becomes visible
  useEffect(() => {
    if (visible && !animationData && !loading) {
      fetchFrames();
    }
  }, [visible, animationData, loading, fetchFrames]);

  // Play/pause interval
  useEffect(() => {
    if (isPlaying && animationData) {
      intervalRef.current = setInterval(() => {
        setCurrentFrame((prev) => {
          const next = prev + 1;
          if (next >= animationData.frames.length) {
            // Reached the end -- stop
            setIsPlaying(false);
            return prev;
          }
          return next;
        });
      }, SPEED_OPTIONS[speed]);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isPlaying, speed, animationData]);

  // Inject edges whenever currentFrame changes
  useEffect(() => {
    if (animationData) {
      injectFrame(currentFrame);
    }
  }, [currentFrame, animationData, injectFrame]);

  // Track animating state in causal store
  useEffect(() => {
    const animating = visible && animationData !== null;
    setIsAnimating(animating);
    return () => setIsAnimating(false);
  }, [visible, animationData, setIsAnimating]);

  // When animation is closed, restore the discovered graph
  const handleClose = useCallback(() => {
    setIsPlaying(false);
    setVisible(false);
    setAnimationData(null);
    setCurrentFrame(0);
    // Restore the discovered graph edges
    if (currentGraph) {
      const { links } = getForceGraphData();
      useGraphStore.setState({ links });
    }
  }, [currentGraph, getForceGraphData]);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setCurrentFrame(0);
  }, []);

  const handlePlayPause = useCallback(() => {
    if (!animationData) return;
    // If at the end, reset to start before playing
    if (currentFrame >= animationData.frames.length - 1) {
      setCurrentFrame(0);
      setIsPlaying(true);
    } else {
      setIsPlaying((prev) => !prev);
    }
  }, [animationData, currentFrame]);

  const handleScrub = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(e.target.value);
      setCurrentFrame(value);
      // Pause when scrubbing
      setIsPlaying(false);
    },
    [],
  );

  if (!isDiscovered) return null;

  // Toggle button when player is hidden
  if (!visible) {
    return (
      <button
        onClick={() => setVisible(true)}
        className="bg-gray-900/95 backdrop-blur border border-gray-700 rounded-lg px-3 py-2 text-gray-300 text-xs font-semibold hover:bg-gray-800/95 transition-colors"
      >
        Edge Discovery Animation
      </button>
    );
  }

  const totalFrames = animationData?.frames.length ?? 0;
  const frame = animationData?.frames[currentFrame];
  const edgeCount = frame?.edge_count ?? 0;
  const totalEdges = animationData?.total_edges ?? 0;
  const survivingEdges = animationData?.surviving_edges ?? 0;
  const atEnd = animationData ? currentFrame >= totalFrames - 1 : false;

  return (
    <div className="absolute bottom-14 left-1/2 -translate-x-1/2 z-20 bg-gray-900/95 backdrop-blur border border-gray-700 rounded-lg px-3 py-2 flex items-center gap-3 min-w-[520px]">
      {loading && (
        <div className="text-gray-400 text-xs">Loading frames...</div>
      )}

      {fetchError && (
        <div className="text-red-400 text-xs flex items-center gap-2">
          <span>Error: {fetchError}</span>
          <button
            onClick={fetchFrames}
            className="text-gray-400 hover:text-gray-200 underline"
          >
            Retry
          </button>
        </div>
      )}

      {animationData && !loading && (
        <>
          {/* Play/Pause */}
          <button
            onClick={handlePlayPause}
            className="text-gray-200 hover:text-white text-sm w-6 h-6 flex items-center justify-center flex-shrink-0"
            title={isPlaying ? "Pause" : atEnd ? "Replay" : "Play"}
          >
            {isPlaying ? "\u23F8" : "\u25B6"}
          </button>

          {/* Scrubber */}
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            value={currentFrame}
            onChange={handleScrub}
            className="flex-1 h-1 accent-purple-500 cursor-pointer"
          />

          {/* Edge count */}
          <div className="text-gray-200 text-xs font-mono flex-shrink-0 w-24 text-center">
            <span className={edgeCount <= survivingEdges ? "text-purple-400" : ""}>
              {edgeCount}
            </span>
            <span className="text-gray-500"> / {totalEdges}</span>
            <span className="text-gray-500 text-[10px]"> edges</span>
          </div>

          {/* Speed control */}
          <select
            value={speed}
            onChange={(e) => setSpeed(e.target.value as SpeedLabel)}
            className="bg-gray-800 border border-gray-600 text-gray-300 text-xs rounded px-1.5 py-1 focus:outline-none focus:border-gray-500 flex-shrink-0"
          >
            {Object.keys(SPEED_OPTIONS).map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </select>

          {/* Reset */}
          <button
            onClick={handleReset}
            className="text-gray-400 hover:text-gray-200 text-xs flex-shrink-0"
            title="Reset to frame 0"
          >
            Reset
          </button>

          {/* Close */}
          <button
            onClick={handleClose}
            className="text-gray-500 hover:text-gray-300 text-xs flex-shrink-0 ml-1"
            title="Close animation player"
          >
            ✕
          </button>
        </>
      )}
    </div>
  );
}
