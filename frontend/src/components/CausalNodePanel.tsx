"use client";

import { useNodeSelection } from "@/hooks/useNodeSelection";
import { useGraphStore } from "@/hooks/useGraphData";
import { useCausalStore } from "@/hooks/useCausalStore";
import { getNodeLabel } from "@/lib/nodeLabels";
import { sentimentToColor, edgeDirectionColor } from "@/lib/graphTransforms";
import type { CausalEdge } from "@/types/graph";

export default function CausalNodePanel() {
  const { selectedNode: rawSelectedNode, clearSelection } = useNodeSelection();
  const nodes = useGraphStore((s) => s.nodes);
  const currentGraph = useCausalStore((s) => s.currentGraph);

  // Use freshest node data from the store
  const selectedNode = rawSelectedNode
    ? nodes.find((n) => n.id === rawSelectedNode.id) ?? rawSelectedNode
    : null;

  if (!selectedNode || !currentGraph) return null;

  // Find the raw CausalNode for z-score and polarity
  const causalNode = currentGraph.nodes.find((n) => n.id === selectedNode.id);
  if (!causalNode) return null;

  const { zscore, polarity, display_sentiment, importance } = causalNode;
  const totalNodes = currentGraph.nodes.length;

  // Compute importance rank
  const sortedByImportance = [...currentGraph.nodes].sort(
    (a, b) => b.importance - a.importance
  );
  const importanceRank =
    sortedByImportance.findIndex((n) => n.id === selectedNode.id) + 1;

  // Polarity explanation
  const polarityLabel =
    polarity > 0
      ? "up = positive"
      : polarity < 0
        ? "up = negative"
        : "unknown";

  // Edge data
  const incomingEdges = currentGraph.edges.filter(
    (e) => e.target === selectedNode.id
  );
  const outgoingEdges = currentGraph.edges.filter(
    (e) => e.source === selectedNode.id
  );

  // Sentiment bar position: map [-1, 1] to [0%, 100%]
  const barPosition = ((display_sentiment + 1) / 2) * 100;

  return (
    <div className="absolute top-4 right-4 w-80 bg-gray-900/95 backdrop-blur border border-gray-700 rounded-lg shadow-xl p-4 text-white z-10 max-h-[calc(100vh-2rem)] overflow-y-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-1">
        <div className="min-w-0 flex-1">
          <h3 className="text-lg font-semibold truncate">
            {getNodeLabel(selectedNode.id)}
          </h3>
          <div className="text-xs text-gray-500 font-mono">{selectedNode.id}</div>
        </div>
        <button
          onClick={clearSelection}
          className="text-gray-400 hover:text-white text-xl leading-none ml-2 flex-shrink-0"
        >
          &times;
        </button>
      </div>

      {/* Sentiment indicator bar */}
      <div className="mb-4 mt-3">
        <div className="flex items-center justify-between text-[10px] text-gray-500 mb-1">
          <span>-1.0 Bearish</span>
          <span
            className="text-xs font-mono font-semibold"
            style={{ color: sentimentToColor(display_sentiment) }}
          >
            {display_sentiment >= 0 ? "+" : ""}
            {display_sentiment.toFixed(3)}
          </span>
          <span>Bullish +1.0</span>
        </div>
        <div className="relative h-2 bg-gradient-to-r from-red-600 via-gray-500 to-green-600 rounded-full">
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 border-white shadow-lg"
            style={{
              left: `${barPosition}%`,
              transform: `translate(-50%, -50%)`,
              backgroundColor: sentimentToColor(display_sentiment),
            }}
          />
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        <div className="bg-gray-800 rounded p-2">
          <div className="text-[10px] text-gray-400 uppercase">Z-Score</div>
          <div className="text-lg font-mono text-gray-200">
            {zscore >= 0 ? "+" : ""}
            {zscore.toFixed(2)}
          </div>
          <div className="text-[10px] text-gray-500">
            {Math.abs(zscore) > 2
              ? "extreme"
              : Math.abs(zscore) > 1
                ? "notable"
                : "normal"}
          </div>
        </div>

        <div className="bg-gray-800 rounded p-2">
          <div className="text-[10px] text-gray-400 uppercase">Polarity</div>
          <div className="text-lg font-mono text-gray-200">
            {polarity > 0 ? "+1" : polarity < 0 ? "-1" : "0"}
          </div>
          <div className="text-[10px] text-gray-500">{polarityLabel}</div>
        </div>

        <div className="bg-gray-800 rounded p-2">
          <div className="text-[10px] text-gray-400 uppercase">Importance</div>
          <div className="text-lg font-mono text-gray-200">
            {importance.toFixed(2)}
          </div>
          <div className="text-[10px] text-gray-500">
            rank #{importanceRank} of {totalNodes}
          </div>
        </div>

        <div className="bg-gray-800 rounded p-2">
          <div className="text-[10px] text-gray-400 uppercase">
            Display Sentiment
          </div>
          <div
            className="text-lg font-mono"
            style={{ color: sentimentToColor(display_sentiment) }}
          >
            {display_sentiment >= 0 ? "+" : ""}
            {display_sentiment.toFixed(3)}
          </div>
          <div className="text-[10px] text-gray-500">z-score * polarity</div>
        </div>
      </div>

      {/* Causal Edges */}
      {(incomingEdges.length > 0 || outgoingEdges.length > 0) && (
        <div className="mb-4">
          <h4 className="text-xs font-semibold text-gray-400 uppercase mb-2">
            Causal Edges ({incomingEdges.length + outgoingEdges.length})
          </h4>

          {incomingEdges.length > 0 && (
            <div className="mb-2">
              <div className="text-[10px] text-gray-500 uppercase mb-1">
                Incoming ({incomingEdges.length})
              </div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {incomingEdges.map((edge, i) => (
                  <EdgeRow
                    key={i}
                    edge={edge}
                    otherNodeId={edge.source}
                    isOutgoing={false}
                  />
                ))}
              </div>
            </div>
          )}

          {outgoingEdges.length > 0 && (
            <div>
              <div className="text-[10px] text-gray-500 uppercase mb-1">
                Outgoing ({outgoingEdges.length})
              </div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {outgoingEdges.map((edge, i) => (
                  <EdgeRow
                    key={i}
                    edge={edge}
                    otherNodeId={edge.target}
                    isOutgoing={true}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* How sentiment is calculated */}
      <div className="mb-2">
        <h4 className="text-xs font-semibold text-gray-400 uppercase mb-1">
          How Sentiment Is Calculated
        </h4>
        <div className="text-[11px] text-gray-400 space-y-1.5 bg-gray-800 rounded p-2.5">
          <p>
            This node&apos;s display color is computed as:{" "}
            <span className="text-gray-200 font-mono">
              z-score x polarity
            </span>
            , clamped to [-1, +1].
          </p>
          <p>
            Polarity is inferred from anchor nodes (S&amp;P 500 = positive,
            Unemployment = negative) via bidirectional propagation through
            causal edges.
          </p>
          <p>
            Z-score = how far the current value is from its 90-day average, in
            standard deviations.
          </p>
        </div>
      </div>
    </div>
  );
}

function EdgeRow({
  edge,
  otherNodeId,
  isOutgoing,
}: {
  edge: CausalEdge;
  otherNodeId: string;
  isOutgoing: boolean;
}) {
  return (
    <div className="text-xs bg-gray-800/60 rounded px-2 py-1.5">
      <div className="flex items-center gap-1.5">
        <span
          className="w-1.5 h-1.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: edgeDirectionColor(edge.direction) }}
        />
        <span className="text-gray-300 truncate">
          {isOutgoing ? (
            <>
              <span className="text-gray-500">-&gt;</span>{" "}
              {getNodeLabel(otherNodeId)}
            </>
          ) : (
            <>
              {getNodeLabel(otherNodeId)}{" "}
              <span className="text-gray-500">-&gt;</span>
            </>
          )}
        </span>
        <span
          className="ml-auto flex-shrink-0 text-[10px]"
          style={{ color: edgeDirectionColor(edge.direction) }}
        >
          {edge.direction}
        </span>
      </div>
      <div className="flex items-center gap-3 mt-0.5 text-[10px] text-gray-500">
        <span>weight: {edge.weight.toFixed(3)}</span>
        <span>lag: {edge.lag}</span>
      </div>
    </div>
  );
}
