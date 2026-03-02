// AetherForge v1.0 — frontend/src/components/XRayGraph.tsx
// ─────────────────────────────────────────────────────────────────
// Glass-box causal graph visualization. Renders the LangGraph
// execution trace as an interactive node-graph using ReactFlow.
// Each node is clickable and shows full policy decision details.
// ─────────────────────────────────────────────────────────────────
import React, { useCallback, useMemo, useState } from "react";
import ReactFlow, {
    Background,
    Controls,
    Handle,
    MiniMap,
    Position,
    type Edge,
    type Node,
} from "reactflow";
import "reactflow/dist/style.css";
import type { CausalGraph, CausalNode } from "../lib/tauri";

interface Props {
    graph: CausalGraph | null;
    onClose: () => void;
}

// ── Custom node component ─────────────────────────────────────────
function XRayNode({ data }: { data: { label: string; latency?: string; type: string } }) {
    const color = data.type === "policy"
        ? "rgba(248,113,113,0.3)"
        : data.type === "output"
            ? "rgba(52,211,153,0.2)"
            : "rgba(139,92,246,0.2)";

    return (
        <div
            className="px-3 py-2 rounded-lg text-xs font-medium cursor-pointer transition-all hover:scale-105"
            style={{
                background: color,
                border: `1px solid ${data.type === "policy" ? "rgba(248,113,113,0.5)" : "rgba(139,92,246,0.4)"}`,
                color: "var(--text-primary)",
                minWidth: 100,
                maxWidth: 160,
            }}
        >
            <Handle type="target" position={Position.Left} style={{ background: "rgba(139,92,246,0.6)" }} />
            <div className="font-semibold truncate">{data.label}</div>
            {data.latency && <div className="text-muted mt-0.5">{data.latency}</div>}
            <Handle type="source" position={Position.Right} style={{ background: "rgba(139,92,246,0.6)" }} />
        </div>
    );
}

const nodeTypes = { xray: XRayNode };

export default function XRayGraph({ graph, onClose }: Props): JSX.Element {
    const [selected, setSelected] = useState<CausalNode | null>(null);

    const { nodes, edges } = useMemo((): { nodes: Node[]; edges: Edge[] } => {
        if (!graph) return { nodes: [], edges: [] };

        const rfNodes: Node[] = graph.nodes.map((n, i) => ({
            id: n.id,
            type: "xray",
            position: { x: i * 180, y: 60 + (i % 2) * 60 },
            data: {
                label: n.id.replace(/_/g, " "),
                latency: n.data?.latency_ms ? `${Number(n.data.latency_ms).toFixed(1)}ms` : undefined,
                type: n.id.includes("colosseum") || n.id.includes("policy")
                    ? "policy"
                    : n.id === "output"
                        ? "output"
                        : "default",
            },
        }));

        const rfEdges: Edge[] = graph.edges.map((e, i) => ({
            id: `e-${i}`,
            source: e.source,
            target: e.target,
            animated: true,
            style: { stroke: "rgba(139,92,246,0.6)", strokeWidth: 1.5 },
        }));

        return { nodes: rfNodes, edges: rfEdges };
    }, [graph]);

    const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
        const found = graph?.nodes.find(n => n.id === node.id) || null;
        setSelected(found);
    }, [graph]);

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="flex items-center justify-between px-3 py-2 border-b" style={{ borderColor: "var(--border-subtle)" }}>
                <div className="flex items-center gap-2">
                    <span className="badge-plasma">X-Ray</span>
                    <span className="text-xs text-secondary">Causal Graph</span>
                    {graph && <span className="text-xs text-muted">{graph.total_latency_ms.toFixed(0)}ms total</span>}
                </div>
                <button onClick={onClose} className="text-muted hover:text-primary text-lg leading-none">×</button>
            </div>

            {/* Graph */}
            <div className="flex-1" style={{ background: "rgba(2,4,8,0.95)" }}>
                {nodes.length > 0 ? (
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        nodeTypes={nodeTypes}
                        onNodeClick={onNodeClick}
                        fitView
                        attributionPosition="bottom-right"
                        style={{ background: "transparent" }}
                    >
                        <Background color="rgba(139,92,246,0.06)" gap={20} />
                        <MiniMap
                            nodeColor={() => "rgba(139,92,246,0.4)"}
                            maskColor="rgba(2,4,8,0.8)"
                            style={{ background: "rgba(5,12,20,0.9)" }}
                        />
                    </ReactFlow>
                ) : (
                    <div className="flex items-center justify-center h-full text-muted text-sm">
                        Send a message with X-Ray ON to see the causal graph
                    </div>
                )}
            </div>

            {/* Selected node detail */}
            {selected && (
                <div className="border-t p-3 max-h-40 overflow-y-auto" style={{ borderColor: "var(--border-subtle)" }}>
                    <p className="text-xs font-semibold text-secondary mb-1">{selected.id}</p>
                    <pre className="text-xs text-muted font-mono whitespace-pre-wrap">
                        {JSON.stringify(selected.data, null, 2)}
                    </pre>
                </div>
            )}
        </div>
    );
}
