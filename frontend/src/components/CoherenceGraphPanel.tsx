import React, { useState, useEffect } from "react";
import { ReactFlow, MiniMap, Controls, Background, useNodesState, useEdgesState } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

interface CoherenceGraphPanelProps {
    sourceName: string;
    onClose: () => void;
}

export function CoherenceGraphPanel({ sourceName, onClose }: CoherenceGraphPanelProps) {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        setIsLoading(true);
        fetch(`/api/v1/ragforge/tree/graph/${encodeURIComponent(sourceName)}`)
            .then(r => r.json())
            .then(data => {
                if (data.nodes && data.nodes.length > 0) {
                    setNodes(data.nodes);
                    setEdges(data.edges);
                }
            })
            .catch(err => console.error("Failed to load graph", err))
            .finally(() => setIsLoading(false));
    }, [sourceName]);

    return (
        <div
            id="coherence-graph-panel"
            style={{
                position: "relative",
                background: "var(--bg-surface, #1e1e2e)",
                border: "1px solid var(--border, #333)",
                borderRadius: "8px",
                padding: "12px",
                marginTop: "12px",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
                height: "500px",
            }}
        >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                    <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--plasma, #00b4ff)", letterSpacing: "1px", textTransform: "uppercase" }}>
                        🕸️ Real-Time Coherence Visualization (SONA)
                    </span>
                    <span style={{ fontSize: "10px", color: "var(--fg-muted, #888)", marginLeft: "8px" }}>
                        {sourceName}
                    </span>
                </div>
                <button
                    onClick={onClose}
                    style={{ background: "transparent", border: "none", color: "var(--fg-muted, #888)", cursor: "pointer", fontSize: "14px" }}
                >
                    ✕
                </button>
            </div>

            <div style={{ flex: 1, border: "1px solid var(--border, #333)", borderRadius: "6px", overflow: "hidden", background: "#f8fafc" }}>
                {isLoading ? (
                    <div style={{ padding: "20px", color: "#64748b", fontSize: "12px" }}>Running RuVector-WASM Graph Calculation...</div>
                ) : nodes.length === 0 ? (
                    <div style={{ padding: "20px", color: "#ef4444", fontSize: "12px" }}>No coherence graph details available.</div>
                ) : (
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        fitView
                        attributionPosition="bottom-right"
                    >
                        <MiniMap />
                        <Controls />
                        <Background color="#cbd5e1" gap={16} />
                    </ReactFlow>
                )}
            </div>
        </div>
    );
}
