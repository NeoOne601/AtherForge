import React, { useState, useEffect, useCallback } from "react";
import { ReactFlow, MiniMap, Controls, Background, useNodesState, useEdgesState } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { GraphToolbar } from "./GraphToolbar";

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
    }, [sourceName, setNodes, setEdges]);

    const onNodeClick = useCallback((_: React.MouseEvent, node: any) => {
        const nId = node.id;
        
        // Find ancestor nodes manually
        const activeNodeIds = new Set<string>([nId, 'root']);
        if (nId.startsWith('chunk_')) {
            const parts = nId.split('_');
            activeNodeIds.add(`cluster_${parts[1]}`);
        } else if (nId.startsWith('cluster_')) {
            // keep root + cluster
        }
        
        // Highlight active edges traversing this path
        setEdges(eds => eds.map(e => {
            let active = false;
            if (activeNodeIds.has(e.source) && activeNodeIds.has(e.target)) {
                active = true;
            }
            return {
                ...e,
                animated: active,
                style: { ...e.style, strokeWidth: active ? 4 : 1, opacity: active ? 1 : 0.2, zIndex: active ? 10 : 1 }
            };
        }));
        
        setNodes(nds => nds.map(n => ({
            ...n,
            style: { ...n.style, opacity: activeNodeIds.has(n.id) ? 1 : 0.3 }
        })));
    }, [setEdges, setNodes]);

    const onPaneClick = useCallback(() => {
        // Reset styles on background pane click
        setEdges(eds => eds.map(e => ({ 
            ...e, 
            animated: e.source === 'root', 
            style: { ...e.style, strokeWidth: e.source === 'root' ? 2 : 1, opacity: 1, zIndex: 1 } 
        })));
        setNodes(nds => nds.map(n => ({ ...n, style: { ...n.style, opacity: 1 } })));
    }, [setEdges, setNodes]);

    const [isMaximized, setIsMaximized] = useState(false);

    return (
        <div
            id="coherence-graph-panel"
            style={{
                position: isMaximized ? "fixed" : "relative",
                top: isMaximized ? 0 : "auto",
                left: isMaximized ? 0 : "auto",
                right: isMaximized ? 0 : "auto",
                bottom: isMaximized ? 0 : "auto",
                zIndex: isMaximized ? 9999 : 1,
                background: "var(--bg-surface, #1e1e2e)",
                border: isMaximized ? "none" : "1px solid var(--border, #333)",
                borderRadius: isMaximized ? "0" : "8px",
                padding: "12px",
                margin: isMaximized ? "0" : "12px 0 0 0",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
                height: isMaximized ? "100vh" : "500px",
                width: isMaximized ? "100vw" : "auto",
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
                <div style={{ display: "flex", gap: "12px" }}>
                    <button
                        onClick={() => setIsMaximized(!isMaximized)}
                        style={{ background: "transparent", border: "none", color: "var(--fg-muted, #888)", cursor: "pointer", fontSize: "14px", display: "flex", alignItems: "center", justifyContent: "center" }}
                        title={isMaximized ? "Restore down" : "Maximize"}
                    >
                        {isMaximized ? "▣" : "🗖"}
                    </button>
                    <button
                        onClick={onClose}
                        style={{ background: "transparent", border: "none", color: "var(--fg-muted, #888)", cursor: "pointer", fontSize: "14px", display: "flex", alignItems: "center", justifyContent: "center" }}
                    >
                        ✕
                    </button>
                </div>
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
                        onNodeClick={onNodeClick}
                        onPaneClick={onPaneClick}
                        fitView
                        attributionPosition="bottom-right"
                    >
                        <GraphToolbar />
                        <MiniMap />
                        <Controls />
                        <Background color="#cbd5e1" gap={16} />
                    </ReactFlow>
                )}
            </div>
        </div>
    );
}
