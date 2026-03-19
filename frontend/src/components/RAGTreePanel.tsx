// AetherForge v1.0 — frontend/src/components/RAGTreePanel.tsx
// ─────────────────────────────────────────────────────────────────
// HTI Document Tree Browser
// Renders the hierarchical document structure returned by the
// /api/v1/ragforge/tree/{source} endpoint.
// ─────────────────────────────────────────────────────────────────
import React, { useState, useEffect, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────────
interface TreeNode {
    id: string;
    label: string;
    path: string;
    parent_id: string | null;
    chunk_count: number;
    pages: number[];
    depth: number;
    children: TreeNode[];
}

interface ChunkPreview {
    text: string;
    meta: Record<string, unknown>;
}

interface RAGTreePanelProps {
    sourceName: string;
    onClose: () => void;
}

// ── Recursive Node Component ─────────────────────────────────────
function TreeNodeItem({
    node,
    selectedId,
    onSelect,
}: {
    node: TreeNode;
    selectedId: string | null;
    onSelect: (node: TreeNode) => void;
}) {
    const [expanded, setExpanded] = useState(node.depth < 1);
    const hasChildren = node.children.length > 0;
    const isSelected = node.id === selectedId;

    return (
        <div style={{ marginLeft: `${node.depth * 14}px` }}>
            <div
                onClick={() => {
                    onSelect(node);
                    if (hasChildren) setExpanded((v) => !v);
                }}
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    padding: "4px 8px",
                    borderRadius: "4px",
                    cursor: "pointer",
                    background: isSelected ? "rgba(var(--plasma-rgb, 100,100,255), 0.15)" : "transparent",
                    border: isSelected ? "1px solid var(--plasma)" : "1px solid transparent",
                    transition: "all 0.15s ease",
                    marginBottom: "2px",
                }}
                title={node.path}
            >
                {/* Expand/Collapse Toggle */}
                <span style={{ fontSize: "10px", opacity: 0.6, minWidth: "12px" }}>
                    {hasChildren ? (expanded ? "▼" : "▶") : "•"}
                </span>

                {/* Node Label */}
                <span
                    style={{
                        flex: 1,
                        fontSize: "12px",
                        color: isSelected ? "var(--plasma)" : "var(--fg)",
                        fontWeight: node.depth === 0 ? 600 : 400,
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                    }}
                >
                    {node.label}
                </span>

                {/* Meta badges */}
                <span
                    style={{
                        fontSize: "9px",
                        color: "var(--fg-muted)",
                        background: "var(--bg-elevated)",
                        borderRadius: "3px",
                        padding: "1px 5px",
                        flexShrink: 0,
                    }}
                >
                    {node.chunk_count} chunks
                </span>
                {node.pages.length > 0 && (
                    <span style={{ fontSize: "9px", color: "var(--aether)", flexShrink: 0 }}>
                        p.{node.pages[0]}{node.pages.length > 1 ? `–${node.pages[node.pages.length - 1]}` : ""}
                    </span>
                )}
            </div>

            {/* Children — animated expand */}
            {hasChildren && expanded && (
                <div style={{ borderLeft: "1px dashed var(--border)", marginLeft: "10px", paddingLeft: "4px" }}>
                    {node.children.map((child) => (
                        <TreeNodeItem
                            key={child.id}
                            node={child}
                            selectedId={selectedId}
                            onSelect={onSelect}
                        />
                    ))}
                </div>
            )}
        </div>
    );
}

// ── Main Panel ───────────────────────────────────────────────────
export function RAGTreePanel({ sourceName, onClose }: RAGTreePanelProps) {
    const [tree, setTree] = useState<TreeNode[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);
    const [chunks, setChunks] = useState<ChunkPreview[]>([]);
    const [chunksLoading, setChunksLoading] = useState(false);

    // Fetch tree on mount
    useEffect(() => {
        setIsLoading(true);
        setError(null);
        fetch(`/api/v1/ragforge/tree/${encodeURIComponent(sourceName)}`)
            .then((r) => r.json())
            .then((data) => {
                setTree(data.tree || []);
            })
            .catch(() => setError("Failed to load document tree."))
            .finally(() => setIsLoading(false));
    }, [sourceName]);

    // Fetch chunks for selected node
    const handleNodeSelect = useCallback(
        async (node: TreeNode) => {
            setSelectedNode(node);
            setChunksLoading(true);
            setChunks([]);
            try {
                const res = await fetch(
                    `/api/v1/ragforge/tree/${encodeURIComponent(sourceName)}/section/${encodeURIComponent(node.id)}`
                );
                const data = await res.json();
                setChunks(data.chunks || []);
            } catch {
                setChunks([]);
            } finally {
                setChunksLoading(false);
            }
        },
        [sourceName]
    );

    return (
        <div
            id="rag-tree-panel"
            style={{
                position: "relative",
                background: "var(--bg-surface)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                padding: "12px",
                marginTop: "12px",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
                maxHeight: "480px",
            }}
        >
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                    <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--plasma)", letterSpacing: "1px", textTransform: "uppercase" }}>
                        🌲 Document Structure
                    </span>
                    <span style={{ fontSize: "10px", color: "var(--fg-muted)", marginLeft: "8px" }}>
                        {sourceName}
                    </span>
                </div>
                <button
                    onClick={onClose}
                    style={{
                        background: "transparent",
                        border: "none",
                        color: "var(--fg-muted)",
                        cursor: "pointer",
                        fontSize: "14px",
                        padding: "0 4px",
                    }}
                    title="Close tree browser"
                >
                    ✕
                </button>
            </div>

            {/* Split: Tree + Preview */}
            <div style={{ display: "flex", gap: "10px", overflow: "hidden", flex: 1 }}>
                {/* Tree Panel */}
                <div
                    style={{
                        flex: "0 0 50%",
                        overflowY: "auto",
                        overflowX: "hidden",
                        paddingRight: "4px",
                        maxHeight: "400px",
                    }}
                >
                    {isLoading ? (
                        <div style={{ color: "var(--fg-muted)", fontSize: "12px", padding: "8px" }}>Loading tree…</div>
                    ) : error ? (
                        <div style={{ color: "var(--ember)", fontSize: "12px", padding: "8px" }}>{error}</div>
                    ) : tree.length === 0 ? (
                        <div style={{ color: "var(--fg-muted)", fontSize: "12px", padding: "8px" }}>
                            No hierarchical sections detected. Re-index with a newer version.
                        </div>
                    ) : (
                        tree.map((node) => (
                            <TreeNodeItem
                                key={node.id}
                                node={node}
                                selectedId={selectedNode?.id ?? null}
                                onSelect={handleNodeSelect}
                            />
                        ))
                    )}
                </div>

                {/* Chunk Preview */}
                <div
                    style={{
                        flex: 1,
                        borderLeft: "1px solid var(--border)",
                        paddingLeft: "10px",
                        overflowY: "auto",
                        maxHeight: "400px",
                    }}
                >
                    {!selectedNode ? (
                        <div style={{ color: "var(--fg-muted)", fontSize: "11px", paddingTop: "8px" }}>
                            ← Select a section to preview its content
                        </div>
                    ) : chunksLoading ? (
                        <div style={{ color: "var(--fg-muted)", fontSize: "12px" }}>Loading chunks…</div>
                    ) : chunks.length === 0 ? (
                        <div style={{ color: "var(--fg-muted)", fontSize: "11px" }}>No content found for this section.</div>
                    ) : (
                        <div>
                            <div style={{ fontSize: "10px", color: "var(--aether)", marginBottom: "6px", fontWeight: 600 }}>
                                {selectedNode.label}
                            </div>
                            {chunks.map((chunk, i) => (
                                <div
                                    key={i}
                                    style={{
                                        background: "var(--bg-elevated)",
                                        border: "1px solid var(--border)",
                                        borderRadius: "4px",
                                        padding: "8px",
                                        marginBottom: "6px",
                                        fontSize: "11px",
                                        lineHeight: "1.5",
                                        color: "var(--fg)",
                                        maxHeight: "100px",
                                        overflowY: "auto",
                                        fontFamily: "monospace",
                                        whiteSpace: "pre-wrap",
                                    }}
                                >
                                    {chunk.text.slice(0, 600)}
                                    {chunk.text.length > 600 && <span style={{ color: "var(--fg-muted)" }}>…</span>}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
