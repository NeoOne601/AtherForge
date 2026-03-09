import React from "react";
import { CausalGraph } from "../types";

export function InlineXRay({ graph }: { graph: CausalGraph }) {
    const typeOf = (id: string) => {
        if (id.includes("colosseum") || id.includes("policy")) return "policy";
        if (id.includes("output")) return "output";
        return "";
    };

    const iconOf = (id: string) => {
        if (id.includes("colosseum")) return "🛡️";
        if (id.includes("intake")) return "📥";
        if (id.includes("router")) return "🔀";
        if (id.includes("module")) return "⚙️";
        if (id.includes("faithful")) return "📊";
        if (id.includes("output")) return "✅";
        return "◆";
    };

    return (
        <div style={{ borderTop: "1px solid var(--glass-border)", padding: "16px 24px", background: "rgba(6,10,18,0.5)" }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--aether-light)", marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
                🔬 X-Ray Causal Trace · {graph.total_latency_ms.toFixed(2)}ms total
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
                {graph.nodes.map((node, i) => (
                    <div key={node.id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <div className={`trace-node ${typeOf(node.id)}`} style={{ margin: 0 }}>
                            <span className="trace-icon">{iconOf(node.id)}</span>
                            <div>
                                <div className="trace-id">{node.id}</div>
                                {node.data.selected_module !== undefined && <div className="trace-detail">→ {String(node.data.selected_module)}</div>}
                                {node.data.score !== undefined && <div className="trace-detail">score: {Number(node.data.score).toFixed(2)}</div>}
                                {node.data.latency_ms !== undefined && <div className="trace-detail">{Number(node.data.latency_ms).toFixed(2)}ms</div>}
                            </div>
                        </div>
                        {i < graph.nodes.length - 1 && (
                            <span style={{ color: "var(--text-muted)", fontSize: 12 }}>→</span>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
