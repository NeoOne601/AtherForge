import { useState, useEffect } from "react";

export function InsightsPanel() {
    const [stats, setStats] = useState<Record<string, unknown> | null>(null);

    useEffect(() => {
        fetch("/api/v1/replay/stats").then(r => r.json()).then(setStats).catch(() => { });
    }, []);

    const insights = [
        { topic: "OPLoRA Math Queries", desc: "Users frequently ask about SVD projector derivations and orthogonality guarantees.", novelty: 0.87 },
        { topic: "Document Search Patterns", desc: "High volume of RAGForge queries around contract terms and compliance documents.", novelty: 0.72 },
        { topic: "Anomaly Detection Use Cases", desc: "WatchTower being used for time-series financial data — a new usage pattern.", novelty: 0.94 },
        { topic: "Policy Customisation", desc: "Several attempts to craft custom Rego rules for domain-specific compliance.", novelty: 0.65 },
    ];

    return (
        <div className="panel-wrapper">
            <div className="panel-title">✨ Insights</div>
            <div className="panel-sub">Weekly novelty report — what the AI is learning from your team's usage</div>

            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-value">{String(stats?.total_records ?? "0")}</div>
                    <div className="stat-label">Total interactions</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{stats?.avg_faithfulness ? (Number(stats.avg_faithfulness) * 100).toFixed(0) + "%" : "—"}</div>
                    <div className="stat-label">Avg faithfulness</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{String(stats?.trained_count ?? "0")}</div>
                    <div className="stat-label">Used in training</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{String(stats?.size_mb ?? "0")} MB</div>
                    <div className="stat-label">Buffer size</div>
                </div>
            </div>

            <div style={{ marginBottom: 12, fontSize: 13, fontWeight: 600, color: "var(--text-secondary)" }}>
                Novel Patterns Detected
            </div>
            {insights.map((ins, i) => (
                <div className="insight-card" key={i}>
                    <div className="insight-topic">{ins.topic}</div>
                    <div className="insight-desc">{ins.desc}</div>
                    <div className="novelty-bar-bg">
                        <div className="novelty-bar" style={{ width: `${ins.novelty * 100}%` }} />
                    </div>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4, fontFamily: "'JetBrains Mono', monospace" }}>
                        novelty {(ins.novelty * 100).toFixed(0)}%
                    </div>
                </div>
            ))}

            <div style={{ marginTop: 20 }}>
                <button
                    className="btn btn-primary"
                    onClick={async () => {
                        await fetch("/api/v1/learning/trigger", { method: "POST" });
                        alert("OPLoRA training job triggered! Check backend logs for progress.");
                    }}
                >
                    ▶ Trigger Training Now
                </button>
            </div>
        </div>
    );
}
