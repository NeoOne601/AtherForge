import { useState, useEffect, useRef } from "react";

interface LiveFolderFile {
    name: string;
    size_kb: number;
    extension: string;
    status: string;
    chunk_count: number;
    image_pages_pending: number;
    document_id: string | null;
    last_error: string | null;
}

const STATUS_DOT: Record<string, { color: string; symbol: string }> = {
    ready:          { color: "#00e676", symbol: "●" },
    partial:        { color: "#ffa500", symbol: "◑" },
    ocr_pending:    { color: "#ffa500", symbol: "○" },
    ocr_running:    { color: "#00b4ff", symbol: "⟳" },
    extracting_text:{ color: "#00b4ff", symbol: "⟳" },
    failed:         { color: "#ff4444", symbol: "✕" },
    not_indexed:    { color: "#666",    symbol: "—" },
    queued:         { color: "#888",    symbol: "↻" },
};
function getStatusStyle(status: string) {
    return STATUS_DOT[status] || { color: "#888", symbol: "?" };
}

export function WatchTowerHUD() {
    const [metrics, setMetrics] = useState<Record<string, any>>({});
    const [history, setHistory] = useState<Record<string, number[]>>({ cpu: [], mem: [], net: [] });
    const [incidents, setIncidents] = useState<any[]>([]);
    const [liveFiles, setLiveFiles] = useState<LiveFolderFile[]>([]);
    const [liveFolderError, setLiveFolderError] = useState<string | null>(null);

    // ── Metrics polling (unchanged) ──────────────────────────────────
    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const res = await fetch("/api/v1/metrics/stream");
                if (res.ok) {
                    const data = await res.json();
                    setMetrics(data);
                    setHistory(prev => {
                        const next = { ...prev };
                        Object.entries(data).forEach(([key, val]: [string, any]) => {
                            if (!next[key]) next[key] = [];
                            next[key] = [...next[key].slice(-19), val.value];
                        });
                        return next;
                    });
                    const newAnomalies: any[] = [];
                    Object.entries(data).forEach(([key, val]: [string, any]) => {
                        if (val.is_anomaly) newAnomalies.push({ key, val });
                    });
                    if (newAnomalies.length > 0) {
                        setIncidents(prev => {
                            let updated = [...prev];
                            newAnomalies.forEach(({ key, val }) => {
                                if (!updated.find(i => i.metric === key && i.status === "active")) {
                                    updated.push({
                                        id: Date.now().toString() + Math.random().toString().slice(2, 6),
                                        metric: key, value: val.value, z_score: val.z_score,
                                        timestamp: new Date().toLocaleTimeString(),
                                        status: "active", rca: null, loading: false
                                    });
                                }
                            });
                            return updated;
                        });
                    }
                }
            } catch (err) {
                console.error("Failed to fetch Live Telemetry", err);
            }
        };
        const interval = setInterval(fetchMetrics, 2000);
        return () => clearInterval(interval);
    }, []);

    // ── LiveFolder real-time file feed ───────────────────────────────
    useEffect(() => {
        const fetchLiveFolder = async () => {
            try {
                const res = await fetch("/api/v1/ragforge/live-folder");
                if (res.ok) {
                    const data = await res.json();
                    setLiveFiles(data.files || []);
                    setLiveFolderError(null);
                } else {
                    setLiveFolderError("Unable to reach LiveFolder API");
                }
            } catch (err) {
                setLiveFolderError("Network error fetching LiveFolder");
            }
        };
        fetchLiveFolder();
        const interval = setInterval(fetchLiveFolder, 5000);
        return () => clearInterval(interval);
    }, []);

    const injectAnomaly = async () => {
        try {
            await fetch("/api/v1/watchtower/inject_anomaly", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ metric: "mem", value: 99.5 })
            });
        } catch (err) { console.error("Failed to inject anomaly", err); }
    };

    const handleAnalyze = async (incident: any) => {
        setIncidents(prev => prev.map(i => i.id === incident.id ? { ...i, loading: true } : i));
        try {
            const res = await fetch("/api/v1/watchtower/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ metric: incident.metric, value: incident.value, z_score: incident.z_score })
            });
            const data = await res.json();
            setIncidents(prev => prev.map(i => i.id === incident.id ? { ...i, loading: false, rca: data } : i));
        } catch (e) {
            console.error(e);
            setIncidents(prev => prev.map(i => i.id === incident.id ? { ...i, loading: false } : i));
        }
    };

    const handleMitigate = async (incident: any, action: string) => {
        try {
            await fetch("/api/v1/watchtower/mitigate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action, target: incident.metric })
            });
            setIncidents(prev => prev.map(i => i.id === incident.id ? { ...i, status: "resolved" } : i));
        } catch (e) { console.error(e); }
    };
    
    const handleRetry = async (doc: LiveFolderFile) => {
        if (!doc.document_id) return;
        try {
            await fetch(`/api/v1/ragforge/documents/${doc.document_id}/retry`, {
                method: "POST"
            });
            // Status will update on next poll
        } catch (err) {
            console.error("Retry failed", err);
        }
    };

    const getMetric = (key: string, defaultVal: number = 0) =>
        metrics[key] || { value: defaultVal, z_score: 0.0, is_anomaly: false };

    const cpu = getMetric("cpu", 45);
    const mem = getMetric("mem", 62);
    const net = getMetric("net", 12);

    return (
        <div className="module-hud" style={{ overflowY: "auto", paddingBottom: "40px" }}>
            <div className="hud-header">
                <div className="hud-title">👁️ Live Telemetry</div>
                <button className="btn btn-ghost" style={{ padding: "4px 10px", fontSize: "11px" }} onClick={injectAnomaly}>
                    ⚠️ Simulate Memory Spike
                </button>
            </div>

            {/* ── System Metrics ──────────────────────────────────────── */}
            <div className="metrics-board">
                {([["CPU Load", cpu, "cpu"], ["Memory Usage", mem, "mem"], ["Network I/O", net, "net"]] as const).map(([label, metric, key]) => (
                    <div key={key} className="metric-card">
                        <div className="metric-header">
                            <span>{label}</span>
                            <span style={{ color: metric.is_anomaly ? "var(--danger)" : "var(--text-muted)" }}>
                                Z: {(metric.z_score > 0 ? "+" : "") + metric.z_score}
                            </span>
                        </div>
                        <div className="metric-value" style={{ color: metric.is_anomaly ? "var(--danger)" : "inherit" }}>
                            {metric.value}{key === "net" ? " MB/s" : "%"}
                        </div>
                        <div className="metric-chart">
                            {(history[key] || []).map((v, i) => (
                                <div key={i}
                                    className={`chart-bar ${metric.is_anomaly && i === history[key].length - 1 ? "anomaly" : ""}`}
                                    style={{ height: `${Math.min(100, v)}%` }}
                                />
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            {/* ── LiveFolder Ingestion Feed ────────────────────────────── */}
            <div style={{ marginTop: "24px", backgroundColor: "rgba(10,12,20,0.6)", borderRadius: "12px", padding: "16px", border: "1px solid var(--border)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                    <h3 style={{ fontSize: "13px", margin: 0, color: "var(--aether)" }}>
                        📂 LiveFolder Ingestion Feed
                    </h3>
                    <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>
                        {liveFiles.length} file{liveFiles.length !== 1 ? "s" : ""} · auto-refreshes every 5s
                    </span>
                </div>
                {liveFolderError ? (
                    <div style={{ color: "var(--danger)", fontSize: "12px" }}>⚠️ {liveFolderError}</div>
                ) : liveFiles.length === 0 ? (
                    <div style={{ color: "var(--text-muted)", fontSize: "12px", textAlign: "center", padding: "16px 0" }}>
                        No files in LiveFolder. Drop files into <code>data/LiveFolder/</code> to auto-ingest.
                    </div>
                ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                        {liveFiles.map((f) => {
                            const s = getStatusStyle(f.status);
                            const isAnimated = f.status === "extracting_text" || f.status === "ocr_running";
                            return (
                                <div key={f.name} style={{
                                    display: "flex", alignItems: "center", gap: "10px",
                                    padding: "8px 10px",
                                    background: "rgba(255,255,255,0.02)",
                                    borderRadius: "8px",
                                    border: `1px solid ${f.status === "failed" ? "rgba(255,68,68,0.2)" : "rgba(255,255,255,0.06)"}`,
                                    fontSize: "12px",
                                }}>
                                    {/* Status dot */}
                                    <span style={{
                                        color: s.color,
                                        fontSize: "14px",
                                        minWidth: "16px",
                                        animation: isAnimated ? "spin 1.2s linear infinite" : "none",
                                        display: "inline-block",
                                    }}>
                                        {s.symbol}
                                    </span>

                                    {/* File name */}
                                    <span style={{ flex: 1, color: "var(--fg)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                                          title={f.name}>
                                        {f.name}
                                    </span>

                                    {/* Size */}
                                    <span style={{ color: "var(--text-muted)", minWidth: "52px", textAlign: "right" }}>
                                        {f.size_kb > 1024 ? `${(f.size_kb/1024).toFixed(1)} MB` : `${f.size_kb} KB`}
                                    </span>

                                    {/* Chunks */}
                                    <span style={{ color: f.chunk_count > 0 ? "var(--aether)" : "var(--text-muted)", minWidth: "72px", textAlign: "right" }}>
                                        {f.chunk_count > 0 ? `${f.chunk_count} chunks` : "—"}
                                    </span>

                                    {/* Status badge */}
                                    <span style={{
                                        color: s.color,
                                        background: `${s.color}18`,
                                        border: `1px solid ${s.color}44`,
                                        borderRadius: "4px",
                                        padding: "1px 6px",
                                        fontSize: "10px",
                                        minWidth: "90px",
                                        textAlign: "center",
                                        whiteSpace: "nowrap",
                                    }}>
                                        {f.status.replace(/_/g, " ")}
                                    </span>

                                    {/* Image pages badge */}
                                    {f.image_pages_pending > 0 && f.status !== "ocr_running" && (
                                        <span style={{
                                            color: "#ffa500",
                                            background: "rgba(255,160,0,0.1)",
                                            border: "1px solid rgba(255,160,0,0.3)",
                                            borderRadius: "4px",
                                            padding: "1px 5px",
                                            fontSize: "10px",
                                            whiteSpace: "nowrap",
                                        }}>
                                            🖼 {f.image_pages_pending} img
                                        </span>
                                    )}

                                    {/* Error tooltip */}
                                    {f.last_error && (
                                        <span title={f.last_error} style={{ cursor: "help", color: "var(--danger)", fontSize: "13px" }}>⚠</span>
                                    )}

                                    {/* Re-Process action */}
                                    {(f.status === "failed" || (f.status === "ready" && f.chunk_count === 0)) && f.document_id && (
                                        <button 
                                            onClick={() => handleRetry(f)}
                                            style={{
                                                background: "rgba(255,255,255,0.05)",
                                                border: "1px solid var(--border)",
                                                color: "var(--fg-muted)",
                                                borderRadius: "4px",
                                                padding: "1px 6px",
                                                fontSize: "10px",
                                                cursor: "pointer"
                                            }}
                                            title="Force re-ingest this file"
                                        >
                                            RE-PROCESS
                                        </button>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* ── Active Incidents ───────────────────────────────────── */}
            {incidents.length > 0 && (
                <div style={{ marginTop: "20px", backgroundColor: "rgba(20,20,30,0.5)", borderRadius: "12px", padding: "16px" }}>
                    <h3 style={{ fontSize: "14px", marginBottom: "12px", color: "var(--text)", borderBottom: "1px solid var(--border)", paddingBottom: "8px" }}>
                        Active Incidents & RCA
                    </h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                        {incidents.slice().reverse().map(incident => (
                            <div key={incident.id} style={{
                                padding: "12px", border: "1px solid var(--border)", borderRadius: "8px",
                                backgroundColor: incident.status === "resolved" ? "rgba(0,255,100,0.05)" : "rgba(255,50,50,0.05)",
                                opacity: incident.status === "resolved" ? 0.6 : 1
                            }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
                                    <div>
                                        <span style={{ fontWeight: "bold", color: incident.status === "resolved" ? "var(--text-muted)" : "var(--danger)", marginRight: "8px" }}>
                                            {incident.metric.toUpperCase()} Spike Detected
                                        </span>
                                        <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>{incident.timestamp}</span>
                                        <div style={{ fontSize: "13px", marginTop: "4px" }}>
                                            Value: {incident.value.toFixed(1)} (Z-Score: +{incident.z_score.toFixed(2)})
                                        </div>
                                    </div>
                                    {incident.status === "active" && (
                                        <div style={{ display: "flex", gap: "8px" }}>
                                            {!incident.rca && !incident.loading && (
                                                <button className="btn btn-ghost" style={{ fontSize: "12px", padding: "4px 8px" }} onClick={() => handleAnalyze(incident)}>
                                                    🔍 Analyze Root Cause
                                                </button>
                                            )}
                                            {incident.loading && <span style={{ fontSize: "12px", color: "var(--brand-glow)" }}>Analyzing 5-Whys...</span>}
                                            {incident.rca && (
                                                <>
                                                    <button className="btn btn-primary" style={{ fontSize: "12px", padding: "4px 8px", backgroundColor: "var(--danger)" }} onClick={() => handleMitigate(incident, "Kill Process")}>
                                                        🛑 Kill Process
                                                    </button>
                                                    <button className="btn btn-ghost" style={{ fontSize: "12px", padding: "4px 8px" }} onClick={() => handleMitigate(incident, "Ignore")}>
                                                        Dismiss
                                                    </button>
                                                </>
                                            )}
                                        </div>
                                    )}
                                    {incident.status === "resolved" && (
                                        <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>✓ Resolved</span>
                                    )}
                                </div>
                                {incident.rca && (
                                    <div style={{ marginTop: "12px", fontSize: "13px", color: "var(--text)", borderLeft: "2px solid var(--brand-glow)", paddingLeft: "12px" }}>
                                        <div style={{ marginBottom: "8px", color: "var(--brand-glow)" }}><strong>Causal Chain Identified:</strong></div>
                                        <ul style={{ paddingLeft: "20px", margin: 0, color: "var(--text-muted)", fontSize: "12px" }}>
                                            {incident.rca.evidence?.map((ev: string, idx: number) => (
                                                <li key={idx} style={{ marginBottom: "4px" }}>{ev}</li>
                                            ))}
                                        </ul>
                                        <div style={{ marginTop: "8px", fontStyle: "italic" }}>
                                            <strong>Mitigation Recommendation: </strong>
                                            {incident.rca.remediation_steps?.[0] || "Terminate offending process."}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
