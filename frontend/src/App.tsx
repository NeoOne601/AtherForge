import { useState, useEffect, useRef, useCallback } from "react";
import "./index.css";

// ── Types ────────────────────────────────────────────────────────
interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    module: string;
    latency_ms?: number;
    faithfulness_score?: number;
    policy_decisions?: PolicyDecision[];
    causal_graph?: CausalGraph;
    blocked?: boolean;
}

interface PolicyDecision {
    allowed: boolean;
    reason: string;
    deny_reasons: string[];
    fsm_state: string;
    latency_ms: number;
}

interface CausalGraph {
    nodes: { id: string; data: Record<string, unknown>; ts: number }[];
    edges: { source: string; target: string }[];
    total_latency_ms: number;
}

interface Module {
    id: string;
    name: string;
    icon: string;
    desc: string;
}

// ── Constants ────────────────────────────────────────────────────
const MODULES: Module[] = [
    { id: "localbuddy", name: "LocalBuddy", icon: "💬", desc: "Conversational AI" },
    { id: "ragforge", name: "RAGForge", icon: "🔍", desc: "Search your docs" },
    { id: "watchtower", name: "WatchTower", icon: "👁️", desc: "Anomaly detection" },
    { id: "streamsync", name: "StreamSync", icon: "⚡", desc: "Event streams" },
    { id: "tunelab", name: "TuneLab", icon: "🎛️", desc: "Fine-tuning control" },
];

const SUGGESTIONS = [
    "Summarise how OPLoRA prevents catastrophic forgetting",
    "What modules does AetherForge have?",
    "Analyse this data for anomalies: 12, 14, 13, 98, 15, 12",
    "How does the Silicon Colosseum protect my data?",
];

// ── Main App ─────────────────────────────────────────────────────
export default function App() {
    const [activeModule, setActiveModule] = useState("localbuddy");
    const [activePanel, setActivePanel] = useState<"chat" | "insights" | "policies">("chat");
    const [xrayOpen, setXrayOpen] = useState(false);
    const [online, setOnline] = useState(false);
    const [cpuPct, setCpuPct] = useState<number | null>(null);

    // Health polling
    useEffect(() => {
        const poll = async () => {
            try {
                const res = await fetch("/api/v1/status", { signal: AbortSignal.timeout(2000) });
                if (res.ok) {
                    const data = await res.json();
                    setOnline(true);
                    setCpuPct(data.cpu_pct ?? null);
                }
            } catch { setOnline(false); }
        };
        poll();
        const id = setInterval(poll, 8000);
        return () => clearInterval(id);
    }, []);

    return (
        <div className="app-shell">
            {/* Header */}
            <header className="app-header">
                <div className="app-logo">
                    <div className="logo-mark">Æ</div>
                    <span className="logo-name">AetherForge v1.0</span>
                </div>

                <div className="header-pills">
                    {cpuPct !== null && (
                        <div className="status-pill">CPU {cpuPct.toFixed(0)}%</div>
                    )}
                    <div className="status-pill">
                        <div className={`status-dot ${online ? "online" : "offline"}`} />
                        {online ? "Online" : "Offline"}
                    </div>
                    <button
                        className={`xray-btn ${xrayOpen ? "active" : ""}`}
                        onClick={() => setXrayOpen(v => !v)}
                    >
                        🔬 X-Ray {xrayOpen ? "ON" : "OFF"}
                    </button>
                </div>
            </header>

            {/* Body */}
            <div className="app-body">
                {/* Sidebar */}
                <aside className="sidebar">
                    <div className="sidebar-section-label">Modules</div>
                    {MODULES.map(m => (
                        <button
                            key={m.id}
                            className={`module-btn ${activeModule === m.id && activePanel === "chat" ? "active" : ""}`}
                            onClick={() => { setActiveModule(m.id); setActivePanel("chat"); }}
                        >
                            <div className="module-icon">{m.icon}</div>
                            <div className="module-info">
                                <div className="module-name">{m.name}</div>
                                <div className="module-desc">{m.desc}</div>
                            </div>
                        </button>
                    ))}

                    <div className="sidebar-divider" />
                    <div className="sidebar-section-label">Views</div>

                    <button className={`nav-btn ${activePanel === "chat" ? "active" : ""}`}
                        onClick={() => setActivePanel("chat")}>
                        💬 Chat
                    </button>
                    <button className={`nav-btn ${activePanel === "insights" ? "active" : ""}`}
                        onClick={() => setActivePanel("insights")}>
                        ✨ Insights
                    </button>
                    <button className={`nav-btn ${activePanel === "policies" ? "active" : ""}`}
                        onClick={() => setActivePanel("policies")}>
                        🛡️ Policies
                    </button>
                </aside>

                {/* Main */}
                <main className="main-panel">
                    {activePanel === "chat" && <ChatPanel module={activeModule} xray={xrayOpen} onXrayData={() => { }} />}
                    {activePanel === "insights" && <InsightsPanel />}
                    {activePanel === "policies" && <PoliciesPanel />}
                </main>

                {/* X-Ray */}
                {xrayOpen && <XRayPanel />}
            </div>
        </div>
    );
}

// ── Module HUD Components ────────────────────────────────────────

export interface RAGDoc {
    name: string;
    status: string;
    tokens: string;
    active: boolean;
}

function RAGForgeHUD({ docs, setDocs }: { docs: RAGDoc[], setDocs: React.Dispatch<React.SetStateAction<RAGDoc[]>> }) {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const uploadFile = async (file: File) => {
        // Optimistic UI update
        setDocs(prev => [...prev.filter(d => d.name !== file.name), { name: file.name, status: "Embedding", tokens: "—", active: true }]);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("/api/v1/ragforge/upload", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();

            if (res.ok) {
                setDocs(prev => prev.map(d => d.name === file.name ? { ...d, status: "Ready", tokens: `~${data.chunks_indexed} chunks` } : d));
            } else {
                setDocs(prev => prev.map(d => d.name === file.name ? { ...d, status: "Failed", tokens: "error" } : d));
            }
        } catch (err) {
            setDocs(prev => prev.map(d => d.name === file.name ? { ...d, status: "Failed", tokens: "network err" } : d));
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    };

    const preventDefault = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); };

    const handleDrop = (e: React.DragEvent) => {
        preventDefault(e);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    };

    const toggleDoc = (name: string) => {
        setDocs(prev => prev.map(d => d.name === name ? { ...d, active: !d.active } : d));
    };

    return (
        <div className="module-hud">
            <div className="hud-header">
                <div className="hud-title">🔍 Knowledge Vault</div>
                <div className="hud-subtitle">Drag & drop files to expand local knowledge</div>
            </div>
            <div style={{ display: "flex", gap: "16px" }}>
                <input type="file" ref={fileInputRef} style={{ display: "none" }} onChange={handleFileChange} />
                <div className="upload-dropzone" style={{ flex: 1 }}
                    onClick={handleUploadClick}
                    onDragEnter={preventDefault} onDragOver={preventDefault} onDragLeave={preventDefault} onDrop={handleDrop}>
                    <div className="upload-icon">📥</div>
                    <div className="upload-text">Drop documents here (or click)</div>
                    <div className="upload-hint">PDF, MD, TXT, CSV (Max 50MB)</div>
                </div>
                <div style={{ flex: 1 }} className="doc-list">
                    {docs.map((d, i) => (
                        <div key={i} className="doc-item" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            <input
                                type="checkbox"
                                checked={d.active}
                                onChange={() => toggleDoc(d.name)}
                                style={{ accentColor: "var(--plasma)", cursor: "pointer" }}
                            />
                            <span className="doc-name" style={{ flex: 1, opacity: d.active ? 1 : 0.5 }}>{d.name}</span>
                            <span className="doc-meta">
                                {d.status === "Ready" ? <span style={{ color: "var(--plasma)" }}>● {d.status}</span> : <span style={{ color: d.status === "Failed" ? "var(--ember)" : "var(--aether)" }}>○ {d.status}</span>}
                                <span>{d.tokens}</span>
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

function WatchTowerHUD() {
    const [metrics, setMetrics] = useState<Record<string, any>>({});
    const [history, setHistory] = useState<Record<string, number[]>>({ cpu: [], mem: [], net: [] });
    const [incidents, setIncidents] = useState<any[]>([]);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const res = await fetch("/api/v1/metrics/stream");
                if (res.ok) {
                    const data = await res.json();
                    setMetrics(data);

                    // Maintain a short history for the sparkline charts
                    setHistory(prev => {
                        const next = { ...prev };
                        Object.entries(data).forEach(([key, val]: [string, any]) => {
                            if (!next[key]) next[key] = [];
                            next[key] = [...next[key].slice(-19), val.value];
                        });
                        return next;
                    });

                    // Check for anomalies to generate incidents
                    const newAnomalies: any[] = [];
                    Object.entries(data).forEach(([key, val]: [string, any]) => {
                        if (val.is_anomaly) newAnomalies.push({ key, val });
                    });

                    if (newAnomalies.length > 0) {
                        setIncidents(prev => {
                            let updated = [...prev];
                            newAnomalies.forEach(({ key, val }) => {
                                // Only create a new incident if there isn't an active one for this metric
                                if (!updated.find(i => i.metric === key && i.status === 'active')) {
                                    updated.push({
                                        id: Date.now().toString() + Math.random().toString().slice(2, 6),
                                        metric: key,
                                        value: val.value,
                                        z_score: val.z_score,
                                        timestamp: new Date().toLocaleTimeString(),
                                        status: 'active',
                                        rca: null,
                                        loading: false
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

    const injectAnomaly = async () => {
        try {
            await fetch("/api/v1/events", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mem: 95.0 + (Math.random() * 5),
                    cpu: 45.0,
                    net: 12.0,
                    event: "simulate_spike",
                    _source: "WatchTower UI"
                })
            });
            // Removed alert so it's less intrusive
        } catch (err) {
            console.error("Failed to inject anomaly", err);
        }
    };

    const handleAnalyze = async (incident: any) => {
        setIncidents(prev => prev.map(i => i.id === incident.id ? { ...i, loading: true } : i));
        try {
            const res = await fetch("/api/v1/watchtower/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    metric: incident.metric,
                    value: incident.value,
                    z_score: incident.z_score
                })
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
            setIncidents(prev => prev.map(i => i.id === incident.id ? { ...i, status: 'resolved' } : i));
        } catch (e) { console.error(e); }
    };

    const getMetric = (key: string, defaultVal: number = 0) => {
        return metrics[key] || { value: defaultVal, z_score: 0.0, is_anomaly: false };
    };

    const cpu = getMetric("cpu", 45);
    const mem = getMetric("mem", 62);
    const net = getMetric("net", 12);

    return (
        <div className="module-hud" style={{ overflowY: 'auto', paddingBottom: '40px' }}>
            <div className="hud-header">
                <div className="hud-title">👁️ Live Telemetry</div>
                <button className="btn btn-ghost" style={{ padding: "4px 10px", fontSize: "11px" }} onClick={injectAnomaly}>
                    ⚠️ Simulate Memory Spike
                </button>
            </div>
            <div className="metrics-board">
                <div className="metric-card">
                    <div className="metric-header">
                        <span>CPU Load</span>
                        <span style={{ color: cpu.is_anomaly ? "var(--danger)" : "var(--text-muted)" }}>Z: {(cpu.z_score > 0 ? "+" : "") + cpu.z_score}</span>
                    </div>
                    <div className="metric-value" style={{ color: cpu.is_anomaly ? "var(--danger)" : "inherit" }}>{cpu.value}%</div>
                    <div className="metric-chart">
                        {(history["cpu"] || []).map((v, i) => <div key={i} className={`chart-bar ${cpu.is_anomaly && i === history["cpu"].length - 1 ? 'anomaly' : ''}`} style={{ height: `${Math.min(100, v)}%` }} />)}
                    </div>
                </div>
                <div className="metric-card">
                    <div className="metric-header">
                        <span>Memory Usage</span>
                        <span style={{ color: mem.is_anomaly ? "var(--danger)" : "var(--text-muted)" }}>Z: {(mem.z_score > 0 ? "+" : "") + mem.z_score}</span>
                    </div>
                    <div className="metric-value" style={{ color: mem.is_anomaly ? "var(--danger)" : "inherit" }}>{mem.value}%</div>
                    <div className="metric-chart">
                        {(history["mem"] || []).map((v, i) => <div key={i} className={`chart-bar ${mem.is_anomaly && i === history["mem"].length - 1 ? 'anomaly' : ''}`} style={{ height: `${Math.min(100, v)}%` }} />)}
                    </div>
                </div>
                <div className="metric-card">
                    <div className="metric-header">
                        <span>Network I/O</span>
                        <span style={{ color: net.is_anomaly ? "var(--danger)" : "var(--text-muted)" }}>Z: {(net.z_score > 0 ? "+" : "") + net.z_score}</span>
                    </div>
                    <div className="metric-value" style={{ color: net.is_anomaly ? "var(--danger)" : "inherit" }}>{net.value} MB/s</div>
                    <div className="metric-chart">
                        {(history["net"] || []).map((v, i) => <div key={i} className={`chart-bar ${net.is_anomaly && i === history["net"].length - 1 ? 'anomaly' : ''}`} style={{ height: `${Math.min(100, v)}%` }} />)}
                    </div>
                </div>
            </div>

            {incidents.length > 0 && (
                <div style={{ marginTop: '20px', backgroundColor: 'rgba(20,20,30,0.5)', borderRadius: '12px', padding: '16px' }}>
                    <h3 style={{ fontSize: '14px', marginBottom: '12px', color: 'var(--text)', borderBottom: '1px solid var(--border)', paddingBottom: '8px' }}>Active Incidents & RCA</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {incidents.slice().reverse().map(incident => (
                            <div key={incident.id} style={{
                                padding: '12px',
                                border: '1px solid var(--border)',
                                borderRadius: '8px',
                                backgroundColor: incident.status === 'resolved' ? 'rgba(0,255,100,0.05)' : 'rgba(255,50,50,0.05)',
                                opacity: incident.status === 'resolved' ? 0.6 : 1
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                                    <div>
                                        <span style={{ fontWeight: 'bold', color: incident.status === 'resolved' ? 'var(--text-muted)' : 'var(--danger)', marginRight: '8px' }}>
                                            {incident.metric.toUpperCase()} Spike Detected
                                        </span>
                                        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{incident.timestamp}</span>
                                        <div style={{ fontSize: '13px', marginTop: '4px' }}>
                                            Value: {incident.value.toFixed(1)} (Z-Score: +{incident.z_score.toFixed(2)})
                                        </div>
                                    </div>

                                    {incident.status === 'active' && (
                                        <div style={{ display: 'flex', gap: '8px' }}>
                                            {!incident.rca && !incident.loading && (
                                                <button className="btn btn-ghost" style={{ fontSize: '12px', padding: '4px 8px' }} onClick={() => handleAnalyze(incident)}>
                                                    🔍 Analyze Root Cause
                                                </button>
                                            )}
                                            {incident.loading && (
                                                <span style={{ fontSize: '12px', color: 'var(--brand-glow)' }}>Analyzing 5-Whys...</span>
                                            )}
                                            {incident.rca && (
                                                <>
                                                    <button className="btn btn-primary" style={{ fontSize: '12px', padding: '4px 8px', backgroundColor: 'var(--danger)' }} onClick={() => handleMitigate(incident, "Kill Process")}>
                                                        🛑 Kill Process
                                                    </button>
                                                    <button className="btn btn-ghost" style={{ fontSize: '12px', padding: '4px 8px' }} onClick={() => handleMitigate(incident, "Ignore")}>
                                                        Dismiss
                                                    </button>
                                                </>
                                            )}
                                        </div>
                                    )}
                                    {incident.status === 'resolved' && (
                                        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>✓ Resolved</span>
                                    )}
                                </div>

                                {incident.rca && (
                                    <div style={{ marginTop: '12px', fontSize: '13px', color: 'var(--text)', borderLeft: '2px solid var(--brand-glow)', paddingLeft: '12px' }}>
                                        <div style={{ marginBottom: '8px', color: 'var(--brand-glow)' }}><strong>Causal Chain Identified:</strong></div>
                                        <ul style={{ paddingLeft: '20px', margin: 0, color: 'var(--text-muted)', fontSize: '12px' }}>
                                            {incident.rca.evidence?.map((ev: string, idx: number) => (
                                                <li key={idx} style={{ marginBottom: '4px' }}>{ev}</li>
                                            ))}
                                        </ul>
                                        <div style={{ marginTop: '8px', fontStyle: 'italic' }}>
                                            <strong>Mitigation Recommendation: </strong>
                                            {incident.rca.remediation_steps?.[0] || 'Terminate offending process.'}
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

function StreamSyncHUD() {
    const [events, setEvents] = useState<any[]>([]);

    useEffect(() => {
        const fetchEvents = async () => {
            try {
                const res = await fetch("/api/v1/events/stream");
                if (res.ok) {
                    const data = await res.json();
                    setEvents(data);
                }
            } catch (err) {
                console.error("Failed to fetch Event Stream", err);
            }
        };

        const interval = setInterval(fetchEvents, 2000);
        return () => clearInterval(interval);
    }, []);

    const formatTime = (ts: number) => {
        const d = new Date(ts * 1000);
        return d.toTimeString().split(' ')[0];
    };

    return (
        <div className="module-hud">
            <div className="hud-header">
                <div className="hud-title">⚡ Event Stream Console</div>
                <div className="hud-subtitle">Listening on POST /api/v1/events</div>
            </div>
            <div className="event-console">
                {events.length === 0 ? (
                    <div style={{ color: "var(--text-muted)", fontSize: "12px", textAlign: "center", padding: "24px" }}>
                        Waiting for events... Send a POST request to /api/v1/events
                    </div>
                ) : (
                    events.map((e, i) => (
                        <div key={i} className={`event-row`}>
                            <span className="event-time">{formatTime(e.timestamp)}</span>
                            <span className="event-source">[{e.source}]</span>
                            <span className="event-payload">{JSON.stringify(e.payload)}</span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

interface ReplayItem {
    id: string;
    timestamp_utc: number;
    module: string;
    prompt: string;
    response: string;
    faithfulness_score: number;
    is_used_for_training: boolean;
}

function TuneLabHUD() {
    const [capacityPct, setCapacityPct] = useState<number>(100);
    const [replaySize, setReplaySize] = useState<number>(0);
    const [isTriggering, setIsTriggering] = useState<boolean>(false);
    const [pendingItems, setPendingItems] = useState<ReplayItem[]>([]);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const capRes = await fetch("/api/v1/learning/capacity");
                if (capRes.ok) {
                    const capData = await capRes.json();
                    setCapacityPct(capData.capacity_pct);
                }
                const repRes = await fetch("/api/v1/replay/stats");
                if (repRes.ok) {
                    const repData = await repRes.json();
                    setReplaySize(repData.total_records);
                }
                const itemsRes = await fetch("/api/v1/replay/items");
                if (itemsRes.ok) {
                    const itemsData = await itemsRes.json();
                    setPendingItems(itemsData);
                }
            } catch (err) {
                console.error("Failed to fetch TuneLab stats or items", err);
            }
        };
        fetchStats();
        // Poll every 5 seconds to catch updates after a training run
        const interval = setInterval(fetchStats, 5000);
        return () => clearInterval(interval);
    }, []);

    const triggerTraining = async () => {
        if (isTriggering) return;
        setIsTriggering(true);
        try {
            await fetch("/api/v1/learning/trigger", { method: "POST" });
            alert("OPLoRA training job triggered! Matrix ranks will update when complete.");
        } catch {
            alert("Failed to trigger training");
        }
        setIsTriggering(false);
    };

    return (
        <div className="module-hud">
            <div className="hud-header">
                <div className="hud-title">🎛️ OPLoRA Training Control</div>
                <button
                    className="btn btn-primary"
                    style={{ padding: "4px 12px", fontSize: "12px", opacity: isTriggering ? 0.7 : 1 }}
                    onClick={triggerTraining}
                    disabled={isTriggering}
                >
                    {isTriggering ? "▶ Compiling..." : "▶ Run Cycle Now"}
                </button>
            </div>
            <div style={{ display: "flex", gap: "24px", alignItems: "center", marginBottom: "24px" }}>
                <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "4px" }}>
                        <span style={{ color: "var(--text-muted)" }}>Matrix Capacity Remaining</span>
                        <span style={{ color: "var(--volt-light)", fontWeight: 600 }}>{capacityPct.toFixed(1)}%</span>
                    </div>
                    <div className="novelty-bar-bg"><div className="novelty-bar" style={{ width: `${capacityPct}%` }} /></div>
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "4px" }}>
                        <span style={{ color: "var(--text-muted)" }}>Replay Buffer Size (Need &gt;10 high-quality)</span>
                        <span style={{ color: "var(--aether-light)", fontWeight: 600 }}>{replaySize.toLocaleString()} items</span>
                    </div>
                    <div className="novelty-bar-bg"><div className="novelty-bar" style={{ width: `${Math.min(100, (replaySize / 10) * 100)}%`, background: "var(--aether)" }} /></div>
                </div>
            </div>

            <div className="hud-title" style={{ fontSize: "14px", borderTop: "1px solid var(--border)", paddingTop: "16px", marginBottom: "12px" }}>
                📖 Pending Knowledge Feed
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px", overflowY: "auto", maxHeight: "400px", paddingRight: "8px" }}>
                {pendingItems.length === 0 ? (
                    <div style={{ color: "var(--text-muted)", fontSize: "12px", textAlign: "center", padding: "24px" }}>
                        No interactions recorded yet. Chat with the AI!
                    </div>
                ) : (
                    pendingItems.map((item) => {
                        // Requirements from BitNetTrainer: min_faithfulness=0.85, not excluded
                        const isFilteredNoise = item.faithfulness_score < 0.85;
                        const isTrained = item.is_used_for_training;
                        let statusColor = "var(--text-muted)", statusText = "";

                        if (isTrained) {
                            statusColor = "var(--aether)"; statusText = "Already Compiled";
                        } else if (isFilteredNoise) {
                            statusColor = "gray"; statusText = "Filtered Noise";
                        } else {
                            statusColor = "var(--volt-light)"; statusText = "Ready for Training";
                        }

                        return (
                            <div key={item.id} style={{
                                padding: "12px",
                                borderRadius: "8px",
                                background: "var(--surface-raised)",
                                border: `1px solid ${isFilteredNoise ? 'transparent' : 'rgba(255,255,255,0.05)'}`,
                                opacity: isFilteredNoise || isTrained ? 0.6 : 1,
                                display: "flex",
                                flexDirection: "column",
                                gap: "6px"
                            }}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                    <span style={{ fontSize: "11px", color: statusColor, fontWeight: 600, display: "flex", alignItems: "center", gap: "4px" }}>
                                        <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: statusColor }} />
                                        {statusText}
                                    </span>
                                    <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>
                                        Faithfulness: {(item.faithfulness_score * 100).toFixed(0)}%
                                    </span>
                                </div>
                                <div style={{ fontSize: "12px", color: "var(--text-bright)", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
                                    <span style={{ color: "var(--text-muted)", marginRight: "8px" }}>User:</span>
                                    {item.prompt}
                                </div>
                                <div style={{ fontSize: "12px", color: "var(--text-muted)", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
                                    <span style={{ marginRight: "8px" }}>AI:</span>
                                    {item.response}
                                </div>
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
}

// ── Chat Panel ───────────────────────────────────────────────────
function ChatPanel({ module, xray }: { module: string; xray: boolean; onXrayData: (g: CausalGraph | null) => void }) {
    // Store messages by module ID so switching tabs doesn't mix conversations
    const [messagesByModule, setMessagesByModule] = useState<Record<string, Message[]>>({
        localbuddy: [],
        ragforge: [],
        watchtower: [],
        streamsync: [],
        tunelab: []
    });

    // Lifted RAG document state
    const [ragDocs, setRagDocs] = useState<RAGDoc[]>([]);

    const messages = messagesByModule[module] || [];

    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [streaming, setStreaming] = useState(true);
    const [xrayGraphByModule, setXrayGraphByModule] = useState<Record<string, CausalGraph | null>>({});

    const xrayGraph = xrayGraphByModule[module] || null;
    const bottomRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const mod = MODULES.find(m => m.id === module)!;

    // Clear input when switching modules to avoid confusion
    useEffect(() => {
        setInput("");
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
        }
    }, [module]);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

    const autosize = () => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 120) + "px";
    };

    const send = useCallback(async () => {
        const text = input.trim();
        if (!text || loading) return;

        const userMsg: Message = {
            id: Date.now().toString(),
            role: "user",
            content: text,
            module,
        };

        setMessagesByModule(prev => ({
            ...prev,
            [module]: [...(prev[module] || []), userMsg]
        }));
        setInput("");
        setLoading(true);
        if (textareaRef.current) textareaRef.current.style.height = "auto";

        try {
            const activeDocs = module === "ragforge" ? ragDocs.filter(d => d.active).map(d => d.name) : [];
            const contextPayload = activeDocs.length > 0 ? { active_docs: activeDocs } : {};

            const res = await fetch("/api/v1/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: `ui-session-${module}`, // Separate backend sessions per module too
                    module,
                    message: text,
                    xray_mode: xray,
                    context: contextPayload,
                }),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            const blocked = data.policy_decisions?.some((p: PolicyDecision) => !p.allowed);
            const aiMsg: Message = {
                id: Date.now().toString() + "-ai",
                role: "assistant",
                content: data.response,
                module: data.module,
                latency_ms: data.latency_ms,
                faithfulness_score: data.faithfulness_score,
                policy_decisions: data.policy_decisions,
                causal_graph: data.causal_graph,
                blocked,
            };

            setMessagesByModule(prev => ({
                ...prev,
                [module]: [...(prev[module] || []), aiMsg]
            }));

            if (data.causal_graph) {
                setXrayGraphByModule(prev => ({ ...prev, [module]: data.causal_graph }));
            }

        } catch (err) {
            setMessagesByModule(prev => ({
                ...prev,
                [module]: [...(prev[module] || []), {
                    id: Date.now().toString() + "-err",
                    role: "assistant",
                    content: `⚠️ Could not reach backend: ${err}. Make sure the FastAPI server is running on port 8765.`,
                    module,
                }]
            }));
        } finally {
            setLoading(false);
        }
    }, [input, loading, module, xray]);

    const onKey = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    };

    return (
        <div className="chat-container">
            {/* Chat header */}
            <div className="chat-header">
                <div className="chat-module-badge">{mod.icon} {mod.name}</div>
                <div className="chat-title">AI Assistant</div>
                <div className="chat-subtitle">All processing is 100% local — your data never leaves this machine</div>
            </div>

            {/* Dynamic Module HUD */}
            {module === "ragforge" && <RAGForgeHUD docs={ragDocs} setDocs={setRagDocs} />}
            {module === "watchtower" && <WatchTowerHUD />}
            {module === "streamsync" && <StreamSyncHUD />}
            {module === "tunelab" && <TuneLabHUD />}

            {/* Messages */}
            <div className="messages-area">
                {messages.length === 0 && !loading ? (
                    <div className="empty-state">
                        <div className="empty-orb">{mod.icon}</div>
                        <div className="empty-title">Welcome to {mod.name}</div>
                        <div className="empty-sub">{mod.desc} — fully local, zero cloud, perpetually learning.</div>
                        <div className="suggestion-chips">
                            {SUGGESTIONS.map((s, i) => (
                                <button key={i} className="chip" onClick={() => { setInput(s); textareaRef.current?.focus(); }}>
                                    {s}
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    <>
                        {messages.map(msg => (
                            <MessageBubble key={msg.id} msg={msg} />
                        ))}
                        {loading && (
                            <div className="message-row">
                                <div className="avatar ai">Æ</div>
                                <div className="bubble ai">
                                    <div className="typing-indicator">
                                        <div className="typing-dot" />
                                        <div className="typing-dot" />
                                        <div className="typing-dot" />
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
                )}
                <div ref={bottomRef} />
            </div>

            {/* Input */}
            <div className="input-area">
                <div className="input-wrapper">
                    <textarea
                        ref={textareaRef}
                        className="msg-input"
                        placeholder={`Message ${mod.name}...`}
                        value={input}
                        rows={1}
                        onChange={e => { setInput(e.target.value); autosize(); }}
                        onKeyDown={onKey}
                    />
                    <button className="send-btn" onClick={send} disabled={!input.trim() || loading}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
                        </svg>
                    </button>
                </div>
                <div className="input-hints">
                    <span className="hint-text">Enter to send · Shift+Enter for new line</span>
                    <label className="stream-toggle">
                        <div className={`toggle-track ${streaming ? "on" : ""}`} onClick={() => setStreaming(v => !v)}>
                            <div className="toggle-thumb" />
                        </div>
                        Stream tokens
                    </label>
                </div>
            </div>

            {/* Inline X-Ray graph (shown below messages when xray=true) */}
            {xray && xrayGraph && (
                <InlineXRay graph={xrayGraph} />
            )}
        </div>
    );
}

// ── Message bubble ───────────────────────────────────────────────
function MessageBubble({ msg }: { msg: Message }) {
    const isUser = msg.role === "user";
    const fScore = msg.faithfulness_score;

    return (
        <div className={`message-row ${isUser ? "user" : ""}`}>
            <div className={`avatar ${isUser ? "user-av" : "ai"}`}>
                {isUser ? "You" : "Æ"}
            </div>
            <div>
                <div className={`bubble ${isUser ? "user-bubble" : "ai"}`}>
                    {msg.content}
                </div>
                {!isUser && (
                    <div className="bubble-meta">
                        <span className="badge module">{msg.module}</span>
                        {msg.latency_ms !== undefined && (
                            <span className="badge latency">{msg.latency_ms.toFixed(0)}ms</span>
                        )}
                        {fScore !== null && fScore !== undefined && (
                            <span className={`badge ${fScore >= 0.92 ? "fidelity-high" : "fidelity-low"}`}>
                                {fScore >= 0.92 ? "✓" : "⚠"} fidelity {(fScore * 100).toFixed(0)}%
                            </span>
                        )}
                        {msg.blocked && (
                            <span className="badge blocked">🛡 policy applied</span>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

// ── Inline X-Ray ─────────────────────────────────────────────────
function InlineXRay({ graph }: { graph: CausalGraph }) {
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

// ── X-Ray side panel ─────────────────────────────────────────────
function XRayPanel() {
    return (
        <div className="xray-panel">
            <div className="xray-header">
                <div className="xray-title">🔬 X-Ray Mode</div>
            </div>
            <div className="xray-body">
                <div className="xray-empty">
                    <div style={{ fontSize: 28 }}>🔬</div>
                    <div style={{ fontWeight: 600, color: "var(--text-secondary)" }}>Send a message</div>
                    <div>The causal reasoning graph will appear here after each response, showing every decision step the AI made.</div>
                </div>
            </div>
        </div>
    );
}

// ── Insights Panel ───────────────────────────────────────────────
function InsightsPanel() {
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

// ── Policies Panel ───────────────────────────────────────────────
function PoliciesPanel() {
    const [policy, setPolicy] = useState("");
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        fetch("/api/v1/policies")
            .then(r => r.json())
            .then(d => setPolicy(d.policy ?? ""))
            .catch(() => { });
    }, []);

    const save = async () => {
        await fetch("/api/v1/policies", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ policy }),
        });
        setDirty(false);
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
    };

    return (
        <div className="panel-wrapper">
            <div className="panel-title">🛡️ Silicon Colosseum</div>
            <div className="panel-sub">Live OPA Rego policy — changes take effect immediately, no restart needed</div>

            <div className="policy-card">
                <div className="policy-toolbar">
                    <span className="policy-lang">OPA Rego</span>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        {dirty && <div className="dirty-dot" title="Unsaved changes" />}
                        {saved && <span style={{ fontSize: 11, color: "var(--plasma)" }}>✓ Saved</span>}
                        <button className="btn btn-ghost" onClick={() => { setPolicy(policy); setDirty(false); }}>Reset</button>
                        <button className="btn btn-primary" onClick={save} disabled={!dirty}>Save & Reload</button>
                    </div>
                </div>
                <textarea
                    className="policy-textarea"
                    value={policy}
                    onChange={e => { setPolicy(e.target.value); setDirty(true); setSaved(false); }}
                    spellCheck={false}
                />
            </div>

            <div style={{ padding: "16px", background: "var(--glass)", border: "1px solid var(--glass-border)", borderRadius: "var(--radius)", fontSize: 12 }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: "var(--text-secondary)" }}>📚 Quick Reference</div>
                {[
                    ["deny_reasons contains r if { input.tool_call_count > N }", "Limit tool calls per turn"],
                    ["deny_reasons contains r if { input.faithfulness_score < 0.92 }", "Block low-confidence outputs"],
                    ["contains(lower(input.message), \"pattern\")", "Detect prohibited content"],
                ].map(([code, desc], i) => (
                    <div key={i} style={{ marginBottom: 10 }}>
                        <code style={{ display: "block", background: "rgba(0,0,0,0.3)", padding: "6px 10px", borderRadius: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#a5b4fc", marginBottom: 3 }}>
                            {code}
                        </code>
                        <div style={{ color: "var(--text-muted)" }}>{desc}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
