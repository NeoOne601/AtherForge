import { useState, useEffect } from "react";

export function WatchTowerHUD() {
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
            await fetch("/api/v1/watchtower/inject_anomaly", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ metric: "mem", value: 99.5 })
            });
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
