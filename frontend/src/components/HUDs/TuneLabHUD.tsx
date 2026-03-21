import { useState, useEffect } from "react";
import { ReplayItem } from "../../types";

export function TuneLabHUD() {
    const [capacityPct, setCapacityPct] = useState<number>(100);
    const [replaySize, setReplaySize] = useState<number>(0);
    const [isTriggering, setIsTriggering] = useState<boolean>(false);
    const [pendingItems, setPendingItems] = useState<ReplayItem[]>([]);
    const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
    const [statusEvents, setStatusEvents] = useState<any[]>([]);

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
                const historyRes = await fetch("/api/v1/learning/history");
                if (historyRes.ok) {
                    setTrainingHistory(await historyRes.json());
                }
                const eventsRes = await fetch("/api/v1/events/stream?limit=10");
                if (eventsRes.ok) {
                    const allEvents = await eventsRes.json();
                    setStatusEvents(allEvents.filter((e: any) => e.source === "TuneLab"));
                }
            } catch (err) {
                console.error("Failed to fetch TuneLab stats or items", err);
            }
        };
        fetchStats();
        // Poll every 3 seconds for tighter feedback during training
        const interval = setInterval(fetchStats, 3000);
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
                        <span style={{ color: "var(--text-muted)" }} title="The AI's short-term learning memory. Training clears this space.">Matrix Capacity Remaining ℹ️</span>
                        <span style={{ color: "var(--volt-light)", fontWeight: 600 }}>{capacityPct.toFixed(1)}%</span>
                    </div>
                    <div className="novelty-bar-bg"><div className="novelty-bar" style={{ width: `${capacityPct}%` }} /></div>
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "4px" }}>
                        <span style={{ color: "var(--text-muted)" }} title="Smart, successful answers saved to be studied and learned permanently.">Replay Buffer Size ℹ️</span>
                        <span style={{ color: "var(--aether-light)", fontWeight: 600 }}>{replaySize.toLocaleString()} items</span>
                    </div>
                    <div className="novelty-bar-bg"><div className="novelty-bar" style={{ width: `${Math.min(100, (replaySize / 10) * 100)}%`, background: "var(--aether)" }} /></div>
                </div>
            </div>

            {/* ── Training Metrics Timeline ─────────────────────── */}
            <div className="hud-title" style={{ fontSize: "14px", borderTop: "1px solid var(--border)", paddingTop: "16px", marginBottom: "4px" }}>
                📊 Training Metrics Timeline
            </div>
            <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "12px" }}>
                Tracks the AI's learning progress. "Loss" going down means the AI is getting smarter and making fewer mistakes.
            </div>
            {trainingHistory.length > 0 ? (
                <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: "12px", padding: "20px", marginBottom: "24px", border: "1px solid rgba(255,255,255,0.1)" }}>
                    <div style={{ display: "flex", alignItems: "flex-end", gap: "6px", height: "120px", marginBottom: "16px", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: "2px" }}>
                        {trainingHistory.map((run, i) => {
                            const h = Math.max(15, Math.min(100, (1 - (run.training_loss || 0)) * 100));
                            return (
                                <div key={i} title={`Task: ${run.task_id}\nLoss: ${run.training_loss?.toFixed(4)}\nSamples: ${run.samples_used}`} style={{
                                    flex: 1,
                                    background: "linear-gradient(to top, var(--brand-glow), var(--volt))",
                                    height: `${h}%`,
                                    borderRadius: "3px 3px 0 0",
                                    minWidth: "12px",
                                    boxShadow: "0 0 10px rgba(var(--brand-glow-rgb, 120, 80, 255), 0.4)",
                                    transition: "height 0.5s ease-out"
                                }} />
                            );
                        })}
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", fontWeight: 500 }}>
                        <span style={{ color: "var(--text-muted)" }}>📈 Performance Evolution (Inverse Loss)</span>
                        <span style={{ color: "var(--volt-bright)" }}>Current Error: {(trainingHistory[trainingHistory.length - 1]?.training_loss * 100).toFixed(1)}%</span>
                    </div>
                </div>
            ) : (
                <div style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)", fontSize: "12px", background: "rgba(0,0,0,0.1)", borderRadius: "12px", marginBottom: "24px" }}>
                    No training history available. Complete your first cycle!
                </div>
            )}

            {/* ── Real-time Status Feed ─────────────────────────── */}
            <div className="hud-title" style={{ fontSize: "14px", borderTop: "1px solid var(--border)", paddingTop: "16px", marginBottom: "4px" }}>
                📡 Live Training Status
            </div>
            <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "12px" }}>
                Watch the AI's internal brain making active parameter adjustments.
            </div>
            <div style={{ background: "rgba(20,20,30,0.4)", borderRadius: "8px", padding: "12px", marginBottom: "24px", border: "1px solid rgba(255,255,255,0.05)", maxHeight: "150px", overflowY: "auto" }}>
                {statusEvents.length === 0 ? (
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", textAlign: "center", padding: "8px" }}>Awaiting activity...</div>
                ) : (
                    statusEvents.map((ev, i) => (
                        <div key={i} style={{ fontSize: "11px", marginBottom: "6px", display: "flex", gap: "8px" }}>
                            <span style={{ color: "var(--text-muted)", opacity: 0.6 }}>{new Date(ev.timestamp * 1000).toLocaleTimeString([], { hour12: false })}</span>
                            <span style={{ color: ev.event_type.includes("fail") ? "var(--danger)" : "var(--brand-glow)", fontWeight: "bold" }}>[{ev.event_type}]</span>
                            <span style={{ color: "var(--text-bright)" }}>{ev.payload?.message || ev.payload?.reason || (ev.event_type === "training_completed" ? `Loss: ${ev.payload?.training_loss?.toFixed(4)}` : JSON.stringify(ev.payload))}</span>
                        </div>
                    ))
                )}
            </div>

            <div className="hud-title" style={{ fontSize: "14px", borderTop: "1px solid var(--border)", paddingTop: "16px", marginBottom: "4px" }}>
                📖 Pending Knowledge Feed
            </div>
            <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "12px" }}>
                A scratchpad of recent good chats. Highly rated answers will be permanently learned in the next cycle.
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px", overflowY: "auto", maxHeight: "400px", paddingRight: "8px" }}>
                {pendingItems.length === 0 ? (
                    <div style={{ color: "var(--text-muted)", fontSize: "12px", textAlign: "center", padding: "24px" }}>
                        No interactions recorded yet. Chat with the AI!
                    </div>
                ) : (
                    pendingItems.map((item) => {
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
