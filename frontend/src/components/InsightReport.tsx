// AetherForge v1.0 — frontend/src/components/InsightReport.tsx
// ─────────────────────────────────────────────────────────────────
// InsightForge weekly novelty report viewer.
// Fetches insights from the backend and displays them as cards
// with novelty score bars, topic badges, and timestamps.
// ─────────────────────────────────────────────────────────────────
import React, { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { getInsights, getReplayStats, triggerTraining, type Insight, type ReplayStats } from "../lib/tauri";

export default function InsightReport(): JSX.Element {
    const [insights, setInsights] = useState<Insight[]>([]);
    const [stats, setStats] = useState<ReplayStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [trainingMsg, setTrainingMsg] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        setLoading(true);
        try {
            const [ins, st] = await Promise.all([getInsights(), getReplayStats()]);
            setInsights(ins);
            setStats(st);
        } catch { /* no backend yet */ }
        setLoading(false);
    }, []);

    useEffect(() => { refresh(); }, [refresh]);

    const handleTriggerTraining = async () => {
        setTrainingMsg("Triggering OPLoRA training job...");
        try {
            const res = await triggerTraining();
            setTrainingMsg(res.message);
            setTimeout(() => setTrainingMsg(null), 4000);
        } catch (e) {
            setTrainingMsg(`Error: ${e}`);
        }
    };

    return (
        <div className="flex flex-col h-full overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: "var(--border-subtle)" }}>
                <div>
                    <h2 className="text-sm font-semibold gradient-text">InsightForge</h2>
                    <p className="text-xs text-muted">Weekly novelty detection & knowledge synthesis</p>
                </div>
                <div className="flex gap-2">
                    <button onClick={refresh} className="btn-ghost text-xs px-3 py-1.5">Refresh</button>
                    <button onClick={handleTriggerTraining} id="trigger-training-btn" className="btn-primary text-xs px-3 py-1.5">
                        ▶ Train Now
                    </button>
                </div>
            </div>

            {/* Training notification */}
            {trainingMsg && (
                <div className="px-4 py-2 text-xs" style={{ background: "rgba(139,92,246,0.1)", borderBottom: "1px solid rgba(139,92,246,0.2)", color: "var(--accent-volt)" }}>
                    {trainingMsg}
                </div>
            )}

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {/* Replay Buffer Stats */}
                {stats && (
                    <div className="glass p-4 grid grid-cols-3 gap-4">
                        {[
                            { label: "Total Interactions", value: stats.total_records.toLocaleString(), color: "var(--accent-volt)" },
                            { label: "Buffer Size", value: `${stats.size_mb} MB`, color: "var(--accent-plasma)" },
                            { label: "Avg Faithfulness", value: `${(stats.avg_faithfulness * 100).toFixed(0)}%`, color: stats.avg_faithfulness >= 0.92 ? "var(--accent-safe)" : "var(--accent-ember)" },
                        ].map(({ label, value, color }) => (
                            <div key={label} className="text-center">
                                <div className="text-xl font-bold" style={{ color }}>{value}</div>
                                <div className="text-xs text-muted mt-0.5">{label}</div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Insights list */}
                {loading ? (
                    <div className="flex justify-center py-8">
                        <div className="spinner" />
                    </div>
                ) : insights.length === 0 ? (
                    <div className="glass p-8 text-center">
                        <p className="text-secondary text-sm">No insights yet.</p>
                        <p className="text-muted text-xs mt-1">Insights appear after the first weekly OPLoRA cycle (Sunday 3 AM) or manual "Train Now".</p>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {insights.map((ins, idx) => (
                            <motion.div
                                key={ins.insight_id}
                                initial={{ opacity: 0, y: 6 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: idx * 0.05 }}
                                className="glass p-4"
                            >
                                <div className="flex items-start justify-between gap-3">
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm font-medium truncate">{ins.title}</p>
                                        <p className="text-xs text-secondary mt-1 leading-relaxed">{ins.summary}</p>
                                    </div>
                                    {/* Novelty score */}
                                    <div className="flex-shrink-0 text-right">
                                        <div className="text-lg font-bold" style={{ color: ins.novelty_score > 0.7 ? "var(--accent-ember)" : "var(--accent-volt)" }}>
                                            {(ins.novelty_score * 100).toFixed(0)}%
                                        </div>
                                        <div className="text-xs text-muted">novelty</div>
                                    </div>
                                </div>
                                {/* Novelty bar */}
                                <div className="mt-3 h-1 rounded-full" style={{ background: "rgba(255,255,255,0.05)" }}>
                                    <div
                                        className="h-full rounded-full transition-all"
                                        style={{
                                            width: `${ins.novelty_score * 100}%`,
                                            background: ins.novelty_score > 0.7
                                                ? "linear-gradient(90deg, #f97316, #fb923c)"
                                                : "linear-gradient(90deg, #8b5cf6, #22d3ee)",
                                        }}
                                    />
                                </div>
                                {/* Topics */}
                                <div className="flex flex-wrap gap-1.5 mt-3">
                                    {ins.topics.map(t => (
                                        <span key={t} className="badge-volt">{t}</span>
                                    ))}
                                    <span className="text-xs text-muted ml-auto">
                                        {new Date(ins.generated_at).toLocaleDateString()}
                                    </span>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
