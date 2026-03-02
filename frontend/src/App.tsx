// AetherForge v1.0 — frontend/src/App.tsx
// ─────────────────────────────────────────────────────────────────
// Root application shell. Manages global state: active module,
// session ID, X-Ray toggle, system health status, and dark theme.
//
// Layout: Three-column glass-panel UI
//   Left:   Sidebar (module nav + system status + X-Ray toggle)
//   Center: Active module panel (chat / insights / policies)
//   Right:  X-Ray causal graph (visible when xray_mode=true)
// ─────────────────────────────────────────────────────────────────
import React, { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ChatInterface from "./components/ChatInterface";
import ModuleTabs from "./components/ModuleTabs";
import XRayGraph from "./components/XRayGraph";
import InsightReport from "./components/InsightReport";
import PolicyEditor from "./components/PolicyEditor";
import {
    checkHealth,
    getSystemStatus,
    newSessionId,
    type CausalGraph,
    type SystemStatus,
} from "./lib/tauri";

// ── App state ─────────────────────────────────────────────────────
type ActivePanel = "chat" | "insights" | "policies";

export default function App(): JSX.Element {
    const [sessionId] = useState<string>(newSessionId);
    const [activeModule, setActiveModule] = useState<string>("localbuddy");
    const [activePanel, setActivePanel] = useState<ActivePanel>("chat");
    const [xrayMode, setXrayMode] = useState<boolean>(false);
    const [causalGraph, setCausalGraph] = useState<CausalGraph | null>(null);
    const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
    const [backendReady, setBackendReady] = useState<boolean>(false);
    const [backendError, setBackendError] = useState<string | null>(null);
    const healthPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // ── Health check on mount ─────────────────────────────────────
    useEffect(() => {
        const check = async () => {
            try {
                await checkHealth();
                setBackendReady(true);
                setBackendError(null);
                fetchStatus();
            } catch {
                setBackendError("Backend not reachable. Run ./run_dev.sh first.");
            }
        };
        check();
        // Poll status every 10s
        healthPollRef.current = setInterval(fetchStatus, 10_000);
        return () => {
            if (healthPollRef.current) clearInterval(healthPollRef.current);
        };
    }, []);

    const fetchStatus = useCallback(async () => {
        try {
            const status = await getSystemStatus();
            setSystemStatus(status);
        } catch { /* non-fatal */ }
    }, []);

    const handleNewGraph = useCallback((graph: CausalGraph) => {
        setCausalGraph(graph);
        if (!xrayMode) setXrayMode(true); // Auto-show X-Ray when graph arrives
    }, [xrayMode]);

    const showXRay = xrayMode && causalGraph !== null;

    return (
        <div className="flex flex-col h-screen overflow-hidden" style={{ background: "var(--bg-primary)" }}>
            {/* ── Title Bar ─────────────────────────────────────────── */}
            <header className="flex items-center justify-between px-4 py-2 border-b" style={{ borderColor: "var(--border-subtle)", background: "rgba(5,12,20,0.95)" }}>
                <div className="flex items-center gap-3">
                    {/* Logo */}
                    <div className="relative w-7 h-7">
                        <div className="absolute inset-0 rounded-lg" style={{ background: "linear-gradient(135deg, #8b5cf6, #22d3ee)" }} />
                        <div className="absolute inset-0.5 rounded-md flex items-center justify-center text-xs font-bold text-white">Æ</div>
                    </div>
                    <span className="font-semibold text-sm tracking-wide gradient-text">AetherForge</span>
                    <span className="text-xs text-muted ml-1">v1.0</span>
                </div>

                <div className="flex items-center gap-4">
                    {/* System status pill */}
                    {systemStatus && (
                        <div className="flex items-center gap-2 text-xs text-secondary">
                            <span className="text-muted">CPU</span>
                            <span style={{ color: systemStatus.cpu_pct > 80 ? "var(--accent-danger)" : "var(--accent-safe)" }}>
                                {systemStatus.cpu_pct.toFixed(0)}%
                            </span>
                            {systemStatus.battery_pct !== null && (
                                <>
                                    <span className="text-muted">·</span>
                                    <span>🔋 {systemStatus.battery_pct?.toFixed(0)}%</span>
                                </>
                            )}
                        </div>
                    )}

                    {/* Backend status */}
                    <div className="flex items-center gap-1.5 text-xs">
                        <span className={`status-dot ${backendReady ? "online" : "offline"}`} />
                        <span className="text-secondary">{backendReady ? "Online" : "Offline"}</span>
                    </div>

                    {/* X-Ray toggle */}
                    <button
                        onClick={() => setXrayMode(v => !v)}
                        className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 ${xrayMode ? "badge-plasma" : "btn-ghost"}`}
                    >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                            <circle cx="12" cy="12" r="3" />
                        </svg>
                        X-Ray {xrayMode ? "ON" : "OFF"}
                    </button>
                </div>
            </header>

            {/* ── Backend error banner ───────────────────────────────── */}
            <AnimatePresence>
                {backendError && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="px-4 py-2 text-xs text-center"
                        style={{ background: "rgba(248,113,113,0.1)", borderBottom: "1px solid rgba(248,113,113,0.3)", color: "var(--accent-danger)" }}
                    >
                        ⚡ {backendError}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── Main layout ───────────────────────────────────────── */}
            <div className="flex flex-1 overflow-hidden">
                {/* Left sidebar */}
                <aside className="w-52 flex-shrink-0 flex flex-col border-r p-3 gap-1" style={{ borderColor: "var(--border-subtle)", background: "rgba(5,12,20,0.6)" }}>
                    <p className="text-muted text-xs font-medium px-2 py-1 mb-1 uppercase tracking-wider">Modules</p>
                    <ModuleTabs active={activeModule} onChange={setActiveModule} />

                    <div className="flex-1" />
                    <div className="border-t pt-2 mt-2" style={{ borderColor: "var(--border-subtle)" }}>
                        {[
                            { id: "chat", label: "Chat", icon: "💬" },
                            { id: "insights", label: "Insights", icon: "✨" },
                            { id: "policies", label: "Policies", icon: "🛡️" },
                        ].map(({ id, label, icon }) => (
                            <button
                                key={id}
                                onClick={() => setActivePanel(id as ActivePanel)}
                                className={`sidebar-item w-full ${activePanel === id ? "active" : ""}`}
                            >
                                <span>{icon}</span>
                                <span>{label}</span>
                            </button>
                        ))}
                    </div>
                </aside>

                {/* Center panel */}
                <main className="flex-1 flex flex-col overflow-hidden">
                    <AnimatePresence mode="wait">
                        {activePanel === "chat" && (
                            <motion.div key="chat" className="flex-1 overflow-hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                                <ChatInterface
                                    sessionId={sessionId}
                                    activeModule={activeModule}
                                    xrayMode={xrayMode}
                                    onNewGraph={handleNewGraph}
                                    enabled={backendReady}
                                />
                            </motion.div>
                        )}
                        {activePanel === "insights" && (
                            <motion.div key="insights" className="flex-1 overflow-hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                                <InsightReport />
                            </motion.div>
                        )}
                        {activePanel === "policies" && (
                            <motion.div key="policies" className="flex-1 overflow-hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                                <PolicyEditor />
                            </motion.div>
                        )}
                    </AnimatePresence>
                </main>

                {/* Right X-Ray panel */}
                <AnimatePresence>
                    {showXRay && (
                        <motion.aside
                            key="xray"
                            initial={{ width: 0, opacity: 0 }}
                            animate={{ width: 380, opacity: 1 }}
                            exit={{ width: 0, opacity: 0 }}
                            transition={{ duration: 0.3, ease: "easeInOut" }}
                            className="flex-shrink-0 border-l overflow-hidden"
                            style={{ borderColor: "var(--border-subtle)", background: "rgba(5,12,20,0.8)" }}
                        >
                            <XRayGraph graph={causalGraph} onClose={() => setXrayMode(false)} />
                        </motion.aside>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
