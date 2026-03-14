import React, { useState, useEffect, useCallback } from "react";
import "./index.css";
import { Module, SessionSummary, StoredMessage, MODULES } from "./types";
import { ChatPanel } from "./components/ChatPanel";
import { XRayPanel } from "./components/Panels/XRayPanel";
import { InsightsPanel } from "./components/Panels/InsightsPanel";
import { PoliciesPanel } from "./components/Panels/PoliciesPanel";
import { SyncPanel } from "./components/Panels/SyncPanel";
import { LoggerPanel } from "./components/Panels/LoggerPanel";
import { SettingsPanel } from "./components/Panels/SettingsPanel";

export default function App() {
    const [activeModule, setActiveModule] = useState("localbuddy");
    const [activePanel, setActivePanel] = useState<"chat" | "insights" | "policies" | "sync" | "settings">("chat");
    const [xrayOpen, setXrayOpen] = useState(false);
    const [online, setOnline] = useState(false);
    const [cpuPct, setCpuPct] = useState<number | null>(null);
    const [visualTheme, setVisualTheme] = useState("Sovereign Dark");

    const getThemeClass = (themeName: string) => {
        const mapping: Record<string, string> = {
            "Sovereign Dark": "theme-sovereign",
            "Nordic Frost": "theme-nordic",
            "Neon Cyberpunk": "theme-cyberpunk",
            "Monochrome Pro": "theme-monochrome",
            "Forest Terminal": "theme-forest"
        };
        return mapping[themeName] || "theme-sovereign";
    };

    // ── Chat Model State ──────────────────────────────────────────
    const [chatModels, setChatModels] = useState<any[]>([]);
    const [selectedChatModel, setSelectedChatModel] = useState<string>("");
    const [isChangingChatModel, setIsChangingChatModel] = useState(false);
    const [webSearchEnabled, setWebSearchEnabled] = useState(false);
    const [deepReasoningEnabled, setDeepReasoningEnabled] = useState(false);
    const [analyticsEnabled, setAnalyticsEnabled] = useState(false);
    const [grammarAssistEnabled, setGrammarAssistEnabled] = useState(false);

    useEffect(() => {
        fetch("/api/v1/chat-models").then(r => r.json()).then(d => {
            setChatModels(d.models || []);
            if (d.selected) setSelectedChatModel(d.selected);
        }).catch(() => { });

        // Fetch visual theme from settings
        fetch("/api/v1/settings")
            .then(r => r.json())
            .then(data => {
                const theme = data.server?.fields?.VISUAL_THEME?.value;
                if (theme) setVisualTheme(theme);
            })
            .catch(() => { });
    }, []);

    const handleChatModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
        const model_id = e.target.value;
        setSelectedChatModel(model_id);
        setIsChangingChatModel(true);
        try {
            await fetch("/api/v1/chat-model-select", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_id })
            });
        } catch (err) { console.error(err); }
        finally { setIsChangingChatModel(false); }
    };

    // ── Session state ─────────────────────────────────────────────
    const [sessions, setSessions] = useState<SessionSummary[]>([]);
    const [sessionsPanelOpen, setSessionsPanelOpen] = useState(true);
    const [activeSessionIds, setActiveSessionIds] = useState<Record<string, string | null>>({
        localbuddy: null,
        ragforge: null,
        watchtower: null,
        streamsync: null,
        tunelab: null
    });
    const [loadedSessionMessages, setLoadedSessionMessages] = useState<StoredMessage[] | null>(null);
    const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
    const [editingTitle, setEditingTitle] = useState("");

    const refreshSessions = useCallback(async () => {
        try {
            const res = await fetch(`/api/v1/sessions?module=${activeModule}`);
            if (res.ok) setSessions(await res.json());
        } catch { /* non-fatal */ }
    }, [activeModule]);

    useEffect(() => { refreshSessions(); }, [refreshSessions]);

    const loadSession = async (sessionId: string) => {
        setActiveSessionIds(prev => ({ ...prev, [activeModule]: sessionId }));
        try {
            const res = await fetch(`/api/v1/sessions/${sessionId}/messages`);
            if (res.ok) setLoadedSessionMessages(await res.json());
        } catch { /* non-fatal */ }
    };

    const deleteSession = async (sessionId: string) => {
        if (!confirm("Delete this session and all its messages?")) return;
        await fetch(`/api/v1/sessions/${sessionId}`, { method: "DELETE" });
        setSessions(s => s.filter(x => x.id !== sessionId));
        if (activeSessionIds[activeModule] === sessionId) {
            setActiveSessionIds(prev => ({ ...prev, [activeModule]: null }));
            setLoadedSessionMessages(null);
        }
    };

    const startRename = (s: SessionSummary) => {
        setEditingSessionId(s.id);
        setEditingTitle(s.title);
    };

    const commitRename = async (sessionId: string) => {
        if (!editingTitle.trim()) return;
        await fetch(`/api/v1/sessions/${sessionId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: editingTitle.trim() }),
        });
        setSessions(s => s.map(x => x.id === sessionId ? { ...x, title: editingTitle.trim() } : x));
        setEditingSessionId(null);
    };

    const exportSession = (sessionId: string, format: "md" | "pdf") => {
        window.open(`/api/v1/sessions/${sessionId}/export?format=${format}`, "_blank");
    };

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
        <div className={`app-shell ${getThemeClass(visualTheme)}`}>
            {/* Header */}
            <header className="app-header">
                <div className="app-logo">
                    <div className="logo-mark">Æ</div>
                    <span className="logo-name">AetherForge v1.0</span>
                </div>

                <div className="header-pills">
                    {/* Web Search Toggle */}
                    <div
                        className={`header-pill-toggle ${webSearchEnabled ? "active" : ""}`}
                        onClick={() => setWebSearchEnabled(!webSearchEnabled)}
                        title="Allow the Agent to search the live web for answers"
                    >
                        <span className="header-pill-label">🌐 Web Grounding</span>
                        <div className="toggle-switch">
                            <div className="toggle-slider" />
                        </div>
                    </div>

                    {/* Analytics Toggle (Only shown when active module is ragforge) */}
                    {activeModule === "ragforge" && (
                        <div
                            className={`header-pill-toggle analytics ${analyticsEnabled ? "active" : ""}`}
                            onClick={() => setAnalyticsEnabled(!analyticsEnabled)}
                            title="Enable DataVault Analytics (charts/pandas) for selected docs"
                        >
                            <span className="header-pill-label">📊 Analytics</span>
                            <div className="toggle-switch">
                                <div className="toggle-slider" />
                            </div>
                        </div>
                    )}

                    {chatModels.length > 0 && (
                        <div className="header-pill-select">
                            <span className="header-pill-label brain">Chat Brain:</span>
                            <select
                                value={selectedChatModel}
                                onChange={handleChatModelChange}
                                disabled={isChangingChatModel}
                                className="chat-brain-select"
                            >
                                {isChangingChatModel && <option value={selectedChatModel}>Downloading... (this will take a minute)</option>}
                                {!isChangingChatModel && chatModels.map(m => (
                                    <option key={m.id} value={m.id}>{m.name}</option>
                                ))}
                            </select>
                        </div>
                    )}
                    {cpuPct !== null && (
                        <div className="status-pill highlight">CPU {cpuPct.toFixed(0)}%</div>
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
                            onClick={() => {
                                setActiveModule(m.id);
                                setActivePanel("chat");
                                const sid = activeSessionIds[m.id];
                                if (sid) {
                                    fetch(`/api/v1/sessions/${sid}/messages`)
                                        .then(r => r.json())
                                        .then(setLoadedSessionMessages)
                                        .catch(() => setLoadedSessionMessages(null));
                                } else {
                                    setLoadedSessionMessages(null);
                                }
                                refreshSessions();
                            }}
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
                    <button className={`nav-btn ${activePanel === "sync" ? "active" : ""}`}
                        onClick={() => setActivePanel("sync")}>
                        🔗 Sync Devices
                    </button>
                    <button className={`nav-btn ${activePanel === "settings" ? "active" : ""}`}
                        onClick={() => setActivePanel("settings")}>
                        ⚙️ Settings
                    </button>

                    <div className="sidebar-divider" />
                    <button
                        className="sidebar-section-label session-panel-toggle"
                        onClick={() => setSessionsPanelOpen(v => !v)}
                    >
                        <span>{sessionsPanelOpen ? "▾" : "▸"}</span>
                        <span>Session History</span>
                    </button>

                    {sessionsPanelOpen && (
                        <div className="session-list">
                            {sessions.length === 0 && (
                                <div className="session-empty">No sessions yet. Start chatting!</div>
                            )}
                            {sessions.map(s => (
                                <div
                                    key={s.id}
                                    className={`session-item ${activeSessionIds[activeModule] === s.id ? "active" : ""}`}
                                    onClick={() => loadSession(s.id)}
                                >
                                    <div className="session-item-body">
                                        {editingSessionId === s.id ? (
                                            <input
                                                className="session-rename-input"
                                                value={editingTitle}
                                                autoFocus
                                                onChange={e => setEditingTitle(e.target.value)}
                                                onBlur={() => commitRename(s.id)}
                                                onKeyDown={e => { if (e.key === "Enter") commitRename(s.id); if (e.key === "Escape") setEditingSessionId(null); }}
                                                onClick={e => e.stopPropagation()}
                                            />
                                        ) : (
                                            <div className="session-title" onDoubleClick={e => { e.stopPropagation(); startRename(s); }}>
                                                {s.title}
                                            </div>
                                        )}
                                        <div className="session-meta">
                                            {new Date(s.updated_at * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                                            {" · "}{s.message_count} msg
                                        </div>
                                    </div>
                                    <div className="session-actions" onClick={e => e.stopPropagation()}>
                                        <button title="Export MD" className="session-action-btn" onClick={() => exportSession(s.id, "md")}>⬇ MD</button>
                                        <button title="Export PDF" className="session-action-btn" onClick={() => exportSession(s.id, "pdf")}>📄 PDF</button>
                                        <button title="Delete" className="session-action-btn session-delete-btn" onClick={() => deleteSession(s.id)}>🗑</button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </aside>

                <main className="main-panel">
                    {activePanel === "chat" && activeModule !== "logger" && <ChatPanel
                        module={activeModule}
                        xray={xrayOpen}
                        onXrayData={() => { }}
                        sessionId={activeSessionIds[activeModule]}
                        preloadedMessages={loadedSessionMessages}
                        onSessionCreated={(id) => {
                            setActiveSessionIds(prev => ({ ...prev, [activeModule]: id }));
                            refreshSessions();
                        }}
                        webSearchEnabled={webSearchEnabled}
                        deepReasoningEnabled={deepReasoningEnabled}
                        onToggleDeepReasoning={() => setDeepReasoningEnabled(!deepReasoningEnabled)}
                        analyticsEnabled={analyticsEnabled}
                        grammarAssist={grammarAssistEnabled}
                        onToggleGrammarAssist={() => setGrammarAssistEnabled(!grammarAssistEnabled)}
                    />}
                    {activePanel === "chat" && activeModule === "logger" && <LoggerPanel />}
                    {activePanel === "insights" && <InsightsPanel />}
                    {activePanel === "policies" && <PoliciesPanel />}
                    {activePanel === "sync" && <SyncPanel />}
                    {activePanel === "settings" && <SettingsPanel />}
                </main>

                {xrayOpen && <XRayPanel />}
            </div>
        </div>
    );
}
