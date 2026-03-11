import React, { useState, useEffect, useRef, useCallback, Suspense, lazy } from "react";
import { Message, RAGDoc, CausalGraph, MODULES, MODULE_SUGGESTIONS, StoredMessage, PolicyDecision } from "../types";
import { MessageBubble } from "./MessageBubble";
import { InlineXRay } from "./InlineXRay";

const RAGForgeHUD = lazy(() => import("./HUDs/RAGForgeHUD").then(m => ({ default: m.RAGForgeHUD })));
const WatchTowerHUD = lazy(() => import("./HUDs/WatchTowerHUD").then(m => ({ default: m.WatchTowerHUD })));
const StreamSyncHUD = lazy(() => import("./HUDs/StreamSyncHUD").then(m => ({ default: m.StreamSyncHUD })));
const TuneLabHUD = lazy(() => import("./HUDs/TuneLabHUD").then(m => ({ default: m.TuneLabHUD })));

interface ChatPanelProps {
    module: string;
    xray: boolean;
    onXrayData: (g: CausalGraph | null) => void;
    sessionId?: string | null;
    preloadedMessages?: StoredMessage[] | null;
    onSessionCreated?: (id: string) => void;
    webSearchEnabled?: boolean;
    deepReasoningEnabled?: boolean;
    onToggleDeepReasoning?: () => void;
    analyticsEnabled?: boolean;
    grammarAssist?: boolean;
    onToggleGrammarAssist?: () => void;
}

export function ChatPanel({
    module, xray,
    sessionId,
    preloadedMessages,
    onSessionCreated,
    webSearchEnabled,
    deepReasoningEnabled,
    analyticsEnabled,
    grammarAssist,
    onToggleGrammarAssist,
    onToggleDeepReasoning,
}: ChatPanelProps) {
    const [messagesByModule, setMessagesByModule] = useState<Record<string, Message[]>>({
        localbuddy: [],
        ragforge: [],
        watchtower: [],
        streamsync: [],
        tunelab: []
    });

    useEffect(() => {
        if (!preloadedMessages || preloadedMessages.length === 0) return;
        const loaded: Message[] = preloadedMessages
            .filter(m => m.role === "user" || m.role === "assistant")
            .map(m => ({
                id: m.id,
                role: m.role as "user" | "assistant",
                content: m.content,
                module,
                latency_ms: (m.metadata?.latency_ms as number) || undefined,
                faithfulness_score: (m.metadata?.faithfulness_score as number) || undefined,
            }));
        setMessagesByModule(prev => ({ ...prev, [module]: loaded }));
    }, [preloadedMessages, module]);

    const currentSessionId = sessionId || `ui-session-${module}`;
    const [ragDocs, setRagDocs] = useState<RAGDoc[]>([]);

    useEffect(() => {
        const fetchRagDocs = () => {
            fetch("/api/v1/ragforge/documents")
                .then(r => r.json())
                .then(data => {
                    if (!data.documents) return;
                    setRagDocs(prev => {
                        const byName = new Map<string, RAGDoc>();
                        prev.forEach(d => byName.set(d.name, d));
                        (data.documents as RAGDoc[]).forEach(d => {
                            const existing = byName.get(d.name);
                            byName.set(d.name, existing ? { ...existing, ...d } : d);
                        });
                        return Array.from(byName.values());
                    });
                })
                .catch(err => console.error("Failed to fetch rag docs", err));
        };

        fetchRagDocs();
        const interval = setInterval(fetchRagDocs, 10000);
        return () => clearInterval(interval);
    }, []);

    const messages = messagesByModule[module] || [];
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [refining, setRefining] = useState(false);
    const [streaming, setStreaming] = useState(true);
    const [showThinking, setShowThinking] = useState(true);
    const [xrayGraphByModule, setXrayGraphByModule] = useState<Record<string, CausalGraph | null>>({});

    const bottomRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const autosize = useCallback(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 120) + "px";
    }, []);

    const handleRefineClick = useCallback(async () => {
        const text = input.trim();
        if (!grammarAssist || !text || text.length < 10 || loading || refining) return;
        setRefining(true);
        try {
            const res = await fetch("/api/v1/learning/refine-text", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });
            if (res.ok) {
                const data = await res.json();
                if (data.refined && data.refined !== input) {
                    setInput(data.refined);
                    autosize();
                }
            }
        } catch (err) {
            console.error("Grammar refinement failed:", err);
        } finally {
            setRefining(false);
        }
    }, [grammarAssist, input, loading, refining, autosize]);

    useEffect(() => {
        if (!grammarAssist || !input.trim() || loading || refining || input.length < 15) return;
        const timer = setTimeout(() => {
            handleRefineClick();
        }, 2500);
        return () => clearTimeout(timer);
    }, [input, grammarAssist, loading, refining, handleRefineClick]);

    const xrayGraph = xrayGraphByModule[module] || null;
    const mod = MODULES.find(m => m.id === module)!;
    const isDashboardOnlyModule = module === "streamsync" || module === "tunelab";

    useEffect(() => {
        setInput("");
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
        }
    }, [module]);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

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
            const contextPayload: Record<string, any> = activeDocs.length > 0 ? { active_docs: activeDocs } : {};
            if (webSearchEnabled) contextPayload.web_search_enabled = true;
            if (deepReasoningEnabled) contextPayload.deep_reasoning = true;

            const targetModule = (module === "ragforge" && analyticsEnabled) ? "analytics" : module;

            const res = await fetch("/api/v1/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    module: targetModule,
                    message: text,
                    xray_mode: xray,
                    context: contextPayload,
                }),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            if (onSessionCreated) onSessionCreated(data.session_id || currentSessionId);

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
    }, [input, loading, module, xray, ragDocs, webSearchEnabled, deepReasoningEnabled, analyticsEnabled, currentSessionId, onSessionCreated]);

    const onKey = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    };

    return (
        <div className="chat-container">
            <div className="chat-header">
                <div className="chat-module-badge">{mod.icon} {mod.name}</div>
                <div className="chat-title">AI Assistant</div>
                <div className="chat-subtitle">All processing is 100% local — your data never leaves this machine</div>
            </div>

            <Suspense fallback={<div className="hud-placeholder pulse">Loading {mod.name} Engine...</div>}>
                {module === "ragforge" && <RAGForgeHUD docs={ragDocs} setDocs={setRagDocs} />}
                {module === "watchtower" && <WatchTowerHUD />}
                {module === "streamsync" && <StreamSyncHUD />}
                {module === "tunelab" && <TuneLabHUD />}
            </Suspense>

            {!isDashboardOnlyModule && (
                <div className="messages-area">
                    {messages.length === 0 && !loading ? (
                        <div className="empty-state">
                            <div className="empty-orb">{mod.icon}</div>
                            <h3 className="empty-title">{mod.name} Agent</h3>
                            <p className="empty-sub">{mod.desc}</p>
                            <div className="suggestion-chips">
                                {(MODULE_SUGGESTIONS[module] || []).map(s => (
                                    <button key={s} className="chip" onClick={() => { setInput(s); textareaRef.current?.focus(); }}>
                                        {s}
                                    </button>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <>
                            {messages.map(m => (
                                <MessageBubble key={m.id} msg={m} showThinking={showThinking} />
                            ))}
                            {loading && (
                                <div className="message-row">
                                    <div className="avatar ai pulse">Æ</div>
                                    <div className="bubble ai" style={{ padding: "12px 16px" }}>
                                        <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "6px" }}>
                                            🧠 Thinking...
                                        </div>
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
            )}

            {isDashboardOnlyModule && (
                <div className="dashboard-placeholder" style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", opacity: 0.5 }}>
                    <div style={{ fontSize: 48, marginBottom: 16 }}>{mod.icon}</div>
                    <div style={{ fontSize: 18, fontWeight: 600 }}>{mod.name} Dashboard</div>
                    <div style={{ fontSize: 13 }}>Interactive AI Chat is disabled for this module.</div>
                    <div ref={bottomRef} />
                </div>
            )}

            {!isDashboardOnlyModule && (
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
                        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                            <span className="hint-text">Enter to send · Shift+Enter for new line</span>

                            <div
                                style={{ display: "flex", alignItems: "center", gap: "4px", cursor: "pointer", opacity: deepReasoningEnabled ? 1 : 0.6 }}
                                onClick={onToggleDeepReasoning}
                                title="Deep Reasoning pass"
                            >
                                <span style={{ fontSize: "10px", fontWeight: 700, color: deepReasoningEnabled ? "var(--brand-glow)" : "var(--text-muted)" }}>🧠 Reasoning</span>
                                <div style={{ width: "20px", height: "12px", background: deepReasoningEnabled ? "rgba(0,255,100,0.2)" : "rgba(255,255,255,0.1)", borderRadius: "10px", position: "relative" }}>
                                    <div style={{ width: "8px", height: "8px", background: deepReasoningEnabled ? "var(--brand-glow)" : "var(--text-muted)", borderRadius: "50%", position: "absolute", top: "2px", left: deepReasoningEnabled ? "10px" : "2px", transition: "all 0.2s" }} />
                                </div>
                            </div>

                            <div
                                style={{ display: "flex", alignItems: "center", gap: "4px", cursor: "pointer", opacity: grammarAssist ? 1 : 0.6 }}
                                onClick={onToggleGrammarAssist}
                                title="Auto-correction"
                            >
                                <span style={{ fontSize: "10px", fontWeight: 700, color: grammarAssist ? "var(--aether-light)" : "var(--text-muted)" }}>✨ Grammar</span>
                                <div style={{ width: "20px", height: "12px", background: grammarAssist ? "rgba(124,58,237,0.2)" : "rgba(255,255,255,0.1)", borderRadius: "10px", position: "relative" }}>
                                    <div style={{ width: "8px", height: "8px", background: grammarAssist ? "var(--aether-light)" : "var(--text-muted)", borderRadius: "50%", position: "absolute", top: "2px", left: grammarAssist ? "10px" : "2px", transition: "all 0.2s" }} />
                                </div>
                            </div>

                            {grammarAssist && (
                                <button
                                    className="btn btn-ghost"
                                    style={{ fontSize: "10px", padding: "0px 6px", height: "18px", background: "rgba(255,255,255,0.05)", border: "1px solid var(--border)", borderRadius: "4px", color: "var(--aether-light)" }}
                                    onClick={handleRefineClick}
                                    disabled={refining || !input.trim() || loading}
                                >
                                    {refining ? "Refining..." : "Refine Now"}
                                </button>
                            )}
                        </div>

                        <label className="stream-toggle" style={{ marginLeft: "12px" }}>
                            <div className={`toggle-track ${showThinking ? "on" : ""}`} onClick={() => setShowThinking(v => !v)}>
                                <div className="toggle-thumb" />
                            </div>
                            Show Thinking
                        </label>
                        <label className="stream-toggle">
                            <div className={`toggle-track ${streaming ? "on" : ""}`} onClick={() => setStreaming(v => !v)}>
                                <div className="toggle-thumb" />
                            </div>
                            Stream
                        </label>
                    </div>
                </div>
            )}

            {xray && xrayGraph && (
                <InlineXRay graph={xrayGraph} />
            )}
        </div>
    );
}
