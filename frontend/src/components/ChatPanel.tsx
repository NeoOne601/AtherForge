import React, { useState, useEffect, useRef, useCallback, Suspense, lazy } from "react";
import { Message, RAGDoc, CausalGraph, MODULES, MODULE_SUGGESTIONS, StoredMessage, PolicyDecision } from "../types";
import { createChatSocket } from "../lib/tauri";
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
    onRequestSession?: () => Promise<string | null>;
    webSearchEnabled?: boolean;
    deepResearchEnabled?: boolean;
    onToggleDeepResearch?: () => void;
    improveAnswerEnabled?: boolean;
    onToggleImproveAnswer?: () => void;
    analyticsEnabled?: boolean;
    grammarAssist?: boolean;
    onToggleGrammarAssist?: () => void;
}

export function ChatPanel({
    module, xray,
    onXrayData,
    sessionId,
    preloadedMessages,
    onSessionCreated,
    onRequestSession,
    webSearchEnabled,
    deepResearchEnabled,
    improveAnswerEnabled,
    analyticsEnabled,
    grammarAssist,
    onToggleGrammarAssist,
    onToggleDeepResearch,
    onToggleImproveAnswer,
}: ChatPanelProps) {
    const [messagesByModule, setMessagesByModule] = useState<Record<string, Message[]>>({
        localbuddy: [],
        ragforge: [],
        watchtower: [],
        streamsync: [],
        tunelab: []
    });

    useEffect(() => {
        if (!preloadedMessages) {
            setMessagesByModule(prev => ({ ...prev, [module]: [] }));
            return;
        }
        const loaded: Message[] = preloadedMessages
            .filter(m => m.role === "user" || m.role === "assistant")
            .map(m => ({
                id: m.id,
                role: m.role as "user" | "assistant",
                content: m.content,
                module,
                latency_ms: (m.metadata?.latency_ms as number) || undefined,
                faithfulness_score: (m.metadata?.faithfulness_score as number) || undefined,
                reasoning_trace: (m.metadata?.reasoning_trace as string) || undefined,
                answer_text: (m.metadata?.answer_text as string) || undefined,
                citations: (m.metadata?.citations as Message["citations"]) || undefined,
                attachments: (m.metadata?.attachments as string[]) || undefined,
                suggestions: (m.metadata?.suggestions as string[]) || undefined,
            }));
        setMessagesByModule(prev => ({ ...prev, [module]: loaded }));
    }, [preloadedMessages, module]);

    const [ragDocs, setRagDocs] = useState<RAGDoc[]>([]);

    useEffect(() => {
        const fetchRagDocs = () => {
            fetch("/api/v1/ragforge/documents")
                .then(r => r.json())
                .then(data => {
                    if (!data.documents) return;
                    setRagDocs((data.documents as any[]).map(d => ({
                        document_id: d.document_id,
                        name: d.name,
                        status: d.status,
                        tokens: d.tokens,
                        active: Boolean(d.selected),
                        file_type: d.file_type,
                        parser: d.parser,
                        chunk_count: d.chunk_count,
                        image_pages_pending: d.image_pages_pending,
                        last_error: d.last_error,
                    })));
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
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        return () => {
            wsRef.current?.close();
        };
    }, []);

    const autosize = useCallback(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 120) + "px";
    }, []);

    const patchMessage = useCallback((moduleId: string, messageId: string, patch: Partial<Message>) => {
        setMessagesByModule(prev => ({
            ...prev,
            [moduleId]: (prev[moduleId] || []).map(msg => msg.id === messageId ? { ...msg, ...patch } : msg),
        }));
    }, []);

    const appendMessageChunk = useCallback((moduleId: string, messageId: string, chunk: string) => {
        setMessagesByModule(prev => ({
            ...prev,
            [moduleId]: (prev[moduleId] || []).map(msg => (
                msg.id === messageId
                    ? { ...msg, content: msg.content + chunk, streaming: true }
                    : msg
            )),
        }));
    }, []);

    const appendReasoningChunk = useCallback((moduleId: string, messageId: string, chunk: string) => {
        setMessagesByModule(prev => ({
            ...prev,
            [moduleId]: (prev[moduleId] || []).map(msg => (
                msg.id === messageId
                    ? { ...msg, reasoning_trace: (msg.reasoning_trace || "") + chunk, streaming: true }
                    : msg
            )),
        }));
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
        wsRef.current?.close();
        wsRef.current = null;
        setLoading(false);
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

        const moduleId = module;
        const userMsg: Message = {
            id: Date.now().toString(),
            role: "user",
            content: text,
            module: moduleId,
        };
        const assistantId = `${Date.now()}-ai`;
        const assistantMsg: Message = {
            id: assistantId,
            role: "assistant",
            content: "",
            module: moduleId,
            streaming,
        };

        setMessagesByModule(prev => ({
            ...prev,
            [moduleId]: [...(prev[moduleId] || []), userMsg, assistantMsg]
        }));
        setInput("");
        setLoading(true);
        if (textareaRef.current) textareaRef.current.style.height = "auto";

        try {
            let resolvedSessionId = sessionId;
            if (!resolvedSessionId && onRequestSession) {
                resolvedSessionId = await onRequestSession();
                if (resolvedSessionId) {
                    onSessionCreated?.(resolvedSessionId);
                }
            }
            if (!resolvedSessionId) {
                throw new Error("A session could not be created.");
            }

            const activeDocs = moduleId === "ragforge" ? ragDocs.filter(d => d.active).map(d => d.name) : [];
            const contextPayload: Record<string, any> = activeDocs.length > 0 ? { active_docs: activeDocs } : {};
            if (webSearchEnabled) contextPayload.web_search_enabled = true;
            if (deepResearchEnabled) contextPayload.deep_research = true;
            if (improveAnswerEnabled) contextPayload.improve_answer = true;
            if (moduleId === "ragforge" && analyticsEnabled) contextPayload.analytics_enabled = true;

            if (streaming) {
                wsRef.current?.close();

                await new Promise<void>((resolve, reject) => {
                    const ws = createChatSocket(
                        resolvedSessionId,
                        (chunk) => {
                            if (chunk.type === "meta") {
                                onSessionCreated?.(chunk.session_id);
                                patchMessage(moduleId, assistantId, { module: chunk.module });
                                return;
                            }

                            if (chunk.type === "reasoning") {
                                appendReasoningChunk(moduleId, assistantId, chunk.content);
                                return;
                            }

                            if (chunk.type === "token") {
                                appendMessageChunk(moduleId, assistantId, chunk.content);
                                return;
                            }

                            if (chunk.type === "tool_start" || chunk.type === "tool_result") {
                                // Silent processing for now or could log to debug
                                return;
                            }

                            if (chunk.type === "done") {
                                const blocked = chunk.policy_decisions?.some((p: PolicyDecision) => !p.allowed);
                                patchMessage(moduleId, assistantId, {
                                    module: chunk.module,
                                    content: chunk.response,
                                    streaming: false,
                                    latency_ms: chunk.latency_ms,
                                    faithfulness_score: chunk.faithfulness_score ?? undefined,
                                    reasoning_trace: chunk.reasoning_summary ?? chunk.reasoning_trace ?? undefined,
                                    answer_text: chunk.answer_text ?? undefined,
                                    policy_decisions: chunk.policy_decisions,
                                    causal_graph: chunk.causal_graph ?? undefined,
                                    tool_calls: chunk.tool_calls,
                                    citations: chunk.citations ?? undefined,
                                    attachments: chunk.attachments ?? undefined,
                                    suggestions: chunk.suggestions ?? undefined,
                                    blocked,
                                });
                                if (chunk.causal_graph) {
                                    setXrayGraphByModule(prev => ({ ...prev, [moduleId]: chunk.causal_graph }));
                                    onXrayData(chunk.causal_graph);
                                }
                                setLoading(false);
                                ws.close();
                                resolve();
                                return;
                            }

                            patchMessage(moduleId, assistantId, {
                                content: `⚠️ ${chunk.content}`,
                                streaming: false,
                            });
                            setLoading(false);
                            ws.close();
                            reject(new Error(chunk.content));
                        },
                        (event) => {
                            reject(new Error(`WebSocket error: ${event.type}`));
                        }
                    );

                    wsRef.current = ws;
                    ws.onclose = () => {
                        if (wsRef.current === ws) {
                            wsRef.current = null;
                        }
                    };
                    ws.onopen = () => {
                        ws.send(JSON.stringify({
                            session_id: resolvedSessionId,
                            module: moduleId,
                            message: text,
                            xray_mode: xray,
                            context: contextPayload,
                        }));
                    };
                });
                return;
            }

            const res = await fetch("/api/v1/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: resolvedSessionId,
                    module: moduleId,
                    message: text,
                    xray_mode: xray,
                    context: contextPayload,
                }),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            if (onSessionCreated) onSessionCreated(data.session_id || resolvedSessionId);

            const blocked = data.policy_decisions?.some((p: PolicyDecision) => !p.allowed);
            patchMessage(moduleId, assistantId, {
                content: data.response,
                module: data.module,
                streaming: false,
                latency_ms: data.latency_ms,
                faithfulness_score: data.faithfulness_score,
                reasoning_trace: data.reasoning_summary ?? data.reasoning_trace ?? undefined,
                answer_text: data.answer_text ?? undefined,
                policy_decisions: data.policy_decisions,
                causal_graph: data.causal_graph,
                tool_calls: data.tool_calls,
                citations: data.citations ?? undefined,
                attachments: data.attachments ?? undefined,
                suggestions: data.suggestions ?? undefined,
                blocked,
            });

            if (data.causal_graph) {
                setXrayGraphByModule(prev => ({ ...prev, [moduleId]: data.causal_graph }));
                onXrayData(data.causal_graph);
            }

        } catch (err) {
            patchMessage(moduleId, assistantId, {
                content: `⚠️ Could not reach backend: ${err}. Make sure the FastAPI server is running on port 8765.`,
                streaming: false,
            });
        } finally {
            setLoading(false);
        }
    }, [
        input,
        loading,
        module,
        xray,
        ragDocs,
        webSearchEnabled,
        deepResearchEnabled,
        improveAnswerEnabled,
        analyticsEnabled,
        sessionId,
        onSessionCreated,
        onRequestSession,
        onXrayData,
        streaming,
        appendMessageChunk,
        patchMessage,
    ]);

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
                            {loading && !messages.some(m => m.streaming) && (
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
                                style={{ display: "flex", alignItems: "center", gap: "4px", cursor: "pointer", opacity: deepResearchEnabled ? 1 : 0.6 }}
                                onClick={onToggleDeepResearch}
                                title="Deep research with planner and VFS"
                            >
                                <span style={{ fontSize: "10px", fontWeight: 700, color: deepResearchEnabled ? "var(--brand-glow)" : "var(--text-muted)" }}>Deep Research</span>
                                <div style={{ width: "20px", height: "12px", background: deepResearchEnabled ? "rgba(0,255,100,0.2)" : "rgba(255,255,255,0.1)", borderRadius: "10px", position: "relative" }}>
                                    <div style={{ width: "8px", height: "8px", background: deepResearchEnabled ? "var(--brand-glow)" : "var(--text-muted)", borderRadius: "50%", position: "absolute", top: "2px", left: deepResearchEnabled ? "10px" : "2px", transition: "all 0.2s" }} />
                                </div>
                            </div>

                            <div
                                style={{ display: "flex", alignItems: "center", gap: "4px", cursor: "pointer", opacity: improveAnswerEnabled ? 1 : 0.6 }}
                                onClick={onToggleImproveAnswer}
                                title="Run a second-pass answer improvement step"
                            >
                                <span style={{ fontSize: "10px", fontWeight: 700, color: improveAnswerEnabled ? "var(--aether-light)" : "var(--text-muted)" }}>Improve Answer</span>
                                <div style={{ width: "20px", height: "12px", background: improveAnswerEnabled ? "rgba(124,58,237,0.2)" : "rgba(255,255,255,0.1)", borderRadius: "10px", position: "relative" }}>
                                    <div style={{ width: "8px", height: "8px", background: improveAnswerEnabled ? "var(--aether-light)" : "var(--text-muted)", borderRadius: "50%", position: "absolute", top: "2px", left: improveAnswerEnabled ? "10px" : "2px", transition: "all 0.2s" }} />
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
                            Show Reasoning
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
