// AetherForge v1.0 — frontend/src/components/ChatInterface.tsx
// ─────────────────────────────────────────────────────────────────
// Primary chat panel. Supports streaming WebSocket responses with
// token-by-token rendering, faithfulness badges, policy decision
// display, and latency indicator.
// ─────────────────────────────────────────────────────────────────
import React, { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    createChatSocket,
    newSessionId,
    sendChat,
    type CausalGraph,
    type ChatResponse,
    type PolicyDecision,
} from "../lib/tauri";

interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    streaming?: boolean;
    latency_ms?: number;
    faithfulness?: number | null;
    policies?: PolicyDecision[];
    module?: string;
}

interface Props {
    sessionId: string;
    activeModule: string;
    xrayMode: boolean;
    onNewGraph: (graph: CausalGraph) => void;
    enabled: boolean;
}

export default function ChatInterface({ sessionId, activeModule, xrayMode, onNewGraph, enabled }: Props): JSX.Element {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isStreaming, setIsStreaming] = useState(false);
    const [useStreaming, setUseStreaming] = useState(true);
    const bottomRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    // Auto-scroll on new messages
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Welcome message on mount
    useEffect(() => {
        setMessages([{
            id: "welcome",
            role: "assistant",
            content: `**Welcome to AetherForge v1.0**\n\nYou are connected to module: **${activeModule}**. All processing is 100% local — your data never leaves this machine.\n\nType a message to begin.`,
            module: activeModule,
        }]);
    }, []);

    const appendChunk = useCallback((msgId: string, chunk: string) => {
        setMessages(prev => prev.map(m =>
            m.id === msgId ? { ...m, content: m.content + chunk, streaming: true } : m
        ));
    }, []);

    const finalizeMessage = useCallback((msgId: string, latency_ms: number) => {
        setMessages(prev => prev.map(m =>
            m.id === msgId ? { ...m, streaming: false, latency_ms } : m
        ));
        setIsStreaming(false);
    }, []);

    const sendMessage = useCallback(async () => {
        const text = input.trim();
        if (!text || isStreaming || !enabled) return;
        setInput("");

        const userMsg: Message = { id: `u-${Date.now()}`, role: "user", content: text };
        const assistantId = `a-${Date.now()}`;
        const assistantMsg: Message = {
            id: assistantId,
            role: "assistant",
            content: "",
            streaming: true,
            module: activeModule,
        };
        setMessages(prev => [...prev, userMsg, assistantMsg]);
        setIsStreaming(true);

        if (useStreaming) {
            // ── WebSocket streaming path ──────────────────────────────
            if (wsRef.current?.readyState !== WebSocket.OPEN) {
                wsRef.current = createChatSocket(
                    sessionId,
                    (chunk) => {
                        if (chunk.type === "token") {
                            appendChunk(assistantId, chunk.content);
                        } else if (chunk.type === "done") {
                            finalizeMessage(assistantId, chunk.latency_ms);
                        } else if (chunk.type === "error") {
                            appendChunk(assistantId, `\n⚠️ Error: ${chunk.content}`);
                            setIsStreaming(false);
                        }
                    }
                );
                // Wait for open
                await new Promise<void>((resolve) => {
                    const ws = wsRef.current!;
                    if (ws.readyState === WebSocket.OPEN) return resolve();
                    ws.addEventListener("open", () => resolve(), { once: true });
                });
            }
                wsRef.current?.send(JSON.stringify({
                    message: text,
                    module: activeModule,
                    xray_mode: xrayMode,
                    protocol: "forensic",
                }));
            } else {
                // ── REST fallback path ────────────────────────────────────
                try {
                    const resp = await sendChat({
                        session_id: sessionId,
                        module: activeModule,
                        message: text,
                        xray_mode: xrayMode,
                        protocol: "forensic",
                    });
                setMessages(prev => prev.map(m =>
                    m.id === assistantId
                        ? {
                            ...m,
                            content: resp.response,
                            streaming: false,
                            latency_ms: resp.latency_ms,
                            faithfulness: resp.faithfulness_score,
                            policies: resp.policy_decisions,
                            module: resp.module,
                        }
                        : m
                ));
                if (resp.causal_graph) onNewGraph(resp.causal_graph);
            } catch (err) {
                setMessages(prev => prev.map(m =>
                    m.id === assistantId
                        ? { ...m, content: `⚠️ Failed to reach backend: ${err}`, streaming: false }
                        : m
                ));
            }
            setIsStreaming(false);
        }
    }, [input, isStreaming, enabled, sessionId, activeModule, xrayMode, useStreaming, appendChunk, finalizeMessage, onNewGraph]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const renderContent = (text: string) => {
        // Simple markdown-lite: bold, code, newlines
        return text
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/`(.+?)`/g, `<code style="background:rgba(0,0,0,0.4);padding:1px 4px;border-radius:3px;font-size:0.85em;font-family:JetBrains Mono,monospace">$1</code>`)
            .replace(/\n/g, "<br/>");
    };

    return (
        <div className="flex flex-col h-full">
            {/* Messages area */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
                <AnimatePresence initial={false}>
                    {messages.map(msg => (
                        <motion.div
                            key={msg.id}
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.2 }}
                            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                            <div className="max-w-[85%]">
                                {msg.role === "user" ? (
                                    <div className="msg-user">{msg.content}</div>
                                ) : (
                                    <div className="msg-assistant">
                                        <div
                                            className="text-sm leading-relaxed"
                                            style={{ color: "var(--text-primary)" }}
                                            dangerouslySetInnerHTML={{ __html: renderContent(msg.content) }}
                                        />
                                        {/* Typing indicator */}
                                        {msg.streaming && msg.content === "" && (
                                            <div className="flex items-center gap-1 mt-2 h-4">
                                                <div className="typing-dot" />
                                                <div className="typing-dot" />
                                                <div className="typing-dot" />
                                            </div>
                                        )}
                                        {/* Metadata row */}
                                        {!msg.streaming && (msg.latency_ms || msg.faithfulness !== undefined) && (
                                            <div className="flex items-center gap-2 mt-2 pt-2 border-t" style={{ borderColor: "var(--border-subtle)" }}>
                                                {msg.module && <span className="badge-volt text-xs">{msg.module}</span>}
                                                {msg.latency_ms && (
                                                    <span className="text-xs text-muted">{msg.latency_ms.toFixed(0)}ms</span>
                                                )}
                                                {msg.faithfulness !== null && msg.faithfulness !== undefined && (
                                                    <span className={`badge text-xs ${msg.faithfulness >= 0.92 ? "badge-safe" : "badge-danger"}`}>
                                                        fidelity {(msg.faithfulness * 100).toFixed(0)}%
                                                    </span>
                                                )}
                                                {msg.policies && msg.policies.some(p => !p.allowed) && (
                                                    <span className="badge-danger text-xs">🛡️ policy applied</span>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>
                <div ref={bottomRef} />
            </div>

            {/* Input area */}
            <div className="px-4 pb-4">
                <div className="glass rounded-xl p-3 flex gap-3 items-end">
                    <textarea
                        ref={inputRef}
                        id="chat-input"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={enabled ? `Message ${activeModule}...` : "Backend offline — run ./run_dev.sh"}
                        disabled={!enabled || isStreaming}
                        rows={1}
                        className="flex-1 bg-transparent text-sm resize-none outline-none"
                        style={{
                            color: "var(--text-primary)",
                            maxHeight: "120px",
                            minHeight: "20px",
                            fontFamily: "Inter, sans-serif",
                        }}
                        onInput={e => {
                            const t = e.target as HTMLTextAreaElement;
                            t.style.height = "auto";
                            t.style.height = `${Math.min(t.scrollHeight, 120)}px`;
                        }}
                    />
                    <button
                        id="send-btn"
                        onClick={sendMessage}
                        disabled={!input.trim() || isStreaming || !enabled}
                        className="btn-primary flex items-center gap-2 py-2 flex-shrink-0 disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none"
                    >
                        {isStreaming ? (
                            <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                        ) : (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
                            </svg>
                        )}
                        Send
                    </button>
                </div>
                <div className="flex items-center justify-between px-1 mt-1">
                    <span className="text-xs text-muted">Enter to send · Shift+Enter for newline</span>
                    <label className="flex items-center gap-1.5 text-xs text-secondary cursor-pointer">
                        <input
                            type="checkbox"
                            checked={useStreaming}
                            onChange={e => setUseStreaming(e.target.checked)}
                            className="w-3 h-3"
                        />
                        Stream tokens
                    </label>
                </div>
            </div>
        </div>
    );
}
