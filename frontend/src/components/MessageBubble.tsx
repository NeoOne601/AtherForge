import React, { useState } from "react";
import { Message } from "../types";
import { md } from "../lib/markdown";
import { ThinkingBlock } from "./ThinkingBlock";

/**
 * Parse <think>...</think> blocks from AI response text.
 * Handles both complete and in-progress reasoning blocks.
 */
function parseThinking(content: string): { thinking: string | null; answer: string } {
    const openTag = "<think>";
    const closeTag = "</think>";
    const start = content.indexOf(openTag);

    if (start === -1) {
        return { thinking: null, answer: content };
    }

    const before = content.slice(0, start).trim();
    const afterOpen = content.slice(start + openTag.length);
    const end = afterOpen.indexOf(closeTag);

    if (end === -1) {
        return {
            thinking: afterOpen.trim(),
            answer: before,
        };
    }

    const thinking = afterOpen.slice(0, end).trim();
    const answer = `${before}\n${afterOpen.slice(end + closeTag.length)}`.trim();
    return { thinking, answer };
}

/**
 * Parse [attachment:filename] tags from AI response.
 * Returns { cleanedAnswer: string, attachments: string[] }.
 */
function parseAttachments(content: string): { cleanedAnswer: string; attachments: string[] } {
    const attachRegex = /\[attachment:([^\]]+)\]/g;
    const attachments: string[] = [];
    let cleanedAnswer = content.replace(attachRegex, (match, filename) => {
        attachments.push(filename);
        return ""; // remove tag from text
    });
    return { cleanedAnswer: cleanedAnswer.trim(), attachments };
}

interface MessageBubbleProps {
    msg: Message;
    showThinking?: boolean;
    onSuggestionClick?: (suggestion: string) => void;
    onSuggestionSubmit?: (suggestion: string) => void;
    onFeedback?: (verdict: "accepted" | "corrected", correctionText?: string) => void;
    isLatestMessage?: boolean;
    isStreaming?: boolean;
}

export function MessageBubble({ msg, showThinking = true, onSuggestionClick, onSuggestionSubmit, onFeedback, isLatestMessage, isStreaming }: MessageBubbleProps) {
    const [suggestionsSubmitted, setSuggestionsSubmitted] = useState(false);
    const [feedbackState, setFeedbackState] = useState<"none" | "accepted" | "correcting" | "corrected">("none");
    const [correctionText, setCorrectionText] = useState("");
    const isUser = msg.role === "user";
    const fScore = msg.faithfulness_score;

    let thinking: string | null = null;
    let answer = msg.content;

    if (!isUser) {
        const parsed = parseThinking(msg.content);
        thinking = msg.thinking ?? msg.reasoning_trace ?? parsed.thinking;
        answer = msg.answer_text ?? parsed.answer;
    }

    const handleSuggestionClick = (suggestion: string) => {
        if (suggestionsSubmitted || isStreaming) return;
        if (onSuggestionSubmit) {
            setSuggestionsSubmitted(true);
            onSuggestionSubmit(suggestion);
        } else {
            onSuggestionClick?.(suggestion);
        }
    };

    // Extract attachments
    const { cleanedAnswer, attachments } = parseAttachments(answer);
    answer = cleanedAnswer;
    const mergedAttachments = Array.from(new Set([...(msg.attachments || []), ...attachments]));
    const citations = msg.citations || [];
    const suggestions = msg.suggestions || [];

    return (
        <div className={`message-row ${isUser ? "user" : ""}`}>
            <div className={`avatar ${isUser ? "user-av" : "ai"}`}>
                {isUser ? "You" : "Æ"}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px", maxWidth: "85%" }}>
                {/* Thinking section — ABOVE the bubble, as a separate collapsible block */}
                {!isUser && thinking && showThinking && (
                    <ThinkingBlock
                        content={thinking}
                        durationMs={msg.thinkingDurationMs}
                        isStreaming={msg.isThinkingStreaming}
                    />
                )}
            <div className={`bubble ${isUser ? "user-bubble" : "ai"}`}>
                {/* Main answer */}
                {isUser ? (
                    <div className="message-text">{msg.content}</div>
                ) : msg.streaming && !thinking && !answer ? (
                    <div className="typing-indicator" style={{ marginTop: "4px" }}>
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                    </div>
                ) : (
                    <div
                        className="message-text markdown-body"
                        dangerouslySetInnerHTML={{ __html: md.render(answer) }}
                    />
                )}

                {/* Attachments */}
                {mergedAttachments.length > 0 && (
                    <div className="attachments-container" style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {mergedAttachments.map((file, i) => {
                            const ext = file.split(".").pop()?.toLowerCase();
                            const url = `http://localhost:8765/api/v1/generated/${file}`;

                            if (ext === "png" || ext === "jpg" || ext === "jpeg") {
                                return (
                                    <div key={i} className="attachment-image-card" style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden', backgroundColor: 'var(--bg-panel)' }}>
                                        <a href={url} target="_blank" rel="noreferrer" style={{ display: 'block' }}>
                                            <img src={url} alt={file} style={{ width: '100%', display: 'block' }} />
                                        </a>
                                        <div style={{ padding: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'var(--bg-main)', borderTop: '1px solid var(--border-color)', fontSize: '13px' }}>
                                            <span style={{ color: 'var(--text-main)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file}</span>
                                            <a href={url} download={file} style={{ color: 'var(--accent)', textDecoration: 'none', fontWeight: 500 }}>Download</a>
                                        </div>
                                    </div>
                                );
                            } else {
                                return (
                                    <div key={i} className="attachment-file-card" style={{ border: '1px solid var(--border-color)', borderRadius: '6px', padding: '10px 14px', backgroundColor: 'var(--bg-main)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                                            <span style={{ fontSize: '16px' }}>📄</span>
                                            <span style={{ color: 'var(--text-main)', fontSize: '14px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{file}</span>
                                        </div>
                                        <a href={url} download={file} style={{ color: 'var(--accent)', textDecoration: 'none', fontSize: '13px', fontWeight: 500, flexShrink: 0 }}>Download</a>
                                    </div>
                                );
                            }
                        })}
                    </div>
                )}

                {!isUser && citations.length > 0 && (
                    <details className="citation-section" style={{ marginTop: "12px" }}>
                        <summary className="thinking-summary">Sources ({citations.length})</summary>
                        <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "10px" }}>
                            {citations.map((citation, index) => {
                                const locatorParts = [citation.source];
                                if (citation.page !== undefined && citation.page !== null && citation.page !== "") {
                                    locatorParts.push(`p.${citation.page}`);
                                }
                                if (citation.section) {
                                    locatorParts.push(citation.section);
                                }

                                return (
                                    <div
                                        key={`${citation.source}-${index}`}
                                        style={{
                                            padding: "10px 12px",
                                            borderRadius: "8px",
                                            border: "1px solid var(--border-color)",
                                            background: "rgba(255,255,255,0.03)",
                                        }}
                                    >
                                        <div style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-main)" }}>
                                            {citation.label || `[${index + 1}]`} {locatorParts.join(" | ")}
                                        </div>
                                        {citation.snippet && (
                                            <div style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "6px", lineHeight: 1.5 }}>
                                                {citation.snippet}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </details>
                )}

                {!isUser && suggestions.length > 0 && !suggestionsSubmitted && (
                    <div style={{ marginTop: "12px", display: "flex", flexWrap: "wrap", gap: "8px" }}>
                        {suggestions.map((suggestion) => (
                            <button
                                key={suggestion}
                                type="button"
                                className="chip"
                                style={{ display: "flex", alignItems: "center", gap: "6px" }}
                                onClick={() => handleSuggestionClick(suggestion)}
                                disabled={isStreaming}
                            >
                                <span>{suggestion}</span>
                                <span style={{ fontSize: "11px", opacity: 0.5, transition: "opacity 0.15s" }}>↗</span>
                            </button>
                        ))}
                    </div>
                )}

                {!isUser && (
                    <div className="bubble-meta">
                        <span className="badge module">{msg.module}</span>
                        {msg.streaming && (
                            <span className="badge latency">streaming</span>
                        )}
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
                        
                        {/* SONA Feedback mechanism */}
                        {!isStreaming && feedbackState !== "none" && feedbackState !== "correcting" ? (
                            <span className="badge" style={{ color: "var(--accent)", marginLeft: "auto" }}>✓ Feedback sent</span>
                        ) : !isStreaming && (
                            <div style={{ display: "flex", gap: "4px", marginLeft: "auto" }}>
                                <button className="icon-btn" style={{ fontSize: "14px", padding: "2px 6px", border: "none", background: "none", cursor: "pointer", opacity: 0.7 }} title="Good response" onClick={() => { setFeedbackState("accepted"); onFeedback?.("accepted"); }}>👍</button>
                                <button className="icon-btn" style={{ fontSize: "14px", padding: "2px 6px", border: "none", background: "none", cursor: "pointer", opacity: 0.7 }} title="Needs correction" onClick={() => setFeedbackState("correcting")}>👎</button>
                            </div>
                        )}
                    </div>
                )}
                
                {feedbackState === "correcting" && (
                    <div style={{ marginTop: "8px", display: "flex", gap: "8px", alignItems: "center", background: "rgba(0,0,0,0.2)", padding: "8px", borderRadius: "6px" }}>
                        <input 
                            type="text" 
                            style={{ flex: 1, padding: "6px 8px", fontSize: "13px", borderRadius: "4px", border: "1px solid var(--border-color)", background: "var(--bg-main)", color: "var(--text-main)" }}
                            placeholder="What was wrong? Provide a correction..." 
                            value={correctionText}
                            onChange={e => setCorrectionText(e.target.value)}
                            onKeyDown={e => {
                                if (e.key === "Enter") {
                                    setFeedbackState("corrected");
                                    onFeedback?.("corrected", correctionText);
                                }
                            }}
                            autoFocus
                        />
                        <button 
                            className="btn" 
                            style={{ padding: "6px 12px", fontSize: "12px", background: "var(--accent)", color: "#000", border: "none", borderRadius: "4px", cursor: "pointer", fontWeight: 600 }}
                            onClick={() => {
                                setFeedbackState("corrected");
                                onFeedback?.("corrected", correctionText);
                            }}
                        >
                            Submit
                        </button>
                    </div>
                )}
            </div>
            </div>
        </div>
    );
}
