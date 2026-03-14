import React from "react";
import { Message } from "../types";
import { md } from "../lib/markdown";

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
}

export function MessageBubble({ msg, showThinking = true }: MessageBubbleProps) {
    const isUser = msg.role === "user";
    const fScore = msg.faithfulness_score;

    let thinking: string | null = null;
    let answer = msg.content;

    if (!isUser) {
        const parsed = parseThinking(msg.content);
        thinking = msg.reasoning_trace ?? parsed.thinking;
        answer = msg.answer_text ?? parsed.answer;
    }

    // Extract attachments
    const { cleanedAnswer, attachments } = parseAttachments(answer);
    answer = cleanedAnswer;
    const mergedAttachments = Array.from(new Set([...(msg.attachments || []), ...attachments]));
    const citations = msg.citations || [];

    return (
        <div className={`message-row ${isUser ? "user" : ""}`}>
            <div className={`avatar ${isUser ? "user-av" : "ai"}`}>
                {isUser ? "You" : "Æ"}
            </div>
            <div className={`bubble ${isUser ? "user-bubble" : "ai"}`}>
                {/* Thinking section (collapsible) */}
                {!isUser && thinking && showThinking && (
                    <details className="thinking-section" open={msg.streaming ? true : undefined}>
                        <summary className="thinking-summary">
                            {msg.streaming ? "🧠 Reasoning..." : "🧠 Reasoning trace"}
                        </summary>
                        <div
                            className="thinking-content markdown-body"
                            dangerouslySetInnerHTML={{ __html: md.render(thinking) }}
                        />
                    </details>
                )}

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
                    </div>
                )}
            </div>
        </div>
    );
}
