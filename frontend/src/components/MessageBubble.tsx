import React from "react";
import { Message } from "../types";
import { md } from "../lib/markdown";

export function MessageBubble({ msg }: { msg: Message }) {
    const isUser = msg.role === "user";
    const fScore = msg.faithfulness_score;

    return (
        <div className={`message-row ${isUser ? "user" : ""}`}>
            <div className={`avatar ${isUser ? "user-av" : "ai"}`}>
                {isUser ? "You" : "Æ"}
            </div>
            <div className={`bubble ${isUser ? "user-bubble" : "ai"}`}>
                {isUser ? (
                    <div className="message-text">{msg.content}</div>
                ) : (
                    <div className="message-text markdown-body" dangerouslySetInnerHTML={{ __html: md.render(msg.content) }} />
                )}

                {!isUser && (
                    <div className="bubble-meta">
                        <span className="badge module">{msg.module}</span>
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
