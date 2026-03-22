// AetherForge v1.0 — frontend/src/components/ThinkingBlock.tsx
// ─────────────────────────────────────────────────────────────────
// Collapsible Chain-of-Thought reasoning block.
// Rendered above the final answer in assistant messages.
// Default: collapsed. Click to expand. Smooth CSS transition.
// ─────────────────────────────────────────────────────────────────
import React, { useState } from "react";

interface ThinkingBlockProps {
    content: string;
    durationMs?: number;
    isStreaming?: boolean;
}

export function ThinkingBlock({ content, durationMs, isStreaming }: ThinkingBlockProps) {
    const [isOpen, setIsOpen] = useState(false);

    const label = isStreaming
        ? "Thinking..."
        : durationMs
            ? `Thought for ${Math.round(durationMs / 1000)}s`
            : "Reasoning";

    return (
        <div
            className="thinking-block"
            style={{
                borderLeft: "2px solid rgba(255,255,255,0.1)",
                background: "rgba(255,255,255,0.03)",
                borderRadius: "0 6px 6px 0",
                paddingLeft: "12px",
                paddingRight: "8px",
                paddingTop: "8px",
                paddingBottom: "8px",
                marginBottom: "12px",
                cursor: "pointer",
                userSelect: "none",
            }}
            onClick={() => setIsOpen(o => !o)}
        >
            {/* Header */}
            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    fontSize: "12px",
                    color: "var(--text-muted, #888)",
                }}
            >
                {isStreaming && !content ? (
                    <span
                        style={{
                            width: "12px",
                            height: "12px",
                            border: "2px solid rgba(255,255,255,0.1)",
                            borderTop: "2px solid rgba(255,255,255,0.4)",
                            borderRadius: "50%",
                            display: "inline-block",
                            animation: "spin 0.8s linear infinite",
                        }}
                    />
                ) : (
                    <span
                        style={{
                            display: "inline-block",
                            transition: "transform 150ms ease",
                            transform: isOpen ? "rotate(90deg)" : "rotate(0deg)",
                            fontSize: "10px",
                        }}
                    >
                        ▶
                    </span>
                )}
                <span>{label}</span>
            </div>

            {/* Body (collapsible) */}
            <div
                style={{
                    maxHeight: isOpen ? "288px" : "0px",
                    overflow: "hidden",
                    transition: "max-height 200ms ease-in-out",
                }}
            >
                <div
                    style={{
                        marginTop: "8px",
                        paddingTop: "8px",
                        borderTop: "1px solid rgba(255,255,255,0.06)",
                    }}
                >
                    <p
                        style={{
                            fontSize: "12px",
                            color: "var(--text-muted, #888)",
                            lineHeight: 1.6,
                            whiteSpace: "pre-wrap",
                            maxHeight: "272px",
                            overflowY: "auto",
                            margin: 0,
                        }}
                    >
                        {content}
                    </p>
                </div>
            </div>
        </div>
    );
}
