import React from "react";

export function XRayPanel() {
    return (
        <div className="xray-panel">
            <div className="xray-header">
                <div className="xray-title">🔬 X-Ray Mode</div>
            </div>
            <div className="xray-body">
                <div className="xray-empty">
                    <div style={{ fontSize: 28 }}>🔬</div>
                    <div style={{ fontWeight: 600, color: "var(--text-secondary)" }}>Send a message</div>
                    <div>The causal reasoning graph will appear here after each response, showing every decision step the AI made.</div>
                </div>
            </div>
        </div>
    );
}
