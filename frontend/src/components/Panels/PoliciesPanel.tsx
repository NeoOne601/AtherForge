import { useState, useEffect } from "react";

export function PoliciesPanel() {
    const [policy, setPolicy] = useState("");
    const [dirty, setDirty] = useState(false);
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        fetch("/api/v1/policies")
            .then(r => r.json())
            .then(d => setPolicy(d.policy ?? ""))
            .catch(() => { });
    }, []);

    const save = async () => {
        await fetch("/api/v1/policies", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ policy }),
        });
        setDirty(false);
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
    };

    return (
        <div className="panel-wrapper">
            <div className="panel-title">🛡️ Silicon Colosseum</div>
            <div className="panel-sub">Live OPA Rego policy — changes take effect immediately, no restart needed</div>

            <div className="policy-card">
                <div className="policy-toolbar">
                    <span className="policy-lang">OPA Rego</span>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        {dirty && <div className="dirty-dot" title="Unsaved changes" />}
                        {saved && <span style={{ fontSize: 11, color: "var(--plasma)" }}>✓ Saved</span>}
                        <button className="btn btn-ghost" onClick={() => { setPolicy(policy); setDirty(false); }}>Reset</button>
                        <button className="btn btn-primary" onClick={save} disabled={!dirty}>Save & Reload</button>
                    </div>
                </div>
                <textarea
                    className="policy-textarea"
                    value={policy}
                    onChange={e => { setPolicy(e.target.value); setDirty(true); setSaved(false); }}
                    spellCheck={false}
                />
            </div>

            <div style={{ padding: "16px", background: "var(--glass)", border: "1px solid var(--glass-border)", borderRadius: "var(--radius)", fontSize: 12 }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: "var(--text-secondary)" }}>📚 Quick Reference</div>
                {[
                    ["deny_reasons contains r if { input.tool_call_count > N }", "Limit tool calls per turn"],
                    ["deny_reasons contains r if { input.faithfulness_score < 0.92 }", "Block low-confidence outputs"],
                    ["contains(lower(input.message), \"pattern\")", "Detect prohibited content"],
                ].map(([code, desc], i) => (
                    <div key={i} style={{ marginBottom: 10 }}>
                        <code style={{ display: "block", background: "rgba(0,0,0,0.3)", padding: "6px 10px", borderRadius: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: "#a5b4fc", marginBottom: 3 }}>
                            {code}
                        </code>
                        <div style={{ color: "var(--text-muted)" }}>{desc}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
