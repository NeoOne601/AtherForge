// AetherForge v1.0 — frontend/src/components/PolicyEditor.tsx
// ─────────────────────────────────────────────────────────────────
// Live OPA Rego policy editor. Fetches current policy from backend,
// allows editing with Monaco (VS Code engine), validates Rego syntax
// on save, and hot-reloads via POST /api/v1/policies.
// ─────────────────────────────────────────────────────────────────
import React, { useCallback, useEffect, useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import { getPolicies, updatePolicies } from "../lib/tauri";

export default function PolicyEditor(): JSX.Element {
    const [policy, setPolicy] = useState<string>("");
    const [original, setOriginal] = useState<string>("");
    const [saving, setSaving] = useState(false);
    const [saveResult, setSaveResult] = useState<{ success: boolean; msg: string } | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        getPolicies()
            .then(({ policy: p }) => { setPolicy(p); setOriginal(p); })
            .catch(() => setPolicy("# Could not load policies — backend offline"))
            .finally(() => setLoading(false));
    }, []);

    const isDirty = policy !== original;

    const handleSave = useCallback(async () => {
        setSaving(true);
        setSaveResult(null);
        try {
            const res = await updatePolicies(policy);
            setSaveResult({
                success: res.success,
                msg: res.success ? "Policy saved and hot-reloaded ✓" : `Error: ${res.error}`,
            });
            if (res.success) setOriginal(policy);
        } catch (e) {
            setSaveResult({ success: false, msg: `Network error: ${e}` });
        }
        setSaving(false);
        setTimeout(() => setSaveResult(null), 5000);
    }, [policy]);

    const handleReset = () => { setPolicy(original); setSaveResult(null); };

    return (
        <div className="flex flex-col h-full overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b flex-shrink-0" style={{ borderColor: "var(--border-subtle)" }}>
                <div>
                    <h2 className="text-sm font-semibold gradient-text">Silicon Colosseum — Policy Editor</h2>
                    <p className="text-xs text-muted">OPA Rego policies evaluated before every tool call</p>
                </div>
                <div className="flex items-center gap-2">
                    {isDirty && (
                        <>
                            <button onClick={handleReset} className="btn-ghost text-xs px-3 py-1.5">Reset</button>
                            <button
                                id="save-policy-btn"
                                onClick={handleSave}
                                disabled={saving}
                                className="btn-primary text-xs px-3 py-1.5 disabled:opacity-60"
                            >
                                {saving ? "Saving..." : "Save & Reload"}
                            </button>
                        </>
                    )}
                    {!isDirty && <span className="text-xs badge-safe">Synced</span>}
                </div>
            </div>

            {/* Save result banner */}
            {saveResult && (
                <div
                    className="px-4 py-2 text-xs flex-shrink-0 transition-all"
                    style={{
                        background: saveResult.success ? "rgba(52,211,153,0.1)" : "rgba(248,113,113,0.1)",
                        borderBottom: `1px solid ${saveResult.success ? "rgba(52,211,153,0.3)" : "rgba(248,113,113,0.3)"}`,
                        color: saveResult.success ? "var(--accent-safe)" : "var(--accent-danger)",
                    }}
                >
                    {saveResult.msg}
                </div>
            )}

            {/* Policy guide */}
            <div className="px-4 py-2 flex-shrink-0 border-b" style={{ borderColor: "var(--border-subtle)", background: "rgba(5,12,20,0.4)" }}>
                <p className="text-xs text-muted">
                    Policies control: tool call budgets, faithfulness thresholds, prohibited patterns, module access.
                    {"  "}
                    <span style={{ color: "var(--accent-plasma)" }}>default allow := false</span> — deny-all by default, rules grant access.
                </p>
            </div>

            {/* Monaco Editor */}
            <div className="flex-1 min-h-0">
                {loading ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="spinner" />
                    </div>
                ) : (
                    <Editor
                        height="100%"
                        defaultLanguage="plaintext"
                        value={policy}
                        onChange={(v) => setPolicy(v ?? "")}
                        theme="vs-dark"
                        options={{
                            fontSize: 13,
                            fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                            minimap: { enabled: false },
                            lineNumbers: "on",
                            wordWrap: "on",
                            scrollBeyondLastLine: false,
                            tabSize: 2,
                            roundedSelection: true,
                            padding: { top: 16 },
                            smoothScrolling: true,
                        }}
                    />
                )}
            </div>
        </div>
    );
}
