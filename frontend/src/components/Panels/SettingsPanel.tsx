import { useState, useEffect } from "react";

interface FieldDef {
    label: string;
    description: string;
    type: "path" | "number" | "select";
    value: any;
    default: any;
    is_saved: boolean;
    options?: string[];
    min?: number;
    max?: number;
    step?: number;
}

interface SettingsGroup {
    label: string;
    description: string;
    fields: Record<string, FieldDef>;
}

interface DiskUsage {
    path: string;
    exists: boolean;
    dir_size_gb?: number;
    disk_total_gb?: number;
    disk_used_gb?: number;
    disk_free_gb?: number;
    disk_pct?: number;
    error?: string;
}

const TAB_KEYS = ["ai_models", "cache_downloads", "data_storage", "server"] as const;

export function SettingsPanel() {
    const [settings, setSettings] = useState<Record<string, SettingsGroup>>({});
    const [diskUsage, setDiskUsage] = useState<Record<string, DiskUsage>>({});
    const [activeTab, setActiveTab] = useState<string>(TAB_KEYS[0]);
    const [editValues, setEditValues] = useState<Record<string, any>>({});
    const [validations, setValidations] = useState<Record<string, boolean | null>>({});
    const [saving, setSaving] = useState(false);
    const [saveMessage, setSaveMessage] = useState<string | null>(null);
    const [hasUnsaved, setHasUnsaved] = useState(false);

    // Load settings on mount
    useEffect(() => {
        fetch("/api/v1/settings")
            .then(r => r.json())
            .then(data => {
                setSettings(data);
                // Initialize edit values from current values
                const vals: Record<string, any> = {};
                for (const group of Object.values(data) as SettingsGroup[]) {
                    for (const [key, field] of Object.entries(group.fields)) {
                        vals[key] = field.value;
                    }
                }
                setEditValues(vals);
            })
            .catch(console.error);

        fetch("/api/v1/settings/disk-usage")
            .then(r => r.json())
            .then(setDiskUsage)
            .catch(console.error);
    }, []);

    const updateField = (key: string, value: any) => {
        setEditValues(prev => ({ ...prev, [key]: value }));
        setHasUnsaved(true);
        setSaveMessage(null);
        setValidations(prev => ({ ...prev, [key]: null })); // Clear validation
    };

    const validatePath = async (key: string) => {
        try {
            const res = await fetch("/api/v1/settings/validate-path", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: editValues[key] }),
            });
            const data = await res.json();
            setValidations(prev => ({
                ...prev,
                [key]: data.exists && data.writable,
            }));
        } catch {
            setValidations(prev => ({ ...prev, [key]: false }));
        }
    };

    const saveSettings = async () => {
        setSaving(true);
        try {
            const res = await fetch("/api/v1/settings", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ settings: editValues }),
            });
            const data = await res.json();
            setSaveMessage(data.message || "Settings saved.");
            setHasUnsaved(false);
        } catch (err) {
            setSaveMessage("Failed to save settings.");
        } finally {
            setSaving(false);
        }
    };

    const resetField = (key: string, defaultVal: any) => {
        setEditValues(prev => ({ ...prev, [key]: defaultVal }));
        setHasUnsaved(true);
        setValidations(prev => ({ ...prev, [key]: null }));
    };

    if (!Object.keys(settings).length) {
        return (
            <div className="panel-wrapper">
                <div className="panel-title">Loading Settings...</div>
            </div>
        );
    }

    const currentGroup = settings[activeTab];

    return (
        <div className="settings-panel">
            {/* Header */}
            <div className="settings-header">
                <div className="settings-title-row">
                    <div>
                        <div className="panel-title">⚙️ Settings</div>
                        <div className="panel-sub">
                            Configure dependency paths, model locations, and server parameters
                        </div>
                    </div>
                    {hasUnsaved && (
                        <div className="settings-restart-badge">⚠️ Unsaved Changes</div>
                    )}
                    {saveMessage && !hasUnsaved && (
                        <div className="settings-restart-badge saved">🔄 Restart Required</div>
                    )}
                </div>

                {/* Sub-tabs */}
                <div className="settings-tabs">
                    {TAB_KEYS.map(tabKey => {
                        const group = settings[tabKey];
                        if (!group) return null;
                        return (
                            <button
                                key={tabKey}
                                className={`settings-tab ${activeTab === tabKey ? "active" : ""}`}
                                onClick={() => setActiveTab(tabKey)}
                            >
                                {group.label}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Content */}
            <div className="settings-content">
                {currentGroup && (
                    <div className="settings-group">
                        <div className="settings-group-desc">{currentGroup.description}</div>

                        {Object.entries(currentGroup.fields).map(([key, field]) => {
                            const du = diskUsage[key];
                            const validation = validations[key];

                            return (
                                <div key={key} className="settings-field">
                                    <div className="settings-field-header">
                                        <div className="settings-field-label">{field.label}</div>
                                        {field.is_saved && (
                                            <span className="settings-saved-badge">User Configured</span>
                                        )}
                                    </div>
                                    <div className="settings-field-desc">{field.description}</div>

                                    <div className="settings-field-input-row">
                                        {field.type === "path" && (
                                            <>
                                                <input
                                                    className="settings-input"
                                                    value={editValues[key] ?? ""}
                                                    onChange={e => updateField(key, e.target.value)}
                                                    spellCheck={false}
                                                />
                                                <button
                                                    className="settings-validate-btn"
                                                    onClick={() => validatePath(key)}
                                                    title="Validate this path"
                                                >
                                                    {validation === null ? "🔍 Validate" :
                                                        validation ? "✅ Valid" : "❌ Invalid"}
                                                </button>
                                                <button
                                                    className="settings-reset-btn"
                                                    onClick={() => resetField(key, field.default)}
                                                    title="Reset to default"
                                                >
                                                    ↻
                                                </button>
                                            </>
                                        )}

                                        {field.type === "number" && (
                                            <input
                                                className="settings-input number"
                                                type="number"
                                                value={editValues[key] ?? ""}
                                                onChange={e => updateField(key, parseFloat(e.target.value))}
                                                min={field.min}
                                                max={field.max}
                                                step={field.step || 1}
                                            />
                                        )}

                                        {field.type === "select" && (
                                            <select
                                                className="settings-select"
                                                value={editValues[key] ?? ""}
                                                onChange={e => updateField(key, e.target.value)}
                                            >
                                                {field.options?.map(opt => (
                                                    <option key={opt} value={opt}>{opt}</option>
                                                ))}
                                            </select>
                                        )}
                                    </div>

                                    {/* Disk usage bar for path fields */}
                                    {field.type === "path" && du && !du.error && du.disk_total_gb && (
                                        <div className="settings-disk-usage">
                                            <div className="settings-disk-bar">
                                                <div
                                                    className={`settings-disk-fill ${(du.disk_pct ?? 0) > 90 ? "critical" : (du.disk_pct ?? 0) > 70 ? "warning" : ""}`}
                                                    style={{ width: `${du.disk_pct}%` }}
                                                />
                                            </div>
                                            <div className="settings-disk-labels">
                                                <span>
                                                    {du.dir_size_gb !== undefined && du.dir_size_gb > 0
                                                        ? `📁 ${du.dir_size_gb} GB used by this folder`
                                                        : "📁 Empty"}
                                                </span>
                                                <span>
                                                    💽 {du.disk_free_gb} GB free / {du.disk_total_gb} GB total
                                                </span>
                                            </div>
                                        </div>
                                    )}

                                    <div className="settings-field-default">
                                        Default: <code>{String(field.default)}</code>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="settings-footer">
                {saveMessage && (
                    <div className="settings-save-message">{saveMessage}</div>
                )}
                <button
                    className={`settings-save-btn ${hasUnsaved ? "active" : ""}`}
                    onClick={saveSettings}
                    disabled={saving || !hasUnsaved}
                >
                    {saving ? "Saving..." : "💾 Save Settings"}
                </button>
            </div>
        </div>
    );
}
