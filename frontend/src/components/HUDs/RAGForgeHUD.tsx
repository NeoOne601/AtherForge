import React, { useState, useEffect, useRef } from "react";
import { RAGDoc } from "../../types";

interface RAGForgeHUDProps {
    docs: RAGDoc[];
    setDocs: React.Dispatch<React.SetStateAction<RAGDoc[]>>;
}

export function RAGForgeHUD({ docs, setDocs }: RAGForgeHUDProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [vlmOptions, setVlmOptions] = useState<any[]>([]);
    const [selectedVlm, setSelectedVlm] = useState<string>("smolvlm-256m");
    const [isSelectingVlm, setIsSelectingVlm] = useState(false);

    useEffect(() => {
        fetch("/api/v1/ragforge/vlm-options")
            .then(res => res.json())
            .then(data => {
                setVlmOptions(data.options || []);
                if (data.selected) {
                    setSelectedVlm(data.selected);
                }
            })
            .catch(err => console.error("Failed to fetch VLM options", err));
    }, []);

    const handleVlmChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
        const vlm_id = e.target.value;
        setSelectedVlm(vlm_id);
        setIsSelectingVlm(true);
        try {
            await fetch("/api/v1/ragforge/vlm-select", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ vlm_id })
            });
        } catch (err) {
            console.error("VLM select failed", err);
        } finally {
            setIsSelectingVlm(false);
        }
    };

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const uploadFile = async (file: File) => {
        setDocs(prev => [...prev.filter(d => d.name !== file.name), { name: file.name, status: "Embedding", tokens: "—", active: true }]);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("/api/v1/ragforge/upload", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();

            if (res.ok) {
                setDocs(prev => prev.map(d => d.name === file.name ? { ...d, status: "Ready", tokens: `~${data.result.chunks_added} chunks` } : d));
            } else {
                setDocs(prev => prev.map(d => d.name === file.name ? { ...d, status: "Failed", tokens: "error" } : d));
            }
        } catch (err) {
            setDocs(prev => prev.map(d => d.name === file.name ? { ...d, status: "Failed", tokens: "network err" } : d));
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    };

    const preventDefault = (e: React.DragEvent) => { e.preventDefault(); e.stopPropagation(); };

    const handleDrop = (e: React.DragEvent) => {
        preventDefault(e);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    };

    const toggleDoc = (name: string) => {
        setDocs(prev => prev.map(d => d.name === name ? { ...d, active: !d.active } : d));
    };

    return (
        <div className="module-hud">
            <div className="hud-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                    <div className="hud-title">🔍 Knowledge Vault</div>
                    <div className="hud-subtitle">Drag & drop files to expand local knowledge</div>
                </div>
                <div style={{ textAlign: "right", fontSize: "12px", display: "flex", flexDirection: "column", gap: "4px" }}>
                    <div style={{ color: "var(--aether)" }}>Vision Language Model</div>
                    <select
                        value={selectedVlm}
                        onChange={handleVlmChange}
                        disabled={isSelectingVlm}
                        className="vlm-select"
                        style={{
                            background: "var(--bg-elevated)", color: "var(--fg)",
                            border: "1px solid var(--border)", padding: "4px 8px", borderRadius: "4px",
                            cursor: isSelectingVlm ? "wait" : "pointer"
                        }}
                    >
                        {vlmOptions.map(o => (
                            <option key={o.id} value={o.id}>
                                {o.name} {o.hardware_rating === "warning" ? "⚠️" : ""}
                            </option>
                        ))}
                    </select>
                </div>
            </div>
            <div style={{ display: "flex", gap: "16px" }}>
                <input type="file" ref={fileInputRef} style={{ display: "none" }} onChange={handleFileChange} />
                <div className="upload-dropzone" style={{ flex: 1 }}
                    onClick={handleUploadClick}
                    onDragEnter={preventDefault} onDragOver={preventDefault} onDragLeave={preventDefault} onDrop={handleDrop}>
                    <div className="upload-icon">📥</div>
                    <div className="upload-text">Drop documents here (or click)</div>
                    <div className="upload-hint">PDF, MD, TXT, CSV (Max 50MB)</div>
                </div>
                <div style={{ flex: 1 }} className="doc-list">
                    {docs.map((d, i) => (
                        <div key={i} className="doc-item" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            <input
                                type="checkbox"
                                checked={d.active}
                                onChange={() => toggleDoc(d.name)}
                                style={{ accentColor: "var(--plasma)", cursor: "pointer" }}
                            />
                            <span className="doc-name" style={{ flex: 1, opacity: d.active ? 1 : 0.5 }}>{d.name}</span>
                            <span className="doc-meta">
                                {d.status === "Ready" ? <span style={{ color: "var(--plasma)" }}>● {d.status}</span> : <span style={{ color: d.status === "Failed" ? "var(--ember)" : "var(--aether)" }}>○ {d.status}</span>}
                                <span>{d.tokens}</span>
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
