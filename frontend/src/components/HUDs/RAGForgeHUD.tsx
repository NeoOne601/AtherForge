import React, { useState, useEffect, useRef } from "react";
import { RAGDoc } from "../../types";
import { RAGTreePanel } from "../RAGTreePanel";

interface RAGForgeHUDProps {
    docs: RAGDoc[];
    setDocs: React.Dispatch<React.SetStateAction<RAGDoc[]>>;
}

export function RAGForgeHUD({ docs, setDocs }: RAGForgeHUDProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [vlmOptions, setVlmOptions] = useState<any[]>([]);
    const [selectedVlm, setSelectedVlm] = useState<string>("smolvlm-256m");
    const [isSelectingVlm, setIsSelectingVlm] = useState(false);
    const [treeBrowserDoc, setTreeBrowserDoc] = useState<string | null>(null);

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
        setDocs(prev => [
            ...prev.filter(d => d.name !== file.name),
            {
                document_id: `pending-${file.name}`,
                name: file.name,
                status: "queued",
                tokens: "—",
                active: true,
            }
        ]);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("/api/v1/ragforge/upload", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();

            if (res.ok) {
                setDocs(prev => prev.map(d => d.name === file.name ? {
                    ...d,
                    document_id: data.document_id,
                    status: data.ingest_status,
                    tokens: `~${data.chunks_added} chunks`,
                    chunk_count: data.chunks_added,
                    image_pages_pending: data.image_pages_pending,
                    last_error: data.last_error,
                } : d));
            } else {
                setDocs(prev => prev.map(d => d.name === file.name ? { ...d, status: "failed", tokens: "error" } : d));
            }
        } catch (err) {
            setDocs(prev => prev.map(d => d.name === file.name ? { ...d, status: "failed", tokens: "network err" } : d));
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

    const toggleDoc = async (doc: RAGDoc) => {
        const nextSelected = !doc.active;
        setDocs(prev => prev.map(d => d.document_id === doc.document_id ? { ...d, active: nextSelected } : d));
        try {
            await fetch(`/api/v1/ragforge/documents/${doc.document_id}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ selected: nextSelected }),
            });
        } catch (err) {
            console.error("Failed to update document selection", err);
            setDocs(prev => prev.map(d => d.document_id === doc.document_id ? { ...d, active: doc.active } : d));
        }
    };

    const handleRetry = async (doc: RAGDoc) => {
        setDocs(prev => prev.map(d => d.document_id === doc.document_id ? { ...d, status: "retrying..." } : d));
        try {
            const res = await fetch(`/api/v1/ragforge/documents/${doc.document_id}/retry`, {
                method: "POST"
            });
            const data = await res.json();
            if (res.ok) {
                setDocs(prev => prev.map(d => d.name === doc.name ? {
                    ...d,
                    status: data.ingest_status,
                    tokens: `~${data.chunks_added} chunks`,
                    chunk_count: data.chunks_added,
                } : d));
            } else {
                setDocs(prev => prev.map(d => d.document_id === doc.document_id ? { ...d, status: "failed" } : d));
            }
        } catch (err) {
            console.error("Retry failed", err);
            setDocs(prev => prev.map(d => d.document_id === doc.document_id ? { ...d, status: "failed" } : d));
        }
    };

    const handleEnrichImages = async (doc: RAGDoc) => {
        setDocs(prev => prev.map(d => d.document_id === doc.document_id ? { ...d, status: "ocr_running" } : d));
        try {
            const res = await fetch(`/api/v1/ragforge/documents/${doc.document_id}/enrich-images`, {
                method: "POST"
            });
            const data = await res.json();
            if (res.ok && data.status === "queued") {
                // Show success feedback — status will update on next poll
                setDocs(prev => prev.map(d => d.document_id === doc.document_id ? {
                    ...d,
                    status: "ocr_running",
                    tokens: `${d.tokens} + ${data.image_pages} img pg(s)`,
                } : d));
            } else if (data.status === "no_images") {
                setDocs(prev => prev.map(d => d.document_id === doc.document_id ? {
                    ...d, status: "ready", image_pages_pending: 0
                } : d));
            }
        } catch (err) {
            console.error("Enrich images failed", err);
            setDocs(prev => prev.map(d => d.document_id === doc.document_id ? { ...d, status: "partial" } : d));
        }
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
                            <option key={o.id} value={o.id} title={o.hardware_message}>
                                {o.name} {o.hardware_rating === "warning" ? "⚠️" : ""}
                            </option>
                        ))}
                    </select>
                    {vlmOptions.find(o => o.id === selectedVlm)?.hardware_rating === "warning" && (
                        <div style={{ color: "var(--ember)", fontSize: "10px", marginTop: "2px", maxWidth: "150px" }}>
                            {vlmOptions.find(o => o.id === selectedVlm)?.hardware_message}
                        </div>
                    )}
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
                    {docs.map((d) => (
                        <div key={d.document_id} className="doc-item" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            <input
                                type="checkbox"
                                checked={d.active}
                                onChange={() => toggleDoc(d)}
                                style={{ accentColor: "var(--plasma)", cursor: "pointer" }}
                            />
                            <span className="doc-name" style={{ flex: 1, opacity: d.active ? 1 : 0.5 }}>{d.name}</span>
                            <span className="doc-meta" style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                                {d.status === "ready"
                                    ? <span style={{ color: "var(--plasma)" }}>● {d.status}</span>
                                    : d.status === "ocr_running" || d.status === "extracting_text"
                                    ? <span style={{ color: "var(--aether)", display: "flex", alignItems: "center", gap: "4px" }}>
                                        <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span>
                                        {d.status}
                                      </span>
                                    : <span style={{ color: d.status === "failed" ? "var(--ember)" : "var(--aether)" }}>○ {d.status}</span>
                                }
                                <span style={{ color: "var(--fg-muted)", fontSize: "11px" }}>{d.tokens}</span>
                                {/* Image pages pending badge */}
                                {(d.image_pages_pending || 0) > 0 && d.status !== "ocr_running" && (
                                    <span title={`${d.image_pages_pending} page(s) contain images that need VLM enrichment`} style={{
                                        background: "rgba(255,160,0,0.12)",
                                        border: "1px solid rgba(255,160,0,0.4)",
                                        borderRadius: "4px",
                                        padding: "1px 5px",
                                        fontSize: "10px",
                                        color: "#ffa500",
                                        whiteSpace: "nowrap",
                                    }}>🖼 {d.image_pages_pending} img pg{(d.image_pages_pending || 0) > 1 ? "s" : ""} need VLM</span>
                                )}
                                {d.status === "ready" && (
                                    <button
                                        id={`tree-btn-${d.document_id}`}
                                        onClick={() => setTreeBrowserDoc(
                                            treeBrowserDoc === d.name ? null : d.name
                                        )}
                                        title="Browse document section tree"
                                        style={{
                                            background: treeBrowserDoc === d.name
                                                ? "rgba(var(--plasma-rgb, 100,100,255), 0.15)"
                                                : "transparent",
                                            border: `1px solid ${treeBrowserDoc === d.name ? "var(--plasma)" : "var(--border)"}`,
                                            borderRadius: "4px",
                                            padding: "2px 6px",
                                            fontSize: "11px",
                                            cursor: "pointer",
                                            color: treeBrowserDoc === d.name ? "var(--plasma)" : "var(--fg-muted)",
                                        }}
                                    >
                                        🌲
                                    </button>
                                )}
                                {/* Enrich Images button — shown for partial + ocr_pending docs with image pages */}
                                {(d.status === "partial" || d.status === "ocr_pending" || d.status === "ready") && (d.image_pages_pending || 0) > 0 && (
                                    <button
                                        onClick={() => handleEnrichImages(d)}
                                        title={`Trigger VLM enrichment for ${d.image_pages_pending} image page(s). Make sure Ollama is running with your selected VLM model.`}
                                        style={{
                                            background: "rgba(0,180,255,0.08)",
                                            border: "1px solid rgba(0,180,255,0.4)",
                                            color: "#00b4ff",
                                            borderRadius: "4px",
                                            padding: "2px 7px",
                                            fontSize: "10px",
                                            cursor: "pointer",
                                            whiteSpace: "nowrap",
                                        }}
                                    >
                                        🖼 Enrich Images
                                    </button>
                                )}
                                {(d.status === "failed" || d.status === "ocr_pending") && (
                                    <button 
                                        onClick={() => handleRetry(d)}
                                        className="repair-btn"
                                        style={{
                                            background: "rgba(255,100,100,0.1)",
                                            border: "1px solid var(--ember)",
                                            color: "var(--ember)",
                                            borderRadius: "4px",
                                            padding: "2px 8px",
                                            fontSize: "10px",
                                            cursor: "pointer",
                                            textTransform: "uppercase",
                                            letterSpacing: "0.5px"
                                        }}
                                    >
                                        Repair
                                    </button>
                                )}
                            </span>
                        </div>
                    ))}
                </div>
            </div>
            {/* HTI Tree Browser — Phase 15 */}
            {treeBrowserDoc && (
                <RAGTreePanel
                    sourceName={treeBrowserDoc}
                    onClose={() => setTreeBrowserDoc(null)}
                />
            )}
        </div>
    );
}
