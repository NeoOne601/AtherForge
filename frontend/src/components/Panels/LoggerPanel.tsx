import { useState, useEffect, useRef } from "react";

// ── Persistent Logger Singleton ──────────────────────────────────────────
// The WebSocket must NOT be closed when this component unmounts (tab switch).
// A module-level singleton keeps the connection alive indefinitely and lets
// multiple mounts/unmounts share the same stream without causing EPIPE.
let _globalWs: WebSocket | null = null;
const _logBuffer: any[] = [];
const _logListeners: Set<(log: any) => void> = new Set();
let _reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function ensureLoggerConnected() {
    if (_globalWs && (_globalWs.readyState === WebSocket.OPEN || _globalWs.readyState === WebSocket.CONNECTING)) {
        return;
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/v1/system/logs`);
    _globalWs = ws;

    ws.onopen = () => {
        if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
    };

    ws.onmessage = (event) => {
        try {
            const log = JSON.parse(event.data);
            // Keep a rolling buffer of last 500 logs for when new components mount
            _logBuffer.push(log);
            if (_logBuffer.length > 500) _logBuffer.shift();
            // Notify all active listeners
            _logListeners.forEach(fn => fn(log));
        } catch (_) {}
    };

    ws.onclose = () => {
        _globalWs = null;
        // Auto-reconnect after 3 seconds regardless of cause (EPIPE, network, nav)
        _reconnectTimer = setTimeout(ensureLoggerConnected, 3000);
    };

    ws.onerror = () => {
        // onclose will handle reconnect
        try { ws.close(); } catch (_) {}
    };
}

// Start the persistent connection immediately when the module loads
ensureLoggerConnected();

export function LoggerPanel() {
    const [logs, setLogs] = useState<any[]>([..._logBuffer]);
    const [autoScroll, setAutoScroll] = useState(true);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Ensure connection is alive when component mounts
        ensureLoggerConnected();

        // Subscribe to new log events
        const handler = (log: any) => {
            setLogs(prev => [...prev.slice(-499), log]);
        };
        _logListeners.add(handler);

        return () => {
            // IMPORTANT: Do NOT close the WebSocket here.
            // Only unsubscribe this component's listener.
            _logListeners.delete(handler);
        };
    }, []);


    useEffect(() => {
        if (autoScroll && scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs, autoScroll]);

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        alert("Log entry copied to clipboard!");
    };

    const levels = ["ERROR", "WARNING", "INFO", "DEBUG"];
    const groupedLogs = levels.reduce((acc, level) => {
        acc[level] = logs.filter(l => l.level === level);
        return acc;
    }, {} as Record<string, any[]>);

    return (
        <div className="panel-wrapper logger-panel-wrapper">
            <div className="panel-title">📜 System Real-time Logs</div>
            <div className="panel-sub">Live diagnostic stream from AetherForge backend — categorized by severity</div>

            <div className="logger-toolbar">
                <button className="btn btn-ghost" onClick={() => setLogs([])}>🗑 Clear View</button>
                <button
                    className={`btn btn-ghost ${autoScroll ? "active" : ""}`}
                    onClick={() => setAutoScroll(!autoScroll)}
                >
                    {autoScroll ? "⏸ Pause Scroll" : "▶ Resume Scroll"}
                </button>
                <div style={{ flex: 1 }} />
                <div className="status-pill">Streaming WebSocket Active</div>
            </div>

            <div className="logger-grid">
                {levels.map(lvl => (
                    <div key={lvl} className={`log-flex-box ${lvl.toLowerCase()}`}>
                        <div className="log-box-header">
                            <span className="log-lvl-label">{lvl}</span>
                            <span className="log-count">{groupedLogs[lvl].length} entries</span>
                        </div>
                        <div className="log-entries-container custom-scrollbar">
                            {groupedLogs[lvl].length === 0 && (
                                <div className="log-empty">No {lvl} logs in buffer</div>
                            )}
                            {groupedLogs[lvl].map((log, i) => (
                                <div
                                    key={i}
                                    className="log-entry-card"
                                    onClick={() => copyToClipboard(`[${new Date(log.timestamp * 1000).toISOString()}] ${log.level} [${log.module}] ${log.message}`)}
                                    title="Click to copy log record"
                                >
                                    <div className="log-meta">
                                        <span className="log-timestamp">{new Date(log.timestamp * 1000).toLocaleTimeString()}</span>
                                        <span className="log-module">{log.module}</span>
                                    </div>
                                    <div className="log-msg">
                                        {log.message}
                                    </div>
                                    {log.pathname && (
                                        <div className="log-source">{log.pathname.split("/").pop()}:{log.lineno}</div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            {/* Also show a unified chronological feed at the bottom */}
            <div style={{ marginTop: 24, marginBottom: 8, fontSize: 13, fontWeight: 600, color: "var(--text-secondary)" }}>
                Chronological Feed
            </div>
            <div className="chronological-feed custom-scrollbar" ref={scrollRef}>
                {logs.map((log, i) => (
                    <div key={i} className={`log-row-stream ${log.level.toLowerCase()}`}>
                        <span className="log-time">[{new Date(log.timestamp * 1000).toLocaleTimeString()}]</span>
                        <span className="log-lvl">[{log.level}]</span>
                        <span className="log-mod">[{log.module}]</span>
                        <span className="log-text">{log.message}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
