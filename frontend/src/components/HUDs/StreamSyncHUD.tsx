import { useState, useEffect } from "react";

export function StreamSyncHUD() {
    const [events, setEvents] = useState<any[]>([]);
    const [rssFeeds, setRssFeeds] = useState<string[]>([]);
    const [newFeedUrl, setNewFeedUrl] = useState("");

    useEffect(() => {
        const fetchEvents = async () => {
            try {
                const res = await fetch("/api/v1/events/stream");
                if (res.ok) {
                    const data = await res.json();
                    setEvents(data);
                }
            } catch (err) {
                console.error("Failed to fetch Event Stream", err);
            }
        };

        const fetchFeeds = async () => {
            try {
                // Corrected endpoint from main.py's StreamSync router
                const res = await fetch("/api/v1/streamsync/rss");
                if (res.ok) {
                    const data = await res.json();
                    setRssFeeds(data.feeds || []);
                }
            } catch (err) {
                console.error("Failed to fetch RSS feeds", err);
            }
        };

        fetchEvents();
        fetchFeeds();
        const eventInterval = setInterval(fetchEvents, 5000);
        const feedInterval = setInterval(fetchFeeds, 30000); // RSS less frequent

        return () => {
            clearInterval(eventInterval);
            clearInterval(feedInterval);
        };
    }, []);

    const handleAddFeed = async () => {
        if (!newFeedUrl) return;
        try {
            const res = await fetch("/api/v1/streamsync/rss/add", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: newFeedUrl })
            });
            if (res.ok) {
                const data = await res.json();
                setRssFeeds(data.feeds);
                setNewFeedUrl("");
            }
        } catch (err) {
            console.error("Failed to add RSS feed", err);
        }
    };

    const handleRemoveFeed = async (url: string) => {
        try {
            const res = await fetch("/api/v1/streamsync/rss/remove", {
                method: "POST", // Changed to POST as per backend implementation
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url })
            });
            if (res.ok) {
                const data = await res.json();
                setRssFeeds(data.feeds);
            }
        } catch (err) {
            console.error("Failed to remove RSS feed", err);
        }
    };

    const formatTime = (ts: number) => {
        const d = new Date(ts * 1000);
        return d.toTimeString().split(' ')[0];
    };

    return (
        <div className="module-hud" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            <div className="hud-header">
                <div className="hud-title">⚡ StreamSync Connector Hub</div>
                <div className="hud-subtitle">Live Data Ingestion Pipeline to RAGForge</div>
            </div>

            {/* Live Folder Watcher Status */}
            <div style={{ background: "rgba(20, 20, 30, 0.6)", padding: "16px", borderRadius: "8px", border: "1px solid var(--border)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
                    <div style={{ color: "var(--plasma)", fontSize: "16px" }}>📁</div>
                    <strong style={{ fontSize: "14px", color: "var(--text)" }}>Live Folder Auto-Sync</strong>
                    <span style={{ marginLeft: "auto", fontSize: "11px", color: "var(--brand-glow)", background: "rgba(0,255,100,0.1)", padding: "2px 6px", borderRadius: "12px", border: "1px solid rgba(0,255,100,0.2)" }}>● Active</span>
                </div>
                <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    Watching your local <code>~/AetherForge/data/LiveFolder</code> directory. Any documents dropped here will automatically be indexed into your AI Brain immediately.
                </div>
            </div>

            {/* RSS Feed Manager */}
            <div style={{ background: "rgba(20, 20, 30, 0.6)", padding: "16px", borderRadius: "8px", border: "1px solid var(--border)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                    <div style={{ color: "var(--ember)", fontSize: "16px" }}>📻</div>
                    <strong style={{ fontSize: "14px", color: "var(--text)" }}>RSS News Poller</strong>
                    <span style={{ marginLeft: "auto", fontSize: "11px", color: "var(--text-muted)" }}>Polls every 30 mins</span>
                </div>

                <div style={{ display: "flex", gap: "8px", marginBottom: "16px" }}>
                    <input
                        type="text"
                        value={newFeedUrl}
                        onChange={(e) => setNewFeedUrl(e.target.value)}
                        placeholder="https://news.ycombinator.com/rss"
                        style={{ flex: 1, background: "rgba(0,0,0,0.2)", border: "1px solid var(--border)", padding: "8px 12px", borderRadius: "6px", color: "var(--text)", fontSize: "13px" }}
                    />
                    <button className="btn btn-primary" onClick={handleAddFeed}>Add Feed</button>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    {rssFeeds.length === 0 ? (
                        <div style={{ fontSize: "12px", color: "var(--text-muted)", fontStyle: "italic" }}>No active RSS feeds.</div>
                    ) : (
                        rssFeeds.map(url => (
                            <div key={url} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: "rgba(0,0,0,0.2)", padding: "8px 12px", borderRadius: "6px", border: "1px solid rgba(255,255,255,0.05)" }}>
                                <span style={{ fontSize: "12px", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{url}</span>
                                <button
                                    onClick={() => handleRemoveFeed(url)}
                                    style={{ background: "none", border: "none", color: "var(--danger)", cursor: "pointer", fontSize: "16px", marginLeft: "12px" }}
                                >
                                    &times;
                                </button>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Global Ingestion Log */}
            <div style={{ marginTop: "8px" }}>
                <div style={{ fontSize: "13px", fontWeight: "bold", color: "var(--text)", marginBottom: "12px", paddingLeft: "4px" }}>
                    Live Ingestion Stream
                </div>
                <div className="event-console" style={{ minHeight: "200px" }}>
                    {events.length === 0 ? (
                        <div style={{ color: "var(--text-muted)", fontSize: "12px", textAlign: "center", padding: "24px" }}>
                            Awaiting events from RSS or Directory watcher...
                        </div>
                    ) : (
                        events.map((e, i) => (
                            <div key={i} className="event-row">
                                <span className="event-time">{formatTime(e.timestamp)}</span>
                                <span className="event-source" style={{
                                    color: e.source === "DirectoryWatcher" ? "var(--plasma)" : "var(--ember)",
                                    fontWeight: "bold"
                                }}>
                                    [{e.source}]
                                </span>
                                <span className="event-payload" style={{
                                    color: e.event_type.includes("failed") ? "var(--danger)" : "var(--text)"
                                }}>
                                    <span style={{ opacity: 0.5, marginRight: "8px" }}>{e.event_type}</span>
                                    {e.payload?.title || e.payload?.filename || JSON.stringify(e.payload)}
                                </span>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
}
