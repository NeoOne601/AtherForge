import { useState, useEffect } from "react";
import QRCode from "react-qr-code";

export function SyncPanel() {
    const [syncInfo, setSyncInfo] = useState<any>(null);
    const [manualUri, setManualUri] = useState("");

    useEffect(() => {
        fetch("/api/v1/sync/info")
            .then(r => r.json())
            .then(setSyncInfo)
            .catch(console.error);
    }, []);

    const pairDevice = async () => {
        try {
            const res = await fetch("/api/v1/sync/pair", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ uri: manualUri })
            });
            if (res.ok) {
                alert("Device Paired Successfully!");
                setManualUri("");
                // Refresh
                fetch("/api/v1/sync/info").then(r => r.json()).then(setSyncInfo);
            } else {
                alert((await res.json()).error);
            }
        } catch (err) { console.error(err); }
    };

    if (!syncInfo) return <div className="panel-wrapper"><div className="panel-title">Loading Sync Engine...</div></div>;
    if (syncInfo.status === "offline") return <div className="panel-wrapper"><div className="panel-title">Sync Engine Offline</div></div>;

    return (
        <div className="panel-wrapper" style={{ overflowY: "auto" }}>
            <div className="panel-title">🔗 Zero-Knowledge Device Sync</div>
            <div className="panel-sub">Peer-to-Peer AP-architecture Database Replication</div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", marginTop: "20px" }}>
                <div className="policy-card" style={{ padding: "24px", border: "1px solid var(--border)", background: "var(--bg-elevated)", borderRadius: "var(--radius)" }}>
                    <div style={{ fontWeight: 600, fontSize: "16px", marginBottom: "16px", color: "var(--fg)" }}>Add Device via QR Code</div>
                    <div style={{ background: "white", padding: "16px", borderRadius: "12px", width: "fit-content", margin: "0 auto" }}>
                        <QRCode value={syncInfo.pairing_uri} size={180} />
                    </div>
                    <div style={{ marginTop: "16px", fontSize: "12px", color: "var(--text-muted)", textAlign: "center" }}>
                        Scan this with the AetherForge mobile app to establish an E2EE connection over your local LAN.
                    </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
                    <div className="policy-card" style={{ padding: "16px", border: "1px solid var(--border)", background: "var(--bg-elevated)", borderRadius: "var(--radius)" }}>
                        <div style={{ fontWeight: 600, marginBottom: "12px", color: "var(--fg)" }}>Manual Pair</div>
                        <input
                            style={{
                                width: "100%", marginBottom: "12px", padding: "10px",
                                background: "var(--bg)", border: "1px solid var(--border)",
                                borderRadius: "8px", color: "var(--fg)", outline: "none"
                            }}
                            placeholder="Paste pairing URI..."
                            value={manualUri}
                            onChange={e => setManualUri(e.target.value)}
                        />
                        <button
                            style={{
                                width: "100%", padding: "10px", borderRadius: "8px",
                                background: manualUri ? "var(--brand-glow)" : "rgba(255,255,255,0.05)",
                                color: manualUri ? "var(--bg)" : "var(--text-muted)", border: "none",
                                cursor: manualUri ? "pointer" : "not-allowed", fontWeight: 600
                            }}
                            onClick={pairDevice} disabled={!manualUri}
                        >
                            Connect Peer
                        </button>
                    </div>

                    <div className="policy-card" style={{ padding: "16px", border: "1px solid var(--border)", background: "var(--bg-elevated)", borderRadius: "var(--radius)" }}>
                        <div style={{ fontWeight: 600, marginBottom: "12px", color: "var(--fg)" }}>Cluster Status ({syncInfo.node_id})</div>

                        <div style={{ fontSize: "13px", color: "var(--text-muted)", marginBottom: "8px" }}>Discovered mDNS Peers:</div>
                        {(!syncInfo.peers || syncInfo.peers.length === 0) ? <div style={{ fontSize: "12px", background: "rgba(255,255,255,0.05)", padding: "8px", borderRadius: "8px", color: "var(--text-muted)" }}>No local peers found.</div> :
                            syncInfo.peers.map((p: any) => (
                                <div key={p.id} style={{ fontSize: "12px", background: "rgba(0,255,100,0.1)", color: "var(--brand-glow)", padding: "8px", borderRadius: "8px", marginBottom: "4px" }}>
                                    ✓ {p.id} ({p.ip}:{p.port})
                                </div>
                            ))
                        }

                        <div style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "16px", marginBottom: "8px" }}>Authorized Devices (E2EE Active):</div>
                        {(!syncInfo.authorized || syncInfo.authorized.length === 0) ? <div style={{ fontSize: "12px", background: "rgba(255,255,255,0.05)", padding: "8px", borderRadius: "8px", color: "var(--text-muted)" }}>Empty list. Scan the QR code.</div> :
                            syncInfo.authorized.map((id: string) => (
                                <div key={id} style={{ fontSize: "12px", background: "rgba(0,150,255,0.1)", color: "#a5b4fc", padding: "8px", borderRadius: "8px", marginBottom: "4px" }}>
                                    🔐 {id}
                                </div>
                            ))
                        }
                    </div>
                </div>
            </div>
        </div>
    );
}
