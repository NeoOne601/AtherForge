// AetherForge v1.0 — frontend/src/lib/tauri.ts
// ─────────────────────────────────────────────────────────────────
// Typed Tauri IPC bridge. All frontend<→>backend communication
// flows through this module, ensuring type safety at the boundary.
//
// In Tauri 2.x, frontend communicates with the FastAPI backend
// via standard HTTP (fetch) since Tauri's CSP allows it. Rust
// commands are used only for OS-level features (tray, notifications).
// ─────────────────────────────────────────────────────────────────

const API_BASE = "http://127.0.0.1:8765";

// ── Type Definitions ─────────────────────────────────────────────

export interface ChatRequest {
    session_id: string;
    module: string;
    message: string;
    xray_mode: boolean;
    protocol?: string;
    context?: Record<string, unknown>;
}

export interface ChatResponse {
    session_id: string;
    response: string;
    module: string;
    latency_ms: number;
    tool_calls: ToolCall[];
    policy_decisions: PolicyDecision[];
    causal_graph: CausalGraph | null;
    faithfulness_score: number | null;
    reasoning_summary: string | null;
    reasoning_trace: string | null;
    answer_text: string | null;
    citations: ChatCitation[];
    attachments: string[];
    suggestions: string[];
}

export interface ToolCall {
    name: string;
    args?: Record<string, unknown>;
    arguments?: Record<string, unknown>;
    result?: unknown;
    latency_ms?: number;
    attachments?: string[];
    citations?: ChatCitation[];
}

export interface ChatCitation {
    source: string;
    page: number | string | null;
    section: string | null;
    snippet: string | null;
    kind: string;
    label?: string | null;
}

export interface PolicyDecision {
    allowed: boolean;
    reason: string;
    deny_reasons: string[];
    policy_version: string;
    latency_ms: number;
    fsm_state: string;
}

export interface CausalGraph {
    nodes: CausalNode[];
    edges: CausalEdge[];
    total_latency_ms: number;
}

export interface CausalNode {
    id: string;
    data: Record<string, unknown>;
    ts: number;
}

export interface CausalEdge {
    source: string;
    target: string;
    label?: string;
}

export interface ModuleInfo {
    id: string;
    name: string;
    description: string;
    icon: string;
}

export interface SystemStatus {
    battery_pct: number | null;
    battery_plugged: boolean | null;
    cpu_pct: number;
    ram_used_gb: number;
    ram_total_gb: number;
    modules: string[];
}

export interface ReplayStats {
    total_records: number;
    size_mb: number;
    modules: Record<string, number>;
    trained_count: number;
    avg_faithfulness: number;
    oldest_record: string | null;
}

export interface Insight {
    insight_id: string;
    title: string;
    summary: string;
    novelty_score: number;
    supporting_records: string[];
    topics: string[];
    generated_at: string;
}

// ── API Client ───────────────────────────────────────────────────

async function apiRequest<T>(
    path: string,
    options: RequestInit = {}
): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { "Content-Type": "application/json", ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const err = await res.text();
        throw new Error(`API ${res.status}: ${err}`);
    }
    return res.json() as Promise<T>;
}

// ── Health ────────────────────────────────────────────────────────

export async function checkHealth(): Promise<{ status: string; startup_ms: number; model: string }> {
    return apiRequest("/health");
}

export async function getSystemStatus(): Promise<SystemStatus> {
    return apiRequest("/api/v1/status");
}

// ── Chat ─────────────────────────────────────────────────────────

export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
    return apiRequest("/api/v1/chat", {
        method: "POST",
        body: JSON.stringify(req),
    });
}

// ── WebSocket streaming chat ──────────────────────────────────────

export type StreamChunk =
    | { type: "meta"; session_id: string; module: string }
    | { type: "reasoning"; content: string }
    | { type: "token"; content: string }
    | { type: "tool_start"; name: string; args: Record<string, unknown> }
    | { type: "tool_result"; name: string; result: string }
    | {
        type: "done";
        session_id: string;
        module: string;
        latency_ms: number;
        faithfulness_score: number | null;
        policy_decisions: PolicyDecision[];
        causal_graph: CausalGraph | null;
        tool_calls: ToolCall[];
        response: string;
        reasoning_summary: string | null;
        reasoning_trace: string | null;
        answer_text: string | null;
        citations: ChatCitation[];
        attachments: string[];
        suggestions: string[];
    }
    | { type: "error"; content: string };

export function createChatSocket(
    sessionId: string,
    onChunk: (chunk: StreamChunk) => void,
    onError?: (err: Event) => void
): WebSocket {
    const ws = new WebSocket(`ws://127.0.0.1:8765/api/v1/ws/${sessionId}`);
    ws.onmessage = (e) => {
        try {
            const chunk = JSON.parse(e.data) as StreamChunk;
            onChunk(chunk);
        } catch {
            /* ignore malformed frames */
        }
    };
    ws.onerror = onError ?? ((e) => console.error("WS error", e));
    return ws;
}

// ── Modules ───────────────────────────────────────────────────────

export async function listModules(): Promise<ModuleInfo[]> {
    return apiRequest("/api/v1/modules");
}

// ── Policies ──────────────────────────────────────────────────────

export async function getPolicies(): Promise<{ policy: string }> {
    return apiRequest("/api/v1/policies");
}

export async function updatePolicies(policy: string): Promise<{ success: boolean; error: string | null }> {
    return apiRequest("/api/v1/policies", {
        method: "POST",
        body: JSON.stringify({ policy }),
    });
}

// ── Learning ──────────────────────────────────────────────────────

export async function triggerTraining(): Promise<{ status: string; message: string }> {
    return apiRequest("/api/v1/learning/trigger", { method: "POST" });
}

export async function getReplayStats(): Promise<ReplayStats> {
    return apiRequest("/api/v1/replay/stats");
}

// ── Insights ──────────────────────────────────────────────────────
// InsightForge stores insights on the backend; we fetch them via API
export async function getInsights(): Promise<Insight[]> {
    // Fallback: read from insights.json via a backend endpoint
    try {
        return apiRequest<Insight[]>("/api/v1/insights");
    } catch {
        return [];
    }
}

// ── Tauri OS commands (native) ────────────────────────────────────

/** Show a native OS notification (uses Tauri plugin) */
export async function showNotification(title: string, body: string): Promise<void> {
    try {
        const { sendNotification } = await import("@tauri-apps/plugin-notification");
        await sendNotification({ title, body });
    } catch {
        // Graceful fallback for web-only mode
        console.log(`[Notification] ${title}: ${body}`);
    }
}

/** Generate a stable session ID for this tab */
export function newSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}
