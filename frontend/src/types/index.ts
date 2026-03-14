export interface PolicyDecision {
    allowed: boolean;
    reason: string;
    deny_reasons: string[];
    fsm_state: string;
    latency_ms: number;
}

export interface CausalGraph {
    nodes: { id: string; data: Record<string, unknown>; ts: number }[];
    edges: { source: string; target: string; label?: string }[];
    total_latency_ms: number;
}

export interface Module {
    id: string;
    name: string;
    icon: string;
    desc: string;
}

export interface SessionSummary {
    id: string;
    module: string;
    title: string;
    created_at: number;
    updated_at: number;
    message_count: number;
}

export interface StoredMessage {
    id: string;
    role: "user" | "assistant";
    content: string;
    ts: number;
    metadata: Record<string, unknown>;
}

export interface RAGDoc {
    name: string;
    status: string;
    tokens: string;
    active: boolean;
}

export interface ReplayItem {
    id: string;
    timestamp_utc: number;
    module: string;
    prompt: string;
    response: string;
    faithfulness_score: number;
    is_used_for_training: boolean;
}

export interface ChatCitation {
    source: string;
    page?: number | string | null;
    section?: string | null;
    snippet?: string | null;
    kind: string;
    label?: string | null;
}

export interface ToolCall {
    name: string;
    arguments?: Record<string, unknown>;
    args?: Record<string, unknown>;
    result?: unknown;
    attachments?: string[];
    citations?: ChatCitation[];
}

export interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    module: string;
    streaming?: boolean;
    latency_ms?: number;
    faithfulness_score?: number;
    reasoning_trace?: string;
    answer_text?: string;
    policy_decisions?: PolicyDecision[];
    causal_graph?: CausalGraph;
    blocked?: boolean;
    tool_calls?: ToolCall[];
    citations?: ChatCitation[];
    attachments?: string[];
}

// ── Constants ────────────────────────────────────────────────────
export const MODULES: Module[] = [
    { id: "localbuddy", name: "LocalBuddy", icon: "💬", desc: "Conversational AI" },
    { id: "ragforge", name: "RAGForge", icon: "🔍", desc: "Private Knowledge RAG" },
    { id: "watchtower", name: "WatchTower", icon: "👁️", desc: "System Observability" },
    { id: "streamsync", name: "StreamSync", icon: "⚡", desc: "Live Data Ingestion" },
    { id: "tunelab", name: "TuneLab", icon: "🎛️", desc: "Self-Optimization (OPLoRA)" },
    { id: "logger", name: "Logger", icon: "📜", desc: "System real-time logs" },
];

export const MODULE_SUGGESTIONS: Record<string, string[]> = {
    localbuddy: [
        "What can you help me with?",
        "Tell me a joke about AI.",
        "How do I use the WatchTower?"
    ],
    ragforge: [
        "What documents have I uploaded?",
        "Summarise the key points from my latest document",
        "Find anything related to OPLoRA in my documents",
        "What are the main topics in the knowledge base?",
    ],
    watchtower: [
        "Why is memory usage so high?",
        "What process is using the most CPU?",
        "Show me the current memory stats",
        "Are there any active anomalies right now?",
    ],
    streamsync: [
        "Show me the recent event stream",
        "Summarise what's in the stream",
        "What sources are sending events?",
        "Clear the event buffer",
    ],
    tunelab: [
        "How many samples are ready for training?",
        "Trigger the OPLoRA compilation cycle",
        "What is the current replay buffer status?",
        "How full is the training queue?",
    ],
    logger: [
        "Show latest logs",
        "Clear log buffer",
    ]
};
