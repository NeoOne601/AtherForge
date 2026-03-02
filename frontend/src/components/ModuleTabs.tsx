// AetherForge v1.0 — frontend/src/components/ModuleTabs.tsx
// Sidebar module selector. Shows all 5 AetherForge modules with
// icons, active state, and animated active indicator.
import React from "react";
import { motion } from "framer-motion";

const MODULES = [
    { id: "ragforge", label: "RAGForge", icon: "🔍", desc: "Local RAG" },
    { id: "localbuddy", label: "LocalBuddy", icon: "💬", desc: "Conversational AI" },
    { id: "watchtower", label: "WatchTower", icon: "👁️", desc: "Anomaly Detection" },
    { id: "streamsync", label: "StreamSync", icon: "⚡", desc: "Event Streams" },
    { id: "tunelab", label: "TuneLab", icon: "🎛️", desc: "Fine-tuning" },
] as const;

interface Props {
    active: string;
    onChange: (id: string) => void;
}

export default function ModuleTabs({ active, onChange }: Props): JSX.Element {
    return (
        <div className="flex flex-col gap-0.5">
            {MODULES.map(mod => (
                <button
                    key={mod.id}
                    id={`module-${mod.id}`}
                    onClick={() => onChange(mod.id)}
                    className={`sidebar-item w-full text-left relative ${active === mod.id ? "active" : ""}`}
                >
                    {active === mod.id && (
                        <motion.div
                            layoutId="active-module"
                            className="absolute inset-0 rounded-lg"
                            style={{ background: "rgba(139,92,246,0.12)", border: "1px solid rgba(139,92,246,0.3)" }}
                            transition={{ type: "spring", stiffness: 400, damping: 35 }}
                        />
                    )}
                    <span className="relative z-10 text-base">{mod.icon}</span>
                    <div className="relative z-10 flex flex-col min-w-0">
                        <span className="text-xs font-medium truncate">{mod.label}</span>
                        <span className="text-xs truncate" style={{ color: "var(--text-muted)", fontSize: "0.68rem" }}>{mod.desc}</span>
                    </div>
                </button>
            ))}
        </div>
    );
}
