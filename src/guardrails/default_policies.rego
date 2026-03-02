# AetherForge v1.0 — src/guardrails/default_policies.rego
# ─────────────────────────────────────────────────────────────────
# Open Policy Agent (OPA) Rego policies for Silicon Colosseum.
#
# Silicon Colosseum is AetherForge's deterministic safety layer.
# Every tool call and agent output passes through these policies
# BEFORE execution. This creates a glass-box audit trail.
#
# Policy structure:
#   - package aetherforge.guardrails (top-level namespace)
#   - allow: default deny; rules grant permission
#   - deny_reasons: collects all reasons for denial (multi-violation)
#
# To edit policies at runtime, use the PolicyEditor UI or:
#   POST /api/v1/policies {"policy": "<new rego>"}
#
# OPA evaluates: POST /v1/data/aetherforge/guardrails/allow
# Input schema:
#   input.session_id       string
#   input.module           string
#   input.message          string
#   input.tool_call_count  int
#   input.faithfulness_score float (optional, post-output check)
# ─────────────────────────────────────────────────────────────────
package aetherforge.guardrails

import rego.v1

# ── Default: deny everything, then selectively allow ─────────────
default allow := false
default deny_reasons := []

# ── Main allow rule ───────────────────────────────────────────────
# Grants access only if NO deny rules fire.
allow if {
    count(deny_reasons) == 0
}

# ── Deny Rule 1: Tool call budget ─────────────────────────────────
# Prevents runaway agent loops. Max 8 tool calls per turn.
# This is the most important safety constraint for autonomous agents.
deny_reasons contains reason if {
    input.tool_call_count > 8
    reason := sprintf(
        "Tool call budget exceeded: %d > 8. Agent loop terminated.",
        [input.tool_call_count]
    )
}

# ── Deny Rule 2: Faithfulness threshold ───────────────────────────
# Blocks outputs that fall below 92% faithfulness.
# Only evaluated when faithfulness_score is present in input
# (i.e., post-output check, not pre-flight).
deny_reasons contains reason if {
    score := input.faithfulness_score
    score < 0.92
    reason := sprintf(
        "Output faithfulness %.2f < 0.92 threshold. Output withheld.",
        [score]
    )
}

# ── Deny Rule 3: Prohibited operations ───────────────────────────
# Hard-block list of operations that are never allowed.
# These can't be overridden by user config — they're baked in.
_prohibited_patterns := [
    "rm -rf",
    "DELETE FROM",
    "DROP TABLE",
    "sudo",
    "__import__",
    "eval(",
    "exec(",
    "os.system",
    "subprocess.call",
]

deny_reasons contains reason if {
    some pattern in _prohibited_patterns
    contains(input.message, pattern)
    reason := sprintf("Prohibited pattern detected in message: '%s'", [pattern])
}

# ── Deny Rule 4: Module access control ───────────────────────────
# Restrict which modules are accessible.
# tunelab requires explicit enterprise mode (future: add RBAC here).
_valid_modules := {"ragforge", "localbuddy", "watchtower", "streamsync", "tunelab", "output_filter"}

deny_reasons contains reason if {
    not _valid_modules[input.module]
    reason := sprintf("Unknown module: '%s'. Valid: %v", [input.module, _valid_modules])
}

# ── Deny Rule 5: Empty / trivial messages ─────────────────────────
deny_reasons contains reason if {
    count(trim_space(input.message)) == 0
    reason := "Empty message rejected"
}

# ── Deny Rule 6: Message length limit ─────────────────────────────
# Prevents prompt injection via extremely long messages.
deny_reasons contains reason if {
    count(input.message) > 16384
    reason := sprintf(
        "Message exceeds 16KB limit: %d chars",
        [count(input.message)]
    )
}

# ── Audit metadata (always populated, used for X-Ray) ────────────
# Every policy decision includes this metadata for the causal graph.
audit := {
    "session_id": input.session_id,
    "module": input.module,
    "tool_call_count": input.tool_call_count,
    "allowed": allow,
    "deny_reasons": deny_reasons,
    "policy_version": "1.0.0",
}
