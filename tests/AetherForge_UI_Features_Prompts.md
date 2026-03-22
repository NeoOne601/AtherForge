# AetherForge — UI Feature Implementation Prompts
## For Gemini Antigravity Sub-Agent Execution
**Features:** Collapsible CoT Reasoning Block + Actionable Suggestion Tiles

---

## DIAGNOSIS SUMMARY

### What is currently broken

**Feature 1 — Chain-of-Thought (CoT) display:**
The CognitiveRAG pipeline's Stage ⑥ generates reasoning steps and embeds them
directly inside the `response` string, like this:

```
response: "**Reasoning:** Step 1: I looked at the hydrostatic table.\nStep 2: ...\n\nThe displacement is 25,839 tonnes."
```

The frontend renders this as a single text block. The user sees reasoning
and the answer mixed together — no separation, no collapse, no visual
hierarchy. This is different from how Claude, ChatGPT, and Gemini show
a collapsed "Thinking..." block above the final answer.

**Feature 2 — Follow-up suggestions:**
The suggestion tiles rendered below each response call `setInputValue(suggestion)`
on click — they populate the text input but the user must still press Enter
to send. Gemini, Claude, and ChatGPT auto-submit on click.

---

## SUB-AGENT 1 — Backend: Separate CoT from response

### Files to attach
- `src/meta_agent.py`
- `src/modules/ragforge/cognitiverag.py`
- `src/routers/chat.py`
- `src/core/container.py`
- `tests/test_cognitiverag.py` (create if not present)

### Prompt

```
You are a senior Python backend engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge
Stack: FastAPI + LangGraph + Python 3.12

CONTEXT:
The CognitiveRAG pipeline's Stage ⑥ (Chain-of-Thought synthesis) currently
embeds reasoning text directly inside the `response` string field, mixed with
the final answer. We need to separate them into two distinct fields.

YOUR TASK — make these changes in this exact order:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Update the AgentResponse model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/meta_agent.py

Find the AgentResponse (or ChatResponse or similar Pydantic model) that is
returned by the chat endpoint. Add a new optional field:

  thinking: Optional[str] = None

This field will hold the CoT reasoning steps. It is separate from `response`
which holds only the clean final answer.

Also find how the response is built after CognitiveRAG runs. It currently looks
like one of:
  result["response"]  or  state["final_response"]  or  output.content

You will see that `response` contains both reasoning text and the answer
concatenated together. We will fix this in Step 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Separate CoT output in cognitiverag.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/modules/ragforge/cognitiverag.py

Find Stage ⑥ — the chain-of-thought synthesis node. It uses the LLM to
generate a response that includes intermediate reasoning steps before the
final answer.

The LLM prompt for this stage likely asks the model to "think step by step"
or "reason through" the answer. The full LLM output is currently assigned
directly to the response field.

Change Stage ⑥ to separate thinking from the final answer using this approach:

  full_output = llm_response.content  # the raw LLM output

  # Extract thinking if delimited (try these patterns in order):
  # Pattern A: <think>...</think> block
  # Pattern B: **Reasoning:**...followed by a blank line
  # Pattern C: Lines starting with "Step N:" until a blank line separator
  # Pattern D (fallback): first 60% of text is thinking, last 40% is answer

  thinking, clean_response = extract_cot(full_output)

  # Return both from this stage node:
  return {
    "thinking": thinking,
    "response": clean_response,
    # ... all other existing return fields unchanged
  }

Write extract_cot(text: str) -> tuple[str, str] as a pure function at the
top of the file. Try Pattern A first (most reliable), fall through to D.
Add unit tests for extract_cot in tests/test_cognitiverag.py covering:
  - text with <think>...</think> tags
  - text with **Reasoning:** prefix
  - text with "Step 1:" lines
  - plain text (fallback to 60/40 split)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Forward thinking in the REST and WS routes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: src/routers/chat.py

Find the POST /api/v1/chat endpoint handler. It currently returns:
  {"response": ..., "causal_graph": ..., "module": ..., "latency_ms": ...}

Add `thinking` to this response:
  {"response": ..., "thinking": result.get("thinking"), "causal_graph": ..., ...}

Find the WebSocket handler at /ws/chat/{session_id}. It currently streams
response tokens. Change the streaming protocol to use typed events:

  WHILE thinking tokens are being generated:
    send: {"type": "thinking", "content": chunk}

  WHEN thinking is complete and answer generation starts:
    send: {"type": "thinking_complete", "duration_ms": thinking_ms}

  WHILE answer tokens are being generated:
    send: {"type": "answer", "content": chunk}

  WHEN done:
    send: {"type": "done", "thinking": full_thinking, "causal_graph": ...}

If the pipeline does not stream separately (generates full response first,
then streams), simulate the split:
  - Send all thinking as one {"type": "thinking", "content": full_thinking}
  - Then stream the answer as {"type": "answer", "content": chunk} tokens
  - Then send {"type": "done", ...}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCEPTANCE CRITERIA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. curl test:
   curl -X POST http://localhost:8765/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"session_id":"test1","module":"ragforge","message":"how does stability work?"}' \
     | python3 -m json.tool

   Response MUST have:
   - "thinking": a non-empty string with reasoning steps
   - "response": the clean answer WITHOUT any CoT prefix

2. "response" field must NOT start with Step, Reasoning:, Let me think,
   First,, or any reasoning marker.

3. pytest tests/test_cognitiverag.py passes, including new extract_cot tests.

4. DO NOT change: OPA policy checks, SAMR faithfulness scoring, causal_graph
   structure, session handling, replay buffer writes, or any non-RAGForge module.
```

---

## SUB-AGENT 2 — Frontend: Collapsible CoT block

### Files to attach
- `frontend/src/components/ChatBubble.tsx` (or the equivalent assistant message component)
- `frontend/src/hooks/useChat.ts` (or `useWebSocket.ts`)
- `frontend/src/types/chat.ts` (or wherever ChatMessage is defined)
- `frontend/src/stores/chatStore.ts` (if it exists)

### Prompt

```
You are a senior React/TypeScript frontend engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge
Stack: React 18, TypeScript 5.5, Tailwind CSS, Shadcn/ui, Tauri 2.1
Frontend root: frontend/src/

CONTEXT:
Sub-agent 1 has added a `thinking` field to the API response and updated the
WebSocket protocol. Your job is to render the thinking as a collapsible block
above the answer — exactly like how Claude, Gemini, and ChatGPT show their
reasoning. Default: collapsed. User clicks to expand.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Add `thinking` to the ChatMessage type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Find the type/interface that defines a chat message.

Add these fields:
  thinking?: string            // CoT text from backend
  thinkingDurationMs?: number  // how long the model thought (for display)
  isThinkingStreaming?: boolean // true while CoT is still streaming in

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Create ThinkingBlock.tsx
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Create: frontend/src/components/ThinkingBlock.tsx

Props:
  content: string
  durationMs?: number
  isStreaming?: boolean

Behaviour:
  - isOpen state starts false (collapsed by default)
  - Clicking the header toggles isOpen
  - Header: chevron (▶, rotates 90° when open) + "Thinking..." if streaming,
    or "Thought for Xs" where X = Math.round(durationMs/1000) if done
  - If isStreaming and content is empty: show a spinner instead of chevron
  - Body: content text, whitespace-pre-wrap, max-h-72 overflow-y-auto
  - Transition: CSS max-height transition for smooth open/close

Tailwind classes:
  Outer wrapper:   border-l-2 border-muted-foreground/20 bg-muted/30
                   rounded-r-md pl-3 pr-2 py-2 mb-3 cursor-pointer
  Header:          flex items-center gap-2 text-xs text-muted-foreground
                   select-none
  Chevron span:    transition-transform duration-150
                   (add rotate-90 class when open)
  Spinner:         w-3 h-3 border-2 border-muted-foreground/20
                   border-t-muted-foreground/60 rounded-full animate-spin
  Body wrapper:    mt-2 pt-2 border-t border-border
  Body text:       text-xs text-muted-foreground leading-relaxed
                   whitespace-pre-wrap max-h-72 overflow-y-auto

Full component code structure:
  import { useState } from "react"

  export function ThinkingBlock({ content, durationMs, isStreaming }: Props) {
    const [isOpen, setIsOpen] = useState(false)
    const label = isStreaming
      ? "Thinking..."
      : durationMs
        ? `Thought for ${Math.round(durationMs / 1000)}s`
        : "Reasoning"

    return (
      <div className="border-l-2 ..." onClick={() => setIsOpen(o => !o)}>
        <div className="flex items-center gap-2 ...">
          {isStreaming && !content
            ? <span className="... animate-spin" />
            : <span className={`... ${isOpen ? "rotate-90" : ""}`}>▶</span>
          }
          <span>{label}</span>
        </div>
        {isOpen && (
          <div className="mt-2 pt-2 border-t ...">
            <p className="text-xs ...">{content}</p>
          </div>
        )}
      </div>
    )
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Integrate into ChatBubble.tsx
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Find the assistant message bubble component.

Before the prose/markdown response block, add:
  {message.thinking && (
    <ThinkingBlock
      content={message.thinking}
      durationMs={message.thinkingDurationMs}
      isStreaming={message.isThinkingStreaming}
    />
  )}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Handle new WS event types in useChat
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In the WebSocket message handler, add cases for the new event types:

  case "thinking":
    updateMessage(id, prev => ({
      ...prev,
      thinking: (prev.thinking ?? "") + msg.content,
      isThinkingStreaming: true,
    }))
    break

  case "thinking_complete":
    updateMessage(id, prev => ({
      ...prev,
      thinkingDurationMs: msg.duration_ms,
      isThinkingStreaming: false,
    }))
    break

  case "answer":
    // existing token accumulation logic unchanged
    break

  case "done":
    updateMessage(id, prev => ({
      ...prev,
      isStreaming: false,
      causal_graph: msg.causal_graph,
    }))
    break

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCEPTANCE CRITERIA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Every AI response with thinking data shows a collapsible block above answer.
2. Block is collapsed by default. Click toggles it open/closed smoothly.
3. Spinner shows while model is still generating the thinking.
4. Answer text below has no reasoning prefixes.
5. npm run type-check passes with no errors.
6. DO NOT change: X-Ray panel, metadata badges, session handling, module tabs.
```

---

## SUB-AGENT 3 — Frontend: Actionable suggestion tiles

### Files to attach
- The suggestion tiles component (search frontend/src/ for "suggestion", "followup", "prompt chip")
- `frontend/src/components/ChatBubble.tsx`
- The main chat view/page component
- `frontend/src/hooks/useChat.ts`

### Prompt

```
You are a senior React/TypeScript frontend engineer working on AetherForge.
Repo: github.com/NeoOne601/AtherForge
Stack: React 18, TypeScript 5.5, Tailwind CSS, Shadcn/ui, Tauri 2.1
Frontend root: frontend/src/

CONTEXT:
After each AI response, suggestion tiles appear. Clicking currently fills
the input but does NOT submit. We need them to auto-submit on click.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Find the suggestion tiles component
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Search frontend/src/ for the component rendering suggestion tiles.
Find the current onClick: it calls setInputValue(suggestion) or similar.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Make suggestions auto-submit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Update the component's props to accept an onSubmit function:

  interface Props {
    suggestions: string[]
    onSubmit: (text: string) => void  // submits the message directly
  }

Change onClick:
  BEFORE: onClick={() => setInputValue(suggestion)}
  AFTER:  onClick={() => handleClick(suggestion)}

  const [submitted, setSubmitted] = useState(false)
  const handleClick = (s: string) => {
    if (submitted) return
    setSubmitted(true)
    onSubmit(s)
  }
  if (submitted) return null  // hide all tiles after one is clicked

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Add ↗ arrow icon to each tile
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Update tile button JSX:
  <button className="... flex items-center gap-1.5 group hover:border-primary ...">
    <span>{suggestion}</span>
    <span className="text-muted-foreground/50 group-hover:text-primary text-xs
      transition-colors">↗</span>
  </button>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Pass onSubmit from parent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Find where the suggestion component is rendered. Pass the actual send function:

  <SuggestedPrompts
    suggestions={message.suggestions}
    onSubmit={sendMessage}   // the function that submits a message
  />

onSubmit must: set input value + send immediately (not just fill the input box).
Also: disable tiles if isStreaming is true for the latest message.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCEPTANCE CRITERIA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Clicking a tile immediately sends that text as a new user message.
2. All tiles disappear after one is clicked.
3. Each tile has a visible ↗ arrow on the right.
4. Tiles are not clickable while a response is still streaming.
5. npm run type-check passes. npm run lint passes.
6. DO NOT change suggestion content, X-Ray panel, or any other component.
```

---

## EXECUTION ORDER FOR GEMINI ANTIGRAVITY

```
Run order:
  1. Sub-agent 1 (backend) — must complete first, defines the `thinking` field
  2. Sub-agent 2 (CoT UI) — depends on Sub-agent 1's backend changes
  3. Sub-agent 3 (suggestions) — can run in parallel with Sub-agent 2

Integration test after all three complete:
  In RAGForge module, ask: "what is the displacement at 8.17m?"

  Verify:
  ✓ A collapsed "Thought for Xs" block appears above the answer
  ✓ Clicking the block expands the reasoning steps
  ✓ The answer text below has no reasoning prefix
  ✓ Follow-up suggestion tiles have ↗ arrows
  ✓ Clicking a tile submits immediately without pressing Enter
  ✓ X-Ray panel still works normally
  ✓ Metadata badges (module, latency, fidelity) unchanged
```

---

*AetherForge UI Features Brief — CoT Collapsible Block + Actionable Suggestions*
*Total files to modify: 6. New files to create: 1 (ThinkingBlock.tsx)*
