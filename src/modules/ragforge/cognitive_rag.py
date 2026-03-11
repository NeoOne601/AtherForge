# AetherForge v1.0 — src/modules/ragforge/cognitive_rag.py
# ─────────────────────────────────────────────────────────────────
# CognitiveRAG™ — Thinking Retrieval Pipeline for Edge Devices
#
# A 7-stage reasoning pipeline that makes RAGForge genuinely THINK
# before answering:
#   ① Query Understanding    — classify query type
#   ② Query Decomposition    — break complex Qs into sub-questions
#   ② HyDE (alt path)        — hypothetical doc for vague queries
#   ③ Multi-path Hybrid Search — dense + sparse per sub-query
#   ④ Evidence Scoring        — rank chunks by sub-question relevance
#   ⑤ Chain-of-Thought        — step-by-step reasoning through evidence
#   ⑥ Self-Verification       — check answer against evidence
#   ⑦ Iterative Re-retrieval  — refine if self-verification fails
#
# Key constraint: runs on 8GB edge devices. All stages reuse the
# already-loaded BitNet model (~950 extra tokens per query, ~0.5s).
# Zero extra RAM. Zero cloud. Fully offline.
#
# Inspired by: Self-RAG, Corrective RAG, HyDE, MiniRAG
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import structlog
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

logger = structlog.get_logger("aetherforge.cognitiverag")

# ── Query types ───────────────────────────────────────────────────
QueryType = Literal["FACTUAL", "COMPARATIVE", "SYNTHESIS", "VAGUE"]


@dataclass
class ThinkingTrace:
    """Captures the full reasoning trace for observability / X-Ray mode."""
    query_type: QueryType = "FACTUAL"
    sub_questions: list[str] = field(default_factory=list)
    hyde_hypothesis: str = ""
    evidence_chunks: int = 0
    reasoning_chain: str = ""
    grounding_score: float = 1.0  # 0.0 (unsupported) to 1.0 (fully supported)
    verification_passed: bool = True
    retrieval_rounds: int = 1
    total_tokens_used: int = 0
    latency_ms: float = 0.0


class CognitiveRAG:
    """
    7-stage thinking retrieval pipeline.

    Accepts a callable LLM function and search function so it can
    reuse the already-loaded BitNet model and vector store — no extra
    memory footprint.
    """

    def __init__(
        self,
        llm_fn: Callable[[list[Any], int | None, float | None], str],
        search_fn: Callable[..., list[Document]],
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        """
        Args:
            llm_fn: Function matching MetaAgent._run_llm_sync signature
            search_fn: Function matching MetaAgent._hybrid_search signature
            embedding_fn: Optional function to embed text (for HyDE)
        """
        self.llm = llm_fn
        self.search = search_fn
        self.embed = embedding_fn

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def think_and_answer(
        self,
        query: str,
        source_filter: str | list[str] | None = None,
        max_retries: int = 1,
    ) -> tuple[str, list[Document], ThinkingTrace]:
        """
        Execute the full CognitiveRAG pipeline.

        Returns:
            (answer_text, evidence_docs, thinking_trace)
        """
        t0 = time.perf_counter()
        trace = ThinkingTrace()

        # ── Stage 1: Query Understanding ─────────────────────────
        trace.query_type = self._classify_query(query)
        logger.info("CognitiveRAG ① QueryType: %s for '%s...'",
                    trace.query_type, query[:60])

        # ── Stage 2: Decompose or HyDE based on query type ──────
        search_queries: list[str] = []

        if trace.query_type in ("COMPARATIVE", "SYNTHESIS"):
            sub_qs = self._decompose_query(query)
            trace.sub_questions = sub_qs
            search_queries = sub_qs
            logger.info("CognitiveRAG ② Decomposed into %d sub-questions", len(sub_qs))

        elif trace.query_type == "VAGUE":
            hypothesis = self._hyde_generate(query)
            trace.hyde_hypothesis = hypothesis
            search_queries = [hypothesis, query]  # search with both
            logger.info("CognitiveRAG ② HyDE hypothesis generated (%d chars)",
                       len(hypothesis))
        else:
            # FACTUAL — direct search
            search_queries = [query]

        # ── Stage 3: Multi-path Hybrid Search ────────────────────
        all_docs = self._multi_path_search(search_queries, source_filter)
        logger.info("CognitiveRAG ③ Retrieved %d unique chunks from %d queries",
                    len(all_docs), len(search_queries))

        if not all_docs:
            trace.latency_ms = (time.perf_counter() - t0) * 1000
            return (
                "No relevant documents found. Please upload documents first.",
                [],
                trace,
            )

        # ── Stage 4: Evidence Scoring ────────────────────────────
        scored_docs = self._score_evidence(query, all_docs)
        top_docs = scored_docs[:5]  # keep top 5 after scoring (reduced from 8 to prevent context overload)
        trace.evidence_chunks = len(top_docs)
        logger.info("CognitiveRAG ④ Top %d evidence chunks scored", len(top_docs))

        # ── Stage 5: Chain-of-Thought Synthesis ──────────────────
        answer, reasoning = self._chain_of_thought(query, top_docs, trace)
        trace.reasoning_chain = reasoning
        logger.info("CognitiveRAG ⑤ CoT synthesis complete (%d chars)", len(answer))

        # ── Stage 6: Self-Verification ───────────────────────────
        grounding_score = self._self_verify(query, answer, top_docs)
        trace.grounding_score = grounding_score
        trace.verification_passed = grounding_score > 0.0
        logger.info("CognitiveRAG ⑥ Self-verification: %s (score=%.1f)",
                    "PASSED ✓" if trace.verification_passed else "FAILED ✗",
                    grounding_score)

        # ── Stage 7: Iterative Re-retrieval (if verification fails)
        if not trace.verification_passed and max_retries > 0:
            logger.info("CognitiveRAG ⑦ Re-retrieving with refined query...")
            trace.retrieval_rounds += 1
            refined_query = self._refine_query(query, answer)
            answer, top_docs, sub_trace = self.think_and_answer(
                refined_query,
                source_filter=source_filter,
                max_retries=max_retries - 1,
            )
            trace.verification_passed = sub_trace.verification_passed

        trace.latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("CognitiveRAG pipeline complete — %.0fms | type=%s | "
                    "evidence=%d | verified=%s | rounds=%d",
                    trace.latency_ms, trace.query_type, trace.evidence_chunks,
                    trace.verification_passed, trace.retrieval_rounds)

        return answer, top_docs, trace

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Query Understanding
    # ─────────────────────────────────────────────────────────────

    def _classify_query(self, query: str) -> QueryType:
        """
        Lightweight query classification using BitNet.
        ~50 tokens output. Uses constrained generation for reliability.
        """
        messages = [
            SystemMessage(content=(
                "Classify this question into EXACTLY ONE category. "
                "Reply with ONLY the category name, nothing else.\n\n"
                "Categories:\n"
                "FACTUAL — asking for specific facts, definitions, names, dates, numbers\n"
                "COMPARATIVE — comparing two or more things, asking about differences/similarities\n"
                "SYNTHESIS — asking for summary, overview, analysis, or combining multiple concepts\n"
                "VAGUE — unclear, abstract, or very broad question that needs interpretation"
            )),
            HumanMessage(content=query),
        ]

        result = self.llm(messages, max_tokens=10, temperature=0.0)
        result = result.strip().upper()

        # Extract the classification from the response
        for qt in ("FACTUAL", "COMPARATIVE", "SYNTHESIS", "VAGUE"):
            if qt in result:
                return qt  # type: ignore[return-value]

        # Default to FACTUAL for unknown classifications
        return "FACTUAL"

    # ─────────────────────────────────────────────────────────────
    # STAGE 2a: Query Decomposition
    # ─────────────────────────────────────────────────────────────

    def _decompose_query(self, query: str) -> list[str]:
        """
        Break a complex query into 2-4 atomic sub-questions.
        ~100 tokens output.
        """
        messages = [
            SystemMessage(content=(
                "Break this complex question into 2-4 simpler, self-contained "
                "sub-questions that together answer the original. "
                "Output ONLY the sub-questions, one per line, numbered 1-4. "
                "No explanations."
            )),
            HumanMessage(content=query),
        ]

        result = self.llm(messages, max_tokens=200, temperature=0.1)

        # Parse numbered lines
        sub_questions = []
        for line in result.strip().split("\n"):
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if cleaned and len(cleaned) > 10:
                sub_questions.append(cleaned)

        # Fallback: if decomposition failed, use original
        if not sub_questions:
            return [query]

        return sub_questions[:4]  # cap at 4

    # ─────────────────────────────────────────────────────────────
    # STAGE 2b: HyDE — Hypothetical Document Embeddings
    # ─────────────────────────────────────────────────────────────

    def _hyde_generate(self, query: str) -> str:
        """
        Generate a hypothetical answer that captures the query's intent.
        This bridges the semantic gap between short queries and long documents.
        ~150 tokens output.
        """
        messages = [
            SystemMessage(content=(
                "You are a research paper expert. Given a vague question, write "
                "a short paragraph (3-4 sentences) that would be a plausible "
                "answer found in a research paper. Do NOT say 'I don't know'. "
                "Write as if you are quoting from an actual document. "
                "Be specific and use technical language."
            )),
            HumanMessage(content=query),
        ]

        return self.llm(messages, max_tokens=200, temperature=0.3)

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: Multi-path Hybrid Search
    # ─────────────────────────────────────────────────────────────

    def _multi_path_search(
        self,
        queries: list[str],
        source_filter: str | list[str] | None = None,
    ) -> list[Document]:
        """
        Run hybrid search for each sub-query and deduplicate results.
        Returns a merged, deduplicated list of all retrieved chunks.
        """
        seen_ids: set[str] = set()
        all_docs: list[Document] = []

        for q in queries:
            try:
                docs = self.search(
                    query=q,
                    k=6,
                    source_filter=source_filter,
                )
                for doc in docs:
                    doc_id = doc.metadata.get("chunk_id", doc.page_content[:80])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            except Exception as e:
                logger.warning("Multi-path search failed for sub-query '%s...': %s",
                             q[:50], e)

        return all_docs

    # ─────────────────────────────────────────────────────────────
    # STAGE 4: Evidence Scoring
    # ─────────────────────────────────────────────────────────────

    def _score_evidence(
        self,
        query: str,
        docs: list[Document],
    ) -> list[Document]:
        """
        Score each chunk's relevance to the original query.
        Uses a fast BitNet call to rate relevance 1-5.
        Returns docs sorted by relevance (highest first).

        For efficiency, we batch all chunks into a single LLM call.
        """
        if len(docs) <= 3:
            return docs  # too few to bother scoring

        # Build a compact evidence list for scoring
        evidence_list = []
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:300].replace("\n", " ")
            evidence_list.append(f"[{i+1}] {snippet}")
        evidence_text = "\n".join(evidence_list)

        messages = [
            SystemMessage(content=(
                "You are rating evidence relevance. Given a question and numbered "
                "evidence snippets, return ONLY the snippet numbers ranked from "
                "MOST to LEAST relevant. Format: comma-separated numbers. "
                "Example: 3,1,5,2,4"
            )),
            HumanMessage(content=(
                f"Question: {query}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"Rank (most to least relevant):"
            )),
        ]

        result = self.llm(messages, max_tokens=50, temperature=0.0)

        # Parse ranking
        try:
            # Extract numbers from the response
            nums = [int(n.strip()) for n in re.findall(r'\d+', result)]
            ranked_docs = []
            seen = set()
            for n in nums:
                idx = n - 1  # 1-indexed to 0-indexed
                if 0 <= idx < len(docs) and idx not in seen:
                    ranked_docs.append(docs[idx])
                    seen.add(idx)

            # Append any docs that weren't ranked
            for i, doc in enumerate(docs):
                if i not in seen:
                    ranked_docs.append(doc)

            return ranked_docs
        except Exception:
            # If parsing fails, return original order
            return docs

    _RAG_PROMPT_VARIANTS = {
        "v1": (
            "You are RAGForge CognitiveRAG — a precise document intelligence "
            "assistant that THINKS before answering.\n\n"
            "CRITICAL RULES:\n"
            "1. Answer EXCLUSIVELY from the Evidence below. NEVER use your prior knowledge or training data.\n"
            "2. If the EXACT answer or supporting facts cannot be found in the Evidence, you MUST output precisely: 'The provided documents do not contain this information.' Do not attempt to guess or hallucinate.\n"
            "3. For specific facts, cite as [1], [2], etc.\n"
            "4. NEVER guess or embellish.\n\n"
            "FORMATTING RULES (strictly follow):\n"
            "- Use **bold** for key terms and section titles.\n"
            "- Use bullet points (- ) for lists of items.\n"
            "- Use numbered lists (1. 2. 3.) for steps or ranked items.\n"
            "- Use a blank line between paragraphs.\n"
            "- Keep responses concise: 1-3 sentences for FACTUAL, structured sections for SYNTHESIS.\n"
            "- DO NOT produce one long run-on paragraph.\n"
            "- For visual elements (figures, tables, charts): describe in structured bullet points covering "
            "  title, axes/headers, key data points, and main takeaway.\n\n"
            "{cot_instruction}\n\n"
            "Evidence:\n{evidence_text}"
        ),
        "v2": (
            "You are RAGForge, an uncompromisingly analytical AI.\n\n"
            "MANDATORY INSTRUCTIONS:\n"
            "Failure to follow these instructions will result in system error.\n"
            "1. Base your answer strictly on the provided 'Evidence'.\n"
            "2. Under no circumstance should you rely on pre-trained knowledge outside the provided Evidence.\n"
            "3. Make liberal use of inline citations (like [1]) directly pointing to the source.\n"
            "4. If there is insufficient data, simply state: 'I cannot answer this based on the current document.'\n\n"
            "OUTPUT SHIELDING:\n"
            "Your objective is high faithfulness. Prefer quoting relevant parts of the text.\n\n"
            "{cot_instruction}\n\n"
            "Evidence Blocks:\n{evidence_text}"
        )
    }

    def think_and_answer(
        self,
        query: str,
        source_filter: str | list[str] | None = None,
        max_retries: int = 1,
        prompt_variant: str = "v1",
    ) -> tuple[str, list[Document], ThinkingTrace]:
        """
        Execute the full CognitiveRAG pipeline.

        Returns:
            (answer_text, evidence_docs, thinking_trace)
        """
        t0 = time.perf_counter()
        trace = ThinkingTrace()

        # ── Stage 1: Query Understanding ─────────────────────────
        trace.query_type = self._classify_query(query)
        logger.info("CognitiveRAG ① QueryType: %s for '%s...'",
                    trace.query_type, query[:40])

        # ── Stage 2: HyDE / Query Decomposition ──────────────────
        sub_queries = [query]
        if trace.query_type == "VAGUE" and self.embed:
            sub_queries.append(self._hyde_generate(query))
            trace.hyde_hypothesis = sub_queries[-1]
            logger.info("CognitiveRAG ② HyDE hyp: %s...", trace.hyde_hypothesis[:40])
        elif trace.query_type in ("COMPARATIVE", "SYNTHESIS"):
            decomposed = self._decompose_query(query)
            if decomposed:
                sub_queries.extend(decomposed)
                trace.sub_questions = decomposed
                logger.info("CognitiveRAG ② Decomposed into %d sub-queries", len(decomposed))

        # ── Stage 3: Multi-path Hybrid Search ────────────────────
        all_candidate_docs: list[Document] = []
        for sq in set(sub_queries):
            docs = self.search(
                query=sq,
                filter_source=source_filter,
                k=6,
                semantic_ratio=0.7
            )
            all_candidate_docs.extend(docs)
        logger.info("CognitiveRAG ③ Retrieved %d total candidate chunks", len(all_candidate_docs))

        # ── Stage 4: Evidence Scoring & Deduplication ────────────
        scored_evidence = self._score_evidence(query, all_candidate_docs)
        if not scored_evidence:
            trace.latency_ms = (time.perf_counter() - t0) * 1000
            error_msg = ("I could not find any relevant information in the provided "
                         "documents to answer your question.")
            return error_msg, [], trace

        trace.evidence_chunks = len(scored_evidence)
        logger.info("CognitiveRAG ④ Filtered down to %d high-quality chunks", trace.evidence_chunks)

        # ── Stage 5: Chain-of-Thought Generation ─────────────────
        evidence_parts = []
        for i, doc in enumerate(scored_evidence):
            meta = doc.metadata
            source = meta.get("source", "Unknown")
            page = meta.get("page", "?")
            section = meta.get("section", "")
            citation = f"[{i+1}] {source} | p.{page}"
            if section:
                citation += f" | §{section[:60]}"
            evidence_parts.append(f"{citation}\n{doc.page_content}")

        evidence_text = "\n\n".join(evidence_parts)

        # Adapt the CoT prompt based on query type
        cot_instruction = self._get_cot_instruction(trace.query_type)
        
        # Load the selected prompt variant
        prompt_template = self._RAG_PROMPT_VARIANTS.get(prompt_variant, self._RAG_PROMPT_VARIANTS["v1"])
        filled_prompt = prompt_template.format(
            cot_instruction=cot_instruction,
            evidence_text=evidence_text
        )

        messages = [
            SystemMessage(content=filled_prompt),
            HumanMessage(content=query),
        ]        docs: list[Document],
        trace: ThinkingTrace,
    ) -> tuple[str, str]:
        """
        Force the LLM to reason step-by-step through the evidence.
        Returns (final_answer, reasoning_chain).
        """
        # Build evidence context with citations
        evidence_parts = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            source = meta.get("source", "Unknown")
            page = meta.get("page", "?")
            section = meta.get("section", "")
            citation = f"[{i+1}] {source} | p.{page}"
            if section:
                citation += f" | §{section[:60]}"
            evidence_parts.append(f"{citation}\n{doc.page_content}")

        evidence_text = "\n\n".join(evidence_parts)

        # Adapt the CoT prompt based on query type
        cot_instruction = self._get_cot_instruction(trace.query_type)

        messages = [
            SystemMessage(content=(
                "You are RAGForge CognitiveRAG — a precise document intelligence "
                "assistant that THINKS before answering.\n\n"
                "CRITICAL RULES:\n"
                "1. Answer EXCLUSIVELY from the Evidence below. NEVER use your prior knowledge or training data.\n"
                "2. If the EXACT answer or supporting facts cannot be found in the Evidence, you MUST output precisely: 'The provided documents do not contain this information.' Do not attempt to guess or hallucinate.\n"
                "3. For specific facts, cite as [1], [2], etc.\n"
                "4. NEVER guess or embellish.\n\n"
                "FORMATTING RULES (strictly follow):\n"
                "- Use **bold** for key terms and section titles.\n"
                "- Use bullet points (- ) for lists of items.\n"
                "- Use numbered lists (1. 2. 3.) for steps or ranked items.\n"
                "- Use a blank line between paragraphs.\n"
                "- Keep responses concise: 1-3 sentences for FACTUAL, structured sections for SYNTHESIS.\n"
                "- DO NOT produce one long run-on paragraph.\n"
                "- For visual elements (figures, tables, charts): describe in structured bullet points covering "
                "  title, axes/headers, key data points, and main takeaway.\n\n"
                f"{cot_instruction}\n\n"
                f"Evidence:\n{evidence_text}"
            )),
            HumanMessage(content=query),
        ]

        result = self.llm(messages, max_tokens=900, temperature=0.1)

        # Extract reasoning chain and final answer using XML tags
        reasoning = ""
        answer = result.strip()
        
        # Try to extract the <think> block
        reasoning_match = re.search(r'<think>(.*?)</think>', result, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            
        # Try to extract the <answer> block
        answer_match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        elif reasoning_match:
            # If there was a reasoning block but no strict <answer> block, 
            # assume everything after </think> is the final answer
            parts = re.split(r'</think>', result, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer = parts[-1].strip()

        # We return (clean_answer, raw_reasoning). 
        # The formatting into <details> tags happens in the caller (MetaAgent)
        # so SAMR-lite can score ONLY the clean_answer.
        return answer, reasoning

    def _get_cot_instruction(self, query_type: QueryType) -> str:
        """Return query-type-specific chain-of-thought + formatting instructions."""
        instructions = {
            "FACTUAL": (
                "CRITICAL: You MUST begin your response with <think>.\n"
                "Think step by step inside <think> and </think> tags:\n"
                "STEP 1: Identify which evidence chunk(s) contain the answer.\n"
                "STEP 2: Extract the exact relevant information.\n"
                "STEP 3: State the answer concisely with citation(s).\n"
                "Then format your final answer inside <answer> and </answer> tags as:\n"
                "  - 1-3 sentences maximum for simple facts\n"
                "  - Bullet points if listing multiple items\n"
                "  - Bold the key finding\n"
            ),
            "COMPARATIVE": (
                "CRITICAL: You MUST begin your response with <think>.\n"
                "Think step by step inside <think> and </think> tags:\n"
                "STEP 1: Identify evidence about each item being compared.\n"
                "STEP 2: List the key similarities and differences found in the evidence.\n"
                "STEP 3: Synthesize a clear comparison.\n"
                "Then format your final answer inside <answer> and </answer> tags with:\n"
                "  **Similarities:** (bullet list)\n"
                "  **Differences:** (bullet list)\n"
                "  **Summary:** (1-2 sentences)\n"
            ),
            "SYNTHESIS": (
                "CRITICAL: You MUST begin your response with <think>.\n"
                "Think step by step inside <think> and </think> tags:\n"
                "STEP 1: Identify the main themes across all evidence chunks.\n"
                "STEP 2: Group related evidence together.\n"
                "STEP 3: Build a structured summary, citing each source.\n"
                "Then format your final answer inside <answer> and </answer> tags with markdown headers, e.g.:\n"
                "  **Overview** (2-3 sentences)\n"
                "  **Key Findings** (numbered list)\n"
                "  **Conclusion** (1-2 sentences)\n"
            ),
            "VAGUE": (
                "CRITICAL: You MUST begin your response with <think>.\n"
                "Think step by step inside <think> and </think> tags:\n"
                "STEP 1: Interpret what the user is most likely asking about.\n"
                "STEP 2: Find the most relevant evidence for that interpretation.\n"
                "STEP 3: Provide a helpful, formatted answer with citations.\n"
                "Format your final answer in <answer> tags as concise bullet points.\n"
            ),
        }
        return instructions.get(query_type, instructions["FACTUAL"])

    # ─────────────────────────────────────────────────────────────
    # STAGE 6: Self-Verification
    # ─────────────────────────────────────────────────────────────

    def _self_verify(
        self,
        query: str,
        answer: str,
        docs: list[Document],
    ) -> float:
        """
        Check if the generated answer is actually supported by the evidence.
        Returns True if verified, False if the answer may be hallucinated.
        ~150 tokens output.
        """
        # Concatenate top evidence snippets for verification
        evidence_snippets = "\n".join(
            doc.page_content[:200] for doc in docs[:5]
        )

        messages = [
            SystemMessage(content=(
                "You are a strict fact-checker. Given an Answer and Evidence, "
                "determine if the Answer is FULLY SUPPORTED by the Evidence.\n\n"
                "Reply with ONLY one word:\n"
                "SUPPORTED — if every claim in the answer appears in the evidence\n"
                "UNSUPPORTED — if the answer makes claims NOT found in the evidence\n"
                "PARTIAL — if some claims are supported but others are not"
            )),
            HumanMessage(content=(
                f"Answer: {answer[:400]}\n\n"
                f"Evidence: {evidence_snippets}"
            )),
        ]

        result = self.llm(messages, max_tokens=10, temperature=0.0)
        result = result.strip().upper()

        score = 1.0
        if "UNSUPPORTED" in result:
            score = 0.0
        elif "PARTIAL" in result:
            score = 0.5
            
        return score

    # ─────────────────────────────────────────────────────────────
    # STAGE 7: Query Refinement (for re-retrieval)
    # ─────────────────────────────────────────────────────────────

    def _refine_query(self, original_query: str, failed_answer: str) -> str:
        """
        Generate a refined search query when self-verification fails.
        Uses the failed answer to understand what information is missing.
        """
        messages = [
            SystemMessage(content=(
                "The following answer was generated but could not be verified "
                "against the source documents. Write a more specific search "
                "query that would find the correct information. "
                "Reply with ONLY the refined query, nothing else."
            )),
            HumanMessage(content=(
                f"Original question: {original_query}\n"
                f"Failed answer: {failed_answer[:200]}\n"
                f"Refined query:"
            )),
        ]

        result = self.llm(messages, max_tokens=100, temperature=0.2)
        refined = result.strip()

        # Fallback to original if refinement is empty or too short
        if len(refined) < 10:
            return original_query

        return refined
