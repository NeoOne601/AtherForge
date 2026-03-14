# AetherForge v1.0 — src/core/grammar.py
from __future__ import annotations

from typing import Any


class GrammarGenerator:
    """
    Generates GBNF (Guided BNF) grammars for llama-cpp-python.
    Forces the model to output valid JSON matching a specific schema.
    """

    @staticmethod
    def generate_tool_grammar(tools: list[dict[str, Any]]) -> str:
        """
        Create a GBNF grammar that restricts output to a single tool-call JSON object.
        We keep argument values generic JSON to stay robust across tools while still
        constraining the model to valid structure and known tool names.
        """
        if not tools:
            return ""

        tool_names = " | ".join(f'"\\"{tool["name"]}\\""' for tool in tools)
        gbnf = [
            "root ::= ws tool_call ws",
            'tool_call ::= "{" ws "\\"name\\"" ws ":" ws tool_name ws "," ws "\\"arguments\\"" ws ":" ws object ws "}"',
            f"tool_name ::= {tool_names}",
            'object ::= "{" ws (member (ws "," ws member)*)? ws "}"',
            'member ::= string ws ":" ws value',
            'array ::= "[" ws (value (ws "," ws value)*)? ws "]"',
            'value ::= string | number | boolean | null | object | array',
            'string ::= "\\"" chars "\\""',
            'chars ::= "" | char chars',
            'char ::= [^"\\\\\\x00-\\x1F] | "\\\\" escape',
            'escape ::= ["\\\\/bfnrt] | "u" hex hex hex hex',
            'hex ::= [0-9a-fA-F]',
            'number ::= "-"? int frac? exp?',
            'int ::= "0" | [1-9] [0-9]*',
            'frac ::= "." [0-9]+',
            'exp ::= [eE] [+-]? [0-9]+',
            'boolean ::= "true" | "false"',
            'null ::= "null"',
            'ws ::= [ \\t\\n\\r]*',
        ]
        return "\n".join(gbnf)

    @staticmethod
    def generate_synthesis_grammar() -> str:
        """
        Grammar for the synthesis pass (reasoning + answer).
        Forces the <think>...</think> structure.
        """
        return (
            'root   ::= think answer\nthink  ::= "<think>" [^<]* "</think>"\nanswer ::= [^\\x00]*'
        )
