"""The three specialized agents from MAPS, as thin wrappers over the backend.

Each agent:
  - formats the current state (question + query pool) into its prompt,
  - calls the shared LLM backend,
  - parses the output into a clean, typed result.

These are deliberately simple in Week 1 (prompt-only). Week 2 adds confidence
extraction and the Orchestrator that decides *which* agent runs *when*.
"""

from __future__ import annotations
import re
from agents.prompts import (
    QUERY_EXTENSION_SYSTEM, JUDGE_SYSTEM, ANSWER_SYSTEM, format_query_pool,
)


class QueryExtensionAgent:
    def __init__(self, backend):
        self.backend = backend

    def run(self, question: str, query_pool) -> str:
        user = f"Question: {question}\nQuery Pool:\n{format_query_pool(query_pool)}"
        out = self.backend.generate(user, system=QUERY_EXTENSION_SYSTEM,
                                    max_tokens=32, temperature=0.0)
        # Enforce the output rule: keywords only, max 8.
        kws = re.findall(r"[A-Za-z0-9']+", out)
        return " ".join(kws[:8]) if kws else question


class JudgeAgent:
    def __init__(self, backend):
        self.backend = backend

    def run(self, question: str, query_pool) -> bool:
        """Returns True if the judge says context is SUFFICIENT."""
        user = f"Question: {question}\nQuery Pool:\n{format_query_pool(query_pool)}"
        out = self.backend.generate(user, system=JUDGE_SYSTEM,
                                    max_tokens=8, temperature=0.0)
        return "insufficient" not in out.lower() and "sufficient" in out.lower()


class AnswerAgent:
    def __init__(self, backend):
        self.backend = backend

    def run(self, question: str, query_pool) -> str:
        system = ANSWER_SYSTEM.format(query_pool=format_query_pool(query_pool))
        user = f"Question: {question}"
        out = self.backend.generate(user, system=system,
                                    max_tokens=64, temperature=0.0)
        # Answer agent is instructed to output only the answer; trim stray prose.
        return out.strip().splitlines()[0].strip() if out.strip() else ""
