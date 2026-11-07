"""
Shared Global Working Memory (SGWM) skeleton.

This is the persistent state your proposal centers on (report Table 3.1). In
Week 1 the scripted pipeline already *populates* it on every step so that the
"complete state at every decision" property is real and loggable. In Week 2 the
Orchestrator will *read* this to choose actions (SPECIALIST_ACTIVATE,
SURGICAL_MICRO_RETRIEVAL, NEGOTIATE, EARLY_STOP).

The six components map 1:1 to the report:
  Q_emb       -> Semantic Anchor        (embedding of the original query)
  H_trace     -> History & Traceability (ordered log of agent activations)
  C_currency  -> Decision Signal        (confidence scores; filled in Week 2)
  Omega_ctx   -> Knowledge State        (retrieved evidence + query pool)
  Psi_query   -> Query State            (counts / complexity)
  Gamma_ops   -> Cost State             (rounds, retrieved-chunk count, tokens)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SGWM:
    question: str
    Q_emb: Optional[Any] = None                       # Semantic Anchor
    H_trace: List[Dict[str, Any]] = field(default_factory=list)  # History
    C_currency: Dict[str, float] = field(default_factory=dict)   # Confidence
    Omega_ctx: List[Dict[str, Any]] = field(default_factory=list)  # query pool
    Psi_query: Dict[str, int] = field(default_factory=dict)      # query state
    Gamma_ops: Dict[str, float] = field(default_factory=lambda: {
        "rounds": 0, "n_chunks": 0, "n_queries": 0})             # cost state

    # --- helpers the pipeline uses to keep state coherent ------------------- #
    def log(self, agent: str, action: str, detail: Any = None):
        self.H_trace.append({"step": len(self.H_trace),
                             "agent": agent, "action": action, "detail": detail})

    def add_query(self, query: str, chunks: List[Dict]):
        self.Omega_ctx.append({"query": query, "chunks": chunks})
        self.Gamma_ops["n_queries"] = len(self.Omega_ctx)
        self.Gamma_ops["n_chunks"] = sum(len(e["chunks"]) for e in self.Omega_ctx)
        self.Psi_query["n_queries"] = self.Gamma_ops["n_queries"]

    @property
    def query_pool(self) -> List[Dict[str, Any]]:
        return self.Omega_ctx

    def snapshot(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "Gamma_ops": dict(self.Gamma_ops),
            "Psi_query": dict(self.Psi_query),
            "C_currency": dict(self.C_currency),
            "n_trace_steps": len(self.H_trace),
            "queries": [e["query"] for e in self.Omega_ctx],
        }
