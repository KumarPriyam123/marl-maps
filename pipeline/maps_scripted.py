"""
Scripted MAPS pipeline -- the Week 1 BASELINE.

Implements the base paper's inference loop (Eq. 8-11):
  Q0,C0 = R(x,k)                                   # initial retrieval (Eq. 8)
  repeat up to MAX_ROUNDS:
      q = Aq(x, Qn, Cn)                            # query extension (Eq. 9)
      Qn+1,Cn+1 = Qn + R(q)
      D = Aj(x, Qn+1, Cn+1)                        # judge (Eq. 10)
      if D == Sufficient: break
  y = Ag(x, Q*, C*)                                # answer (Eq. 11)

It returns the answer AND a full trace (via the SGWM), so Week 2's Orchestrator
and the over-search / efficiency metrics have everything they need.
"""

from __future__ import annotations
from typing import Dict, Any, List
from memory.sgwm import SGWM


def run_maps(question: str, retriever, q_agent, j_agent, a_agent,
             top_k: int = 3, max_rounds: int = 3,
             max_chunks: int = 20) -> Dict[str, Any]:
    sgwm = SGWM(question=question)

    # --- Eq. 8: seed the query pool with the raw question ------------------- #
    chunks = retriever.retrieve(question, k=top_k)
    sgwm.add_query(question, chunks)
    sgwm.log("retriever", "initial_retrieve", {"query": question,
                                               "n": len(chunks)})

    rounds = 0
    judged_sufficient = False
    for r in range(max_rounds):
        rounds = r + 1
        sgwm.Gamma_ops["rounds"] = rounds

        # --- Eq. 10: judge current pool ------------------------------------ #
        if j_agent.run(question, sgwm.query_pool):
            sgwm.log("judge", "sufficient", None)
            judged_sufficient = True
            break
        sgwm.log("judge", "insufficient", None)

        if sgwm.Gamma_ops["n_chunks"] >= max_chunks:
            sgwm.log("pipeline", "chunk_cap_reached", max_chunks)
            break

        # --- Eq. 9: extend query, retrieve, append ------------------------- #
        new_q = q_agent.run(question, sgwm.query_pool)
        new_chunks = retriever.retrieve(new_q, k=top_k)
        sgwm.add_query(new_q, new_chunks)
        sgwm.log("query_agent", "new_query", {"query": new_q,
                                              "n": len(new_chunks)})

    # --- Eq. 11: answer from the optimized pool ---------------------------- #
    answer = a_agent.run(question, sgwm.query_pool)
    sgwm.log("answer_agent", "final_answer", answer)

    return {
        "question": question,
        "answer": answer,
        "rounds": rounds,
        "judged_sufficient": judged_sufficient,
        "queries": [e["query"] for e in sgwm.query_pool],
        "n_chunks": sgwm.Gamma_ops["n_chunks"],
        "sgwm": sgwm.snapshot(),
        "trace": sgwm.H_trace,
    }
