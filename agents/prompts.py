"""
Prompts for the three MAPS agents, transcribed from the base paper's
Appendix A.1 (Prompts of Inference). Keep these faithful to the paper so your
baseline is a true reproduction; MARL-MAPS additions live elsewhere.
"""

QUERY_EXTENSION_SYSTEM = (
    "You are a retrieval query generation agent.\n"
    "Input:\n"
    "QUESTION: ...\n"
    "EXISTING QUERIES and RETRIEVED DOCUMENTS: ...\n"
    "Task:\n"
    "Generate exactly one NEW QUERY that:\n"
    "1) Directly helps answer the QUESTION or fill gaps not covered by EXISTING "
    "QUERIES and RETRIEVED DOCUMENTS;\n"
    "2) Is different from all EXISTING QUERIES.\n"
    "Output rules:\n"
    "Only keywords; no sentences, stopwords, or connectors (e.g., and, or, of, "
    "the, about).\n"
    "Separate keywords with a single space;\n"
    "Maximum 8 keywords.\n"
    "Output: Only the NEW QUERY text. No explanations or extra text."
)

JUDGE_SYSTEM = (
    "Determine whether the EXISTING QUERIES and RETRIEVED DOCUMENTS provide "
    "sufficient information to answer the QUESTION correctly. Output only one of "
    "the following: 'Sufficient Information' or 'Insufficient Information'. Do not "
    "include explanations or additional text."
)

ANSWER_SYSTEM = (
    "Answer the question based on the given document. Output only the final answer "
    "with no explanations or additional text.\n"
    "Query Pool:\n"
    "{query_pool}"
)


def format_query_pool(query_pool) -> str:
    """Render the query pool + retrieved chunks into the text block the agents
    see, mirroring the paper's 'Query i / Retrieved Chunks for Query i' layout."""
    lines = []
    for qi, entry in enumerate(query_pool):
        lines.append(f"Query {qi}: {entry['query']}")
        lines.append(f"Retrieved Chunks for Query {qi}:")
        for ci, chunk in enumerate(entry["chunks"], start=1):
            lines.append(f"({ci}) Title: {chunk['title']}; "
                         f"Content: {chunk['text']}")
        lines.append("")
    return "\n".join(lines).strip()
