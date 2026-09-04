"""The retrieval tool.

Exposing retrieval as a tool rather than as a separate graph branch is what
keeps the agent a single clean ReAct loop instead of a router - see
docs/adr/0002. It is not one of the "two backend tools" the brief asks for;
those are get_claim_status and submit_claim. This is the knowledge tool.

The instruction about exclusions in the docstring is load-bearing, not
decoration. sample_policy.md says gradual leaks and flood damage are strictly
excluded, and the default failure of a small model asked "is water damage
covered?" is a confident yes. That sentence, the ground node, and eval cases
EV-04/EV-05 are three independent defences against the single most likely
wrong answer this system can give.
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.tools import StructuredTool

from libs.contracts import Chunk, SearchPolicyArgs


class PolicySearcher(Protocol):
    """Whatever can answer a hybrid search - the retrieval service client in
    production, an in-memory index in tests."""

    async def search(self, query: str, top_k: int) -> list[Chunk]: ...


def build_policy_tool(searcher: PolicySearcher) -> StructuredTool:
    async def search_policy_documents(query: str, top_k: int = 3) -> dict[str, Any]:
        chunks = await searcher.search(query, top_k)
        return {
            "chunks": [
                {
                    "text": c.text,
                    "section_title": c.section_title,
                    "source_file": c.source_file,
                    "citation": c.citation,
                    "score": c.score,
                }
                for c in chunks
            ],
            "citations": list(dict.fromkeys(c.citation for c in chunks)),
        }

    search_policy_documents.__doc__ = """Search the OmniCare policy documents for coverage rules, limits and exclusions.

    Use this for ANY question that touches the policy document, including:
      - what is or is not covered, deductibles, caps, exclusions
      - appraisal or documentation requirements
      - "what does Section N say", "summarise section N", "read me the part
        about X" - any request to quote, summarise or explain a section
      - a question about a section that may not exist

    Always call this before answering. Never answer from memory, and never say
    a section does not exist without searching first.

    Pay close attention to exclusions in the returned text. If a section says
    something is excluded, say so plainly and say which section says it. Do not
    soften an exclusion into "may be covered" or "it depends" - a policyholder
    acting on that would be misled.

    Args:
        query: The policyholder's coverage question, rephrased for search.
            Include the concrete nouns they used ("burst pipe", "jewelry",
            "flood") rather than generic wording.
        top_k: How many policy sections to retrieve. Default 3.

    Returns:
        chunks, each with text, section_title, source_file and citation; and
        citations, the deduplicated list. Cite every section you draw on,
        exactly as given in the citation field. Do not cite a section that is
        not in this result - invented citations are stripped before the answer
        reaches the policyholder, which will make your answer look incomplete.
    """

    return StructuredTool.from_function(
        coroutine=search_policy_documents,
        name="search_policy_documents",
        description=search_policy_documents.__doc__,
        args_schema=SearchPolicyArgs,
    )
