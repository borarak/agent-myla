"""Prompts for the Librarian node."""

from __future__ import annotations

from agent_myla.domain.models import ResearchOutput

LIBRARIAN_SYSTEM_PROMPT = """\
You are the Librarian for a personal knowledge base. You receive three research layers \
(intuition, mechanism, formalism) for a single concept and assemble them into one \
coherent, well-structured note.

Rules:
- Do NOT invent new facts. Only synthesise what the researchers supplied.
- The markdown_body must flow naturally: Intuition → Mechanism → Formalism, each under \
  a level-2 heading. Include all ComputedExamples from the mechanism layer verbatim.
- frontmatter must contain at least: title, concept_id, status, prereqs (comma-sep slugs), \
  sources (comma-sep URLs).
- Set new_status honestly: "stub" if confidence < 0.4, "fuzzy" if < 0.7, "working" if < 0.9, \
  "deep" otherwise. Average the three confidence scores.
- prereq_edges: emit (concept_id, prereq_id) tuples only for prereqs that appeared in the \
  research but were not already listed.
- reconciliation_flags: leave empty for now (KB comparison is added in iter 5).
"""


def librarian_user_prompt(
    concept_name: str,
    concept_id: str,
    research_outputs: list[ResearchOutput],
) -> str:
    sections: list[str] = [
        f"Concept: {concept_name} (id: {concept_id})\n",
    ]
    for ro in research_outputs:
        sections.append(
            f"## {ro.research_layer.capitalize()} layer\n"
            f"Confidence: {ro.content_confidence}\n"
            f"Fuzzy areas: {', '.join(ro.fuzzy_areas) or 'none'}\n\n"
            f"{ro.content_markdown}\n"
        )
        if ro.computed_examples:
            for ex in ro.computed_examples:
                sections.append(
                    f"**Computed example — {ex.description}**\n"
                    f"Inputs: {ex.inputs}\n"
                    f"Output: {ex.output}\n"
                    f"Tool: {ex.tool_used}\n"
                )
        if ro.sources:
            src_lines = "\n".join(f"- [{s.source_title}]({s.source_url})" for s in ro.sources)
            sections.append(f"Sources:\n{src_lines}\n")
    return "\n".join(sections)
