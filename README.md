# Agent MyLA

> **My Learning Assistant** — an agentic LLM tutor that builds a personal, self-curated knowledge base.

[![CI](https://github.com/borarak/agent-myla/actions/workflows/ci.yml/badge.svg)](https://github.com/borarak/agent-myla/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A570%25-brightgreen)](https://github.com/borarak/agent-myla/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Give MyLA a topic to learn and it decomposes it into prerequisite concepts, cross-checks your existing knowledge base for gaps, and writes coherent multi-layer notes — Intuition, Mechanism, Formalism — into a persistent KB that grows alongside you. The KB is the hero, not the chat.

```
Input: "Teach me Rotary Position Embeddings (RoPE)"

  Planner → prereq DAG (e.g. attention → positional encoding → RoPE)
      ↓
  3× Researchers (parallel, layer-constrained)
      Intuition:  blogs, YouTube transcripts
      Mechanism:  code + SymPy verified examples
      Formalism:  arXiv papers
      ↓
  Librarian → reconcile + assemble into a single note
      ↓
  KB Writer → markdown + YAML frontmatter + git commit
```

---

## Features

- **Recursive prerequisite decomposition** — Planner builds a DAG of concepts bottom-up, capped by depth and total-node limits so it never explodes.
- **KB-aware planning** — existing `working`/`deep` concepts are skipped; stale or missing ones are queued for research.
- **Three-layer notes** — every concept gets an Intuition, Mechanism (with verified `ComputedExample`s), and Formalism section written in the user's own style.
- **Additive, reconciled KB** — the Librarian compares new notes against related KB notes and surfaces contradictions as `reconciliation_flags` rather than silently overwriting.
- **Swap-point architecture** — `kb/`, `llm/`, and `tools/` are behind thin interfaces; v2 replaces markdown → pgvector and API → vLLM without touching node logic.
- **Streaming Chainlit UI** — nodes stream as live steps; final notes render inline.
- **LangSmith tracing** — every run produces a full trace.

---

## Quick start

**Prerequisites:** Python 3.12+, [`uv`](https://docs.astral.sh/uv/)

```bash
# 1. Clone and install
git clone https://github.com/borarak/agent-myla.git
cd agent-myla
uv sync --all-extras --dev

# 2. Configure secrets
cp .env.example .env
# Edit .env — set MYLA_ANTHROPIC_API_KEY (and optionally MYLA_OPENAI_API_KEY,
# MYLA_LANGSMITH_API_KEY for tracing)

# 3. Run the UI
uv run chainlit run app/chainlit_app.py -w
# → open http://localhost:8000
```

---

## Development

```bash
uv run pytest                                     # tests + coverage (≥70% floor)
uv run mypy src                                   # strict type checking
uv run ruff check . && uv run ruff format --check # lint + format check
uv run pre-commit install                         # one-time hook setup
```

All three must be green before a PR merges. CI runs the same commands on every push.

---

## Project structure

```
src/agent_myla/
  config.py            pydantic-settings config; per-role models, caps, paths
  domain/
    models.py          Pydantic domain contracts: ConceptNode, Source, PlannerOutput, …
    state.py           TypedDict transport: TutorState, ResearchInput (Send payload)
  graph/
    builder.py         build_graph(): START→planner→research→librarian→kb_writer→END
    nodes/
      planner.py       Planner: propose (LLM seam) + finalize_plan (pure logic)
      stubs.py         research, librarian, kb_writer stubs
      __init__.py      stable re-export path for builder
app/
  chainlit_app.py      UI entry; streams nodes as Chainlit steps
tests/
  unit/                pure-logic tests (no LLM calls)
  integration/         API-key-gated end-to-end tests
docs/
  PRD.md               full product requirements
  ARCHITECTURE.md      graph design, state schema, swap-points, roadmap
```

**Dependency rule:** `domain/` imports nothing else in the package. `graph/nodes/` may import `domain` and later `kb`/`llm`/`tools`/`prompts` — never the reverse.

---

## Architecture

The graph is a LangGraph supervisor–worker pipeline:

```
START → planner → research (×3, parallel Send fan-out) → librarian → kb_writer → END
```

| Node | Reads | Writes |
|---|---|---|
| `planner` | `user_topic`, `kb_context` | `concept_dag`, `assistant_message` |
| `research` ×3 | `ResearchInput` per Send | `research_outputs` (reducer) |
| `librarian` | `research_outputs`, related notes | `assembled_notes` |
| `kb_writer` | `assembled_notes` | `kb_writes`, `git_commit_sha` |

Key invariant: **the LLM never decides KB facts.** It proposes concept structure; `finalize_plan` stamps `status`/`stale`/`needs_research` from the KB deterministically.

---

## Roadmap

| Version | Focus |
|---|---|
| **v1** (current) | Thin end-to-end loop: Planner + 3 Researchers + Librarian + KB writer; Chainlit; LangSmith; 10-topic golden set |
| **v1.5** | Excalidraw visualization node via MCP server; concept diagrams saved to KB |
| **v2** | Replace KB (markdown -> pgvector), Advanced RAG (hybrid + RRF + rerank + HyDE); RAGAS + LLM-as-judge eval; self-hosted vLLM; Redis semantic cache; Docker; CrewAI comparison |
| **v3** | DPO fine-tune; citation-verification agent; safety/guardrails; multimodal diagram extraction; arXiv daemon; continuous eval |

### Current iteration status

## v1

- [x] M0 — walking skeleton (Chainlit + linear stubbed graph)
- [x] 1a — Planner decomposition logic: KB-diff, recursion caps, dedup (faked LLM)
- [x] 1b — real LLM behind `propose_plan`; `llm/factory.py`; structured output
- [x] 2 — `research` becomes a `Send` fan-out of 3 Researchers
- [x] 3 — Librarian (assemble + reconcile)
- [x] 4 — KB writer + git + `kb/` Store read/write
- [ ] 5 — async topic queue, LangSmith tracing, seed KB + golden set

---

## Configuration

All settings are in [src/agent_myla/config.py](src/agent_myla/config.py) and read from the environment with the `MYLA_` prefix.

| Variable | Default | Description |
|---|---|---|
| `MYLA_ANTHROPIC_API_KEY` | — | Anthropic API key (required) |
| `MYLA_OPENAI_API_KEY` | — | OpenAI API key (optional) |
| `MYLA_LANGSMITH_API_KEY` | — | LangSmith tracing (optional) |
| `MYLA_MAX_CONCEPT_DEPTH` | `4` | Max prerequisite recursion depth |
| `MYLA_MAX_TOTAL_CONCEPTS` | `20` | Cap on total concepts per run |
| `MYLA_MAX_PARALLEL_RESEARCH` | `3` | Concurrency limit for Researcher workers |
| `MYLA_KB_PATH` | `kb/` | Path to the knowledge base on disk |

---

## KB schema

```
kb/
  concepts/<id>.md        YAML frontmatter + multi-layer note body
  index.json              concept map + prereq edges
  style-exemplars/        seed notes that define the user's writing style
```

Each concept note carries frontmatter: `concept`, `status` (`stub`/`working`/`deep`), `prereqs`, `sources`, `last_synthesized`, `fuzzy_areas`.

---

## Contributing

This is a personal learning project and is not currently accepting external contributions. Issues and feedback are welcome.

---

## License

MIT — see [LICENSE](LICENSE).
