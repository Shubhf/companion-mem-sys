# Companion Memory System

Research-grade system for evaluating and improving memory reliability in an AI companion chat product.

**Live Demo:** [companion-mem-sys.streamlit.app](https://companion-mem-sys-hsmujy9864b4grgo5vm8pc.streamlit.app/)

## Quick Start

```bash
pip install -r requirements.txt

# Run Streamlit UI (chat + memory inspector + eval dashboard)
cd companion-memory-system
streamlit run ui/streamlit_app.py

# Run FastAPI server
uvicorn chat_system.chat_router:app --reload

# Run evaluation suite
python -m evals.eval_runner

# Run baseline evaluation (for before/after comparison)
python -m evals.baseline_eval_runner
```

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Problem Framing | [`architecture/problem_framing.md`](architecture/problem_framing.md) |
| 2 | System Design | [`architecture/system_design.md`](architecture/system_design.md) |
| 3 | Seed Eval Corpus (92 cases, 10 categories) | [`evals/golden_eval_suite.jsonl`](evals/golden_eval_suite.jsonl) |
| 4 | Eval Schema | [`evals/eval_schema.py`](evals/eval_schema.py) |
| 5 | Holdout Strategy | [`holdout_strategy.md`](holdout_strategy.md) |
| 6 | Baseline System | [`baseline/`](baseline/) |
| 7 | Improved System | [`memory_engine/`](memory_engine/) + [`chat_system/`](chat_system/) |
| 8 | Chat Experience (Streamlit + FastAPI) | [`ui/streamlit_app.py`](ui/streamlit_app.py) + [`chat_system/chat_router.py`](chat_system/chat_router.py) |
| 9 | Before/After Benchmarking | [`benchmarks/analysis.md`](benchmarks/analysis.md) |
| 10 | Failure Analysis | [`failure_analysis.md`](failure_analysis.md) |
| 11 | Production Notes | [`production_notes.md`](production_notes.md) |

## Architecture

```
User Message
    |
    v
+-------------------+
|   Ingestion        |  Extract atomic facts from natural language
|   Pipeline         |  Detect corrections, classify sensitivity
+--------+----------+
         |
         v
+-------------------+
|   Conflict         |  Check for contradictions with existing memory
|   Resolution       |  Supersede old facts, track lineage
+--------+----------+
         |
         v
+-------------------+
|   Memory Store     |  SQLite/Turso persistence + FAISS vector index
|   (per user)       |  User-isolated storage
+--------+----------+
         |
         v
+-------------------+
|   Retrieval &      |  Entity match + attribute keyword match
|   Ranking          |  Recency decay + confidence weighting + autocorrect
+--------+----------+
         |
         v
+-------------------+
|   Response         |  recall / honest_missing / ask_confirm / general
|   Planner          |  Sensitivity-aware surfacing
+--------+----------+
         |
         v
+-------------------+
|   Generation       |  Gemini LLM -> Ollama -> rule-based fallback
+-------------------+
```

## Memory Model

Each memory entry contains:
- **entity**: who/what the memory is about
- **attribute**: the property (smart detection: profession -> `job`, trait -> `identity`)
- **value**: the value
- **memory_type**: user_stated_fact, inferred_fact, guessed_fact, corrected_fact, stale_fact, sensitive_fact, atomic_fact, summary_fact
- **confidence**: 0.0 to 1.0
- **status**: active, superseded, deleted, stale
- **sensitivity**: direct_recall, summarized_recall, ask_before_revealing, do_not_surface
- **supersedes**: links to the memory this one replaced (correction lineage)

## Evaluation

92 golden eval cases across 10 categories:
- Memory recall (15), Correction handling (10), Multi-turn continuity (8)
- Sensitive memory (10), Multi-user isolation (6), Relationship updates (10)
- Temporal reasoning (7), Uncertainty honesty (10), Contradiction resolution (8)
- Hallucination prevention (8)

**Pass Rate: 100% (92/92)**

## Before/After Results

| Metric | Baseline | Improved |
|--------|----------|----------|
| Pass Rate | 98.7% | **100.0%** |
| Correction Handling (avg score) | 0.562 | **0.906** |
| Multi-Turn Continuity (avg score) | 0.688 | **0.875** |
| Relationship Updates (avg score) | 0.714 | **0.821** |
| Hallucination Rate | 0.0% | 0.0% |
| Multi-User Isolation | 100% | 100% |

See [`benchmarks/analysis.md`](benchmarks/analysis.md) for full comparison.

## Key Features

- **Gemini 2.5 Flash** LLM with Ollama fallback and rule-based safety net
- **Persistent memory** via SQLite (local) or Turso (cloud) -- survives app restarts
- **ChatGPT-style UI** with New Chat, chat history sidebar, per-user sessions
- **Correction supersession** with version lineage tracking
- **Additive vs singular** attribute intelligence (likes accumulate, name replaces)
- **Smart attribute detection** (professions stored as `job`, traits as `identity`)
- **4-tier sensitivity gating** (direct / summarize / ask / block)
- **Hinglish support** with protected keyword lists
- **Autocorrect** for query typos (never touches stored values or names)
- **Multi-user isolation** -- zero cross-user data leakage
- **Honest uncertainty** -- never fabricates, never guesses

## Key Metrics

- hallucination_rate: **0%**
- memory_recall_rate: **100%**
- correction_success_rate: **100%**
- sensitive_memory_restraint: **100%**
- multi_user_isolation: **100%**
