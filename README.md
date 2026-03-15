# Companion Memory System

Research-grade system for evaluating and improving memory reliability in an AI companion chat product.

## Quick Start

```bash
pip install -r requirements.txt

# Run Streamlit UI (chat experience + memory inspector + eval dashboard)
cd companion-memory-system
streamlit run ui/streamlit_app.py

# Run FastAPI server
uvicorn chat_system.chat_router:app --reload

# Run improved system evaluation
python -m evals.eval_runner

# Run baseline evaluation (for before/after comparison)
python -m evals.baseline_eval_runner
```

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Problem Framing | [`architecture/problem_framing.md`](architecture/problem_framing.md) |
| 2 | System Design | [`architecture/system_design.md`](architecture/system_design.md) |
| 3 | Seed Eval Corpus (87 cases, 10 categories) | [`evals/golden_eval_suite.jsonl`](evals/golden_eval_suite.jsonl) |
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
    │
    ▼
┌─────────────────┐
│   Ingestion      │  Extract atomic facts from natural language
│   Pipeline       │  Detect corrections, classify sensitivity
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Conflict       │  Check for contradictions with existing memory
│   Resolution     │  Supersede old facts, track lineage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Memory Store   │  SQLite persistence + FAISS vector index
│   (per user)     │  User-isolated storage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval &    │  Entity match + semantic similarity
│   Ranking        │  Recency decay + confidence weighting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Response       │  recall / honest_missing / ask_confirm / general
│   Planner        │  Sensitivity-aware surfacing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generation     │  LLM or rule-based fallback
└─────────────────┘
```

## Memory Model

Each memory entry contains:
- **entity**: who/what the memory is about
- **attribute**: the property
- **value**: the value
- **memory_type**: user_stated_fact, inferred_fact, guessed_fact, corrected_fact, stale_fact, sensitive_fact, atomic_fact, summary_fact
- **confidence**: 0.0 to 1.0
- **status**: active, superseded, deleted, stale
- **sensitivity**: direct_recall, summarized_recall, ask_before_revealing, do_not_surface
- **supersedes**: links to the memory this one replaced (correction lineage)

## Evaluation

77 golden eval cases across 10 categories:
- Memory recall (10), Correction handling (8), Multi-turn continuity (8)
- Sensitive memory (8), Multi-user isolation (6), Relationship updates (7)
- Temporal reasoning (7), Uncertainty honesty (8), Contradiction resolution (7)
- Hallucination prevention (8)

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

## Key Metrics

- hallucination_rate
- memory_recall_rate
- correction_success_rate
- sensitive_memory_restraint
- multi_user_isolation
