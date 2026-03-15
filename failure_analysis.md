# Failure Analysis

## Current State

The improved system achieves 100% pass rate (77/77) on the visible golden eval suite. However, a perfect score on a visible set does not mean the system is production-ready. This document identifies what still fails, what could fail under novel conditions, and where hallucinations remain possible.

## Known Limitations

### 1. Response Quality Is Rule-Based, Not LLM-Generated

**What happens**: Without an LLM backend configured, the system falls back to template responses like "Based on what I remember: user's species is hamster." These are correct but not natural.

**Why it matters**: The eval suite scores on factual accuracy (substring match, absence check), not on conversational warmth or naturalness. A production companion needs both.

**Risk level**: Medium. Factual correctness is preserved, but user experience degrades.

**What would fix it**: LLM integration with memory-augmented system prompts. The response planner already selects the right strategy — the generation layer just needs a capable LLM.

### 2. Regex-Based Extraction Has Coverage Gaps

**What happens**: The ingestion pipeline uses 40+ regex patterns to extract facts and detect corrections. Novel phrasings that don't match any pattern are missed.

**Examples that would fail**:
- "Scratch that — Spark is actually a guinea pig" (no "not" pattern, no "nahi" pattern)
- "FYI I go by Daredevil now, not Rocky" (inverted correction order)
- "Lol jk about the hamster thing, he's a chinchilla" (informal retraction)
- "Between you and me, I never actually liked pizza" (implicit preference negation)

**Risk level**: High in production. Users don't follow correction templates.

**What would fix it**: LLM-based extraction as primary, with regex as fast-path fallback. The architecture already supports this (`llm_fn` parameter in ingestion) but it's not wired up.

### 3. Temporal Reasoning Is Static

**What happens**: The system stores temporal facts as raw strings ("January 2024", "next month", "2 weeks ago") but doesn't resolve them to absolute dates at storage time.

**Examples that would fail**:
- User says "I'm going to Goa next month" in January. In March, the system still says "next month."
- User says "I started keto 2 weeks ago" in June. In December, the system still says "2 weeks ago."
- "Is my birthday coming up?" requires knowing today's date — the rule-based system may not have it.

**Risk level**: Medium. Temporal staleness creates confusion but doesn't fabricate information.

**What would fix it**: Resolve relative dates to absolute dates at ingestion time. Add a staleness detector that flags temporal facts older than their natural expiry.

### 4. Sensitive Memory Boundary Cases

**What happens**: Sensitivity classification uses keyword matching (e.g., "salary", "therapy", "password"). Edge cases slip through.

**Examples that would fail**:
- "I make 2L a month" — no "salary" keyword, but clearly income information
- "My shrink says I should journal more" — "shrink" not in sensitivity keywords
- "I voted BJP last election" — political info, but no "political_view" keyword
- "I've been clean for 6 months" — sobriety context, highly sensitive, not caught

**Risk level**: High for trust. Missing sensitivity classification means the system may surface intimate facts directly.

**What would fix it**: LLM-based sensitivity classification, or a broader keyword dictionary with context-aware rules.

### 5. Multi-User Isolation Under Shared Entities

**What happens**: User isolation works at the storage layer via `user_id` scoping. But if two users share entity names (e.g., both have a pet named "Max"), the system correctly isolates them — however, the eval suite only tests straightforward isolation.

**Untested edge cases**:
- Two users in the same conversation context (e.g., a group chat scenario)
- User A mentions User B by name — should this create a memory about B?
- Admin/debug views that aggregate across users

**Risk level**: Low for current architecture (single-user sessions), high if extended to multi-party.

### 6. Contradiction Resolution Defaults to Last-Write-Wins

**What happens**: When the system detects contradicting memories (e.g., city = "Chennai" and city = "Hyderabad"), it supersedes the older one. But some contradictions are genuinely ambiguous.

**Examples that would fail**:
- User has two homes: "I live in Delhi during the week and Shimla on weekends"
- User's answer depends on context: "I like coffee in winter and chai in summer"
- Gradual change: "I'm mostly vegetarian now, but I eat fish sometimes"

**Risk level**: Medium. Forcing single-value resolution loses nuance.

**What would fix it**: Support multi-valued attributes with context tags. Allow the response planner to present both values when ambiguity is genuine.

### 7. Eval Scoring Has False Passes

**What happens**: Some cases in the benchmark results show `passed: true` but `score: 0.5`, with failure reasons like "Found disallowed: ['is']" or "Found disallowed: ['user']". These are scoring artifacts — the disallowed term extraction picks up common words that appear in the response legitimately.

**Specific cases**:
- `correct_06`: score 0.5, disallowed "is" and "user" — these are grammatical, not factual violations
- `sensitive_06`: score 0.5, disallowed "political" — but the response correctly asks before revealing
- `sensitive_08`: score 0.5, disallowed "religion" — same pattern

**Risk level**: Low for the system, but undermines eval credibility. A reviewer might question whether 100% pass rate is meaningful when some cases score 0.5.

**What would fix it**: Refine `_extract_disallowed_terms()` to filter common English words more aggressively. Or switch to LLM-judge scoring for combined cases.

### 8. No Conversation-Level Memory Decay

**What happens**: All active memories have equal status regardless of age. A fact stored 2 years ago competes equally with one stored yesterday (retrieval uses recency weighting, but storage doesn't expire facts).

**Risk level**: Low in short-lived sessions, high in long-lived companion relationships.

**What would fix it**: Periodic staleness sweeps that mark old memories as STALE unless recently accessed. The `conflict_resolution.py` already has `detect_staleness()` but it's not run automatically.

## Where Hallucinations Can Still Happen

Despite 0% hallucination rate on the visible suite, hallucinations are possible:

1. **LLM generation layer**: When an LLM is configured, it may add details beyond what the memory system provides. The response planner constrains the strategy, but the LLM may embellish within a "recall" strategy.

2. **Conversation history echo**: The `history_recall` strategy returns conversation history verbatim. If a user said something false ("I told you I'm a doctor"), the system may echo it as memory.

3. **Inference from partial facts**: If the system knows "user lives in Jaipur" and is asked "what restaurants near me?", the rule-based system correctly refuses. But an LLM might generate restaurant names based on the city.

4. **Temporal projection**: "Is my birthday coming up?" requires live date awareness. Without grounding, any answer is a guess.

## Summary of Remaining Risks

| Category | Severity | Likelihood | Mitigation Available |
|----------|----------|------------|---------------------|
| Novel correction phrasing | High | High | LLM extraction |
| Temporal staleness | Medium | High | Absolute date resolution |
| Sensitivity keyword gaps | High | Medium | LLM classification |
| Eval scoring artifacts | Low | Certain | Refine term extraction |
| LLM hallucination in generation | High | Medium | Constrained prompting |
| Multi-value attributes | Medium | Medium | Schema extension |
| Conversation history as false memory | Medium | Low | History validation |

## What We Would Ship Now vs Block

### Ship Now
- Rule-based memory system with current eval coverage
- Sensitive memory gating (4-tier)
- Multi-user isolation
- Correction handling for common patterns
- Honest missing-memory responses

### Block From Shipping
- LLM generation without hallucination guardrails
- Temporal reasoning on relative dates (needs resolution at ingestion)
- Sensitive classification without broader keyword coverage
- Any deployment without the holdout eval passing at 85%+
