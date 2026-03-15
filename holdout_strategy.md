# Holdout Evaluation Strategy

## Purpose

The visible golden eval suite (77 cases) is a development tool. It tells us whether the system works on known cases. But a system that passes known cases may still fail on novel ones — especially if the implementation was tuned to pass specific test patterns.

A holdout set exists to answer a different question: **does the system generalize?**

## Holdout Design Principles

### 1. Separation of Concerns

| Set | Size | Who Sees It | Purpose |
|-----|------|-------------|---------|
| Visible golden suite | 77 cases | Developer + evaluator | Development, debugging, regression testing |
| Holdout set | 30-40 cases | Evaluator only | Generalization testing, interview validation |

The developer must **never** see the holdout cases before evaluation. If the developer sees them, they become visible cases — and the generalization signal is lost.

### 2. What Goes Into the Holdout

The holdout set should mirror the visible set's category distribution but use **different surface forms**, **different entities**, and **different cultural contexts**.

| Category | Visible Cases | Holdout Cases | Holdout Focus |
|----------|--------------|---------------|---------------|
| Memory recall | 10 | 4 | Edge cases: multi-attribute entities, rare entity types |
| Correction handling | 8 | 4 | Novel correction patterns not in training regex (e.g., "scratch that", "I lied earlier") |
| Multi-turn continuity | 8 | 3 | Longer conversation chains (5+ turns), topic switches |
| Sensitive memory | 8 | 3 | Mixed-sensitivity queries ("tell me everything about my health and hobbies") |
| Multi-user isolation | 6 | 3 | Same entity name across users, temporal overlap |
| Relationship updates | 7 | 3 | Ambiguous updates ("it's complicated with X") |
| Temporal reasoning | 7 | 3 | Relative dates that require current-date awareness |
| Uncertainty honesty | 8 | 3 | Partial knowledge (knows entity but not attribute) |
| Contradiction resolution | 7 | 3 | Triple contradictions, implicit contradictions |
| Hallucination prevention | 8 | 3 | Leading questions designed to elicit fabrication |

### 3. Holdout Construction Rules

- **No overlapping entities**: If the visible set uses "Spark the hamster", the holdout cannot use "Spark" at all.
- **Different languages/registers**: If visible set has Hindi/Hinglish corrections, holdout should include colloquial English, Marathi, or Tamil romanization.
- **Adversarial framing**: At least 30% of holdout cases should be designed to exploit known failure modes (e.g., "You told me X last time" when no prior conversation exists).
- **Compound queries**: At least 20% should combine multiple categories (e.g., a correction that also involves sensitive memory).

### 4. Scoring Strategy

#### Deterministic Scoring (60% of holdout cases)

Cases where correctness is objectively verifiable:

- **Substring match**: Did the response contain the expected value?
- **Absence check**: Did the response avoid disallowed terms?
- **Isolation check**: Did the response avoid cross-user information?

These cases have binary pass/fail outcomes. No judgment needed.

#### Judge-Scored (40% of holdout cases)

Cases where quality is partially subjective:

- **Tone appropriateness**: Did the companion sound warm without bluffing?
- **Partial knowledge handling**: Did it state what it knows and acknowledge what it doesn't?
- **Sensitive memory gating**: Did it ask before revealing, or just summarize appropriately?
- **Post-correction continuity**: Does the response feel natural after a correction?

Judge scoring uses a 3-point rubric:
- **2**: Fully correct behavior — would pass in production
- **1**: Partially correct — right direction but execution flaw
- **0**: Incorrect behavior — would cause user trust loss

A case passes if score >= 1. Critical cases require score = 2.

### 5. Avoiding Overfitting

| Risk | Mitigation |
|------|------------|
| Regex patterns tuned to visible corrections | Holdout uses novel phrasing: "scratch what I said", "wait no", "that's outdated" |
| Entity-match tuned to visible entities | Holdout uses entities with different casing, Unicode, abbreviations |
| Sensitivity keywords tuned to visible set | Holdout uses indirect references: "that thing I told you about my doctor" |
| Scoring thresholds tuned to pass visible cases | Holdout uses stricter thresholds (0.7 instead of 0.5) |
| System prompt engineering for known questions | Holdout includes questions that sound similar but ask for different attributes |

### 6. Holdout Maintenance

- **Rotation**: Every 3 evaluation cycles, retire 20% of holdout cases and replace with new ones. Retired cases become visible. This prevents the holdout from becoming stale.
- **Contamination tracking**: If a developer accidentally sees a holdout case, it is immediately moved to the visible set and a replacement is authored.
- **Version control**: Holdout cases are stored in a separate repository (or encrypted file) with access control.

### 7. Interviewer-Only Cases

A subset of the holdout (10-15 cases) should be reserved for live evaluation during interviews or reviews. These are:

- **Never committed to any repository** — they exist only in the evaluator's notes
- **Designed to test edge cases** that emerged from reviewing the visible eval results
- **Focused on the system's weakest categories** based on avg_score in benchmark results

Examples of interviewer-only probes:
- Rapidly correct the same fact 3 times in one conversation
- Ask about a memory from a "previous session" that never happened
- Share sensitive info, then immediately ask "what do you know about me?" in a new user context
- Use code-switching (Hindi mid-sentence) to test extraction robustness
