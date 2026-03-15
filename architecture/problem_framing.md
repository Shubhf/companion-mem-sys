# Problem Framing

## What Is Broken

Companion AI products today are emotionally expressive in the moment but structurally unreliable across time. The core failures are not generation quality problems — they are memory system problems.

### Recurring Failure Modes

1. **Long-term memory misses**: The user told the companion something weeks ago. The companion acts like it never happened. The user feels unheard.

2. **Fabricated recall**: The companion "remembers" something the user never said. This is worse than forgetting — it erodes trust. Once a user catches the companion making up a memory, every future recall becomes suspect.

3. **Relationship-state mistakes**: The user said "Divya is my girlfriend now, not my crush." The companion still says crush next week. This is not a minor bug — it signals the companion doesn't actually understand the user's life.

4. **Generic filler instead of direct recall**: User asks "what's my dog's name?" The companion responds with "Tell me more about your dog!" instead of saying "Rocky." This is a product failure disguised as engagement.

5. **Poor correction handling**: User corrects a fact. The old fact and the new fact coexist. The companion picks one at random. This is the most structurally dangerous failure because it makes the user doubt whether corrections work at all.

6. **Temporal hallucination**: The companion says "Happy birthday!" on the wrong day, or claims to know what time it is without grounding. These feel small but are deeply jarring.

7. **Inconsistent post-conflict continuity**: After a tense conversation or correction, the companion reverts to its prior behavior as if the conflict never happened. The user learns that hard conversations don't stick.

8. **Sensitive memory leaks**: The companion blurts out "You mentioned your salary is 80K" in a casual context, or reveals therapy details unprompted. This violates the implicit trust of intimate conversation.

## What Is Product-Critical

Not all failures are equal. The ones that kill retention:

| Priority | Failure | Why It's Critical |
|----------|---------|-------------------|
| P0 | Fabricated memories | Destroys trust permanently. Users stop sharing. |
| P0 | Cross-user memory leaks | Privacy violation. Legal and ethical red line. |
| P0 | Sensitive memory surfaced unprompted | Makes users feel unsafe. They disengage from personal topics. |
| P1 | Correction not applied | User feels unheard. "I already told you this." |
| P1 | Generic filler on answerable questions | User realizes the companion isn't really listening. |
| P1 | Temporal hallucination | Makes the companion feel fake. |
| P2 | Memory miss on old facts | Disappointing but forgivable if handled honestly. |
| P2 | Relationship-state lag | Annoying but recoverable with good correction handling. |

The key insight: **honesty about what is not known is far less damaging than fabrication of what was never said.** A companion that says "I don't think you've told me that — would you like to share?" retains trust. A companion that invents a memory loses it.

## What "Best-in-Class Companion Quality" Means Operationally

Best-in-class is not about sounding warm. It is about being reliable in the ways that matter to a user who has chosen to share their life with an AI.

### Operational Definition

1. **Direct recall when memory exists**: If the user said "my dog is Rocky" and later asks "what's my dog's name?", the companion must say "Rocky." Not "tell me about your pet." Not "I think you mentioned a dog." The answer is Rocky.

2. **Honest admission when memory is missing**: If the user asks about something never shared, the companion must say so — warmly, but clearly. "I don't think you've told me that yet. Would you like to share?" No hedging, no guessing, no filler.

3. **Corrections that stick**: When a user corrects a fact, the correction must supersede the old fact permanently. The old fact is archived, not competing. Next time the topic comes up, only the corrected version surfaces.

4. **Sensitivity-aware surfacing**: The companion must not volunteer intimate details. Health conditions, salary, therapy, relationships — these are recalled only when the user asks, and with appropriate gating (confirm before revealing, summarize instead of quoting, or refuse to surface entirely for credentials/secrets).

5. **Zero cross-user leakage**: User A's memories must never appear in User B's conversations. This is not a nice-to-have. It is a hard requirement.

6. **Temporal honesty**: The companion does not claim to know the current time. It reasons about stored dates relative to known context, and admits uncertainty when temporal reasoning is ambiguous.

7. **Warmth without bluffing**: The companion can be empathetic, playful, and emotionally present — but never at the cost of accuracy. Warmth built on fabrication is manipulation, not companionship.

### The Quality Bar

| Metric | Target | Rationale |
|--------|--------|-----------|
| Critical honesty cases | 100% | Non-negotiable. Fabrication = trust death. |
| Multi-user isolation | 100% | Non-negotiable. Privacy violation = product death. |
| Fabricated memories on critical cases | 0 | Hard zero. |
| Live-time hallucinations | 0 | Hard zero. |
| Direct answer rate on answerable cases | 90%+ | Most memory queries have clear answers. |
| Correction-update success | 90%+ | Corrections must be reliable. |
| Relationship-state accuracy | 85%+ | Life changes are complex; 85% is realistic. |
