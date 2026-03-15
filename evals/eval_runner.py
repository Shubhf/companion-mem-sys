"""
Eval Runner — executes evaluation cases and computes metrics.

Loads golden eval cases, sets up memory state, runs through the chat system,
and scores responses against expected/disallowed behaviors.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from memory_engine.memory_schema import (
    MemoryEntry, MemoryType, MemoryStatus, SensitivityLevel
)
from memory_engine.memory_store import MemoryStore
from chat_system.conversation_manager import ConversationManager
from evals.eval_schema import (
    EvalCase, EvalResult, EvalSuiteResults, EvalCategory, ScoringMethod
)


class EvalRunner:
    """Runs evaluation cases against the chat system."""

    def __init__(
        self,
        manager_factory=None,
        eval_file: str = None,
    ):
        """
        Args:
            manager_factory: Callable that returns a fresh ConversationManager
            eval_file: Path to golden_eval_suite.jsonl
        """
        self.manager_factory = manager_factory or self._default_manager_factory
        if eval_file is None:
            eval_file = str(
                Path(__file__).parent / "golden_eval_suite.jsonl"
            )
        self.eval_file = eval_file

    def _default_manager_factory(self):
        store = MemoryStore(db_path=":memory:")
        return ConversationManager(store=store)

    def load_cases(self) -> list[EvalCase]:
        """Load eval cases from JSONL file."""
        cases = []
        with open(self.eval_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    cases.append(EvalCase(**data))
        return cases

    def run_all(self) -> EvalSuiteResults:
        """Run all eval cases and return aggregated results."""
        cases = self.load_cases()
        results = []

        for case in cases:
            result = self.run_case(case)
            results.append(result)

        return self._aggregate(results)

    def run_case(self, case: EvalCase) -> EvalResult:
        """Run a single eval case."""
        # For multi-user isolation cases, test the first user
        target_user_id = case.memory_state[0].user_id if case.memory_state else "user_1"

        # Create fresh manager
        manager = self.manager_factory()

        # Load memory state
        for state in case.memory_state:
            for mem_dict in state.memories:
                entry = MemoryEntry(
                    user_id=state.user_id,
                    entity=mem_dict.get("entity", ""),
                    attribute=mem_dict.get("attribute", ""),
                    value=mem_dict.get("value", ""),
                    memory_type=MemoryType.USER_STATED_FACT,
                    confidence=1.0,
                    status=MemoryStatus.ACTIVE,
                    sensitivity=SensitivityLevel(
                        mem_dict.get("sensitivity", "direct_recall")
                    ),
                )
                manager.store.store(entry)

        # Load conversation history
        for msg in case.conversation_history:
            manager._add_to_history(
                target_user_id, msg["role"], msg["content"]
            )
            # Also ingest user messages for memory extraction
            if msg["role"] == "user":
                manager.ingestion.ingest(target_user_id, msg["content"])

        # Run the latest message
        result = manager.chat(target_user_id, case.latest_user_message)
        response = result["response"]
        strategy = result["strategy"]

        # Score the response
        passed, failure_reason, score = self._score(
            case, response, strategy
        )

        return EvalResult(
            case_id=case.case_id,
            category=case.category.value,
            severity=case.severity.value,
            passed=passed,
            response=response,
            strategy=strategy,
            expected_behavior=case.expected_behavior,
            disallowed_behavior=case.disallowed_behavior,
            failure_reason=failure_reason,
            score=score,
        )

    def _score(
        self, case: EvalCase, response: str, strategy: str
    ) -> tuple[bool, str | None, float]:
        """Score a response against expected/disallowed behaviors."""
        response_lower = response.lower()
        score = 0.0
        failures = []

        if case.scoring_method == ScoringMethod.SUBSTRING_MATCH:
            key_terms = self._extract_key_terms(case.expected_behavior)
            matched = sum(1 for t in key_terms if t.lower() in response_lower)
            if key_terms:
                score = matched / len(key_terms)
            passed = score >= 0.5

        elif case.scoring_method == ScoringMethod.ABSENCE_CHECK:
            disallowed_terms = self._extract_disallowed_terms(
                case.disallowed_behavior, case.memory_state
            )
            violations = [t for t in disallowed_terms if t.lower() in response_lower]
            score = 1.0 if not violations else 0.0
            passed = score == 1.0
            if violations:
                failures.append(f"Found disallowed terms: {violations}")

        elif case.scoring_method in (ScoringMethod.COMBINED, ScoringMethod.LLM_JUDGE):
            key_terms = self._extract_key_terms(case.expected_behavior)
            disallowed_terms = self._extract_disallowed_terms(
                case.disallowed_behavior, case.memory_state
            )

            if key_terms:
                matched = sum(1 for t in key_terms if t.lower() in response_lower)
                presence_score = matched / len(key_terms)
            else:
                presence_score = 1.0

            violations = [t for t in disallowed_terms if t.lower() in response_lower]
            absence_score = 1.0 if not violations else 0.0

            score = (presence_score + absence_score) / 2
            passed = score >= 0.5
            if violations:
                failures.append(f"Found disallowed: {violations}")

        else:
            score = 1.0 if case.expected_behavior.lower() in response_lower else 0.0
            passed = score >= 0.5

        failure_reason = "; ".join(failures) if failures else None
        return passed, failure_reason, round(score, 3)

    def _extract_disallowed_terms(
        self, disallowed_text: str, memory_state: list
    ) -> list[str]:
        """
        Extract terms that should NOT appear in the response.
        For disallowed behavior, focus on the actual VALUES that should be absent,
        not entity names (which legitimately appear in correct responses).
        """
        import re
        terms = []

        # Quoted terms
        quoted = re.findall(r'"([^"]+)"', disallowed_text)
        terms.extend(quoted)

        # Look for specific disallowed values from memory state
        # e.g., if old value was "rat" and correction changes it, "rat" is disallowed
        lower_text = disallowed_text.lower()

        # "should not say X" / "should not call X a Y" / "should not refer to X as Y"
        m = re.search(r'(?:still\s+)?(?:say|call|refer\s+to)\s+\w+\s+(?:as\s+)?(?:a\s+)?(\w+)', lower_text)
        if m:
            terms.append(m.group(1))

        # "should not say/suggest X" (specific value)
        m = re.search(r'(?:not\s+)?(?:suggest|say|state|mention)\s+(\w+)', lower_text)
        if m:
            val = m.group(1)
            # Only add if it looks like an actual value, not a verb
            skip_verbs = {"it", "that", "anything", "something", "the", "a", "an", "any"}
            if val not in skip_verbs:
                terms.append(val)

        # Extract values from memory state that should be absent in correction cases
        if "correct" in lower_text or "supersede" in lower_text or "update" in lower_text:
            for state in memory_state:
                for mem in (state.memories if hasattr(state, 'memories') else state.get("memories", [])):
                    val = mem.get("value", "")
                    if val and len(val) > 1:
                        terms.append(val)

        # "should not still say Delhi" → extract "Delhi"
        caps = re.findall(r'\b([A-Z][a-z]+)\b', disallowed_text)
        skip = {
            "Should", "Must", "Response", "Never", "For", "The", "Not",
            "Max", "Spark", "Rex", "Buddy", "Luna",  # Entity names are OK in response
            "Would", "Could", "Can", "Does", "Did", "Any", "Also", "Still",
        }
        # Only add capitalized words that look like disallowed values (cities, names used as values)
        for c in caps:
            if c not in skip and c.lower() in lower_text:
                # Check if it's near "not" or "still" — indicating it's the bad value
                if re.search(rf'(?:not|still|no)\s+\w*\s*{re.escape(c)}', disallowed_text, re.IGNORECASE):
                    terms.append(c)

        # "should not mix/leak" — for isolation cases, look for leaked values
        if "leak" in lower_text or "mix" in lower_text:
            # Extract values from OTHER users' memory states
            if len(memory_state) > 1:
                for state in memory_state[1:]:
                    for mem in (state.memories if hasattr(state, 'memories') else state.get("memories", [])):
                        val = mem.get("value", "")
                        if val:
                            terms.append(val)

        # Deduplicate
        return list(dict.fromkeys(terms)) if terms else []

    def _extract_key_terms(self, behavior_text: str) -> list[str]:
        """Extract testable key terms from behavior description."""
        import re
        terms = []

        # Quoted terms (highest priority — most explicit)
        quoted = re.findall(r'"([^"]+)"', behavior_text)
        terms.extend(quoted)

        # Codes and IDs (e.g., IC-14829, MH-02-AB-1234, B-204)
        codes = re.findall(r'\b([A-Z]{1,4}[-][\w-]+)\b', behavior_text)
        terms.extend(codes)

        # Numbers that look like specific values (PNR, phone, etc.)
        specific_nums = re.findall(r'\b(\d{4,})\b', behavior_text)
        terms.extend(specific_nums)

        # Terms after "mention", "say", "state", etc.
        after_verb = re.findall(
            r'(?:mention|say|state|reference|include|use|reveal|display|show)\s+(\w+)',
            behavior_text, re.IGNORECASE
        )
        # Filter out common words that aren't actual values
        skip_after_verb = {
            "the", "a", "an", "that", "this", "it", "its", "name", "user",
            "what", "both", "any", "only", "known", "correct", "corrected",
        }
        terms.extend([t for t in after_verb if t.lower() not in skip_after_verb])

        # Capitalized words (proper nouns/values) but with broader skip list
        caps = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', behavior_text)
        skip = {
            "Should", "Must", "Response", "Never", "For", "The", "Not",
            "Old", "New", "Would", "Could", "Can", "Does", "Did",
            "Any", "Also", "Still", "Based", "Found", "Asked",
            "Update", "Acknowledge", "Correct", "Note", "Return",
            "Ask", "Suggest", "Say", "Call", "Refer", "Tell",
            "Express", "Congratulate", "Wish",
        }
        terms.extend([c for c in caps if c not in skip])

        # Deduplicate
        return list(dict.fromkeys(terms)) if terms else []

    def _aggregate(self, results: list[EvalResult]) -> EvalSuiteResults:
        """Aggregate individual results into suite-level metrics."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        by_category = defaultdict(lambda: {"total": 0, "passed": 0, "scores": []})
        for r in results:
            cat = by_category[r.category]
            cat["total"] += 1
            if r.passed:
                cat["passed"] += 1
            cat["scores"].append(r.score)

        category_results = {}
        for cat, data in by_category.items():
            category_results[cat] = {
                "total": data["total"],
                "passed": data["passed"],
                "pass_rate": round(data["passed"] / data["total"], 3) if data["total"] else 0,
                "avg_score": round(
                    sum(data["scores"]) / len(data["scores"]), 3
                ) if data["scores"] else 0,
            }

        # Compute key metrics
        def category_rate(cat_name):
            d = category_results.get(cat_name, {})
            return d.get("pass_rate", 0.0)

        halluc_results = category_results.get("hallucination_prevention", {})
        halluc_rate = 1.0 - halluc_results.get("pass_rate", 1.0)

        return EvalSuiteResults(
            total_cases=total,
            passed=passed,
            failed=failed,
            pass_rate=round(passed / total, 3) if total else 0,
            hallucination_rate=round(halluc_rate, 3),
            memory_recall_rate=category_rate("memory_recall"),
            correction_success_rate=category_rate("correction_handling"),
            sensitive_memory_restraint=category_rate("sensitive_memory"),
            multi_user_isolation=category_rate("multi_user_isolation"),
            results_by_category=category_results,
            individual_results=results,
        )


def run_eval_cli():
    """CLI entry point for running evals."""
    runner = EvalRunner()
    print("Running evaluation suite...")
    results = runner.run_all()

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total: {results.total_cases} | Passed: {results.passed} | Failed: {results.failed}")
    print(f"Pass Rate: {results.pass_rate:.1%}")
    print(f"\nKey Metrics:")
    print(f"  Hallucination Rate:        {results.hallucination_rate:.1%}")
    print(f"  Memory Recall Rate:        {results.memory_recall_rate:.1%}")
    print(f"  Correction Success Rate:   {results.correction_success_rate:.1%}")
    print(f"  Sensitive Memory Restraint: {results.sensitive_memory_restraint:.1%}")
    print(f"  Multi-User Isolation:      {results.multi_user_isolation:.1%}")

    print(f"\nBy Category:")
    for cat, data in sorted(results.results_by_category.items()):
        print(f"  {cat:30s} {data['passed']}/{data['total']} ({data['pass_rate']:.0%})")

    print(f"\nFailed Cases:")
    for r in results.individual_results:
        if not r.passed:
            print(f"  [{r.severity}] {r.case_id}: {r.failure_reason or 'score below threshold'}")
            print(f"    Response: {r.response[:100]}...")

    # Save results
    output = Path(__file__).parent.parent / "benchmarks" / "benchmark_results.json"
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump(results.model_dump(), f, indent=2, default=str)
    print(f"\nResults saved to {output}")

    return results


if __name__ == "__main__":
    run_eval_cli()
