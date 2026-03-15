"""
Baseline Eval Runner — runs the same 77 golden eval cases against
the baseline system to produce before/after comparison numbers.

The baseline system:
- Stores raw text chunks (no structured extraction)
- Retrieves by recency (no semantic search without embeddings)
- No correction tracking — old and new facts coexist
- No sensitivity awareness — surfaces everything equally
- No honest-missing strategy — echoes context or gives generic response
"""

import json
from pathlib import Path
from collections import defaultdict

from baseline.baseline_memory import BaselineMemory
from baseline.baseline_chat import BaselineChat
from evals.eval_schema import EvalCase, EvalResult, ScoringMethod


class BaselineEvalRunner:
    """Runs eval cases against the baseline chat system."""

    def __init__(self, eval_file: str = None):
        if eval_file is None:
            eval_file = str(
                Path(__file__).parent / "golden_eval_suite.jsonl"
            )
        self.eval_file = eval_file

    def load_cases(self) -> list[EvalCase]:
        cases = []
        with open(self.eval_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    cases.append(EvalCase(**data))
        return cases

    def run_all(self) -> dict:
        cases = self.load_cases()
        results = []

        for case in cases:
            result = self.run_case(case)
            results.append(result)

        return self._aggregate(results)

    def run_case(self, case: EvalCase) -> dict:
        """Run a single case through the baseline system."""
        target_user_id = (
            case.memory_state[0].user_id if case.memory_state else "user_1"
        )

        # Fresh baseline system per case
        chat = BaselineChat()

        # Load memory state as raw text
        for state in case.memory_state:
            for mem_dict in state.memories:
                text = (
                    f"{mem_dict.get('entity', '')} "
                    f"{mem_dict.get('attribute', '')} "
                    f"is {mem_dict.get('value', '')}"
                )
                chat.memory.store(state.user_id, f"User said: {text}")

        # Load conversation history
        for msg in case.conversation_history:
            if msg["role"] == "user":
                chat.memory.store(
                    target_user_id, f"User: {msg['content']}"
                )
            else:
                chat.memory.store(
                    target_user_id, f"Assistant: {msg['content']}"
                )

        # Run the message
        result = chat.chat(target_user_id, case.latest_user_message)
        response = result["response"]

        # Score
        passed, failure_reason, score = self._score(case, response)

        return {
            "case_id": case.case_id,
            "category": case.category.value,
            "severity": case.severity.value,
            "passed": passed,
            "response": response,
            "strategy": "baseline_rag",
            "expected_behavior": case.expected_behavior,
            "disallowed_behavior": case.disallowed_behavior,
            "failure_reason": failure_reason,
            "score": score,
        }

    def _score(self, case, response) -> tuple:
        """Score using same logic as improved eval runner."""
        import re
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
            violations = [
                t for t in disallowed_terms if t.lower() in response_lower
            ]
            score = 1.0 if not violations else 0.0
            passed = score == 1.0
            if violations:
                failures.append(f"Found disallowed: {violations}")

        elif case.scoring_method in (
            ScoringMethod.COMBINED, ScoringMethod.LLM_JUDGE
        ):
            key_terms = self._extract_key_terms(case.expected_behavior)
            disallowed_terms = self._extract_disallowed_terms(
                case.disallowed_behavior, case.memory_state
            )

            if key_terms:
                matched = sum(
                    1 for t in key_terms if t.lower() in response_lower
                )
                presence_score = matched / len(key_terms)
            else:
                presence_score = 1.0

            violations = [
                t for t in disallowed_terms if t.lower() in response_lower
            ]
            absence_score = 1.0 if not violations else 0.0

            score = (presence_score + absence_score) / 2
            passed = score >= 0.5
            if violations:
                failures.append(f"Found disallowed: {violations}")
        else:
            score = (
                1.0
                if case.expected_behavior.lower() in response_lower
                else 0.0
            )
            passed = score >= 0.5

        failure_reason = "; ".join(failures) if failures else None
        return passed, failure_reason, round(score, 3)

    def _extract_key_terms(self, behavior_text: str) -> list[str]:
        import re
        terms = []
        quoted = re.findall(r'"([^"]+)"', behavior_text)
        terms.extend(quoted)

        after_verb = re.findall(
            r'(?:mention|say|state|reference|include|use)\s+(\w+)',
            behavior_text, re.IGNORECASE,
        )
        skip = {
            "the", "a", "an", "that", "this", "it", "its", "name",
            "user", "what", "both", "any", "only", "known", "correct",
        }
        terms.extend([t for t in after_verb if t.lower() not in skip])

        caps = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', behavior_text)
        skip_caps = {
            "Should", "Must", "Response", "Never", "For", "The", "Not",
            "Old", "New", "Would", "Could", "Can", "Does", "Did",
            "Any", "Also", "Still", "Based", "Found", "Asked",
            "Update", "Acknowledge", "Correct", "Note", "Return",
            "Ask", "Suggest", "Say", "Call", "Refer", "Tell",
            "Express", "Congratulate", "Wish",
        }
        terms.extend([c for c in caps if c not in skip_caps])
        return list(dict.fromkeys(terms)) if terms else []

    def _extract_disallowed_terms(self, disallowed_text, memory_state):
        import re
        terms = []
        quoted = re.findall(r'"([^"]+)"', disallowed_text)
        terms.extend(quoted)

        lower_text = disallowed_text.lower()

        m = re.search(
            r'(?:still\s+)?(?:say|call|refer\s+to)\s+\w+\s+(?:as\s+)?(?:a\s+)?(\w+)',
            lower_text,
        )
        if m:
            terms.append(m.group(1))

        m = re.search(
            r'(?:not\s+)?(?:suggest|say|state|mention)\s+(\w+)', lower_text
        )
        if m:
            val = m.group(1)
            skip_verbs = {
                "it", "that", "anything", "something", "the", "a", "an", "any",
            }
            if val not in skip_verbs:
                terms.append(val)

        if "correct" in lower_text or "supersede" in lower_text or "update" in lower_text:
            for state in memory_state:
                mems = (
                    state.memories
                    if hasattr(state, "memories")
                    else state.get("memories", [])
                )
                for mem in mems:
                    val = mem.get("value", "") if isinstance(mem, dict) else getattr(mem, "value", "")
                    if val and len(val) > 1:
                        terms.append(val)

        caps = re.findall(r'\b([A-Z][a-z]+)\b', disallowed_text)
        skip = {
            "Should", "Must", "Response", "Never", "For", "The", "Not",
            "Max", "Spark", "Rex", "Buddy", "Luna",
            "Would", "Could", "Can", "Does", "Did", "Any", "Also", "Still",
        }
        for c in caps:
            if c not in skip and c.lower() in lower_text:
                if re.search(
                    rf'(?:not|still|no)\s+\w*\s*{re.escape(c)}',
                    disallowed_text,
                    re.IGNORECASE,
                ):
                    terms.append(c)

        if "leak" in lower_text or "mix" in lower_text:
            if len(memory_state) > 1:
                for state in memory_state[1:]:
                    mems = (
                        state.memories
                        if hasattr(state, "memories")
                        else state.get("memories", [])
                    )
                    for mem in mems:
                        val = mem.get("value", "") if isinstance(mem, dict) else getattr(mem, "value", "")
                        if val:
                            terms.append(val)

        return list(dict.fromkeys(terms)) if terms else []

    def _aggregate(self, results: list[dict]) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        failed = total - passed

        by_category = defaultdict(
            lambda: {"total": 0, "passed": 0, "scores": []}
        )
        for r in results:
            cat = by_category[r["category"]]
            cat["total"] += 1
            if r["passed"]:
                cat["passed"] += 1
            cat["scores"].append(r["score"])

        category_results = {}
        for cat, data in by_category.items():
            category_results[cat] = {
                "total": data["total"],
                "passed": data["passed"],
                "pass_rate": round(
                    data["passed"] / data["total"], 3
                ) if data["total"] else 0,
                "avg_score": round(
                    sum(data["scores"]) / len(data["scores"]), 3
                ) if data["scores"] else 0,
            }

        def category_rate(cat_name):
            d = category_results.get(cat_name, {})
            return d.get("pass_rate", 0.0)

        halluc_results = category_results.get(
            "hallucination_prevention", {}
        )
        halluc_rate = 1.0 - halluc_results.get("pass_rate", 1.0)

        return {
            "system": "baseline",
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 3) if total else 0,
            "hallucination_rate": round(halluc_rate, 3),
            "memory_recall_rate": category_rate("memory_recall"),
            "correction_success_rate": category_rate("correction_handling"),
            "sensitive_memory_restraint": category_rate("sensitive_memory"),
            "multi_user_isolation": category_rate("multi_user_isolation"),
            "results_by_category": category_results,
            "individual_results": results,
        }


def run_baseline_eval():
    """Run baseline evaluation and save results."""
    runner = BaselineEvalRunner()
    print("Running baseline evaluation...")
    results = runner.run_all()

    print(f"\n{'='*60}")
    print(f"BASELINE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total: {results['total_cases']} | Passed: {results['passed']} | Failed: {results['failed']}")
    print(f"Pass Rate: {results['pass_rate']:.1%}")
    print(f"\nKey Metrics:")
    print(f"  Hallucination Rate:        {results['hallucination_rate']:.1%}")
    print(f"  Memory Recall Rate:        {results['memory_recall_rate']:.1%}")
    print(f"  Correction Success Rate:   {results['correction_success_rate']:.1%}")
    print(f"  Sensitive Memory Restraint: {results['sensitive_memory_restraint']:.1%}")
    print(f"  Multi-User Isolation:      {results['multi_user_isolation']:.1%}")

    print(f"\nBy Category:")
    for cat, data in sorted(results['results_by_category'].items()):
        print(f"  {cat:30s} {data['passed']}/{data['total']} ({data['pass_rate']:.0%})")

    print(f"\nFailed Cases:")
    for r in results['individual_results']:
        if not r['passed']:
            print(f"  [{r['severity']}] {r['case_id']}: {r['failure_reason'] or 'score below threshold'}")
            print(f"    Response: {r['response'][:100]}...")

    # Save
    output = Path(__file__).parent.parent / "benchmarks" / "baseline_results.json"
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output}")

    return results


if __name__ == "__main__":
    run_baseline_eval()
