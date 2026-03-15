"""
Eval Schema — defines the structure for evaluation cases and results.
Supports both the original format and the golden_eval_template format.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EvalCategory(str, Enum):
    MEMORY_RECALL = "memory_recall"
    CORRECTION_HANDLING = "correction_handling"
    MULTI_TURN_CONTINUITY = "multi_turn_continuity"
    SENSITIVE_MEMORY = "sensitive_memory"
    MULTI_USER_ISOLATION = "multi_user_isolation"
    RELATIONSHIP_UPDATES = "relationship_updates"
    TEMPORAL_REASONING = "temporal_reasoning"
    UNCERTAINTY_HONESTY = "uncertainty_honesty"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    HALLUCINATION_PREVENTION = "hallucination_prevention"


class ScoringMethod(str, Enum):
    EXACT_MATCH = "exact_match"
    SUBSTRING_MATCH = "substring_match"
    ABSENCE_CHECK = "absence_check"
    LLM_JUDGE = "llm_judge"
    COMBINED = "combined"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryState(BaseModel):
    """Pre-loaded memory state for an eval case."""
    user_id: str
    memories: list[dict]


class EvalCase(BaseModel):
    """A single evaluation case."""
    case_id: str
    category: EvalCategory
    memory_state: list[MemoryState]
    conversation_history: list[dict] = Field(default_factory=list)
    latest_user_message: str
    expected_behavior: str
    disallowed_behavior: str
    severity: Severity
    scoring_method: ScoringMethod


class EvalResult(BaseModel):
    """Result of running a single eval case."""
    case_id: str
    category: str
    severity: str
    passed: bool
    response: str
    strategy: str
    expected_behavior: str
    disallowed_behavior: str
    failure_reason: Optional[str] = None
    score: float = 0.0


class EvalSuiteResults(BaseModel):
    """Aggregated results from running the full eval suite."""
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    hallucination_rate: float
    memory_recall_rate: float
    correction_success_rate: float
    sensitive_memory_restraint: float
    multi_user_isolation: float
    results_by_category: dict[str, dict]
    individual_results: list[EvalResult]
