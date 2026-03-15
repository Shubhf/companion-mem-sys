"""
Memory Schema — defines structured memory types, entry model, and enums.

Each memory is an atomic fact about a user, with provenance tracking,
confidence scoring, correction lineage, and sensitivity classification.
"""

from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class MemoryType(str, Enum):
    """How the memory was derived."""
    USER_STATED_FACT = "user_stated_fact"       # User explicitly said it
    INFERRED_FACT = "inferred_fact"             # Derived from context
    GUESSED_FACT = "guessed_fact"               # Low-confidence guess
    CORRECTED_FACT = "corrected_fact"           # Correction of prior memory
    STALE_FACT = "stale_fact"                   # Outdated information
    SENSITIVE_FACT = "sensitive_fact"           # Requires access control
    ATOMIC_FACT = "atomic_fact"                 # Single indivisible fact
    SUMMARY_FACT = "summary_fact"              # Aggregation of multiple facts


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory entry."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    STALE = "stale"


class SensitivityLevel(str, Enum):
    """Controls how memory can be surfaced in responses."""
    DIRECT_RECALL = "direct_recall"            # Can be stated directly
    SUMMARIZED_RECALL = "summarized_recall"    # Surface only in summary form
    ASK_BEFORE_REVEALING = "ask_before_revealing"  # Confirm before surfacing
    DO_NOT_SURFACE = "do_not_surface"          # Never surface in responses


class Severity(str, Enum):
    """Severity level for eval cases."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryEntry(BaseModel):
    """A single atomic memory about a user."""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    entity: str                    # Who/what the memory is about (e.g., "Spark")
    attribute: str                 # The property (e.g., "species")
    value: str                     # The value (e.g., "hamster")
    memory_type: MemoryType = MemoryType.USER_STATED_FACT
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "user_message"   # Where the memory came from
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: MemoryStatus = MemoryStatus.ACTIVE
    sensitivity: SensitivityLevel = SensitivityLevel.DIRECT_RECALL
    supersedes: Optional[str] = None  # memory_id of the entry this replaces
    embedding: Optional[list[float]] = None  # For retrieval

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MemoryQuery(BaseModel):
    """Query parameters for memory retrieval."""
    user_id: str
    query_text: str
    entity: Optional[str] = None
    top_k: int = 5
    include_superseded: bool = False


class MemoryCorrection(BaseModel):
    """Represents a correction to an existing memory."""
    user_id: str
    old_entity: str
    old_attribute: str
    new_entity: Optional[str] = None
    new_attribute: Optional[str] = None
    new_value: str
    source: str = "user_correction"
