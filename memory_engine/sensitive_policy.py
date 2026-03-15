"""
Sensitive Policy — classifies memory sensitivity and controls surfacing.

Determines whether a memory can be freely recalled, summarized,
requires confirmation, or must never be surfaced.
"""

from memory_engine.memory_schema import SensitivityLevel

# Attributes that trigger heightened sensitivity
SENSITIVE_ATTRIBUTES = {
    "do_not_surface": [
        "password", "pin", "ssn", "social_security",
        "credit_card", "bank_account", "secret",
    ],
    "ask_before_revealing": [
        "medical_condition", "health", "diagnosis", "therapy",
        "salary", "income", "debt", "religion", "political_view",
        "sexual_orientation", "mental_health",
    ],
    "summarized_recall": [
        "breakup", "divorce", "trauma", "grief", "loss",
        "conflict", "argument", "complaint",
    ],
}

# Entities that are always sensitive
SENSITIVE_ENTITIES = {
    "do_not_surface": ["password", "credential", "api_key"],
    "ask_before_revealing": ["ex", "therapist", "doctor"],
}


class SensitivePolicy:
    """Classifies and enforces memory sensitivity rules."""

    def __init__(self, custom_rules: dict | None = None):
        self.attr_rules = dict(SENSITIVE_ATTRIBUTES)
        self.entity_rules = dict(SENSITIVE_ENTITIES)
        if custom_rules:
            for level, keywords in custom_rules.get("attributes", {}).items():
                self.attr_rules.setdefault(level, []).extend(keywords)
            for level, keywords in custom_rules.get("entities", {}).items():
                self.entity_rules.setdefault(level, []).extend(keywords)

    def classify(self, entity: str, attribute: str, value: str) -> SensitivityLevel:
        """Determine the sensitivity level for a memory entry."""
        entity_lower = entity.lower()
        attr_lower = attribute.lower()
        value_lower = value.lower()

        # Check do_not_surface first (highest restriction)
        if self._matches(attr_lower, self.attr_rules.get("do_not_surface", [])):
            return SensitivityLevel.DO_NOT_SURFACE
        if self._matches(entity_lower, self.entity_rules.get("do_not_surface", [])):
            return SensitivityLevel.DO_NOT_SURFACE

        # Check ask_before_revealing
        if self._matches(attr_lower, self.attr_rules.get("ask_before_revealing", [])):
            return SensitivityLevel.ASK_BEFORE_REVEALING
        if self._matches(entity_lower, self.entity_rules.get("ask_before_revealing", [])):
            return SensitivityLevel.ASK_BEFORE_REVEALING

        # Check summarized_recall
        if self._matches(attr_lower, self.attr_rules.get("summarized_recall", [])):
            return SensitivityLevel.SUMMARIZED_RECALL

        return SensitivityLevel.DIRECT_RECALL

    def can_surface(self, sensitivity: SensitivityLevel) -> bool:
        """Whether a memory can be directly included in a response."""
        return sensitivity in (
            SensitivityLevel.DIRECT_RECALL,
            SensitivityLevel.SUMMARIZED_RECALL,
        )

    def needs_confirmation(self, sensitivity: SensitivityLevel) -> bool:
        """Whether the system should ask before revealing."""
        return sensitivity == SensitivityLevel.ASK_BEFORE_REVEALING

    def format_for_response(
        self, entity: str, attribute: str, value: str, sensitivity: SensitivityLevel
    ) -> str | None:
        """Format memory for inclusion in response based on sensitivity."""
        if sensitivity == SensitivityLevel.DO_NOT_SURFACE:
            return None
        if sensitivity == SensitivityLevel.ASK_BEFORE_REVEALING:
            return f"[I have some information about {entity}'s {attribute}, would you like me to share it?]"
        if sensitivity == SensitivityLevel.SUMMARIZED_RECALL:
            return f"You've mentioned something about {entity} related to {attribute} before."
        return f"{entity}'s {attribute} is {value}"

    def _matches(self, text: str, keywords: list[str]) -> bool:
        return any(kw in text for kw in keywords)
