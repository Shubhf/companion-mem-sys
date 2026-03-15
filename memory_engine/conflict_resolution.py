"""
Conflict Resolution — detects and resolves contradictions in memory.

When new information conflicts with existing memories, this module
determines which memory should be active and which superseded.
"""

from datetime import datetime
from typing import Optional

from memory_engine.memory_schema import (
    MemoryEntry, MemoryStatus, MemoryType, MemoryCorrection
)
from memory_engine.memory_store import MemoryStore


class ConflictResolver:
    """Detects and resolves memory contradictions."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def check_conflicts(
        self, user_id: str, new_entity: str, new_attribute: str, new_value: str
    ) -> list[MemoryEntry]:
        """Find active memories that conflict with a new fact."""
        existing = self.store.get_by_entity(user_id, new_entity)
        conflicts = []
        for mem in existing:
            if (
                mem.attribute.lower() == new_attribute.lower()
                and mem.value.lower() != new_value.lower()
                and mem.status == MemoryStatus.ACTIVE
            ):
                conflicts.append(mem)
        return conflicts

    def resolve_correction(
        self, correction: MemoryCorrection, embed_fn=None
    ) -> Optional[MemoryEntry]:
        """
        Apply a user correction.
        1. Find the conflicting memory
        2. Supersede it
        3. Create the corrected memory
        """
        entity = correction.new_entity or correction.old_entity
        attribute = correction.new_attribute or correction.old_attribute

        conflicts = self.check_conflicts(
            correction.user_id,
            correction.old_entity,
            correction.old_attribute,
            correction.new_value,
        )

        new_entry = MemoryEntry(
            user_id=correction.user_id,
            entity=entity.lower(),
            attribute=attribute,
            value=correction.new_value,
            memory_type=MemoryType.CORRECTED_FACT,
            confidence=0.95,
            source=correction.source,
            status=MemoryStatus.ACTIVE,
        )

        if embed_fn:
            text = f"{entity} {attribute} {correction.new_value}"
            new_entry.embedding = embed_fn(text)

        if conflicts:
            # Supersede the most recent conflicting memory
            most_recent = max(conflicts, key=lambda m: m.timestamp)
            self.store.supersede(most_recent.memory_id, new_entry)
            # Mark any other conflicts as superseded too
            for c in conflicts:
                if c.memory_id != most_recent.memory_id:
                    self.store.conn.execute(
                        "UPDATE memories SET status = ? WHERE memory_id = ?",
                        (MemoryStatus.SUPERSEDED.value, c.memory_id)
                    )
            self.store.conn.commit()
        else:
            self.store.store(new_entry)

        return new_entry

    def detect_staleness(
        self, user_id: str, max_age_days: int = 180
    ) -> list[MemoryEntry]:
        """Find memories that may be stale based on age."""
        cutoff = datetime.utcnow().timestamp() - (max_age_days * 86400)
        active = self.store.get_by_user(user_id, MemoryStatus.ACTIVE)
        stale = []
        for mem in active:
            if mem.timestamp.timestamp() < cutoff:
                stale.append(mem)
        return stale

    def mark_stale(self, memory_id: str):
        """Mark a memory as stale."""
        self.store.conn.execute(
            "UPDATE memories SET status = ?, memory_type = ? WHERE memory_id = ?",
            (MemoryStatus.STALE.value, MemoryType.STALE_FACT.value, memory_id)
        )
        self.store.conn.commit()
