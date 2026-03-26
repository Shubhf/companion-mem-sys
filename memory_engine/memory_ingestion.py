"""
Memory Ingestion — extracts atomic facts from user messages.

Uses LLM to decompose natural language into structured (entity, attribute, value)
triples. Detects corrections and flags sensitive content.
"""

import json
import re
from datetime import datetime
from typing import Optional

from memory_engine.memory_schema import (
    MemoryEntry, MemoryType, MemoryStatus, SensitivityLevel
)
from memory_engine.memory_store import MemoryStore
from memory_engine.sensitive_policy import SensitivePolicy

# Words that indicate the message is a question, not a fact
QUESTION_STARTERS = {
    "what", "when", "where", "who", "whom", "how", "why", "which",
    "do", "does", "did", "can", "could", "will", "would",
    "is", "are", "am", "was", "were", "tell", "should",
    "have", "has", "suggest", "recommend",
}


class MemoryIngestionPipeline:
    """Extracts structured memories from conversational text."""

    def __init__(
        self,
        store: MemoryStore,
        embed_fn=None,
        llm_extract_fn=None,
    ):
        self.store = store
        self.embed_fn = embed_fn
        self.sensitive_policy = SensitivePolicy()

        # Build extraction chain: LLM (if available) → rule-based fallback
        self._llm_fn = llm_extract_fn
        self._rule_fn = self._rule_based_extract

        # Try to auto-detect Gemini extractor if no llm_extract_fn provided
        if not self._llm_fn:
            try:
                from memory_engine.llm_extractor import create_gemini_extractor
                self._llm_fn = create_gemini_extractor()
            except Exception:
                pass

        self.llm_extract_fn = self._chained_extract

    def _chained_extract(self, message: str) -> list[dict]:
        """Try LLM extraction first, fall back to rule-based."""
        if self._llm_fn:
            try:
                results = self._llm_fn(message)
                if results:  # LLM returned facts
                    return results
            except Exception:
                pass  # Fall through to rule-based
        return self._rule_fn(message)

    def ingest(
        self, user_id: str, message: str, source: str = "user_message"
    ) -> list[MemoryEntry]:
        """
        Main ingestion entry point.
        1. Extract facts from message
        2. Detect corrections
        3. Check sensitivity
        4. Store with embeddings
        """
        raw_facts = self.llm_extract_fn(message)
        entries = []

        for fact in raw_facts:
            entity = fact.get("entity", "").lower().strip()
            attribute = fact.get("attribute", "").strip().lower()
            value = fact.get("value", "").strip()
            is_correction = fact.get("is_correction", False)

            if not entity or not attribute or not value:
                continue

            # Clean up trailing filler words from values
            value = re.sub(r'\s+(?:now|ab|abhi|bhi|hai|hain|tha|thi)$', '', value, flags=re.IGNORECASE).strip()
            # Clean up leading/trailing punctuation and commas
            value = value.strip('.,!?; ')
            fact["value"] = value

            if not value:
                continue

            sensitivity = self.sensitive_policy.classify(entity, attribute, value)
            confidence = self._compute_confidence(fact, source)
            memory_type = MemoryType.USER_STATED_FACT

            entry = MemoryEntry(
                user_id=user_id,
                entity=entity,
                attribute=attribute,
                value=value,
                memory_type=memory_type,
                confidence=confidence,
                source=source,
                sensitivity=sensitivity,
                status=MemoryStatus.ACTIVE,
            )

            # Compute embedding
            if self.embed_fn:
                text = f"{entity} {attribute} {value}"
                entry.embedding = self.embed_fn(text)

            # Check for exact duplicate (same entity+attribute+value already active)
            existing = self.store.get_by_entity(user_id, entity)
            if any(
                m.attribute.lower() == attribute.lower()
                and m.value.lower() == value.lower()
                and m.status == MemoryStatus.ACTIVE
                for m in existing
            ):
                # Already stored — skip
                continue

            # Handle correction: supersede conflicting memories
            if is_correction:
                self._handle_correction(user_id, entity, attribute, entry)
            else:
                conflict = self._find_conflict(existing, attribute)
                if conflict:
                    self.store.supersede(conflict.memory_id, entry)
                else:
                    self.store.store(entry)

            entries.append(entry)

        return entries

    def _handle_correction(
        self, user_id: str, entity: str, attribute: str, new_entry: MemoryEntry
    ):
        """Find and supersede the old memory, store the correction."""
        existing = self.store.get_by_entity(user_id, entity)

        # Try exact attribute match first
        old = self._find_conflict(existing, attribute)

        # If no exact match, try broader match: any identity/species/type attribute
        if not old:
            identity_attrs = {"identity", "species", "type", "kind", "breed", "role"}
            if attribute in identity_attrs:
                for mem in existing:
                    if mem.attribute.lower() in identity_attrs and mem.status == MemoryStatus.ACTIVE:
                        old = mem
                        break

        # Do NOT blindly supersede unrelated attributes.
        # Only supersede when we found a matching attribute.

        if old:
            new_entry.memory_type = MemoryType.CORRECTED_FACT
            self.store.supersede(old.memory_id, new_entry)
        else:
            new_entry.memory_type = MemoryType.CORRECTED_FACT
            self.store.store(new_entry)

    # Attributes that are additive — multiple values can coexist
    ADDITIVE_ATTRIBUTES = {
        "likes", "hobby", "hobbies", "interest", "interests",
        "allergies", "allergy", "language", "languages",
        "skill", "skills", "sibling", "siblings",
    }

    # Attributes that are singular — new value replaces old
    SINGULAR_ATTRIBUTES = {
        "name", "city", "age", "job", "birthday", "partner",
        "relationship_status", "car", "identity", "species",
        "type", "kind", "breed", "role", "favorite_color",
        "favorite_food", "favorite_movie", "pet_name",
        "roommate", "manager", "gym_buddy", "best_friend",
        "living_situation", "diet", "morning_drink",
        "dog_name", "cat_name",
    }

    def _find_conflict(
        self, existing: list[MemoryEntry], attribute: str
    ) -> Optional[MemoryEntry]:
        """
        Find an active memory that conflicts with a new fact.
        Additive attributes (likes, hobbies) only conflict on same value.
        Singular attributes (name, city) conflict on same attribute.
        """
        attr_lower = attribute.lower()
        identity_attrs = {"identity", "species", "type", "kind", "breed", "role"}

        # Additive attributes: only conflict if exact same value exists
        if attr_lower in self.ADDITIVE_ATTRIBUTES:
            return None  # Never conflict — allow multiple values

        for mem in existing:
            mem_attr = mem.attribute.lower()
            if mem.status != MemoryStatus.ACTIVE:
                continue
            # Exact attribute match
            if mem_attr == attr_lower:
                return mem
            # Identity group match
            if mem_attr in identity_attrs and attr_lower in identity_attrs:
                return mem
        return None

    def _compute_confidence(self, fact: dict, source: str) -> float:
        """Assign confidence based on source and extraction method."""
        base = 1.0 if source == "user_message" else 0.7
        if fact.get("is_correction"):
            base = min(base, 0.95)
        if fact.get("hedged"):
            base *= 0.6
        return round(base, 2)

    # Hinglish question words that indicate questions, not facts
    HINGLISH_QUESTION_WORDS = {
        "kya", "kaun", "kahan", "kab", "kaise", "kitna", "kitni", "kitne",
        "kiska", "kiski", "kiske", "konsa", "konsi",
        "batav", "batao", "bata", "bolo", "boldo", "bolna",
        "whom",
    }

    def _is_question(self, text: str) -> bool:
        """Check if text is a question rather than a statement."""
        text = text.strip()
        if text.endswith("?"):
            return True
        first_word = text.split()[0].lower().rstrip("'s") if text.split() else ""
        if first_word in QUESTION_STARTERS:
            return True
        # Hinglish questions: "Vuskaa naam batav", "Mera naam kya hai"
        words = set(w.lower() for w in text.split())
        if words & self.HINGLISH_QUESTION_WORDS:
            return True
        return False

    def _rule_based_extract(self, message: str) -> list[dict]:
        """
        Rule-based fact extraction.
        Handles many natural language patterns for stating facts and corrections.
        """
        message = message.strip()

        # Split multi-sentence messages
        # Split on ". " / "! " / "? " first
        sentences = re.split(r'[.!?]\s+', message.rstrip('.!? '))
        # Further split on " and " and ", " to handle compound sentences
        # e.g. "My name is X, I like coffee" or "My name is X and I live in Y"
        expanded = []
        for s in sentences:
            # Split on ", I " or ", my " or " and I " or " and my "
            parts = re.split(r',\s+(?=[Ii]\s|[Mm]y\s|[Mm]er[aie]\s)|\s+and\s+(?=[Ii]\s|[Mm]y\s|[Mm]er[aie]\s)', s)
            expanded.extend(parts)
        sentences = expanded if len(expanded) > 1 else sentences
        if len(sentences) == 1:
            sentences = [message]

        # If the ENTIRE message is a single-sentence question, skip
        if len(sentences) == 1 and self._is_question(message):
            return []

        all_facts = []

        # First pass: check for correction patterns across entire message
        # But only if the message looks like a correction (not a compound intro)
        if not re.search(r'\band\b', message, re.IGNORECASE) or re.search(
            r'(?:not|nahi|actually|no\b|isn\'t|quit|sold|moved|stopped|broke)',
            message, re.IGNORECASE
        ):
            correction_facts = self._extract_corrections(message)
            if correction_facts:
                return correction_facts

        # Second pass: extract facts from each sentence/clause
        for sent in sentences:
            sent = sent.strip()
            if not sent or self._is_question(sent):
                continue
            facts = self._extract_single_sentence(sent)
            all_facts.extend(facts)

        # Detect hedging
        hedge_words = ["i think", "maybe", "probably", "shayad", "lagta hai"]
        for fact in all_facts:
            if any(h in message.lower() for h in hedge_words):
                fact["hedged"] = True

        return all_facts

    def _extract_corrections(self, message: str) -> list[dict]:
        """Detect and extract correction patterns."""
        msg_lower = message.lower()

        # Pattern: "X is not Y. X is Z" / "X nahi hai Y. X hai Z"
        m = re.search(
            r'(\w+)\s+(?:mera\s+)?(?:is\s+not|isn\'t|nahi\s+hai|nahi)\s+'
            r'(?:a\s+|an\s+|my\s+|mera\s+|meri\s+)?(\w+)[\.,!]?\s*'
            r'(?:\1|it\'s|it\s+is|he\'s|she\'s|woh|wo)?\s*'
            r'(?:is\s+|hai\s+)?(?:a\s+|an\s+|my\s+|mera\s+|meri\s+)?(\w[\w\s]*?)[\.,!?]?$',
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": m.group(1).strip().lower(),
                "attribute": "identity",
                "value": m.group(3).strip(),
                "is_correction": True,
            }]

        # Pattern: "No no, X is not a Y. X is my Z."
        m = re.search(
            r'(?:no\s+(?:no\s*,?\s*)?)?(\w+)\s+(?:is\s+not|isn\'t|nahi\s+hai)\s+'
            r'(?:a\s+|an\s+|my\s+)?(\w+)[\.,!]?\s*'
            r'\1\s+(?:is|hai)\s+(?:a\s+|an\s+|my\s+|mera\s+|meri\s+)?(.+?)[\.,!?]?$',
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": m.group(1).strip().lower(),
                "attribute": "identity",
                "value": m.group(3).strip(),
                "is_correction": True,
            }]

        # Pattern: "X mera Y nahi hai. X Z hai."
        m = re.search(
            r'(\w+)\s+(?:mera|meri|mere)\s+(\w+)\s+nahi\s+hai[\.,!]?\s*'
            r'(\w+)\s+(\w+)\s+hai',
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": m.group(3).strip().lower(),
                "attribute": "identity",
                "value": m.group(4).strip(),
                "is_correction": True,
            }]

        # Pattern: "Actually, X is Y" / "No, X is Y"
        m = re.search(
            r'(?:actually|no|nahi|nah|wait|sorry|correction)[,\s]+\s*'
            r'(?:I\s+)?(?:moved?\s+to\s+|live\s+in\s+|\'m\s+in\s+)(.+?)[\.,!?]?$',
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "city",
                "value": m.group(1).strip(),
                "is_correction": True,
            }]

        m = re.search(
            r'(?:actually|no|nahi|nah|wait|sorry|correction)[,\s]+\s*'
            r'(?:my\s+|mera\s+|meri\s+)?(\w[\w\s]*?)\s+(?:is|hai)\s+'
            r'(?:a\s+|an\s+)?(.+?)[\.,!?]?$',
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": m.group(1).strip().lower(),
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "I don't like X anymore, I'm into Y now"
        m = re.search(
            r"(?:i\s+)?(?:don'?t\s+like|stopped\s+(?:liking|eating|doing))\s+(\w+)\s+"
            r"(?:anymore|now|these\s+days)[,\.\s]+\s*"
            r"(?:i'?m?\s+(?:into|like|prefer|love)|now\s+(?:i\s+)?(?:like|prefer))\s+"
            r"(.+?)(?:\s+now)?[\.,!?]?$",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "preference",
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "I sold my X. I drive/have Y now."
        m = re.search(
            r"(?:i\s+)?(?:sold|got\s+rid\s+of|gave\s+away)\s+(?:my\s+)?(\w+)[\.,!]?\s*"
            r"(?:i\s+)?(?:drive|have|got|use|own)\s+(?:a\s+)?(.+?)(?:\s+now)?[\.,!?]?$",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "car",
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "I quit X. I'm a Y now."
        m = re.search(
            r"(?:i\s+)?(?:quit|left|stopped)\s+(\w+)[\.,!]?\s*"
            r"(?:i'?m?\s+)?(?:a\s+|an\s+)?(.+?)(?:\s+now)?[\.,!?]?$",
            message, re.IGNORECASE
        )
        if m:
            val = m.group(2).strip()
            if len(val.split()) <= 5:
                return [{
                    "entity": "user",
                    "attribute": "job",
                    "value": val,
                    "is_correction": True,
                }]

        # Pattern: "X and I broke up. I'm seeing Y now."
        m = re.search(
            r"(\w+)\s+and\s+I\s+(?:broke\s+up|split|separated|aren'?t\s+(?:together|friends))[\.,!]?\s*"
            r"(?:i'?m?\s+)?(?:seeing|dating|with|now\s+with)\s+(\w+)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "partner",
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "X moved out" / "X stopped coming"
        m = re.search(
            r"(\w+)\s+(?:moved\s+out|stopped\s+coming|left|passed\s+away|died)[\.,!]?\s*"
            r"(?:i\s+)?(?:now\s+)?(?:go|live|adopted|have|got)\s+(?:with\s+|a\s+)?(?:\w+\s+(?:named|called)\s+)?(\w+)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "companion",
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "I moved to X" / "I shifted to X" (explicit relocation = correction)
        # Note: "I live in X" is NOT a correction — it's handled in _extract_single_sentence
        m = re.search(
            r"(?:i\s+)?(?:moved?\s+to|shifted\s+to|relocated\s+to)\s+(.+?)(?:\s+now)?[\.,!?]?$",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "city",
                "value": m.group(1).strip(),
                "is_correction": True,
            }]

        # Pattern: "I got married" / "I'm married now"
        m = re.search(
            r"(?:i\s+)?(?:got\s+married|'m\s+married|am\s+married)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "relationship_status",
                "value": "married",
                "is_correction": True,
            }]

        # Pattern: "I just turned X!" (age update)
        m = re.search(
            r"(?:i\s+)?(?:just\s+)?turned\s+(\d+)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "age",
                "value": m.group(1).strip(),
                "is_correction": True,
            }]

        # Pattern: "X passed away... I adopted Y named Z"
        m = re.search(
            r"(\w+)\s+passed\s+away[\.,!]?\s*(?:.*?)?(?:i\s+)?adopted\s+(?:a\s+)?(\w+)\s+named\s+(\w+)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "pet",
                "value": f"{m.group(2)} named {m.group(3)}",
                "is_correction": True,
            }]

        # Pattern: "I have a new X, her/his name is Y"
        m = re.search(
            r"(?:i\s+have\s+)?(?:a\s+)?new\s+(\w+)\s+(?:now)?[,\s]*(?:her|his|their)?\s*name\s+is\s+(\w+)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": m.group(1).strip().lower(),
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "X and I had a big fight. We're not friends anymore."
        m = re.search(
            r"(\w+)\s+and\s+I\s+(?:had\s+a\s+(?:big\s+)?fight|fell\s+out|aren'?t\s+friends)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "ex_friend",
                "value": m.group(1).strip(),
                "is_correction": True,
            }]

        # Pattern: "I live alone now, X moved out"
        m = re.search(
            r"(?:i\s+)?live\s+alone\s+now",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "living_situation",
                "value": "lives alone",
                "is_correction": True,
            }]

        # Pattern: "My mom just had a baby boy! I have a brother now too!"
        m = re.search(
            r"(?:i\s+have\s+)?(?:a\s+)?(?:brother|sister|sibling)\s+(?:now|too)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "new_sibling",
                "value": message.strip(),
                "is_correction": False,
            }]

        # Pattern: "now it is reduced/dropped to X" / "it is now X" (pronoun-based numeric update)
        m = re.search(
            r"(?:now\s+)?(?:it|ye|yeh|wo|woh)\s+(?:is\s+)?(?:reduced|dropped|down|came\s+down|decreased|increased|gone\s+up|went\s+up)?\s*(?:to\s+)?(\d+[\.\d]*)(?:\s*(?:kg|kgs|lbs|pounds|cm|ft|feet|inches))?",
            message, re.IGNORECASE
        )
        if m:
            val = m.group(1).strip()
            # Detect what "it" refers to from context or use generic
            attr = "update"
            attr_keywords = {
                "weight": ["weight", "wajan", "wazan", "vajan", "kg", "kgs", "lbs", "pounds", "reduced", "dropped"],
                "age": ["age", "umar", "umr", "birthday", "turned"],
                "salary": ["salary", "pay", "income", "tankhwah", "ctc", "package"],
                "height": ["height", "lambai", "kad", "cm", "ft", "feet", "inches"],
                "score": ["score", "marks", "grade", "number", "percentage", "cgpa"],
            }
            for attr_name, keywords in attr_keywords.items():
                if any(w in msg_lower for w in keywords):
                    attr = attr_name
                    break
            return [{
                "entity": "user",
                "attribute": attr,
                "value": val,
                "is_correction": True,
            }]

        # Pattern: "I switched to X, Y was giving me problems"
        m = re.search(
            r"(?:i\s+)?(?:switched|changed|moved)\s+to\s+(.+?)[,\.]",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "preference",
                "value": m.group(1).strip(),
                "is_correction": True,
            }]

        # Pattern: "I started eating X recently"
        m = re.search(
            r"(?:i\s+)?started\s+(?:eating|drinking|doing|learning|playing)\s+(.+?)(?:\s+recently|\s+lately)?[\.,!?]?$",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "recent_change",
                "value": m.group(0).strip(),
                "is_correction": True,
            }]

        # Pattern: "I used to like X but now I prefer Y"
        m = re.search(
            r"(?:i\s+)?used\s+to\s+(?:like|love|prefer|enjoy)\s+(\w+)\s+"
            r"but\s+(?:now\s+)?(?:i\s+)?(?:prefer|like|love)\s+(\w+)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "preference",
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "Ab se mera naam/name X hai/h" (Hinglish correction with full phrase)
        m = re.search(
            r"(?:ab\s+(?:se\s+)?|now\s+)(?:mera|meri|my)\s+(?:naam|name)\s+(\w+)\s*(?:hai|h|he|hain|is)?[\.,!?]?$",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": "name",
                "value": m.group(1).strip(),
                "is_correction": True,
            }]

        # Pattern: "Ab se mera X Y hai/h" (Hinglish correction with attribute)
        m = re.search(
            r"(?:ab\s+(?:se\s+)?|now\s+)(?:mera|meri|my)\s+(\w+)\s+(\w+)\s*(?:hai|h|he|hain|is)?[\.,!?]?$",
            message, re.IGNORECASE
        )
        if m:
            attr = m.group(1).strip().lower()
            val = m.group(2).strip()
            return [{
                "entity": "user",
                "attribute": attr,
                "value": val,
                "is_correction": True,
            }]

        # Pattern: "Mera X ab Y hai" / "mera weight ab 70 hai" (subject + ab + value)
        m = re.search(
            r"(?:my|mera|meri)\s+(\w+)\s+(?:ab|now)\s+(\w+)\s*(?:hai|h|he|hain|is)?[\.,!?]?$",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": m.group(1).strip().lower(),
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "Ab X Y hai" / "Ab se X yaad rakhna" (Hinglish "now X is Y")
        m = re.search(
            r"(?:ab\s+(?:se\s+)?|now\s+)(\w+)\s+(?:hai|h|he|hain|is|yaad\s+rakhna|remember)",
            message, re.IGNORECASE
        )
        if m:
            val = m.group(1).strip()
            # Try to figure out the attribute from context
            attr = "update"
            attr_keywords = {
                "nickname": ["nickname", "naam", "name", "bolna"],
                "role": ["captain", "role", "position"],
                "relationship": ["crush", "girlfriend", "gf", "bf", "boyfriend"],
                "weight": ["weight", "wajan", "wazan", "vajan"],
                "age": ["age", "umar", "umr"],
                "salary": ["salary", "pay", "income", "tankhwah"],
                "height": ["height", "lambai", "kad"],
                "score": ["score", "marks", "grade"],
            }
            for attr_name, keywords in attr_keywords.items():
                if any(w in msg_lower for w in keywords):
                    attr = attr_name
                    break
            return [{
                "entity": "user",
                "attribute": attr,
                "value": val,
                "is_correction": True,
            }]

        # Pattern: "Mera X pehle Y tha, ... ab Z hai" (temporal update with subject)
        m = re.search(
            r"(?:mera|meri|my)\s+(\w+)\s+pehle\s+[\w\s,]+?ab\s+(\w+)\s+(?:hai|h|he|hain)",
            message, re.IGNORECASE
        )
        if m:
            attr = m.group(1).strip().lower()
            return [{
                "entity": "user",
                "attribute": attr,
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "Pehle X tha/thi, ab Y hai" (Before X, now Y — no subject)
        m = re.search(
            r"pehle\s+(?:main\s+|mera\s+)?(?:\w+\s+)?(\w+)\s+(?:tha|thi|the)[\.,]?\s*"
            r"(?:but\s+|lekin\s+|par\s+)?ab\s+(\w[\w\s]*?)\s+(?:hai|h|he|hain|hoon|leta|leti|karta|karti)",
            message, re.IGNORECASE
        )
        if m:
            # Try to detect the subject/attribute from earlier in the message
            attr = "status"
            attr_keywords = {
                "weight": ["weight", "wajan", "wazan", "vajan"],
                "age": ["age", "umar", "umr"],
                "salary": ["salary", "pay", "income", "tankhwah"],
                "height": ["height", "lambai", "kad"],
                "score": ["score", "marks", "grade", "number"],
                "rank": ["rank", "position"],
                "role": ["captain", "leader", "manager", "boss"],
                "routine": ["routine", "schedule", "daily"],
                "diet": ["diet", "khana", "eating", "food"],
                "hobby": ["hobby", "play", "game"],
            }
            for attr_name, keywords in attr_keywords.items():
                if any(w in msg_lower for w in keywords):
                    attr = attr_name
                    break
            return [{
                "entity": "user",
                "attribute": attr,
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "Divya ab meri girlfriend hai, crush nahi"
        m = re.search(
            r"(\w+)\s+(?:ab\s+)?(?:meri|mera)\s+(\w+)\s+(?:hai|h|he|hain)[\.,]?\s*(?:\w+\s+)?(?:nahi|not|mat)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": m.group(1).strip().lower(),
                "attribute": "relationship",
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "Mera nickname X hai" (simple Hinglish assignment that implies correction)
        m = re.search(
            r"(?:mera|meri)\s+(\w+)\s+(\w+)\s+(?:hai|h|he|hain)[\.,]?\s*(?:.*?)(?:ab\s+se|yaad\s+rakh)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": m.group(1).strip().lower(),
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "My X's name is actually Y, not Z"
        m = re.search(
            r"(?:my\s+)?(\w+)(?:'s)?\s+name\s+is\s+actually\s+(\w+),?\s+not\s+(\w+)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": f"{m.group(1).strip().lower()}_name",
                "value": m.group(2).strip(),
                "is_correction": True,
            }]

        # Pattern: "X stopped coming to Y. Now I go with Z."
        m = re.search(
            r"(\w+)\s+stopped\s+coming\s+(?:to\s+)?(\w+)[\.,!]?\s*"
            r"(?:now\s+)?(?:i\s+)?(?:go|train|work|play)\s+with\s+(\w+)",
            message, re.IGNORECASE
        )
        if m:
            return [{
                "entity": "user",
                "attribute": f"{m.group(2).strip().lower()}_buddy",
                "value": m.group(3).strip(),
                "is_correction": True,
            }]

        return []

    def _extract_single_sentence(self, sent: str) -> list[dict]:
        """Extract facts from a single sentence."""
        sent = sent.strip()
        if not sent or self._is_question(sent):
            return []

        facts = []

        # Ordered by specificity — most specific first

        # "store X" / "remember X" / "note X" — direct storage commands
        m = re.search(
            r"^(?:store|remember|note|save|yaad\s+rakh)\s+(.+?)$",
            sent, re.IGNORECASE
        )
        if m:
            rest = m.group(1).strip()
            # "store crush name is IRA" / "store X name is Y"
            m2 = re.search(
                r"(\w[\w\s]*?)\s+(?:name\s+is|is\s+named|naam\s+hai|naam)\s+(\w+)",
                rest, re.IGNORECASE
            )
            if m2:
                facts.append({
                    "entity": "user",
                    "attribute": f"{m2.group(1).strip().lower().replace(' ', '_')}_name",
                    "value": m2.group(2).strip(),
                    "is_correction": False,
                })
                return facts
            # "store X is Y" / "remember X Y" (e.g. "remember dadaji service number IC-14829")
            m2 = re.search(r"(\w[\w\s]*?)\s+(?:is|hai)\s+(.+?)$", rest, re.IGNORECASE)
            if m2:
                facts.append({
                    "entity": "user",
                    "attribute": m2.group(1).strip().lower().replace(" ", "_"),
                    "value": m2.group(2).strip(),
                    "is_correction": False,
                })
                return facts
            # "remember dadaji service number IC-14829" — key-value with no "is"
            m2 = re.search(r"(\w[\w\s]+?)\s+([\w][\w\-@#.]+)$", rest, re.IGNORECASE)
            if m2:
                facts.append({
                    "entity": "user",
                    "attribute": m2.group(1).strip().lower().replace(" ", "_"),
                    "value": m2.group(2).strip(),
                    "is_correction": False,
                })
                return facts
            # "store crush" / "store X" — bare topic
            if len(rest.split()) <= 3:
                facts.append({
                    "entity": "user",
                    "attribute": "noted_topic",
                    "value": rest,
                    "is_correction": False,
                })
                return facts

        # "My name is Y" / "Mera naam Y hai/h" (user's own name — check first)
        # English order: my name is X
        m = re.search(
            r"(?:my|mera|meri)\s+(?:name|naam)\s+(?:is|hai|h|he|hain)\s+(\w+)",
            sent, re.IGNORECASE
        )
        if not m:
            # Hindi order: mera naam X hai/h (verb must be a separate word)
            m = re.search(
                r"(?:my|mera|meri)\s+(?:name|naam)\s+(\w+)\s+(?:hai|h|he|hain|is)[\.,!?]?$",
                sent, re.IGNORECASE
            )
        if not m:
            # Bare form: mera naam X (no verb)
            m = re.search(
                r"(?:my|mera|meri)\s+(?:name|naam)\s+(\w+)[\.,!?]?$",
                sent, re.IGNORECASE
            )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "name",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "My X's name is Y" / "My X is named Y" (possessive — crush's name, dog's name)
        m = re.search(
            r"(?:my|mera|meri|mere)\s+(\w+)(?:'s)?\s+(?:name\s+is|is\s+named|is\s+called|naam\s+(?:hai|h|he|hain)|naam)\s+(\w+)",
            sent, re.IGNORECASE
        )
        if m:
            noun = m.group(1).strip().lower()
            if noun != "name":  # Avoid "my name name is X"
                facts.append({
                    "entity": "user",
                    "attribute": f"{noun}_name",
                    "value": m.group(2).strip(),
                    "is_correction": False,
                })
                return facts

        # "X name is Y" / "X ka naam Y hai" (without "my" prefix — e.g. "crush name is Ira")
        m = re.search(
            r"^(\w[\w\s]*?)\s+(?:name\s+is|ka\s+naam|naam\s+(?:hai|h|he|hain))\s+(\w+)",
            sent, re.IGNORECASE
        )
        if m:
            attr_key = m.group(1).strip().lower().replace(" ", "_")
            skip_words = {"what", "which", "kya", "kaun", "kiska", "my", "mera", "meri"}
            if attr_key not in skip_words:
                facts.append({
                    "entity": "user",
                    "attribute": f"{attr_key}_name",
                    "value": m.group(2).strip(),
                    "is_correction": False,
                })
                return facts

        # "My X is Y" / "My X is now Y" / "Mera X hai Y"
        m = re.search(
            r"(?:my|mera|meri|mere)\s+(\w[\w\s]*?)\s+(?:is|hai|h|he|hain|ka naam)\s+(?:now\s+|ab\s+)?(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            attr = m.group(1).strip().lower().replace(" ", "_")
            # Clean up: "my name" → "name", not "my_name"
            attr = re.sub(r'^my_', '', attr)
            facts.append({
                "entity": "user",
                "attribute": attr,
                "value": m.group(2).strip(),
                "is_correction": False,
            })
            return facts

        # "my X Y Z" — where Z looks like a value (number, code, proper noun)
        # e.g. "my dadaji number 12345567", "my wifi password Spark@2024"
        m = re.search(
            r"(?:my|mera|meri|mere)\s+(\w[\w\s]*?)\s+([\w][\w@#\-\.]+)$",
            sent, re.IGNORECASE
        )
        if m:
            attr = m.group(1).strip().lower().replace(" ", "_")
            val = m.group(2).strip()
            # Only if the value looks like a specific datum (has digits, special chars, or is capitalized)
            if (any(c.isdigit() for c in val) or
                any(c in val for c in '@#-_.') or
                (val[0].isupper() and len(val) > 1)):
                facts.append({
                    "entity": "user",
                    "attribute": attr,
                    "value": val,
                    "is_correction": False,
                })
                return facts

        # "I'm a X" / "I am a X"
        m = re.search(
            r"(?:i'?m|i\s+am)\s+(?:a\s+|an\s+)?(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            val = m.group(1).strip()
            # Avoid capturing emotional states or actions
            if not any(w in val.lower() for w in [
                "here", "feeling", "going", "planning", "looking",
                "learning", "training", "seeing", "stressed",
                "allergic", "into",
            ]):
                # Determine if this is a profession or a trait
                profession_keywords = {
                    "engineer", "developer", "designer", "doctor", "nurse",
                    "teacher", "professor", "student", "researcher", "scientist",
                    "analyst", "consultant", "manager", "architect", "writer",
                    "artist", "musician", "chef", "lawyer", "accountant",
                    "freelancer", "intern", "founder", "ceo", "cto",
                    "programmer", "data", "software", "devops", "ai", "ml",
                    "web", "full", "front", "back", "cloud", "product",
                    "marketing", "sales", "hr", "finance", "mechanic",
                    "pilot", "journalist", "photographer", "filmmaker",
                    "pharmacist", "dentist", "surgeon", "therapist",
                    "entrepreneur", "businessman", "businesswoman",
                    "tutor", "coach", "trainer", "plumber", "electrician",
                    "carpenter", "driver", "officer", "soldier",
                }
                val_lower = val.lower()
                is_profession = any(kw in val_lower for kw in profession_keywords)

                # Trait keywords (vegan, vegetarian, etc.) stay as identity
                attr = "job" if is_profession else "identity"

                facts.append({
                    "entity": "user",
                    "attribute": attr,
                    "value": val,
                    "is_correction": False,
                })
                return facts

        # "I live in X" / "I'm from X" / "I stay in X"
        m = re.search(
            r"(?:i\s+)?(?:live\s+in|stay\s+in|reside\s+in|'m\s+from|am\s+from)\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "city",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "I work at/as X" / "I'm working at X"
        m = re.search(
            r"(?:i\s+)?(?:work\s+(?:at|as|for|in)|'m\s+working\s+(?:at|as|for))\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "job",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "I'm allergic to X"
        m = re.search(
            r"(?:i'?m|i\s+am)\s+allergic\s+to\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "allergies",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "I'm learning X"
        m = re.search(
            r"(?:i'?m|i\s+am)\s+learning\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "learning",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "I'm training for X"
        m = re.search(
            r"(?:i'?m|i\s+am)\s+(?:training|preparing)\s+for\s+(?:a\s+)?(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "training",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "I'm planning a trip to X"
        m = re.search(
            r"(?:i'?m|i\s+am)\s+planning\s+(?:a\s+)?(?:trip|visit|vacation)\s+to\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "travel_plan",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "I'm feeling stressed about my X"
        m = re.search(
            r"(?:i'?m|i\s+am)\s+(?:feeling\s+)?(?:stressed|worried|anxious|nervous)\s+about\s+(?:my\s+)?(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "concern",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "I have X siblings/brothers/sisters: Y and Z"
        m = re.search(
            r"(?:i\s+have)\s+(?:\w+\s+)?(?:siblings?|brothers?|sisters?)[,:\s]+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "siblings",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "I have a X named/called Y"
        m = re.search(
            r"(?:i\s+have|mere\s+paas)\s+(?:a\s+)?(\w+)\s+(?:named|called|naam)\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": m.group(2).strip().lower(),
                "attribute": "species",
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            facts.append({
                "entity": "user",
                "attribute": "pet",
                "value": f"{m.group(1).strip()} named {m.group(2).strip()}",
                "is_correction": False,
            })
            return facts

        # "X is my Y"
        m = re.search(
            r"(\w+)\s+(?:is|hai)\s+(?:my|mera|meri|mere)\s+(\w[\w\s]*?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": m.group(2).strip().lower().replace(" ", "_"),
                "value": m.group(1).strip(),
                "is_correction": False,
            })
            return facts

        # "My favorite X is Y"
        m = re.search(
            r"(?:my|mera|meri)\s+(?:favorite|favourite|fav)\s+(\w+)\s+(?:is|hai)\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": f"favorite_{m.group(1).strip().lower()}",
                "value": m.group(2).strip(),
                "is_correction": False,
            })
            return facts

        # "It's my X exam" / "It is my X exam" / "It's my X"
        m = re.search(
            r"(?:it'?s|it\s+is)\s+(?:my\s+)?(.+?)(?:\s+(exam|test|quiz))?[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            val = m.group(1).strip()
            suffix = m.group(2) or ""
            full_val = f"{val} {suffix}".strip() if suffix else val
            if len(full_val.split()) <= 4 and full_val.lower() not in {"a", "an", "the", "my"}:
                facts.append({
                    "entity": "user",
                    "attribute": "current_topic",
                    "value": full_val,
                    "is_correction": False,
                })
                return facts

        # "X ka/ki Y hai Z"
        m = re.search(
            r"(\w+)\s+(?:ka|ki|ke)\s+(\w+)\s+(?:hai|is)\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": m.group(1).strip().lower(),
                "attribute": m.group(2).strip().lower(),
                "value": m.group(3).strip(),
                "is_correction": False,
            })
            return facts

        # "I also love/like X"
        m = re.search(
            r"(?:i\s+)?(?:also\s+)?(?:love|like|enjoy|prefer)\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            val = m.group(1).strip()
            if len(val.split()) <= 4:
                facts.append({
                    "entity": "user",
                    "attribute": "likes",
                    "value": val,
                    "is_correction": False,
                })
                return facts

        # "X is a Y" / "X hai Y" (generic, last resort)
        m = re.search(
            r"^(\w+)\s+(?:is\s+a|is\s+an|is|hai)\s+(?:a\s+|an\s+)?(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m:
            entity = m.group(1).strip()
            value = m.group(2).strip()
            # Skip if entity looks like a pronoun or common word
            skip_entities = {"it", "this", "that", "there", "here", "he", "she", "they", "we", "i"}
            if entity.lower() not in skip_entities and len(value.split()) <= 5:
                facts.append({
                    "entity": entity.lower(),
                    "attribute": "identity",
                    "value": value,
                    "is_correction": False,
                })
                return facts

        # "In X" / "Next X" (temporal info attached to conversation)
        m = re.search(
            r"^(?:in|next|this|last)\s+(\w+)$",
            sent, re.IGNORECASE
        )
        if m:
            facts.append({
                "entity": "user",
                "attribute": "timeframe",
                "value": sent.strip(),
                "is_correction": False,
            })
            return facts

        # "A web scraper for X" — context continuation
        m = re.search(
            r"^(?:a\s+)?(\w[\w\s]+?)\s+(?:for|about|on)\s+(.+?)[\.,!?]?$",
            sent, re.IGNORECASE
        )
        if m and len(sent.split()) <= 8:
            facts.append({
                "entity": "user",
                "attribute": "project",
                "value": sent.strip(),
                "is_correction": False,
            })
            return facts

        return facts


def create_llm_extract_fn(llm_client=None):
    """
    Factory for LLM-based fact extraction.
    Returns a function that uses an LLM to extract structured facts.
    """
    if llm_client is None:
        return None

    def extract(message: str) -> list[dict]:
        prompt = f"""Extract atomic facts from this message as JSON.
Each fact should have: entity, attribute, value, is_correction (bool).

Message: "{message}"

Return a JSON array of facts. If no facts, return [].
Example: [{{"entity": "spark", "attribute": "species", "value": "hamster", "is_correction": false}}]
"""
        try:
            response = llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return []

    return extract
