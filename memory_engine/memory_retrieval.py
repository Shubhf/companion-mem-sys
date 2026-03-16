"""
Memory Retrieval — ranked memory retrieval with multi-signal scoring.

Combines entity matching, attribute keyword matching, full-text keyword search,
semantic similarity, recency, and confidence to produce ranked results.
Includes autocorrect for query typos.
"""

import re
from datetime import datetime
from typing import Optional

from memory_engine.memory_schema import (
    MemoryEntry, MemoryQuery, MemoryStatus, SensitivityLevel
)
from memory_engine.memory_store import MemoryStore
from memory_engine.sensitive_policy import SensitivePolicy

try:
    from autocorrect import Speller
    _speller = Speller(lang='en')
    AUTOCORRECT_AVAILABLE = True
except ImportError:
    AUTOCORRECT_AVAILABLE = False


# Scoring weights
WEIGHT_ENTITY_MATCH = 0.35
WEIGHT_SEMANTIC = 0.25
WEIGHT_RECENCY = 0.20
WEIGHT_CONFIDENCE = 0.20

# Maps query keywords to attribute patterns they might match
QUERY_ATTRIBUTE_MAP = {
    "color": ["color", "favorite_color", "colour"],
    "colour": ["color", "favorite_color", "colour"],
    "name": ["name", "pet_name", "dog_name", "cat_name"],
    "city": ["city", "location", "place", "lives"],
    "live": ["city", "location", "place", "living_situation"],
    "birthday": ["birthday", "dob", "birth_date", "born"],
    "age": ["age", "years_old"],
    "hobby": ["hobby", "hobbies", "likes", "interest"],
    "hobbies": ["hobby", "hobbies", "likes", "interest"],
    "job": ["job", "work", "profession", "occupation", "career", "identity"],
    "work": ["job", "work", "profession", "occupation", "career", "identity"],
    "profession": ["job", "work", "profession", "occupation", "career", "identity"],
    "career": ["job", "work", "profession", "occupation", "career", "identity"],
    "occupation": ["job", "work", "profession", "occupation", "career", "identity"],
    "pet": ["pet", "dog", "cat", "animal", "species", "pet_name"],
    "dog": ["pet", "dog", "species", "breed", "dog_name"],
    "cat": ["pet", "cat", "species", "breed", "cat_name"],
    "breed": ["breed", "species"],
    "food": ["food", "favorite_food", "diet", "preference"],
    "movie": ["movie", "favorite_movie", "likes", "film"],
    "movies": ["movie", "favorite_movie", "likes", "film"],
    "language": ["language", "languages", "speaks"],
    "languages": ["language", "languages", "speaks"],
    "allergy": ["allergy", "allergies", "allergic"],
    "allergies": ["allergy", "allergies", "allergic"],
    "allergic": ["allergy", "allergies", "allergic"],
    "friend": ["friend", "best_friend", "friends"],
    "partner": ["partner", "boyfriend", "girlfriend", "spouse", "husband", "wife"],
    "married": ["relationship_status", "partner", "spouse"],
    "car": ["car", "vehicle", "drive", "drives"],
    "drive": ["car", "vehicle", "drive", "drives"],
    "sibling": ["sibling", "siblings", "brother", "sister", "sibling_count"],
    "siblings": ["sibling", "siblings", "brother", "sister", "sibling_count"],
    "brother": ["sibling", "siblings", "brother", "new_sibling"],
    "sister": ["sibling", "siblings", "sister", "sibling_count"],
    "health": ["health", "medical_condition", "diagnosis", "allergy", "allergies"],
    "salary": ["salary", "income", "pay", "compensation"],
    "travel": ["travel", "travel_plan", "vacation", "trip"],
    "exam": ["exam", "test", "exam_date", "concern"],
    "diet": ["diet", "food", "eating", "vegetarian"],
    "roommate": ["roommate", "housemate", "living_situation"],
    "manager": ["manager", "boss", "supervisor"],
    "gym": ["gym", "gym_buddy", "exercise", "fitness"],
    "anniversary": ["anniversary", "anniversary_date"],
    "membership": ["membership", "subscription"],
    "beliefs": ["religion", "political_view", "beliefs"],
    "views": ["religion", "political_view", "beliefs", "views"],
    "therapy": ["therapy", "therapist", "mental_health"],
    "password": ["password", "pin", "credential"],
    "phone": ["phone", "phone_number", "mobile"],
    "blood": ["blood_type", "blood"],
    "camera": ["camera", "photography", "equipment"],
    "hospital": ["hospital", "workplace", "clinic"],
    "routine": ["routine", "daily_routine", "schedule"],
    "restaurant": ["restaurant", "dining", "food"],
    "vaccination": ["vaccination", "vaccine", "shots"],
    "college": ["college", "university", "school", "education"],
    # Hinglish keywords
    "naam": ["name", "pet_name", "dog_name", "cat_name", "user_name"],
    "kaam": ["job", "work", "profession", "work_context"],
    "ghar": ["city", "location", "home", "living_situation"],
    "kahan": ["city", "location", "home", "living_situation"],
    "rehta": ["city", "location", "home", "living_situation"],
    "rehti": ["city", "location", "home", "living_situation"],
    "khana": ["food", "favorite_food", "diet", "likes"],
    "dost": ["friend", "best_friend", "friends"],
    "bhai": ["friend", "brother", "sibling"],
    "behen": ["sister", "sibling"],
    "shaadi": ["relationship_status", "partner", "marriage"],
    "padhai": ["education", "college", "exam", "learning"],
    "sehat": ["health", "medical_condition", "allergy"],
    "paisa": ["salary", "income", "financial"],
    "gaadi": ["car", "vehicle", "bike"],
    "jaanwar": ["pet", "species", "pet_name"],
    "vuskaa": ["name", "pet_name", "species", "identity"],
    "uska": ["name", "pet_name", "species", "identity"],
    "iska": ["name", "pet_name", "species", "identity"],
    # Relationships
    "crush": ["crush", "crush_name", "relationship", "partner"],
    "gf": ["crush", "crush_name", "girlfriend", "partner", "partner_name"],
    "bf": ["crush", "crush_name", "boyfriend", "partner", "partner_name"],
    "girlfriend": ["crush", "crush_name", "girlfriend", "partner", "partner_name"],
    "boyfriend": ["crush", "crush_name", "boyfriend", "partner", "partner_name"],
    "wife": ["wife", "partner", "partner_name", "spouse"],
    "husband": ["husband", "partner", "partner_name", "spouse"],
    "papa": ["father", "father_name", "dad"],
    "mummy": ["mother", "mother_name", "mom"],
    "dost": ["friend", "best_friend", "friend_name"],
    "behen": ["sister", "sister_name", "sibling"],
    "bhai": ["brother", "brother_name", "sibling"],
    "dadaji": ["grandfather", "dadaji", "service_number", "dadaji_service_number", "dadaji_ka_number"],
    "daadaji": ["grandfather", "dadaji", "service_number", "dadaji_service_number", "dadaji_ka_number"],
    "dadadji": ["grandfather", "dadaji", "service_number", "dadaji_service_number", "dadaji_ka_number"],
    "dadaju": ["grandfather", "dadaji", "service_number", "dadaji_service_number", "dadaji_ka_number"],
    "boss": ["boss", "manager", "boss_name"],
    "service": ["service_number", "dadaji_service_number"],
    "number": ["service_number", "phone", "phone_number", "dadaji_service_number"],
    "password": ["password", "wifi_password", "my_wifi_password", "pin"],
    "wifi": ["wifi_password", "my_wifi_password"],
    # Suggestion/action queries → preference attributes
    "have": ["likes", "preference", "favorite_food", "food", "diet", "morning_drink"],
    "eat": ["likes", "preference", "favorite_food", "food", "diet"],
    "drink": ["likes", "preference", "morning_drink", "diet"],
    "watch": ["likes", "favorite_movie", "movie"],
    "read": ["likes", "favorite_book", "book"],
    "play": ["likes", "hobby", "hobbies", "sport"],
    "cook": ["likes", "favorite_food", "food"],
    "snack": ["likes", "preference", "food", "allergies", "allergy"],
    "dinner": ["likes", "preference", "food", "favorite_food", "diet"],
    "lunch": ["likes", "preference", "food", "favorite_food", "diet"],
    "breakfast": ["likes", "preference", "food", "morning_drink", "diet"],
    "prefer": ["likes", "preference"],
    "favorite": ["likes", "preference", "favorite_color", "favorite_food", "favorite_movie"],
    "suggest": ["likes", "preference", "hobby", "hobbies", "diet", "food", "identity"],
    "recommend": ["likes", "preference", "hobby", "hobbies", "diet", "food", "identity"],
    "dishes": ["diet", "food", "favorite_food", "likes", "identity", "preference"],
    "dish": ["diet", "food", "favorite_food", "likes", "identity", "preference"],
    "order": ["likes", "preference", "diet", "food", "morning_drink"],
    "vegan": ["diet", "identity", "food"],
    "vegetarian": ["diet", "identity", "food"],
    "morning": ["morning_drink", "likes", "preference"],
    "humid": ["likes", "preference", "morning_drink"],
}


class MemoryRetriever:
    """Retrieves and ranks memories for a user query."""

    def __init__(self, store: MemoryStore, embed_fn=None):
        self.store = store
        self.embed_fn = embed_fn
        self.sensitive_policy = SensitivePolicy()

    def retrieve(self, query: MemoryQuery) -> list[tuple[MemoryEntry, float]]:
        """
        Retrieve ranked memories for a query.
        Returns list of (memory, score) tuples sorted by relevance.
        """
        candidates = []

        # 1. Direct entity match
        entities = self._extract_entities(query.query_text)
        for entity in entities:
            matches = self.store.get_by_entity(query.user_id, entity)
            for mem in matches:
                if mem.status == MemoryStatus.ACTIVE:
                    candidates.append((mem, "entity_match"))

        # 2. "my/I" → search entity "user" + ALL entities with attribute keyword matching
        if self._refers_to_self(query.query_text):
            # Search ALL active memories for this user, not just entity="user"
            all_user_memories = self.store.get_by_user(query.user_id, MemoryStatus.ACTIVE)
            query_keywords = self._extract_query_keywords(query.query_text)
            attr_patterns = set()
            for kw in query_keywords:
                if kw in QUERY_ATTRIBUTE_MAP:
                    attr_patterns.update(QUERY_ATTRIBUTE_MAP[kw])

            for mem in all_user_memories:
                if mem.status != MemoryStatus.ACTIVE:
                    continue
                if mem.memory_id in {c[0].memory_id for c in candidates}:
                    continue
                # Check if memory attribute matches any keyword pattern
                mem_attr = mem.attribute.lower()
                if attr_patterns and any(
                    pat in mem_attr or mem_attr in pat
                    for pat in attr_patterns
                ):
                    candidates.append((mem, "attribute_match"))
                # Also check if any query keyword appears in the attribute
                elif any(kw in mem_attr or mem_attr.startswith(kw) for kw in query_keywords):
                    candidates.append((mem, "attribute_match"))

        # 3. Full-text keyword search across ALL active memories
        all_memories = self.store.get_by_user(query.user_id, MemoryStatus.ACTIVE)
        query_words = set(w.lower() for w in re.findall(r'\b\w+\b', query.query_text))
        query_words -= self._stop_words()
        for mem in all_memories:
            if mem.memory_id in {c[0].memory_id for c in candidates}:
                continue
            # Check if query keywords appear in memory value or attribute
            mem_text = f"{mem.entity} {mem.attribute} {mem.value}".lower()
            overlap = query_words & set(re.findall(r'\b\w+\b', mem_text))
            if overlap:
                candidates.append((mem, "keyword_match"))

        # 4. Semantic similarity via FAISS
        if self.embed_fn:
            query_emb = self.embed_fn(query.query_text)
            similar = self.store.search_similar(
                query.user_id, query_emb, top_k=query.top_k * 2
            )
            for mem, sim_score in similar:
                if mem.status == MemoryStatus.ACTIVE:
                    candidates.append((mem, "semantic", sim_score))

        # 5. Score and deduplicate
        scored = self._score_candidates(candidates, query)

        # 6. Filter by sensitivity
        filtered = self._filter_sensitive(scored)

        return filtered[:query.top_k]

    def retrieve_for_response(
        self, user_id: str, message: str, top_k: int = 5
    ) -> list[dict]:
        """
        High-level retrieval for the chat system.
        Returns formatted memory context for response generation.
        """
        query = MemoryQuery(user_id=user_id, query_text=message, top_k=top_k)
        results = self.retrieve(query)

        context = []
        for mem, score in results:
            formatted = self.sensitive_policy.format_for_response(
                mem.entity, mem.attribute, mem.value, mem.sensitivity
            )
            if formatted:
                context.append({
                    "memory_id": mem.memory_id,
                    "text": formatted,
                    "entity": mem.entity,
                    "attribute": mem.attribute,
                    "value": mem.value,
                    "confidence": mem.confidence,
                    "score": score,
                    "sensitivity": mem.sensitivity.value,
                    "needs_confirmation": self.sensitive_policy.needs_confirmation(
                        mem.sensitivity
                    ),
                })
        return context

    def _refers_to_self(self, text: str) -> bool:
        """Check if the query refers to the user themselves."""
        self_words = {"my", "i", "me", "i'm", "mine", "i've", "i'd", "im",
                      "mera", "meri", "mere", "mujhe", "am", "main", "hoon",
                      "batav", "bata", "batao", "bolo", "boldo",
                      "vuskaa", "uska", "uski", "iska", "iski",
                      # These indicate queries about user's stored data
                      "dadaji", "daadaji", "dadaju", "dada", "papa", "mummy",
                      "nana", "nani", "mere", "suggest", "recommend"}
        words = set(w.lower().rstrip("'s") for w in text.split())
        return bool(words & self_words)

    # Words that should NOT be autocorrected (Hinglish, names, etc.)
    _NO_AUTOCORRECT = {
        "mera", "meri", "mere", "kya", "hai", "hain", "kaun", "kahan",
        "kab", "kaise", "kitna", "kitni", "batav", "batao", "bata",
        "bolo", "boldo", "naam", "vuskaa", "uska", "uski", "iska",
        "dadaji", "daadaji", "dadaju", "dada", "papa", "mummy",
        "nana", "nani", "rehta", "rehti", "ghar", "khana", "paisa",
        "gaadi", "shaadi", "padhai", "sehat", "dost", "bhai", "behen",
        "hoon", "main", "tujhe", "kaam",
    }

    def _autocorrect_word(self, word: str) -> str:
        """Autocorrect a word if it's an English typo, skip Hinglish."""
        if not AUTOCORRECT_AVAILABLE:
            return word
        if word in self._NO_AUTOCORRECT:
            return word
        if len(word) <= 2:
            return word
        # Only correct if the word looks like English (no Hindi words)
        corrected = _speller(word)
        return corrected if corrected else word

    def _extract_query_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from a query, with autocorrect for typos."""
        words = re.findall(r'\b[A-Za-z]\w+\b', text)
        light_stop = {
            "the", "is", "are", "was", "were", "what", "who", "how",
            "does", "do", "did", "can", "will", "my", "your", "his",
            "her", "its", "about", "tell", "me", "know", "remember",
            "been", "being", "you", "that", "this", "with", "from",
            "for", "not", "yet", "any", "some", "also", "too", "and",
            "or", "but", "when", "where", "which", "would", "could",
            "should", "shall", "may", "might", "am", "i", "a", "an",
            "our", "we", "they", "them", "their", "there", "here",
            "of", "in", "on", "at", "to", "by", "up", "out",
            "so", "if", "just", "now", "then", "than",
        }
        keywords = []
        for w in words:
            wl = w.lower()
            if wl in light_stop:
                continue
            # Try autocorrect: "professin" → "profession"
            corrected = self._autocorrect_word(wl)
            keywords.append(corrected)
            # Also keep original if different (for Hinglish/names)
            if corrected != wl:
                keywords.append(wl)
        return keywords

    def _extract_entities(self, text: str) -> list[str]:
        """Extract potential entity names from query text."""
        words = re.findall(r'\b[A-Za-z]\w+\b', text)
        stop = self._stop_words()
        entities = [w.lower() for w in words if w.lower() not in stop]
        return list(dict.fromkeys(entities))

    def _stop_words(self) -> set:
        return {
            "the", "is", "are", "was", "were", "what", "who", "how",
            "does", "do", "did", "can", "will", "my", "your", "his",
            "her", "its", "about", "tell", "me", "know", "remember",
            "have", "has", "had", "been", "being", "mera", "meri",
            "mere", "kya", "hai", "hain", "ka", "ki", "ke", "naam",
            "you", "that", "this", "with", "from", "for", "not",
            "still", "yet", "any", "some", "also", "too", "and",
            "or", "but", "when", "where", "which", "would", "could",
            "should", "shall", "may", "might", "am", "i", "a", "an",
            "our", "we", "they", "them", "their", "there", "here",
            "going", "been", "come", "coming", "get", "getting",
            "over", "ever", "everything", "anything", "something",
            "suggest", "recommend", "give", "please", "want",
            "need", "like", "think", "believe", "called", "named",
            "of", "in", "on", "at", "to", "by", "up", "out",
            "so", "if", "just", "don", "didn", "doesn", "won",
            "now", "then", "than", "more", "most", "much", "many",
            "very", "really", "quite",
        }

    def _score_candidates(
        self,
        candidates: list[tuple],
        query: MemoryQuery,
    ) -> list[tuple[MemoryEntry, float]]:
        """Score and deduplicate candidate memories."""
        scores: dict[str, tuple[MemoryEntry, float]] = {}

        now = datetime.utcnow()

        for item in candidates:
            mem = item[0]
            match_type = item[1]
            sim_score = item[2] if len(item) > 2 else 0.0

            mid = mem.memory_id
            if mid in scores:
                existing_score = scores[mid][1]
                scores[mid] = (mem, existing_score + 0.15)
                continue

            score = 0.0

            # Match type bonuses
            if match_type == "entity_match":
                score += WEIGHT_ENTITY_MATCH
            elif match_type == "attribute_match":
                score += WEIGHT_ENTITY_MATCH * 0.9  # Almost as good as entity match
            elif match_type == "keyword_match":
                score += WEIGHT_ENTITY_MATCH * 0.5
            elif match_type == "semantic":
                score += WEIGHT_SEMANTIC * sim_score

            # Recency score (exponential decay, half-life = 30 days)
            age_seconds = (now - mem.timestamp).total_seconds()
            age_days = max(age_seconds / 86400, 0)
            recency = 2 ** (-age_days / 30)
            score += WEIGHT_RECENCY * recency

            # Confidence
            score += WEIGHT_CONFIDENCE * mem.confidence

            scores[mid] = (mem, round(score, 4))

        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return ranked

    def _filter_sensitive(
        self, results: list[tuple[MemoryEntry, float]]
    ) -> list[tuple[MemoryEntry, float]]:
        """Remove memories that should not be surfaced."""
        return [
            (mem, score) for mem, score in results
            if mem.sensitivity != SensitivityLevel.DO_NOT_SURFACE
        ]
