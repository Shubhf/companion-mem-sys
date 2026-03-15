"""
Conversation Manager — maintains multi-turn state and orchestrates the pipeline.

Coordinates: ingestion → retrieval → planning → generation.
Manages conversation history per user session.
Also searches conversation history as a fallback when structured memory misses.
"""

import re
from datetime import datetime
from typing import Optional

from memory_engine.memory_store import MemoryStore
from memory_engine.memory_ingestion import MemoryIngestionPipeline
from memory_engine.memory_retrieval import MemoryRetriever
from memory_engine.conflict_resolution import ConflictResolver
from chat_system.response_planner import ResponsePlanner


class ConversationManager:
    """Manages multi-turn conversations with memory integration."""

    def __init__(
        self,
        store: MemoryStore,
        embed_fn=None,
        llm_fn=None,
        llm_extract_fn=None,
    ):
        self.store = store
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn

        self.ingestion = MemoryIngestionPipeline(
            store=store, embed_fn=embed_fn, llm_extract_fn=llm_extract_fn
        )
        self.retriever = MemoryRetriever(store=store, embed_fn=embed_fn)
        self.conflict_resolver = ConflictResolver(store=store)
        self.planner = ResponsePlanner()

        self.histories: dict[str, list[dict]] = {}

    def chat(self, user_id: str, message: str) -> dict:
        """
        Process a user message through the full pipeline.
        Returns response dict with text, plan strategy, and extracted memories.
        """
        # 1. Ingest: extract and store memories from user message
        extracted = self.ingestion.ingest(user_id, message)

        # 2. Retrieve: find relevant memories for the message
        memory_context = self.retriever.retrieve_for_response(
            user_id, message, top_k=5
        )

        # 2b. Always include user's name if we have it and it's not already in context
        if memory_context:
            has_name = any(m.get("attribute") == "name" for m in memory_context)
            if not has_name:
                name_memories = self.store.get_by_entity(user_id, "user")
                for nm in name_memories:
                    if nm.attribute == "name" and nm.status.value == "active":
                        from memory_engine.sensitive_policy import SensitivePolicy
                        sp = SensitivePolicy()
                        formatted = sp.format_for_response(nm.entity, nm.attribute, nm.value, nm.sensitivity)
                        if formatted:
                            memory_context.append({
                                "memory_id": nm.memory_id,
                                "text": formatted,
                                "entity": nm.entity,
                                "attribute": nm.attribute,
                                "value": nm.value,
                                "confidence": nm.confidence,
                                "score": 0.3,
                                "sensitivity": nm.sensitivity.value,
                                "needs_confirmation": False,
                            })
                        break

        # 3. If no structured memory found, search conversation history
        history_context = []
        if not memory_context:
            history_context = self._search_history(user_id, message)

        # 4. Plan: decide response strategy
        plan = self.planner.plan(message, memory_context, history_context)

        # 5. Generate response
        response_text = self._generate(user_id, plan)

        # 6. Update conversation history
        self._add_to_history(user_id, "user", message)
        self._add_to_history(user_id, "assistant", response_text)

        return {
            "response": response_text,
            "strategy": plan.strategy,
            "memories_used": memory_context,
            "memories_extracted": [
                {
                    "entity": m.entity,
                    "attribute": m.attribute,
                    "value": m.value,
                    "type": m.memory_type.value,
                }
                for m in extracted
            ],
        }

    def _search_history(self, user_id: str, query: str) -> list[dict]:
        """
        Search conversation history for relevant information.
        Used as fallback when structured memory retrieval returns nothing.
        """
        history = self.get_history(user_id)
        if not history:
            return []

        query_words = set(
            w.lower() for w in re.findall(r'\b\w{3,}\b', query)
        )
        # Remove very common words
        query_words -= {
            "what", "when", "where", "who", "how", "why", "which",
            "the", "and", "for", "are", "was", "were", "that", "this",
            "with", "have", "has", "had", "from", "been", "does", "did",
            "can", "will", "you", "your", "about", "tell", "know",
            "remember", "going", "planning",
            # Hinglish common words that cause false matches
            "mera", "meri", "mere", "kya", "hai", "hain", "bata",
            "batao", "bolo", "kitna", "kitni", "kaun", "kahan",
        }

        relevant = []
        for msg in history:
            if msg["role"] != "user":
                continue
            msg_words = set(
                w.lower() for w in re.findall(r'\b\w{3,}\b', msg["content"])
            )
            overlap = query_words & msg_words
            # Require at least 1 meaningful keyword overlap
            if overlap and len(overlap) >= 1:
                relevant.append({
                    "text": msg["content"],
                    "role": msg["role"],
                    "overlap": list(overlap),
                })

        # Only return history context if we found specific keyword matches
        # Do NOT dump all messages as context — that causes false recalls
        return relevant[:10]

    def _generate(self, user_id: str, plan) -> str:
        """Generate response using LLM or fallback."""
        if self.llm_fn:
            history = self.get_history(user_id)
            messages = [{"role": "system", "content": plan.system_prompt}]
            for msg in history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": plan.user_message})

            try:
                return self.llm_fn(messages)
            except Exception as e:
                return f"[LLM Error: {e}] " + self._fallback_response(plan)

        return self._fallback_response(plan)

    def _fallback_response(self, plan) -> str:
        """Rule-based fallback when no LLM is available."""
        if plan.strategy == "recall" and plan.memory_context:
            # Extract user name for personalization
            name = None
            other_facts = []
            for m in plan.memory_context:
                if m.get("attribute") == "name":
                    name = m["value"]
                else:
                    other_facts.append(m["text"])

            if not other_facts:
                other_facts = [m["text"] for m in plan.memory_context]

            prefix = f"{name}, based" if name else "Based"
            return f"{prefix} on what I remember: " + "; ".join(other_facts) + "."

        if plan.strategy == "history_recall" and plan.history_context:
            hist_texts = [h["text"] for h in plan.history_context[:5]]
            return "Based on our conversation: " + "; ".join(hist_texts) + "."

        if plan.strategy == "honest_missing":
            return (
                "I don't think you've told me that yet. "
                "Would you like to share?"
            )

        if plan.strategy == "ask_confirm":
            topics = [m["attribute"] for m in plan.needs_confirmation]
            return (
                f"I have some information about {', '.join(topics)}, "
                "but it's sensitive. Would you like me to share it?"
            )

        # Friendly fallback for general/greeting messages
        msg_lower = plan.user_message.lower().strip().rstrip("?!. ")
        if any(g in msg_lower for g in ["hi", "hello", "hey", "howdy", "namaste", "hola"]):
            return "Hey there! How's your day going? What would you like to chat about?"
        if any(g in msg_lower for g in ["how are", "how r u", "kaise ho"]):
            return "I'm doing great, thanks for asking! How about you? What's on your mind today?"
        if any(g in msg_lower for g in ["bye", "goodbye", "see you", "take care"]):
            return "Goodbye! It was great chatting with you. See you next time!"
        if any(g in msg_lower for g in ["thanks", "thank you", "shukriya"]):
            return "You're welcome! Happy to help. Anything else you'd like to talk about?"
        return "I'm here! What would you like to talk about?"

    def get_history(self, user_id: str) -> list[dict]:
        return self.histories.get(user_id, [])

    def _add_to_history(self, user_id: str, role: str, content: str):
        if user_id not in self.histories:
            self.histories[user_id] = []
        self.histories[user_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def clear_history(self, user_id: str):
        self.histories.pop(user_id, None)

    def get_user_memories(self, user_id: str) -> list[dict]:
        """Get all active memories for a user (for UI inspection)."""
        entries = self.store.get_by_user(user_id)
        return [
            {
                "memory_id": e.memory_id,
                "entity": e.entity,
                "attribute": e.attribute,
                "value": e.value,
                "type": e.memory_type.value,
                "confidence": e.confidence,
                "status": e.status.value,
                "sensitivity": e.sensitivity.value,
                "timestamp": e.timestamp.isoformat(),
                "supersedes": e.supersedes,
            }
            for e in entries
        ]
