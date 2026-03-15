"""
Baseline Chat — simple RAG-style chat with no structured memory.

Stores raw conversation text and retrieves similar chunks for context.
No fact extraction, no correction handling, no sensitivity awareness.
"""

from baseline.baseline_memory import BaselineMemory


class BaselineChat:
    """Baseline chat system using raw text retrieval."""

    def __init__(self, embed_fn=None, llm_fn=None):
        self.memory = BaselineMemory()
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.histories: dict[str, list[dict]] = {}

    def chat(self, user_id: str, message: str) -> dict:
        """Process a message using baseline approach."""
        # Store the message
        embedding = self.embed_fn(message) if self.embed_fn else None
        self.memory.store(user_id, f"User: {message}", embedding)

        # Retrieve similar past context
        context_chunks = []
        if self.embed_fn:
            query_emb = self.embed_fn(message)
            results = self.memory.search(user_id, query_emb, top_k=5)
            context_chunks = [r["text"] for r in results]
        else:
            all_mem = self.memory.get_all(user_id)
            context_chunks = [m["text"] for m in all_mem[:5]]

        # Generate response
        response = self._generate(user_id, message, context_chunks)

        # Store response
        self.memory.store(
            user_id, f"Assistant: {response}",
            self.embed_fn(response) if self.embed_fn else None
        )

        self.histories.setdefault(user_id, []).append(
            {"role": "user", "content": message}
        )
        self.histories[user_id].append(
            {"role": "assistant", "content": response}
        )

        return {
            "response": response,
            "strategy": "baseline_rag",
            "memories_used": [{"text": c} for c in context_chunks],
            "memories_extracted": [],
        }

    def _generate(self, user_id: str, message: str, context: list[str]) -> str:
        """Generate response with context."""
        if self.llm_fn:
            context_block = "\n".join(context) if context else "No prior context."
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly AI companion. Use the following "
                        "conversation context to answer the user's question.\n\n"
                        f"Context:\n{context_block}"
                    ),
                },
                {"role": "user", "content": message},
            ]
            try:
                return self.llm_fn(messages)
            except Exception:
                pass

        # Fallback: echo context
        if context:
            return f"Based on our conversations: {'; '.join(context[:3])}"
        return "I'm here to chat! Tell me more about yourself."
