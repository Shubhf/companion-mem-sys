"""
Response Planner — decides how to respond based on available memory and history.

Implements the core response logic:
- Memory exists → use it
- History exists → use it
- Memory missing → honestly say so
- Never fabricate or guess
"""

from dataclasses import dataclass, field

from memory_engine.memory_schema import SensitivityLevel


@dataclass
class ResponsePlan:
    """Plan for generating a response."""
    strategy: str  # "recall", "history_recall", "honest_missing", "ask_confirm", "general"
    memory_context: list[dict] = field(default_factory=list)
    history_context: list[dict] = field(default_factory=list)
    needs_confirmation: list[dict] = field(default_factory=list)
    user_message: str = ""
    system_prompt: str = ""


class ResponsePlanner:
    """Plans response strategy based on retrieved memory and history context."""

    SYSTEM_PROMPT_BASE = """You are a friendly AI companion with persistent memory.
Rules:
1. If you have memory about what the user asks, use it naturally.
2. If you do NOT have memory about it, say so honestly. Example: "I don't think you've told me that yet."
3. NEVER fabricate or guess user information.
4. NEVER pretend to remember something you don't have in memory.
5. Be warm and conversational.
6. If a memory requires confirmation before sharing, ask the user first.
"""

    def plan(
        self, user_message: str, memory_context: list[dict],
        history_context: list[dict] | None = None,
    ) -> ResponsePlan:
        """
        Create a response plan based on query, available memory, and history.
        """
        history_context = history_context or []

        if memory_context:
            # Separate memories needing confirmation
            direct = []
            needs_confirm = []
            for mem in memory_context:
                if mem.get("needs_confirmation"):
                    needs_confirm.append(mem)
                else:
                    direct.append(mem)

            if needs_confirm and not direct:
                return self._plan_ask_confirm(user_message, needs_confirm)

            return self._plan_with_memory(user_message, direct, needs_confirm)

        if history_context:
            return self._plan_with_history(user_message, history_context)

        return self._plan_no_memory(user_message)

    def _plan_with_memory(
        self, message: str, direct: list[dict], needs_confirm: list[dict]
    ) -> ResponsePlan:
        """Plan response when relevant memories exist."""
        memory_lines = []
        for mem in direct:
            memory_lines.append(f"- {mem['text']} (confidence: {mem['confidence']:.0%})")

        memory_block = "\n".join(memory_lines)
        system = (
            self.SYSTEM_PROMPT_BASE
            + f"\n\nRelevant memories about this user:\n{memory_block}\n\n"
            "Use these memories naturally in your response."
        )

        return ResponsePlan(
            strategy="recall",
            memory_context=direct,
            needs_confirmation=needs_confirm,
            user_message=message,
            system_prompt=system,
        )

    def _plan_with_history(
        self, message: str, history_context: list[dict]
    ) -> ResponsePlan:
        """Plan response when conversation history has relevant info."""
        history_lines = [f"- User said: \"{h['text']}\"" for h in history_context[:5]]
        history_block = "\n".join(history_lines)

        is_memory_question = self._is_memory_question(message)

        if is_memory_question:
            system = (
                self.SYSTEM_PROMPT_BASE
                + f"\n\nRelevant conversation history:\n{history_block}\n\n"
                "The user is asking about something discussed in conversation. "
                "Use the conversation history above to answer."
            )
            strategy = "history_recall"
        else:
            system = (
                self.SYSTEM_PROMPT_BASE
                + f"\n\nRecent conversation context:\n{history_block}\n\n"
                "Use conversation context to respond naturally."
            )
            strategy = "history_recall"

        return ResponsePlan(
            strategy=strategy,
            history_context=history_context,
            user_message=message,
            system_prompt=system,
        )

    def _plan_no_memory(self, message: str) -> ResponsePlan:
        """Plan response when no relevant memory or history exists."""
        is_memory_question = self._is_memory_question(message)

        if is_memory_question:
            system = (
                self.SYSTEM_PROMPT_BASE
                + "\n\nYou have NO relevant memories about what the user is asking. "
                "Respond honestly that you don't have this information. "
                "Example: 'I don't think you've told me that yet. Would you like to tell me?'"
            )
            strategy = "honest_missing"
        else:
            system = (
                self.SYSTEM_PROMPT_BASE
                + "\n\nNo specific memories are relevant to this message. "
                "Respond naturally as a friendly companion."
            )
            strategy = "general"

        return ResponsePlan(
            strategy=strategy,
            user_message=message,
            system_prompt=system,
        )

    def _plan_ask_confirm(
        self, message: str, needs_confirm: list[dict]
    ) -> ResponsePlan:
        """Plan response when memories need confirmation before revealing."""
        topics = [m["attribute"] for m in needs_confirm]
        topic_str = ", ".join(topics)
        system = (
            self.SYSTEM_PROMPT_BASE
            + f"\n\nYou have sensitive information about: {topic_str}. "
            "Ask the user if they'd like you to share what you remember "
            "about these topics before revealing details."
        )
        return ResponsePlan(
            strategy="ask_confirm",
            needs_confirmation=needs_confirm,
            user_message=message,
            system_prompt=system,
        )

    def _is_greeting_or_smalltalk(self, message: str) -> bool:
        """Check if the message is a greeting or small talk, not a memory query."""
        msg_lower = message.lower().strip().rstrip("?!. ")
        greetings = [
            "hi", "hello", "hey", "howdy", "sup", "yo", "hola",
            "good morning", "good afternoon", "good evening", "good night",
            "how are you", "how are u", "how r u", "how's it going",
            "what's up", "whats up", "wassup", "how do you do",
            "how have you been", "how you doing", "how are things",
            "nice to meet you", "thanks", "thank you", "bye", "goodbye",
            "see you", "take care", "kaise ho", "kya haal hai",
            "namaste", "salam", "ok", "okay", "sure", "yes", "no",
            "hii", "hiii", "hiiii", "heyy", "heyyy",
            "hi how are u", "hi how are you", "hello how are you",
            "hey how are you", "hi there", "hello there",
        ]
        return any(msg_lower == g or msg_lower.startswith(g + " ") or msg_lower.startswith(g + ",") for g in greetings)

    def _is_memory_question(self, message: str) -> bool:
        """Check if the user is asking about something that would need memory."""
        # Greetings and small talk are never memory questions
        if self._is_greeting_or_smalltalk(message):
            return False

        indicators = [
            "what is", "what's", "what are", "who is", "do you know",
            "do you remember", "remember", "tell me about", "tell me",
            "kya hai", "kaun hai", "bata", "yaad", "what do you know",
            "where do", "where did", "when is", "when did", "when was",
            "how old", "how long", "how many", "how much",
            "what did", "what was", "what were", "which",
            "do i", "am i", "is my", "have i", "had i",
            "what should", "can you", "summarize", "summary",
            "what movies", "what languages", "what's my",
        ]
        msg_lower = message.lower()

        # Only treat "?" as memory question if it contains memory indicators
        if any(ind in msg_lower for ind in indicators):
            return True

        # A bare "?" question without indicators is likely small talk
        return False
