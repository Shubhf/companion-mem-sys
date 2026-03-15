"""
Improved Memory Pipeline — full-featured memory system.

Compared to baseline, this adds:
- Atomic fact extraction from natural language
- Correction tracking with supersession lineage
- Contradiction detection and resolution
- Sensitivity classification and access control
- Multi-signal ranked retrieval
- Honest uncertainty handling

This module provides a factory function that wires up all components.
"""

from memory_engine.memory_store import MemoryStore
from memory_engine.memory_ingestion import MemoryIngestionPipeline
from memory_engine.memory_retrieval import MemoryRetriever
from memory_engine.conflict_resolution import ConflictResolver
from memory_engine.sensitive_policy import SensitivePolicy
from chat_system.conversation_manager import ConversationManager


def create_improved_pipeline(
    db_path: str = "memories.db",
    embed_fn=None,
    llm_fn=None,
    llm_extract_fn=None,
) -> ConversationManager:
    """
    Factory function to create a fully-configured improved pipeline.

    Architecture:
    1. Ingestion: User message → atomic fact extraction → structured MemoryEntry
    2. Validation: Conflict detection, correction handling, sensitivity check
    3. Storage: SQLite persistence + FAISS vector index
    4. Retrieval: Multi-signal ranked retrieval (entity match + semantic + recency + confidence)
    5. Planning: Response strategy selection (recall / honest_missing / ask_confirm)
    6. Generation: Context-aware response with memory grounding

    Args:
        db_path: Path to SQLite database
        embed_fn: Function to compute embeddings (text -> list[float])
        llm_fn: Function to generate responses (messages -> str)
        llm_extract_fn: Function to extract facts from text (text -> list[dict])

    Returns:
        ConversationManager wired with all components
    """
    store = MemoryStore(db_path=db_path)

    manager = ConversationManager(
        store=store,
        embed_fn=embed_fn,
        llm_fn=llm_fn,
        llm_extract_fn=llm_extract_fn,
    )

    return manager


def create_improved_pipeline_with_embeddings(
    db_path: str = "memories.db",
    llm_fn=None,
    llm_extract_fn=None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> ConversationManager:
    """
    Factory with automatic sentence-transformers embedding setup.

    Requires: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model)

        def embed_fn(text: str) -> list[float]:
            return model.encode(text, normalize_embeddings=True).tolist()

    except ImportError:
        print(
            "Warning: sentence-transformers not installed. "
            "Running without embeddings."
        )
        embed_fn = None

    return create_improved_pipeline(
        db_path=db_path,
        embed_fn=embed_fn,
        llm_fn=llm_fn,
        llm_extract_fn=llm_extract_fn,
    )


class ImprovedPipelineComparison:
    """
    Utility to run both baseline and improved systems side-by-side
    for comparison benchmarking.
    """

    def __init__(self, improved_manager: ConversationManager, baseline_chat):
        self.improved = improved_manager
        self.baseline = baseline_chat

    def compare(self, user_id: str, message: str) -> dict:
        """Run the same message through both systems."""
        improved_result = self.improved.chat(user_id, message)
        baseline_result = self.baseline.chat(user_id, message)

        return {
            "message": message,
            "improved": {
                "response": improved_result["response"],
                "strategy": improved_result["strategy"],
                "memories_extracted": improved_result["memories_extracted"],
                "memories_used": len(improved_result["memories_used"]),
            },
            "baseline": {
                "response": baseline_result["response"],
                "strategy": baseline_result["strategy"],
                "memories_used": len(baseline_result["memories_used"]),
            },
        }
