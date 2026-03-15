"""
Chat Router — FastAPI endpoints for the companion memory system.

Provides REST API for chat, memory inspection, user management, and evals.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from memory_engine.memory_store import MemoryStore
from chat_system.conversation_manager import ConversationManager

app = FastAPI(
    title="Companion Memory System",
    description="Research-grade memory system for AI companion chat",
    version="0.1.0",
)

# Global instances (initialized in startup)
store: Optional[MemoryStore] = None
manager: Optional[ConversationManager] = None


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    strategy: str
    memories_used: list[dict]
    memories_extracted: list[dict]


class MemoryInspectResponse(BaseModel):
    user_id: str
    memory_count: int
    memories: list[dict]


@app.on_event("startup")
async def startup():
    global store, manager
    store = MemoryStore(db_path="memories.db")

    # Try to load Gemini LLM
    llm_fn = None
    try:
        from llm_provider import create_gemini_llm_fn
        llm_fn = create_gemini_llm_fn()
        print("Gemini LLM loaded successfully")
    except Exception as e:
        print(f"Running without LLM (Gemini not available: {e})")

    manager = ConversationManager(store=store, llm_fn=llm_fn)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message and get a response with memory integration."""
    if not manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    result = manager.chat(req.user_id, req.message)
    return ChatResponse(**result)


@app.get("/memories/{user_id}", response_model=MemoryInspectResponse)
async def get_memories(user_id: str):
    """Inspect all active memories for a user."""
    if not manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    memories = manager.get_user_memories(user_id)
    return MemoryInspectResponse(
        user_id=user_id,
        memory_count=len(memories),
        memories=memories,
    )


@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """Get conversation history for a user."""
    if not manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    return {"user_id": user_id, "history": manager.get_history(user_id)}


@app.delete("/memories/{user_id}")
async def delete_memories(user_id: str):
    """Delete all memories for a user (privacy/testing)."""
    if not store:
        raise HTTPException(status_code=503, detail="System not initialized")
    store.delete_user_memories(user_id)
    return {"status": "deleted", "user_id": user_id}


@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear conversation history for a user."""
    if not manager:
        raise HTTPException(status_code=503, detail="System not initialized")
    manager.clear_history(user_id)
    return {"status": "cleared", "user_id": user_id}


@app.get("/stats")
async def stats():
    """Get system statistics."""
    if not store:
        raise HTTPException(status_code=503, detail="System not initialized")
    return {
        "total_active_memories": store.count(),
    }
