# System Design

## Pipeline Stages

### 1. Ingestion
- Receives raw user messages
- Extracts atomic (entity, attribute, value) triples via rule-based patterns or LLM
- Detects corrections ("X is not Y, X is Z")
- Detects hedging ("I think", "maybe") → lowers confidence

### 2. Memory Extraction
- Decomposes multi-fact messages into individual MemoryEntry objects
- Each fact gets a unique memory_id, timestamp, confidence score

### 3. Memory Validation
- ConflictResolver checks new facts against existing memories
- If conflict found: old memory → status=superseded, new memory → supersedes=old_id
- SensitivePolicy classifies each fact into sensitivity tiers

### 4. Storage
- SQLite: durable, queryable, supports user isolation via user_id index
- FAISS: in-memory vector index per user for semantic search
- Embeddings stored as JSON in SQLite, loaded into FAISS on startup

### 5. Retrieval
- Multi-signal ranking: entity_match (0.35) + semantic (0.25) + recency (0.20) + confidence (0.20)
- Recency uses exponential decay with 30-day half-life
- Filters out DO_NOT_SURFACE memories
- Returns top-k with scores

### 6. Ranking
- Deduplication across match types (entity + semantic)
- Boost for memories matching on multiple signals
- Active-status-only filtering

### 7. Response Planning
- **recall**: memories exist → include in system prompt
- **honest_missing**: user asks about unknown info → admit honestly
- **ask_confirm**: sensitive memories → ask before revealing
- **general**: no memory-related query → normal conversation

### 8. Generation
- LLM-based with memory-augmented system prompt
- Rule-based fallback when no LLM configured
- Never fabricates, never guesses

## Multi-User Isolation
- All queries scoped by user_id
- Separate FAISS indices per user
- No cross-user data leakage possible at the storage layer

## Correction Flow
```
User: "Spark is my rat"  →  memory(spark, species, rat, active)
User: "No, Spark is a hamster"  →  memory(spark, species, rat, superseded)
                                     memory(spark, species, hamster, active, supersedes=old_id)
```

## Sensitivity Tiers
| Tier | Behavior |
|------|----------|
| direct_recall | State freely |
| summarized_recall | Vague reference only |
| ask_before_revealing | Confirm with user first |
| do_not_surface | Never include in response |
