# Production Thinking

## Latency

### Current Architecture
| Stage | Expected Latency | Notes |
|-------|-----------------|-------|
| Ingestion (regex extraction) | 5-15ms | CPU-bound, 40+ patterns, runs per message |
| Conflict resolution | 2-10ms | SQLite query + comparison |
| Storage (SQLite + FAISS) | 1-5ms | Local write, vector index update |
| Retrieval (entity + semantic) | 10-30ms | FAISS search + SQLite filter |
| Response planning | 1-3ms | Rule-based strategy selection |
| Generation (rule-based) | < 1ms | Template fill |
| Generation (LLM) | 500-2000ms | Depends on model and provider |

**Total without LLM**: ~20-60ms — well within interactive thresholds.
**Total with LLM**: ~550-2100ms — acceptable for chat, but needs streaming.

### Optimizations for Production
- **Embedding computation**: Batch embeddings at ingestion time, cache per message. Use a lightweight model (all-MiniLM-L6-v2, 384-dim) for sub-10ms embedding.
- **FAISS warm-up**: Pre-load per-user indices into memory on session start. For cold starts, use SQLite entity match as fallback while FAISS loads.
- **LLM streaming**: Stream generation tokens to reduce perceived latency. First token in ~200ms; full response complete by 1-2s.

## Cost

### Per-Message Cost Breakdown (with LLM)
| Component | Cost | Notes |
|-----------|------|-------|
| Embedding (local model) | ~$0 | Runs on inference server, amortized |
| Embedding (API, e.g., OpenAI) | ~$0.0001/msg | ~500 tokens/message |
| SQLite storage | ~$0 | Local, no per-query cost |
| LLM generation (Gemini Flash) | ~$0.001-0.003/msg | ~500 input + 200 output tokens |
| LLM generation (Gemini Pro) | ~$0.005-0.015/msg | Higher quality, higher cost |

**Estimated cost per active user per month** (assuming 50 messages/day):
- With Gemini Flash: ~$1.50-4.50/month
- With Gemini Pro: ~$7.50-22.50/month
- Without LLM (rule-based): ~$0 (compute only)

### Cost Control
- Use rule-based responses for simple recall and honest-missing cases (no LLM needed).
- Reserve LLM for complex generation: multi-fact responses, sensitive gating, emotional nuance.
- Cache frequent query patterns (e.g., "what's my name?" → direct lookup, no LLM).

## Scale

### Current Limits
- **SQLite**: Single-writer, adequate for single-user local deployment. Not suitable for multi-user server.
- **FAISS in-memory**: Scales linearly with memory count per user. At 10K memories per user, index is ~15MB.
- **Regex extraction**: CPU-bound, single-threaded. At 40+ patterns per message, throughput is ~1000 messages/sec on a single core.

### Scaling Path
| Stage | Current | 100 Users | 10K Users | 1M Users |
|-------|---------|-----------|-----------|----------|
| Storage | SQLite (local) | PostgreSQL | PostgreSQL + read replicas | Distributed store (CockroachDB / DynamoDB) |
| Vector index | FAISS in-memory | FAISS per-user, loaded on demand | Pinecone / Qdrant (managed) | Pinecone / Weaviate with sharding |
| Embeddings | Local model | Shared GPU inference server | Batched API calls | Dedicated embedding service |
| LLM | API calls | API with rate limiting | API with load balancing | Self-hosted or dedicated throughput |

### User Isolation at Scale
- Current: `user_id` column in SQLite, separate FAISS indices.
- At scale: Row-level security in PostgreSQL, namespace isolation in vector DB, tenant-scoped API keys.
- Critical invariant: **no query should ever omit the user_id filter.** This must be enforced at the ORM/query-builder level, not at the application level.

## Observability

### What to Monitor

| Signal | Metric | Alert Threshold |
|--------|--------|----------------|
| Memory recall accuracy | % of recall-strategy responses containing expected terms | < 85% over 1h window |
| Hallucination rate | % of responses flagged by post-generation fact-check | > 1% over 1h window |
| Correction success | % of corrections that result in supersession | < 90% |
| Sensitive memory violations | Count of DO_NOT_SURFACE values appearing in responses | > 0 (immediate alert) |
| Cross-user leakage | Count of responses containing another user's entity values | > 0 (P0 incident) |
| Ingestion failures | % of messages with zero extracted facts (for fact-bearing messages) | > 30% |
| Latency p99 | End-to-end response time | > 3s |
| Error rate | 5xx responses / total requests | > 1% |

### Logging Strategy
- **Structured logs** for every pipeline stage: ingestion result, conflict check result, retrieval scores, strategy selection, generation.
- **Audit trail** for memory mutations: every store, supersede, and delete operation logged with timestamp, user_id, and triggering message.
- **No PII in logs**: Log memory_ids and entity names, never raw values. Sensitive values are hashed in logs.
- **Eval regression dashboard**: Run golden eval suite on every deploy. Compare pass rate and avg_score to previous deploy. Block rollout if regression detected.

## Rollback Strategy

### Memory System Rollback
- **Schema migrations**: All SQLite/PostgreSQL schema changes use versioned migrations (Alembic or equivalent). Rollback = apply reverse migration.
- **Memory data**: Memory writes are append-only (supersession, not deletion). Rolling back the code does not require rolling back data.
- **FAISS indices**: Rebuilt from SQLite source of truth on startup. No separate rollback needed.

### Application Rollback
- **Blue-green deployment**: New version runs alongside old version. Traffic shifted gradually. If eval regression detected, shift traffic back.
- **Feature flags**: New extraction patterns, sensitivity rules, and response strategies gated behind flags. Disable without full rollback.
- **Canary deploys**: New version tested on 5% of traffic for 1 hour before full rollout. Monitor hallucination rate and recall accuracy during canary.

### Rollback Triggers
| Condition | Action |
|-----------|--------|
| Golden eval pass rate drops below 95% | Automatic rollback |
| Any sensitive memory violation in production | Immediate rollback + P0 investigation |
| Cross-user leakage detected | Immediate rollback + data audit |
| p99 latency > 5s for 10 minutes | Automatic rollback |
| Error rate > 5% for 5 minutes | Automatic rollback |

## Privacy and Deletion

### Data Retention
- **Conversation history**: Retained for 90 days by default. Configurable per user.
- **Memory entries**: Retained until explicitly deleted by user or superseded by correction.
- **Superseded memories**: Retained for audit trail (30 days), then hard-deleted.
- **Embeddings**: Deleted when source memory is deleted. FAISS index rebuilt.

### User Deletion (Right to be Forgotten)
1. **DELETE /user/{user_id}** endpoint triggers full deletion:
   - All memory entries (active, superseded, stale, deleted)
   - All conversation history
   - All FAISS index entries
   - All audit log entries (anonymized, not deleted — retain structure for system health)
2. **Verification**: After deletion, attempt retrieval for user_id. Must return empty. Run as automated check.
3. **Propagation**: If using distributed storage, deletion must propagate to all replicas within 24 hours (GDPR requirement).

### Sensitive Data Handling
- **DO_NOT_SURFACE memories** (passwords, credit cards, SSNs): Encrypted at rest with per-user keys. Never included in logs, analytics, or model training data.
- **ASK_BEFORE_REVEALING memories**: Stored normally but flagged. Response planner enforces gating.
- **No training on user data**: Memory content is never used for model fine-tuning without explicit opt-in.

### Compliance
- GDPR: Right to access (export all memories as JSON), right to deletion, right to correction (user can edit memories via UI).
- CCPA: Same deletion and access rights. No sale of personal data.
- SOC 2: Audit logs for all data access. Encryption at rest and in transit.
