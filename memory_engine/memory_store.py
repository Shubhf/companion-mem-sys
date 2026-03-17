"""
Memory Store — SQLite/Turso-backed persistence with FAISS vector index.

Handles CRUD operations for memory entries with full user isolation.
Supports both local SQLite and cloud Turso (libSQL) for persistent storage.
The FAISS index is rebuilt on startup from stored embeddings.
"""

import sqlite3
import json
import os
import numpy as np
from datetime import datetime
from typing import Optional
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from memory_engine.memory_schema import (
    MemoryEntry, MemoryStatus, MemoryType, SensitivityLevel
)

EMBEDDING_DIM = 768  # Gemini text-embedding-004


class TursoHTTPConnection:
    """
    SQLite-compatible wrapper around Turso's HTTP API.
    No native library needed — works on any Python version.
    """

    def __init__(self, url: str, token: str):
        import httpx
        # Convert libsql:// to https://
        self.base_url = url.replace("libsql://", "https://").rstrip("/")
        self.token = token
        self.client = httpx.Client(timeout=30)
        self.row_factory = None

    def execute(self, sql: str, params: tuple = None):
        """Execute a SQL statement via Turso HTTP API."""
        body = {"statements": [{"q": sql}]}
        if params:
            body["statements"][0]["params"] = [
                {"type": "text", "value": str(p)} if p is not None else {"type": "null"}
                for p in params
            ]

        resp = self.client.post(
            f"{self.base_url}/v3/pipeline",
            json=body,
            headers={"Authorization": f"Bearer {self.token}"},
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Turso API error: {resp.status_code} {resp.text[:200]}")

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return _TursoResult([], [])

        result = results[0].get("response", {}).get("result", {})
        cols = [c["name"] for c in result.get("cols", [])]
        rows_raw = result.get("rows", [])

        rows = []
        for row in rows_raw:
            values = [cell.get("value") for cell in row]
            if self.row_factory == sqlite3.Row:
                rows.append(_TursoRow(cols, values))
            else:
                rows.append(values)

        return _TursoResult(rows, cols)

    def commit(self):
        pass  # HTTP API auto-commits


class _TursoResult:
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols
        self.lastrowid = None

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _TursoRow:
    """Mimics sqlite3.Row for compatibility."""
    def __init__(self, cols, values):
        self._data = dict(zip(cols, values))

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self._data.values())[key]
        return self._data.get(key)

    def keys(self):
        return self._data.keys()


def _connect_db(db_path: str, turso_url: str = None, turso_token: str = None):
    """Connect to Turso (cloud) or local SQLite."""
    url = turso_url or os.environ.get("TURSO_DATABASE_URL")
    token = turso_token or os.environ.get("TURSO_AUTH_TOKEN")

    if url and token:
        try:
            conn = TursoHTTPConnection(url, token)
            conn.row_factory = sqlite3.Row
            # Test connection
            conn.execute("SELECT 1")
            return conn, "turso"
        except Exception as e:
            print(f"Turso connection failed ({e}), falling back to local SQLite")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"


class MemoryStore:
    """SQLite/Turso + FAISS memory store with per-user isolation."""

    def __init__(self, db_path: str = "memories.db",
                 turso_url: str = None, turso_token: str = None):
        self.db_path = db_path
        self.conn, self.db_type = _connect_db(db_path, turso_url, turso_token)
        self._create_tables()
        self._faiss_indices: dict[str, faiss.IndexFlatIP] = {}
        self._id_maps: dict[str, list[str]] = {}
        if FAISS_AVAILABLE:
            self._rebuild_faiss_indices()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                entity TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'user_message',
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                sensitivity TEXT NOT NULL DEFAULT 'direct_recall',
                supersedes TEXT,
                embedding TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_status
            ON memories(user_id, status)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_entity
            ON memories(user_id, entity)
        """)
        self.conn.commit()

    def _rebuild_faiss_indices(self):
        """Rebuild FAISS indices from stored embeddings."""
        if not FAISS_AVAILABLE:
            return
        rows = self.conn.execute(
            "SELECT memory_id, user_id, embedding FROM memories "
            "WHERE status = 'active' AND embedding IS NOT NULL"
        ).fetchall()

        user_embeddings: dict[str, list] = {}
        user_ids_map: dict[str, list[str]] = {}

        for row in rows:
            uid = row["user_id"]
            emb = json.loads(row["embedding"])
            user_embeddings.setdefault(uid, []).append(emb)
            user_ids_map.setdefault(uid, []).append(row["memory_id"])

        for uid, embs in user_embeddings.items():
            arr = np.array(embs, dtype=np.float32)
            faiss.normalize_L2(arr)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            index.add(arr)
            self._faiss_indices[uid] = index
            self._id_maps[uid] = user_ids_map[uid]

    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns memory_id."""
        emb_json = json.dumps(entry.embedding) if entry.embedding else None
        self.conn.execute(
            "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                entry.memory_id, entry.user_id, entry.entity,
                entry.attribute, entry.value, entry.memory_type.value,
                entry.confidence, entry.source, entry.timestamp.isoformat(),
                entry.status.value, entry.sensitivity.value,
                entry.supersedes, emb_json
            )
        )
        self.conn.commit()

        # Update FAISS index
        if FAISS_AVAILABLE and entry.embedding and entry.status == MemoryStatus.ACTIVE:
            self._add_to_faiss(entry.user_id, entry.memory_id, entry.embedding)

        return entry.memory_id

    def _add_to_faiss(self, user_id: str, memory_id: str, embedding: list[float]):
        arr = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(arr)
        if user_id not in self._faiss_indices:
            self._faiss_indices[user_id] = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._id_maps[user_id] = []
        self._faiss_indices[user_id].add(arr)
        self._id_maps[user_id].append(memory_id)

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a single memory by ID."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_by_user(
        self, user_id: str, status: Optional[MemoryStatus] = MemoryStatus.ACTIVE
    ) -> list[MemoryEntry]:
        """Get all memories for a user, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE user_id = ? AND status = ? "
                "ORDER BY timestamp DESC",
                (user_id, status.value)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY timestamp DESC",
                (user_id,)
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_by_entity(
        self, user_id: str, entity: str
    ) -> list[MemoryEntry]:
        """Get active memories about a specific entity for a user."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE user_id = ? AND entity = ? "
            "AND status = 'active' ORDER BY timestamp DESC",
            (user_id, entity.lower())
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def supersede(self, old_memory_id: str, new_entry: MemoryEntry) -> str:
        """Mark old memory as superseded and store the new one."""
        self.conn.execute(
            "UPDATE memories SET status = ? WHERE memory_id = ?",
            (MemoryStatus.SUPERSEDED.value, old_memory_id)
        )
        new_entry.supersedes = old_memory_id
        new_entry.memory_type = MemoryType.CORRECTED_FACT
        result = self.store(new_entry)
        # Rebuild FAISS for this user since we invalidated an entry
        if FAISS_AVAILABLE:
            self._rebuild_user_faiss(new_entry.user_id)
        return result

    def _rebuild_user_faiss(self, user_id: str):
        """Rebuild FAISS index for a single user."""
        if not FAISS_AVAILABLE:
            return
        rows = self.conn.execute(
            "SELECT memory_id, embedding FROM memories "
            "WHERE user_id = ? AND status = 'active' AND embedding IS NOT NULL",
            (user_id,)
        ).fetchall()
        if not rows:
            self._faiss_indices.pop(user_id, None)
            self._id_maps.pop(user_id, None)
            return
        embs = [json.loads(r["embedding"]) for r in rows]
        ids = [r["memory_id"] for r in rows]
        arr = np.array(embs, dtype=np.float32)
        faiss.normalize_L2(arr)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(arr)
        self._faiss_indices[user_id] = index
        self._id_maps[user_id] = ids

    def search_similar(
        self, user_id: str, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[MemoryEntry, float]]:
        """Search for similar memories using FAISS. Returns (entry, score) pairs."""
        if not FAISS_AVAILABLE or user_id not in self._faiss_indices:
            return []
        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        index = self._faiss_indices[user_id]
        k = min(top_k, index.ntotal)
        if k == 0:
            return []
        scores, indices = index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            mid = self._id_maps[user_id][idx]
            entry = self.get(mid)
            if entry and entry.status == MemoryStatus.ACTIVE:
                results.append((entry, float(score)))
        return results

    def delete_user_memories(self, user_id: str):
        """Delete all memories for a user (for testing/privacy)."""
        self.conn.execute(
            "DELETE FROM memories WHERE user_id = ?", (user_id,)
        )
        self.conn.commit()
        self._faiss_indices.pop(user_id, None)
        self._id_maps.pop(user_id, None)

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        emb = json.loads(row["embedding"]) if row["embedding"] else None
        return MemoryEntry(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            entity=row["entity"],
            attribute=row["attribute"],
            value=row["value"],
            memory_type=MemoryType(row["memory_type"]),
            confidence=row["confidence"],
            source=row["source"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            status=MemoryStatus(row["status"]),
            sensitivity=SensitivityLevel(row["sensitivity"]),
            supersedes=row["supersedes"],
            embedding=emb,
        )

    def count(self, user_id: Optional[str] = None) -> int:
        if user_id:
            row = self.conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE user_id = ? AND status = 'active'",
                (user_id,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE status = 'active'"
            ).fetchone()
        return row["c"]
