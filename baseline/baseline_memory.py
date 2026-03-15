"""
Baseline Memory — simple text storage with similarity search.

No structured extraction, no correction tracking, no sensitivity handling.
Just stores raw conversation text and retrieves via embedding similarity.
"""

import json
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

EMBEDDING_DIM = 384


class BaselineMemory:
    """Stores raw conversation chunks and retrieves by similarity."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._indices: dict[str, faiss.IndexFlatIP] = {}
        self._id_maps: dict[str, list[int]] = {}
        self._next_id = 0

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS baseline_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                embedding TEXT
            )
        """)
        self.conn.commit()

    def store(self, user_id: str, text: str, embedding: Optional[list[float]] = None):
        """Store a raw text chunk."""
        emb_json = json.dumps(embedding) if embedding else None
        cursor = self.conn.execute(
            "INSERT INTO baseline_memories (user_id, text, timestamp, embedding) VALUES (?,?,?,?)",
            (user_id, text, datetime.utcnow().isoformat(), emb_json)
        )
        self.conn.commit()
        row_id = cursor.lastrowid

        if FAISS_AVAILABLE and embedding:
            self._add_to_index(user_id, row_id, embedding)

    def _add_to_index(self, user_id: str, row_id: int, embedding: list[float]):
        arr = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(arr)
        if user_id not in self._indices:
            self._indices[user_id] = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._id_maps[user_id] = []
        self._indices[user_id].add(arr)
        self._id_maps[user_id].append(row_id)

    def search(
        self, user_id: str, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """Search for similar text chunks."""
        if not FAISS_AVAILABLE or user_id not in self._indices:
            # Fallback: return most recent
            rows = self.conn.execute(
                "SELECT * FROM baseline_memories WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, top_k)
            ).fetchall()
            return [{"text": r["text"], "score": 1.0} for r in rows]

        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        index = self._indices[user_id]
        k = min(top_k, index.ntotal)
        if k == 0:
            return []
        scores, indices = index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row_id = self._id_maps[user_id][idx]
            row = self.conn.execute(
                "SELECT * FROM baseline_memories WHERE id = ?", (row_id,)
            ).fetchone()
            if row:
                results.append({"text": row["text"], "score": float(score)})
        return results

    def get_all(self, user_id: str) -> list[dict]:
        """Get all stored text for a user."""
        rows = self.conn.execute(
            "SELECT * FROM baseline_memories WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,)
        ).fetchall()
        return [{"text": r["text"], "timestamp": r["timestamp"]} for r in rows]

    def clear(self, user_id: str):
        self.conn.execute(
            "DELETE FROM baseline_memories WHERE user_id = ?", (user_id,)
        )
        self.conn.commit()
        self._indices.pop(user_id, None)
        self._id_maps.pop(user_id, None)
