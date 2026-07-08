import psycopg2
import psycopg2.extras
import uuid
import numpy as np
from typing import List, Dict, Any


import os

class VectorStore:
    def __init__(self):
        dbname = os.environ.get("DB_NAME", "vectordb")
        user = os.environ.get("DB_USER", "myuser")
        password = os.environ.get("DB_PASSWORD", "mypassword")
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "5432")

        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.conn.autocommit = False

        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id          uuid        PRIMARY KEY,
                    chunk_id    int,
                    text        text,
                    embedding   vector(384),
                    metadata    jsonb
                )
            """)
            self.conn.commit()

    def _cursor(self):
        """Return a fresh cursor (re-opens connection if closed)."""
        dbname = os.environ.get("DB_NAME", "vectordb")
        user = os.environ.get("DB_USER", "myuser")
        password = os.environ.get("DB_PASSWORD", "mypassword")
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "5432")

        if self.conn.closed:
            self.conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )
        return self.conn.cursor()

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index a list of document dicts produced by pipeline.process().

        Each dict must contain:
            - "text"      : str
            - "embedding" : np.ndarray | list
            - "metadata"  : dict   (chunk_id, pdf_path, etc.)
        """
        if not documents:
            return

        rows = []
        for doc in documents:
            chunk_id = doc.get("metadata", {}).get("chunk_id", 0)
            rows.append((
                str(uuid.uuid4()),
                chunk_id,
                doc["text"],
                doc["embedding"].tolist() if isinstance(doc["embedding"], np.ndarray) else doc["embedding"],
                psycopg2.extras.Json(doc.get("metadata", {}))
            ))

        with self._cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO documents (id, chunk_id, text, embedding, metadata) VALUES %s",
                rows
            )
            self.conn.commit()

    def add_document(self, text: str, embedding: np.ndarray, metadata: Dict = None):
        """Insert a single document chunk."""
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO documents (id, chunk_id, text, embedding, metadata) VALUES (%s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), 0, text, embedding_list, psycopg2.extras.Json(metadata or {}))
            )
            self.conn.commit()

    def search(self, embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Find top_k nearest chunks using cosine distance.

        Returns a list of dicts with keys: text, chunk_id, metadata, distance.
        """
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT text, chunk_id, metadata, embedding <=> %s::vector AS distance
                FROM documents
                ORDER BY distance ASC
                LIMIT %s
                """,
                (embedding_list, top_k)
            )
            rows = cur.fetchall()

        return [
            {"text": row[0], "chunk_id": row[1], "metadata": row[2], "distance": row[3]}
            for row in rows
        ]

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()

