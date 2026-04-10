"""Database access layer for conversation memory."""

from __future__ import annotations

from typing import List, Tuple

import asyncpg


class Database:
    """Async PostgreSQL wrapper for storing and reading chat memory."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(dsn=self._database_url, min_size=1, max_size=5)
        await self._init_schema()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _init_schema(self) -> None:
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized.")

        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """

        create_index_query = """
        CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created_at
        ON chat_messages (user_id, created_at DESC);
        """

        async with self._pool.acquire() as conn:
            await conn.execute(create_table_query)
            await conn.execute(create_index_query)

    async def add_message(self, user_id: int, role: str, content: str) -> None:
        """Persist one message row."""
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized.")

        query = """
        INSERT INTO chat_messages (user_id, role, content)
        VALUES ($1, $2, $3);
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, user_id, role, content)

    async def get_recent_messages(self, user_id: int, limit: int = 20) -> List[Tuple[str, str]]:
        """Read the latest messages for one user.

        Returns list of (role, content), oldest to newest.
        """
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized.")

        query = """
        SELECT role, content
        FROM chat_messages
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2;
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, user_id, limit)

        return [(row["role"], row["content"]) for row in reversed(rows)]
