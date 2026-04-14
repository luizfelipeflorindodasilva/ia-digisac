import os
import sqlite3
from datetime import datetime, timezone

DATABASE_URL = os.getenv("DATABASE_URL", "")


def _is_pg() -> bool:
    return DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")


def _ph() -> str:
    """Placeholder para query: %s (PostgreSQL) ou ? (SQLite)."""
    return "%s" if _is_pg() else "?"


def get_conn():
    if _is_pg():
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    return sqlite3.connect("logs.db")


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    if _is_pg():
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id  VARCHAR(36) PRIMARY KEY,
                started_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                msg_count   INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          SERIAL PRIMARY KEY,
                session_id  VARCHAR(36) NOT NULL,
                role        VARCHAR(20) NOT NULL,
                content     TEXT NOT NULL,
                created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id  TEXT PRIMARY KEY,
                started_at  TEXT,
                updated_at  TEXT,
                msg_count   INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT
            )
        """)
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_conversation(session_id: str):
    ph = _ph()
    conn = get_conn()
    cur = conn.cursor()
    now = _now()
    if _is_pg():
        cur.execute(f"""
            INSERT INTO conversations (session_id, started_at, updated_at, msg_count)
            VALUES ({ph}, NOW(), NOW(), 0)
            ON CONFLICT (session_id) DO UPDATE
            SET updated_at = NOW(),
                msg_count  = conversations.msg_count + 1
        """, (session_id,))
    else:
        cur.execute(f"""
            INSERT INTO conversations (session_id, started_at, updated_at, msg_count)
            VALUES ({ph}, {ph}, {ph}, 0)
            ON CONFLICT (session_id) DO UPDATE
            SET updated_at = {ph},
                msg_count  = conversations.msg_count + 1
        """, (session_id, now, now, now))
    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    ph = _ph()
    conn = get_conn()
    cur = conn.cursor()
    now = _now()
    cur.execute(
        f"INSERT INTO messages (session_id, role, content, created_at) VALUES ({ph},{ph},{ph},{ph})",
        (session_id, role, content, now)
    )
    conn.commit()
    conn.close()
    upsert_conversation(session_id)


def get_all_conversations() -> list[dict]:
    conn = get_conn()
    if not _is_pg():
        conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT c.session_id, c.started_at, c.updated_at,
               COUNT(m.id) AS msg_count
        FROM conversations c
        LEFT JOIN messages m ON m.session_id = c.session_id
        GROUP BY c.session_id, c.started_at, c.updated_at
        ORDER BY c.updated_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [{"session_id": r[0], "started_at": r[1], "updated_at": r[2], "msg_count": r[3]} for r in rows]


def get_conversation_messages(session_id: str) -> list[dict]:
    ph = _ph()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"SELECT role, content, created_at FROM messages WHERE session_id = {ph} ORDER BY id ASC",
        (session_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]
