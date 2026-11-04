"""Orders, notifications, and policy search over SQLite or Postgres."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from typing import Any, Protocol

from arl.db.policies import POLICIES
from arl.db.search import VECTOR_DIM, IndexedDoc, hash_embed, hybrid_scores, rank
from arl.domain.models import (
    Customer,
    Notification,
    Order,
    OrderItem,
    PolicyHit,
    Refund,
    Shipment,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  phone TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
  id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL,
  status TEXT NOT NULL,
  total_cents INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  final_sale INTEGER NOT NULL,
  items_json TEXT NOT NULL,
  notes TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shipments (
  order_id TEXT PRIMARY KEY,
  carrier TEXT NOT NULL,
  tracking_number TEXT NOT NULL,
  status TEXT NOT NULL,
  eta TEXT
);
CREATE TABLE IF NOT EXISTS refunds (
  id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL,
  amount_cents INTEGER NOT NULL,
  reason TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
  id TEXT PRIMARY KEY,
  customer_id TEXT NOT NULL,
  channel TEXT NOT NULL,
  destination TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS policies (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  category TEXT NOT NULL,
  body TEXT NOT NULL,
  embedding_json TEXT NOT NULL
);
"""


class Cursor(Protocol):
    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any: ...
    def fetchall(self) -> list[Any]: ...
    def fetchone(self) -> Any: ...


class Database:
    """Thin wrapper so the repository can talk to sqlite3 or psycopg."""

    def __init__(self, conn: Any, dialect: str, today: date) -> None:
        self.conn = conn
        self.dialect = dialect
        self.today = today

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        statement = sql if self.dialect == "sqlite" else sql.replace("?", "%s")
        cursor = self.conn.execute(statement, params)
        rows = cursor.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(dict(row))
            else:
                out.append(dict(row))
        return out

    def execute_write(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        statement = sql if self.dialect == "sqlite" else sql.replace("?", "%s")
        self.conn.execute(statement, params)
        self.conn.commit()

    def executescript(self, script: str) -> None:
        if self.dialect == "sqlite":
            self.conn.executescript(script)
            self.conn.commit()
            return
        for chunk in script.split(";"):
            statement = chunk.strip()
            if statement:
                self.conn.execute(statement)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def reopen_sqlite(self, path: str) -> None:
        """Point this handle at a new SQLite file. Used by the eval harness."""
        if self.dialect != "sqlite":
            raise RuntimeError("reopen_sqlite requires the sqlite dialect")
        self.conn.close()
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self.conn = conn


def connect_sqlite(path: str, today: date) -> Database:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return Database(conn, "sqlite", today)


def connect_postgres(url: str, today: date) -> Database:
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(url, row_factory=dict_row)
    return Database(conn, "postgres", today)


def connect(url: str, today: date) -> Database:
    if url.startswith("sqlite:///"):
        path = url.removeprefix("sqlite:///")
        return connect_sqlite(path, today)
    if url.startswith("postgres://") or url.startswith("postgresql://"):
        return connect_postgres(url, today)
    raise ValueError(f"unsupported database url: {url}")


class OpsRepository:
    """All agent-facing reads and writes go through this object."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def migrate(self) -> None:
        self.db.executescript(SCHEMA)
        if self.db.dialect == "postgres":
            self._migrate_pgvector()

    def _migrate_pgvector(self) -> None:
        self.db.execute_write("CREATE EXTENSION IF NOT EXISTS vector")
        self.db.execute_write(f"ALTER TABLE policies ADD COLUMN IF NOT EXISTS embedding vector({VECTOR_DIM})")

    def customer_count(self) -> int:
        rows = self.db.execute("SELECT COUNT(*) AS c FROM customers")
        return int(rows[0]["c"])

    def upsert_customer(self, customer: Customer) -> None:
        self.db.execute_write(
            """
            INSERT INTO customers (id, name, email, phone) VALUES (?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET name = excluded.name,
              email = excluded.email, phone = excluded.phone
            """,
            (customer.id, customer.name, customer.email, customer.phone),
        )

    def upsert_order(
        self,
        *,
        order_id: str,
        customer_id: str,
        status: str,
        total_cents: int,
        created_at: date,
        final_sale: bool,
        items: list[OrderItem],
        notes: str,
    ) -> None:
        payload = json.dumps([item.model_dump() for item in items])
        self.db.execute_write(
            """
            INSERT INTO orders
              (id, customer_id, status, total_cents, created_at, final_sale, items_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
              customer_id = excluded.customer_id,
              status = excluded.status,
              total_cents = excluded.total_cents,
              created_at = excluded.created_at,
              final_sale = excluded.final_sale,
              items_json = excluded.items_json,
              notes = excluded.notes
            """,
            (
                order_id,
                customer_id,
                status,
                total_cents,
                created_at.isoformat(),
                int(final_sale),
                payload,
                notes,
            ),
        )

    def upsert_shipment(self, shipment: Shipment) -> None:
        self.db.execute_write(
            """
            INSERT INTO shipments (order_id, carrier, tracking_number, status, eta)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (order_id) DO UPDATE SET
              carrier = excluded.carrier,
              tracking_number = excluded.tracking_number,
              status = excluded.status,
              eta = excluded.eta
            """,
            (
                shipment.order_id,
                shipment.carrier,
                shipment.tracking_number,
                shipment.status,
                shipment.eta,
            ),
        )

    def insert_refund(self, refund: Refund) -> None:
        self.db.execute_write(
            """
            INSERT INTO refunds (id, order_id, amount_cents, reason, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                refund.id,
                refund.order_id,
                refund.amount_cents,
                refund.reason,
                refund.status,
                refund.created_at,
            ),
        )

    def upsert_policy(self, policy_id: str, title: str, category: str, body: str) -> None:
        embedding = hash_embed(f"{title} {body}")
        self.db.execute_write(
            """
            INSERT INTO policies (id, title, category, body, embedding_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
              title = excluded.title,
              category = excluded.category,
              body = excluded.body,
              embedding_json = excluded.embedding_json
            """,
            (policy_id, title, category, body, json.dumps(embedding)),
        )
        if self.db.dialect == "postgres":
            self._write_pgvector(policy_id, embedding)

    def _write_pgvector(self, policy_id: str, embedding: list[float]) -> None:
        from pgvector.psycopg import register_vector

        register_vector(self.db.conn)
        literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
        self.db.execute_write(
            "UPDATE policies SET embedding = ?::vector WHERE id = ?",
            (literal, policy_id),
        )

    def get_customer(self, customer_id: str) -> Customer | None:
        rows = self.db.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
        if not rows:
            return None
        return Customer.model_validate(rows[0])

    def list_refunds(self, order_id: str) -> list[Refund]:
        rows = self.db.execute(
            "SELECT * FROM refunds WHERE order_id = ? ORDER BY id",
            (order_id,),
        )
        return [Refund.model_validate(row) for row in rows]

    def get_order(self, order_id: str, customer_id: str) -> Order | None:
        rows = self.db.execute(
            "SELECT * FROM orders WHERE id = ? AND customer_id = ?",
            (order_id, customer_id),
        )
        if not rows:
            return None
        return self._order_from_row(rows[0])

    def list_orders(self, customer_id: str, status: str | None = None) -> list[Order]:
        if status:
            rows = self.db.execute(
                """
                SELECT * FROM orders WHERE customer_id = ? AND status = ? ORDER BY id
                """,
                (customer_id, status),
            )
        else:
            rows = self.db.execute(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY id",
                (customer_id,),
            )
        return [self._order_from_row(row) for row in rows]

    def get_shipment(self, order_id: str, customer_id: str) -> Shipment | None:
        order = self.get_order(order_id, customer_id)
        if order is None:
            return None
        rows = self.db.execute("SELECT * FROM shipments WHERE order_id = ?", (order_id,))
        if not rows:
            return None
        return Shipment.model_validate(rows[0])

    def create_refund(
        self,
        *,
        order_id: str,
        amount_cents: int,
        reason: str,
        today: date,
    ) -> Refund:
        existing = self.list_refunds(order_id)
        refund = Refund(
            id=f"RF-{order_id}-{len(existing) + 1}",
            order_id=order_id,
            amount_cents=amount_cents,
            reason=reason,
            status="completed",
            created_at=today.isoformat(),
        )
        self.insert_refund(refund)
        order_rows = self.db.execute("SELECT total_cents, status FROM orders WHERE id = ?", (order_id,))
        if order_rows and amount_cents == int(order_rows[0]["total_cents"]):
            self.db.execute_write(
                "UPDATE orders SET status = ? WHERE id = ?",
                ("refunded", order_id),
            )
        return refund

    def add_notification(
        self,
        *,
        customer_id: str,
        channel: str,
        destination: str,
        subject: str,
        body: str,
        today: date,
    ) -> Notification:
        count_row = self.db.execute("SELECT COUNT(*) AS c FROM notifications")
        number = int(count_row[0]["c"]) + 1
        note = Notification(
            id=f"NT-{number}",
            customer_id=customer_id,
            channel=channel,
            destination=destination,
            subject=subject,
            body=body,
            created_at=today.isoformat(),
        )
        self.db.execute_write(
            """
            INSERT INTO notifications
              (id, customer_id, channel, destination, subject, body, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note.id,
                note.customer_id,
                note.channel,
                note.destination,
                note.subject,
                note.body,
                note.created_at,
            ),
        )
        return note

    def get_policy(self, policy_id: str) -> dict[str, Any] | None:
        rows = self.db.execute(
            "SELECT id, title, category, body FROM policies WHERE id = ?",
            (policy_id,),
        )
        return rows[0] if rows else None

    def search_policies(self, query: str, k: int = 3) -> list[PolicyHit]:
        rows = self.db.execute("SELECT id, title, category, body, embedding_json FROM policies")
        if not rows:
            return []
        docs = [IndexedDoc(doc_id=str(row["id"]), text=f"{row['title']} {row['body']}") for row in rows]
        embeddings = {str(row["id"]): _load_embedding(str(row["embedding_json"])) for row in rows}
        vector_scores = self._pgvector_scores(query) if self.db.dialect == "postgres" else None
        scores = hybrid_scores(query, docs, embeddings, vector_scores=vector_scores)
        by_id = {str(row["id"]): row for row in rows}
        hits: list[PolicyHit] = []
        for doc_id, score in rank(scores, k):
            row = by_id[doc_id]
            body = str(row["body"])
            hits.append(
                PolicyHit(
                    policy_id=doc_id,
                    title=str(row["title"]),
                    category=str(row["category"]),
                    snippet=body[:240],
                    score=round(score, 6),
                )
            )
        return hits

    def _pgvector_scores(self, query: str) -> dict[str, float] | None:
        try:
            from pgvector.psycopg import register_vector
        except ImportError:  # pragma: no cover
            return None
        register_vector(self.db.conn)
        literal = "[" + ",".join(f"{value:.8f}" for value in hash_embed(query)) + "]"
        rows = self.db.execute(
            "SELECT id, (embedding <=> ?::vector) AS distance FROM policies WHERE embedding IS NOT NULL",
            (literal,),
        )
        # pgvector cosine distance is 1 - cosine for normalized vectors.
        return {str(row["id"]): 1.0 - float(row["distance"]) for row in rows}

    def snapshot(self) -> dict[str, Any]:
        orders = self.db.execute("SELECT id, status, total_cents FROM orders ORDER BY id")
        refunds = self.db.execute("SELECT id, order_id, amount_cents, reason, status FROM refunds ORDER BY id")
        notes = self.db.execute(
            """
            SELECT id, customer_id, channel, destination, subject, body
            FROM notifications ORDER BY id
            """
        )
        return {
            "orders": {str(row["id"]): {"status": row["status"], "total_cents": row["total_cents"]} for row in orders},
            "refunds": [dict(row) for row in refunds],
            "notifications": [dict(row) for row in notes],
        }

    def seed_policies(self) -> None:
        for policy in POLICIES:
            self.upsert_policy(policy.id, policy.title, policy.category, policy.body)

    def _order_from_row(self, row: dict[str, Any]) -> Order:
        created = date.fromisoformat(str(row["created_at"]))
        items = [OrderItem.model_validate(item) for item in json.loads(str(row["items_json"]))]
        order_id = str(row["id"])
        return Order(
            id=order_id,
            customer_id=str(row["customer_id"]),
            status=str(row["status"]),
            total_cents=int(row["total_cents"]),
            created_at=created,
            final_sale=bool(row["final_sale"]),
            items=items,
            notes=str(row["notes"]),
            age_days=(self.db.today - created).days,
            refunds=self.list_refunds(order_id),
        )


def _load_embedding(payload: str) -> list[float]:
    raw = json.loads(payload)
    if not isinstance(raw, list):
        return []
    return [float(value) for value in raw]
