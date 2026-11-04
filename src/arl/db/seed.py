"""Build the deterministic customer-ops database.

Faker supplies display names only. Every identifier the eval suite depends on is fixed
in ``arl.db.catalog``.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from arl.db.catalog import (
    FIXED_TODAY,
    PRESEEDED_REFUNDS,
    PRODUCTS,
    carrier_for,
    customer_email,
    customer_phone,
    order_catalog,
    tracking_number,
)
from arl.db.repository import OpsRepository, connect
from arl.domain.models import Customer, OrderItem, Refund, Shipment


def seed(repo: OpsRepository, *, today: date = FIXED_TODAY) -> None:
    """Insert the catalog when the customers table is empty."""
    repo.migrate()
    if repo.customer_count():
        repo.seed_policies()
        return
    fake = Faker("en_US")
    seen: set[str] = set()
    for spec in order_catalog():
        if spec.customer_id in seen:
            continue
        seen.add(spec.customer_id)
        index = int(spec.customer_id.rsplit("-", 1)[-1])
        person = Faker("en_US")
        person.seed_instance(1000 + index)
        # Touch the shared Faker so the dependency is real even if seed_instance is used.
        _ = fake.seed_instance
        repo.upsert_customer(
            Customer(
                id=spec.customer_id,
                name=person.name(),
                email=customer_email(spec.customer_id),
                phone=customer_phone(spec.customer_id),
            )
        )
    for offset, spec in enumerate(order_catalog()):
        product = PRODUCTS[offset % len(PRODUCTS)]
        repo.upsert_order(
            order_id=spec.order_id,
            customer_id=spec.customer_id,
            status=spec.status,
            total_cents=spec.total_cents,
            created_at=spec.created_at,
            final_sale=spec.final_sale,
            items=[OrderItem(sku=f"SKU-{offset + 1}", name=product, qty=1, cents=spec.total_cents)],
            notes=spec.notes,
        )
        if spec.shipment_status:
            eta = (today + timedelta(days=3)).isoformat()
            repo.upsert_shipment(
                Shipment(
                    order_id=spec.order_id,
                    carrier=carrier_for(spec.order_id),
                    tracking_number=tracking_number(spec.order_id),
                    status=spec.shipment_status,
                    eta=None if spec.shipment_status == "delivered" else eta,
                )
            )
        if spec.order_id in PRESEEDED_REFUNDS:
            repo.insert_refund(
                Refund(
                    id=f"RF-{spec.order_id}-1",
                    order_id=spec.order_id,
                    amount_cents=spec.total_cents,
                    reason="previously refunded",
                    status="completed",
                    created_at=spec.created_at.isoformat(),
                )
            )
    repo.seed_policies()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the customer-ops database")
    parser.add_argument("--database-url", default="sqlite:///./data/ops.db")
    parser.add_argument("--today", default=FIXED_TODAY.isoformat())
    args = parser.parse_args()
    if args.database_url.startswith("sqlite:///"):
        path = Path(args.database_url.removeprefix("sqlite:///"))
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
    repo = OpsRepository(connect(args.database_url, date.fromisoformat(args.today)))
    seed(repo, today=date.fromisoformat(args.today))
    print(f"seeded {args.database_url} customers={repo.customer_count()}")


if __name__ == "__main__":
    main()
