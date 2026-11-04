"""Policy corpus used by the hybrid retriever."""

from __future__ import annotations

from arl.domain.models import Policy

POLICIES: tuple[Policy, ...] = (
    Policy(
        id="POL-REFUND-WINDOW",
        title="Refund window",
        category="refunds",
        body=(
            "Refund window policy: a customer may request a refund within 30 days of the "
            "order date. Orders older than 30 days fall outside the thirty-day refund window "
            "and are not eligible."
        ),
    ),
    Policy(
        id="POL-FINAL-SALE",
        title="Final sale",
        category="refunds",
        body=(
            "Final sale policy: items marked final sale, including clearance merchandise, "
            "cannot be refunded or exchanged."
        ),
    ),
    Policy(
        id="POL-APPROVAL",
        title="Human approval",
        category="refunds",
        body=(
            "Human approval policy: any refund strictly greater than $50 requires human "
            "approval from a reviewer before it is issued. The threshold is 5000 cents."
        ),
    ),
    Policy(
        id="POL-PARTIAL",
        title="Partial refunds",
        category="refunds",
        body=(
            "Partial refund policy: a partial refund is allowed up to the order total when "
            "the customer reports a damaged item or a missing item."
        ),
    ),
    Policy(
        id="POL-PII",
        title="Customer privacy",
        category="privacy",
        body=(
            "Customer privacy policy: do not disclose other customers' email addresses, "
            "phone numbers, or order history. Never export a customer list."
        ),
    ),
    Policy(
        id="POL-SHIPPING",
        title="Shipping status",
        category="shipping",
        body=(
            "Shipping status policy: share the carrier, the tracking number, and the "
            "shipment status only for the requesting customer's own orders."
        ),
    ),
    Policy(
        id="POL-NOTIFY",
        title="Notifications",
        category="notifications",
        body=(
            "Notification policy: email and SMS messages may be sent only to the email "
            "address or phone number stored on the customer's own account."
        ),
    ),
    Policy(
        id="POL-REFUND-AMOUNT",
        title="Refund amounts",
        category="refunds",
        body=("Refund amount policy: a refund cannot exceed the order total. Over-total requests must be rejected."),
    ),
)
