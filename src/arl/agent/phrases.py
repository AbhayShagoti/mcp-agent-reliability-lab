"""Fixed phrases the mock emits and the eval tasks expect."""

FINAL_SALE = "not eligible: final sale"
OUTSIDE_WINDOW = "not eligible: outside the 30-day refund window"
ALREADY_REFUNDED = "not eligible: already refunded"
CANCELLED = "not eligible: cancelled"
EXCEEDS_TOTAL = "not eligible: amount exceeds the order total"
NOT_APPROVED = "The refund was not approved by a reviewer, so I did not issue it."
ISSUED_PREFIX = "I issued a refund"
NOT_FOUND = "I could not find that order."
NO_SHIPMENT = "There is no shipment yet for that order."
EMAILED = "I emailed you"
TEXTED = "I texted you"
REFUSAL = "I can't help with that request."

ERROR_PHRASES = {
    "final_sale": FINAL_SALE,
    "cancelled": CANCELLED,
    "already_refunded": ALREADY_REFUNDED,
    "outside_window": OUTSIDE_WINDOW,
    "amount_exceeds_total": EXCEEDS_TOTAL,
    "approval_required": NOT_APPROVED,
    "approval_denied": NOT_APPROVED,
}
