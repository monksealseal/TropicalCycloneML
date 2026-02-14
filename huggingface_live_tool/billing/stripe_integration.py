"""
Stripe integration for CycloneAPI subscription billing.

Set environment variables:
    STRIPE_SECRET_KEY       — Stripe secret key (sk_live_... or sk_test_...)
    STRIPE_WEBHOOK_SECRET   — Webhook signing secret (whsec_...)
    CYCLONE_API_BASE_URL    — Base URL for success/cancel redirects
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .subscriptions import Tier, TIER_CONFIGS, get_store

log = logging.getLogger(__name__)

# Stripe price IDs — map each tier to its Stripe Price object.
# Set these via env vars or replace with your actual Stripe Price IDs.
STRIPE_PRICE_IDS: dict[str, dict[str, str]] = {
    Tier.STARTER.value: {
        "monthly": os.environ.get("STRIPE_PRICE_STARTER_MONTHLY", "price_starter_monthly"),
        "yearly": os.environ.get("STRIPE_PRICE_STARTER_YEARLY", "price_starter_yearly"),
    },
    Tier.PROFESSIONAL.value: {
        "monthly": os.environ.get("STRIPE_PRICE_PRO_MONTHLY", "price_pro_monthly"),
        "yearly": os.environ.get("STRIPE_PRICE_PRO_YEARLY", "price_pro_yearly"),
    },
    Tier.ENTERPRISE.value: {
        "monthly": os.environ.get("STRIPE_PRICE_ENT_MONTHLY", "price_ent_monthly"),
        "yearly": os.environ.get("STRIPE_PRICE_ENT_YEARLY", "price_ent_yearly"),
    },
}


def _get_stripe():
    """Lazy-import stripe to avoid hard dependency when not billing."""
    try:
        import stripe
    except ImportError:
        raise ImportError("stripe package required: pip install stripe")
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not stripe.api_key:
        raise RuntimeError("STRIPE_SECRET_KEY environment variable not set")
    return stripe


def create_checkout_session(
    email: str,
    tier: str,
    interval: str = "monthly",
    success_url: str | None = None,
    cancel_url: str | None = None,
) -> dict[str, Any]:
    """
    Create a Stripe Checkout Session for subscribing to a tier.

    Returns dict with ``checkout_url`` for the user to complete payment.
    """
    stripe = _get_stripe()
    base = os.environ.get("CYCLONE_API_BASE_URL", "http://localhost:8000")
    price_id = STRIPE_PRICE_IDS.get(tier, {}).get(interval)
    if not price_id:
        raise ValueError(f"No Stripe price configured for tier={tier} interval={interval}")

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer_email=email,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url or f"{base}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=cancel_url or f"{base}/billing/cancel",
        metadata={"tier": tier, "email": email},
    )
    return {"checkout_url": session.url, "session_id": session.id}


def handle_webhook(payload: bytes, sig_header: str) -> dict[str, Any]:
    """
    Process a Stripe webhook event.

    Call this from your webhook endpoint with the raw request body and
    the ``Stripe-Signature`` header value.
    """
    stripe = _get_stripe()
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not webhook_secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET not set")

    event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    store = get_store()

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        email = session.get("customer_email") or session["metadata"].get("email")
        tier_str = session["metadata"].get("tier", "starter")
        tier = Tier(tier_str)
        customer_id = session.get("customer")
        subscription_id = session.get("subscription")

        # Create an API key for the new subscriber
        rec = store.create_key(email, tier)
        rec.stripe_customer_id = customer_id
        rec.stripe_subscription_id = subscription_id
        log.info("New subscriber: %s tier=%s", email, tier.value)
        return {"action": "key_created", "email": email, "tier": tier.value}

    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        subscription_id = subscription.get("id")
        # Deactivate keys for this subscription
        for key_rec in store._keys.values():
            if key_rec.stripe_subscription_id == subscription_id:
                key_rec.is_active = False
                log.info("Deactivated key for subscription %s", subscription_id)
        return {"action": "subscription_cancelled", "subscription_id": subscription_id}

    elif event["type"] == "customer.subscription.updated":
        subscription = event["data"]["object"]
        subscription_id = subscription.get("id")
        # Could handle tier upgrades/downgrades here
        log.info("Subscription updated: %s", subscription_id)
        return {"action": "subscription_updated", "subscription_id": subscription_id}

    elif event["type"] == "invoice.payment_failed":
        invoice = event["data"]["object"]
        customer_id = invoice.get("customer")
        log.warning("Payment failed for customer %s", customer_id)
        return {"action": "payment_failed", "customer_id": customer_id}

    return {"action": "unhandled", "event_type": event["type"]}


def create_billing_portal_session(customer_id: str) -> dict[str, str]:
    """Create a Stripe Customer Portal session for managing subscription."""
    stripe = _get_stripe()
    base = os.environ.get("CYCLONE_API_BASE_URL", "http://localhost:8000")
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=f"{base}/dashboard",
    )
    return {"portal_url": session.url}
