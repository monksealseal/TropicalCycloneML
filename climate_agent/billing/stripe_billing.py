"""Stripe billing integration for subscription management.

Handles customer creation, subscription lifecycle, webhook processing,
and checkout session generation.
"""

from __future__ import annotations

import os
from typing import Any

import stripe
from sqlalchemy.orm import Session

from climate_agent.db.models import PlanTier, PLAN_LIMITS, User

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# These would be created in Stripe Dashboard and stored as env vars
STRIPE_PRICE_IDS = {
    PlanTier.STARTER: {
        "monthly": os.environ.get("STRIPE_PRICE_STARTER_MONTHLY", "price_starter_monthly"),
        "yearly": os.environ.get("STRIPE_PRICE_STARTER_YEARLY", "price_starter_yearly"),
    },
    PlanTier.PROFESSIONAL: {
        "monthly": os.environ.get("STRIPE_PRICE_PRO_MONTHLY", "price_pro_monthly"),
        "yearly": os.environ.get("STRIPE_PRICE_PRO_YEARLY", "price_pro_yearly"),
    },
}

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")


def create_stripe_customer(user: User) -> str:
    """Create a Stripe customer for a user and return the customer ID."""
    customer = stripe.Customer.create(
        email=user.email,
        name=user.name,
        metadata={"user_id": user.id, "plan": user.plan},
    )
    return customer.id


def create_checkout_session(
    db: Session,
    user: User,
    plan: PlanTier,
    billing_period: str = "monthly",
) -> dict[str, str]:
    """Create a Stripe Checkout Session for upgrading to a paid plan.

    Args:
        db: Database session.
        user: User initiating the checkout.
        plan: Target plan tier.
        billing_period: 'monthly' or 'yearly'.

    Returns:
        Dict with 'checkout_url' and 'session_id'.
    """
    if plan not in STRIPE_PRICE_IDS:
        raise ValueError(f"No Stripe price configured for plan: {plan}")

    price_id = STRIPE_PRICE_IDS[plan].get(billing_period)
    if not price_id:
        raise ValueError(f"No {billing_period} price for plan {plan}")

    # Create or retrieve Stripe customer
    if not user.stripe_customer_id:
        customer_id = create_stripe_customer(user)
        user.stripe_customer_id = customer_id
        db.commit()
    else:
        customer_id = user.stripe_customer_id

    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=f"{BASE_URL}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{BASE_URL}/billing/cancel",
        metadata={"user_id": user.id, "plan": plan.value},
        subscription_data={"metadata": {"user_id": user.id, "plan": plan.value}},
    )

    return {"checkout_url": session.url, "session_id": session.id}


def create_billing_portal_session(user: User) -> dict[str, str]:
    """Create a Stripe Billing Portal session for managing subscriptions."""
    if not user.stripe_customer_id:
        raise ValueError("User has no Stripe customer record")

    session = stripe.billing_portal.Session.create(
        customer=user.stripe_customer_id,
        return_url=f"{BASE_URL}/dashboard",
    )
    return {"portal_url": session.url}


def handle_webhook(payload: bytes, sig_header: str, db: Session) -> dict[str, Any]:
    """Process a Stripe webhook event.

    Handles subscription lifecycle events to keep user plans in sync.
    """
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        raise ValueError(f"Webhook verification failed: {e}")

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(data, db)
    elif event_type == "customer.subscription.updated":
        _handle_subscription_updated(data, db)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(data, db)
    elif event_type == "invoice.payment_failed":
        _handle_payment_failed(data, db)

    return {"event_type": event_type, "handled": True}


def _handle_checkout_completed(session_data: dict, db: Session) -> None:
    """Activate a subscription after successful checkout."""
    user_id = session_data.get("metadata", {}).get("user_id")
    plan = session_data.get("metadata", {}).get("plan")
    subscription_id = session_data.get("subscription")

    if not user_id or not plan:
        return

    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.plan = plan
        user.stripe_subscription_id = subscription_id
        db.commit()


def _handle_subscription_updated(subscription_data: dict, db: Session) -> None:
    """Update user plan when subscription changes."""
    user_id = subscription_data.get("metadata", {}).get("user_id")
    plan = subscription_data.get("metadata", {}).get("plan")

    if not user_id:
        return

    user = db.query(User).filter(User.id == user_id).first()
    if user and plan:
        user.plan = plan
        user.stripe_subscription_id = subscription_data.get("id")
        db.commit()


def _handle_subscription_deleted(subscription_data: dict, db: Session) -> None:
    """Downgrade user to free plan when subscription is cancelled."""
    user_id = subscription_data.get("metadata", {}).get("user_id")
    if not user_id:
        return

    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.plan = PlanTier.FREE.value
        user.stripe_subscription_id = None
        db.commit()


def _handle_payment_failed(invoice_data: dict, db: Session) -> None:
    """Handle failed payment (log, notify - don't immediately downgrade)."""
    # In production: send email notification, retry logic handled by Stripe
    pass


def get_plan_pricing() -> list[dict[str, Any]]:
    """Return pricing information for all plans."""
    plans = []
    for tier, limits in PLAN_LIMITS.items():
        plan_info = {
            "id": tier.value,
            "name": limits["name"],
            "description": limits["description"],
            "price": {
                "monthly_cents": limits["price_monthly_cents"],
                "monthly_display": f"${limits['price_monthly_cents'] / 100:.0f}/mo"
                if limits["price_monthly_cents"] > 0
                else "Free",
                "yearly_cents": limits["price_yearly_cents"],
                "yearly_display": f"${limits['price_yearly_cents'] / 100:.0f}/yr"
                if limits["price_yearly_cents"] > 0
                else "Free",
            },
            "limits": {
                "requests_per_month": limits["requests_per_month"],
                "requests_per_minute": limits["requests_per_minute"],
                "agent_turns_per_request": limits["agent_turns_per_request"],
            },
            "features": {
                "visualizations": limits["visualizations"],
                "forecasting": limits["forecasting"],
                "priority_support": limits["priority_support"],
            },
        }
        if tier == PlanTier.ENTERPRISE:
            plan_info["price"]["monthly_display"] = "Custom"
            plan_info["price"]["yearly_display"] = "Custom"
        plans.append(plan_info)
    return plans
