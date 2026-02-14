"""
TerraScore API — AI Suitability Assessment as a Service.

Production REST API for the AI Suitability Framework. Provides
programmatic access to earth science AI suitability assessments
with tiered access, structured JSON responses, and batch evaluation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Optional

from flask import Flask, request, jsonify, render_template, abort

from ai_suitability_framework.framework import (
    AISuitabilityFramework,
    ProblemSpecification,
    AssessmentResult,
    Recommendation,
)
from ai_suitability_framework.scoring import SuitabilityScorer, SuitabilityClass, ApproachType
from ai_suitability_framework.heuristics import EarthScienceHeuristics, GateStatus

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)
app.config["SECRET_KEY"] = os.environ.get("TERRASCORE_SECRET_KEY", "dev-key-change-in-production")


# ---------------------------------------------------------------------------
# API key store (in production, back this with a database)
# ---------------------------------------------------------------------------

class Tier(Enum):
    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

TIER_LIMITS = {
    Tier.FREE: {"requests_per_day": 10, "batch_size": 1, "weight_profiles": ["default"]},
    Tier.PROFESSIONAL: {"requests_per_day": 500, "batch_size": 10, "weight_profiles": "all"},
    Tier.ENTERPRISE: {"requests_per_day": 50000, "batch_size": 100, "weight_profiles": "all"},
}

# In-memory store for demo; replace with database in production
_api_keys: dict = {}
_usage: dict = {}  # key -> {"date": str, "count": int}


def generate_api_key(tier: Tier, owner: str) -> str:
    """Generate a new API key for the given tier."""
    raw = f"{owner}-{tier.value}-{uuid.uuid4()}-{time.time()}"
    key = f"ts_{tier.value[:3]}_{hashlib.sha256(raw.encode()).hexdigest()[:32]}"
    _api_keys[key] = {"tier": tier, "owner": owner, "created": datetime.now(timezone.utc).isoformat()}
    return key


def _get_tier(api_key: str) -> Optional[Tier]:
    """Look up tier for an API key."""
    entry = _api_keys.get(api_key)
    if entry:
        return entry["tier"]
    return None


def _check_rate_limit(api_key: str, tier: Tier) -> bool:
    """Check if the key is within its daily rate limit."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    usage = _usage.get(api_key, {"date": today, "count": 0})
    if usage["date"] != today:
        usage = {"date": today, "count": 0}
    limit = TIER_LIMITS[tier]["requests_per_day"]
    if usage["count"] >= limit:
        return False
    usage["count"] += 1
    _usage[api_key] = usage
    return True


# Seed a demo key for the free tier
_demo_key = generate_api_key(Tier.FREE, "demo")


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------

def require_api_key(f):
    """Decorator to require and validate API key."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not api_key:
            return jsonify({"error": "Missing API key. Include X-API-Key header or api_key parameter."}), 401

        tier = _get_tier(api_key)
        if tier is None:
            return jsonify({"error": "Invalid API key."}), 403

        if not _check_rate_limit(api_key, tier):
            limit = TIER_LIMITS[tier]["requests_per_day"]
            return jsonify({
                "error": f"Rate limit exceeded. {tier.value} tier allows {limit} requests/day.",
                "upgrade_url": "/pricing",
            }), 429

        request.tier = tier
        request.api_key = api_key
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_assessment(result: AssessmentResult) -> dict:
    """Convert AssessmentResult to JSON-serializable dict."""
    dimensions = {}
    for name, ds in result.composite_score.dimension_scores.items():
        dimensions[name] = {
            "score": round(ds.score, 4),
            "confidence": round(ds.confidence, 4),
            "rationale": ds.rationale,
            "sub_scores": {k: round(v, 4) for k, v in ds.sub_scores.items()},
        }

    gates = []
    for gr in result.gate_results:
        gate = {
            "gate_name": gr.gate_name,
            "status": gr.status.value,
            "message": gr.message,
            "severity": round(gr.severity, 4),
        }
        if gr.condition:
            gate["condition"] = gr.condition
        gates.append(gate)

    return {
        "problem_name": result.problem_name,
        "overall_score": round(result.overall_score, 4),
        "overall_confidence": round(result.composite_score.overall_confidence, 4),
        "suitability_class": result.suitability_class.value,
        "suitability_label": result.suitability_class.label,
        "recommended_approach": result.recommended_approach.value,
        "recommended_approach_label": result.recommended_approach.label,
        "recommendation": result.final_recommendation.value,
        "recommendation_label": result.final_recommendation.label,
        "dimensions": dimensions,
        "gates": gates,
        "gate_summary": {
            "total": result.gate_summary["total_gates"],
            "passed": len(result.gate_summary["passed"]),
            "warnings": len(result.gate_summary["warnings"]),
            "failed": len(result.gate_summary["failed"]),
            "conditional": len(result.gate_summary["conditions"]),
            "overall_status": result.gate_summary["overall_gate_status"].value,
            "max_severity": round(result.gate_summary["max_severity"], 4),
        },
        "strengths": result.composite_score.strengths,
        "weaknesses": result.composite_score.weaknesses,
        "critical_dimensions": result.composite_score.critical_dimensions,
    }


def _parse_problem_spec(data: dict) -> ProblemSpecification:
    """Parse JSON request body into ProblemSpecification."""
    # Extract only the fields that ProblemSpecification accepts
    valid_fields = set(ProblemSpecification.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return ProblemSpecification(**filtered)


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

@app.route("/")
def landing():
    """Serve the landing page."""
    return render_template("index.html", demo_key=_demo_key)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/v1/assess", methods=["POST"])
@require_api_key
def assess():
    """
    Perform a single AI suitability assessment.

    Expects JSON body with ProblemSpecification fields.
    Returns structured assessment result.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    if "name" not in data:
        return jsonify({"error": "Field 'name' is required"}), 400

    weight_profile = data.pop("weight_profile", "default")
    tier = request.tier

    # Check weight profile access
    allowed = TIER_LIMITS[tier]["weight_profiles"]
    if allowed != "all" and weight_profile not in allowed:
        return jsonify({
            "error": f"Weight profile '{weight_profile}' requires Professional tier or above.",
            "allowed_profiles": allowed,
            "upgrade_url": "/pricing",
        }), 403

    try:
        spec = _parse_problem_spec(data)
    except TypeError as e:
        return jsonify({"error": f"Invalid specification: {e}"}), 400

    framework = AISuitabilityFramework(weight_profile=weight_profile)
    result = framework.assess(spec)

    return jsonify({
        "status": "success",
        "assessment": _serialize_assessment(result),
        "meta": {
            "api_version": "1.0",
            "weight_profile": weight_profile,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    })


@app.route("/api/v1/assess/batch", methods=["POST"])
@require_api_key
def assess_batch():
    """
    Perform batch assessment of multiple problems.

    Expects JSON body with {"problems": [...], "weight_profile": "default"}.
    Returns comparative analysis.
    """
    data = request.get_json()
    if not data or "problems" not in data:
        return jsonify({"error": "Request body must include 'problems' array"}), 400

    tier = request.tier
    max_batch = TIER_LIMITS[tier]["batch_size"]
    problems = data["problems"]

    if len(problems) > max_batch:
        return jsonify({
            "error": f"Batch size {len(problems)} exceeds {tier.value} tier limit of {max_batch}.",
            "upgrade_url": "/pricing",
        }), 403

    weight_profile = data.get("weight_profile", "default")
    framework = AISuitabilityFramework(weight_profile=weight_profile)

    results = []
    assessments = []
    for problem_data in problems:
        if "name" not in problem_data:
            return jsonify({"error": "Each problem must include 'name' field"}), 400
        try:
            spec = _parse_problem_spec(problem_data)
        except TypeError as e:
            return jsonify({"error": f"Invalid specification for '{problem_data.get('name', '?')}': {e}"}), 400
        result = framework.assess(spec)
        results.append(result)
        assessments.append(_serialize_assessment(result))

    # Comparative ranking
    ranked = sorted(assessments, key=lambda a: a["overall_score"], reverse=True)

    return jsonify({
        "status": "success",
        "assessments": ranked,
        "comparison": {
            "count": len(ranked),
            "score_range": {
                "min": round(min(a["overall_score"] for a in ranked), 4),
                "max": round(max(a["overall_score"] for a in ranked), 4),
                "mean": round(sum(a["overall_score"] for a in ranked) / len(ranked), 4),
            },
            "recommendation_distribution": _count_recommendations(ranked),
        },
        "meta": {
            "api_version": "1.0",
            "weight_profile": weight_profile,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    })


@app.route("/api/v1/dimensions", methods=["GET"])
def list_dimensions():
    """List all scoring dimensions with descriptions."""
    scorer = SuitabilityScorer()
    dims = []
    for name, dim in scorer._dimensions.items():
        dims.append({
            "name": name,
            "description": dim.description,
            "default_weight": round(scorer.weights.get(name, 0.0), 4),
        })
    return jsonify({"dimensions": dims})


@app.route("/api/v1/gates", methods=["GET"])
def list_gates():
    """List all heuristic decision gates."""
    gate_names = [
        "Well-Defined Objective",
        "Training Distribution Representativeness",
        "Rare Event Handling",
        "Analytical Solution Check",
        "Timescale Assessment",
        "Counterfactual Reasoning",
        "Parameterization Opportunity",
        "Conservation Law Requirements",
        "Coupled System Complexity",
        "Observational Coverage",
        "Emulation vs Discovery",
    ]
    return jsonify({"gates": gate_names, "total": len(gate_names)})


@app.route("/api/v1/weight-profiles", methods=["GET"])
def list_weight_profiles():
    """List available weight profiles."""
    profiles = SuitabilityScorer.weight_profiles()
    return jsonify({"profiles": {k: {dk: round(dv, 4) for dk, dv in v.items()} for k, v in profiles.items()}})


@app.route("/api/v1/examples", methods=["GET"])
def list_examples():
    """List available pre-configured examples."""
    from ai_suitability_framework.examples import ALL_EXAMPLES
    examples = []
    for key, fn in ALL_EXAMPLES.items():
        spec = fn()
        examples.append({
            "key": key,
            "name": spec.name,
            "description": spec.description,
            "timescale": spec.target_timescale,
            "purpose": spec.purpose,
        })
    return jsonify({"examples": examples})


@app.route("/api/v1/examples/<name>/assess", methods=["GET"])
@require_api_key
def assess_example(name: str):
    """Run assessment on a pre-configured example."""
    from ai_suitability_framework.examples import ALL_EXAMPLES
    if name not in ALL_EXAMPLES:
        return jsonify({"error": f"Unknown example '{name}'", "available": list(ALL_EXAMPLES.keys())}), 404

    weight_profile = request.args.get("weight_profile", "default")
    spec = ALL_EXAMPLES[name]()
    framework = AISuitabilityFramework(weight_profile=weight_profile)
    result = framework.assess(spec)

    return jsonify({
        "status": "success",
        "assessment": _serialize_assessment(result),
        "meta": {
            "api_version": "1.0",
            "example": name,
            "weight_profile": weight_profile,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    })


@app.route("/api/v1/quick-score", methods=["POST"])
@require_api_key
def quick_score():
    """
    Simplified endpoint returning just the score and recommendation.
    Designed for integration into CI/CD pipelines and decision workflows.
    """
    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "JSON body with 'name' field required"}), 400

    try:
        spec = _parse_problem_spec(data)
    except TypeError as e:
        return jsonify({"error": str(e)}), 400

    framework = AISuitabilityFramework()
    result = framework.assess(spec)

    return jsonify({
        "name": result.problem_name,
        "score": round(result.overall_score, 4),
        "recommendation": result.final_recommendation.value,
        "approach": result.recommended_approach.value,
        "gate_status": result.gate_summary["overall_gate_status"].value,
        "proceed": result.final_recommendation in (
            Recommendation.PROCEED,
            Recommendation.PROCEED_WITH_CAUTION,
        ),
    })


# ---------------------------------------------------------------------------
# Key management (in production, protect these endpoints)
# ---------------------------------------------------------------------------

@app.route("/api/v1/keys/generate", methods=["POST"])
def generate_key():
    """Generate a new API key (demo endpoint)."""
    data = request.get_json() or {}
    owner = data.get("owner", "anonymous")
    tier_str = data.get("tier", "free")
    try:
        tier = Tier(tier_str)
    except ValueError:
        return jsonify({"error": f"Invalid tier '{tier_str}'. Options: free, professional, enterprise"}), 400

    key = generate_api_key(tier, owner)
    return jsonify({
        "api_key": key,
        "tier": tier.value,
        "owner": owner,
        "limits": TIER_LIMITS[tier],
    })


# ---------------------------------------------------------------------------
# Pricing page
# ---------------------------------------------------------------------------

@app.route("/pricing")
def pricing():
    """Pricing information."""
    return render_template("pricing.html")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_recommendations(assessments: list) -> dict:
    """Count recommendation distribution in batch results."""
    counts = {}
    for a in assessments:
        label = a["recommendation_label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def create_app():
    """Application factory."""
    return app


if __name__ == "__main__":
    port = int(os.environ.get("TERRASCORE_PORT", 5000))
    print(f"\n  TerraScore API starting on http://localhost:{port}")
    print(f"  Demo API key: {_demo_key}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
