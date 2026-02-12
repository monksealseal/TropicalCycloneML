"""
Core scoring engine for AI suitability assessment.

Computes a composite suitability score from individual dimension scores,
handles weight calibration, and provides interpretive classification of
the overall assessment.

The composite score S is computed as:

    S = (Σ_i w_i * s_i * c_i) / (Σ_i w_i * c_i)

where:
    w_i : weight assigned to dimension i
    s_i : score of dimension i ∈ [0, 1]
    c_i : confidence of dimension i ∈ [0, 1]

This confidence-weighted formulation ensures that uncertain dimension
scores contribute less to the final assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ai_suitability_framework.dimensions import (
    DimensionScore,
    ScoringDimension,
    DataFidelity,
    ProblemLearnability,
    SampleSufficiency,
    PhysicsBaselineComparison,
    ComputationalJustification,
    PhysicalConsistency,
    UncertaintyQuantification,
    ExtrapolationRisk,
)


class SuitabilityClass(Enum):
    """Classification of overall AI suitability."""

    STRONG_CANDIDATE = "strong_candidate"
    HYBRID_RECOMMENDED = "hybrid_recommended"
    CAUTIOUS_APPLICATION = "cautious_application"
    PHYSICS_PREFERRED = "physics_preferred"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    @property
    def label(self) -> str:
        labels = {
            "strong_candidate": "Strong ML Candidate",
            "hybrid_recommended": "Hybrid Physics-ML Recommended",
            "cautious_application": "Cautious ML Application",
            "physics_preferred": "Physics-Based Approach Preferred",
            "insufficient_evidence": "Insufficient Evidence for Assessment",
        }
        return labels[self.value]


class ApproachType(Enum):
    """Recommended modeling approach based on assessment."""

    END_TO_END_ML = "end_to_end_ml"
    HYBRID_PHYSICS_ML = "hybrid_physics_ml"
    ML_PARAMETERIZATION = "ml_parameterization"
    ML_EMULATION = "ml_emulation"
    PHYSICS_WITH_ML_DA = "physics_with_ml_da"
    PHYSICS_BASED = "physics_based"

    @property
    def label(self) -> str:
        labels = {
            "end_to_end_ml": "End-to-End ML (Foundation Model / Digital Twin)",
            "hybrid_physics_ml": "Hybrid Physics-ML Model",
            "ml_parameterization": "ML Parameterization within Physics Model",
            "ml_emulation": "ML Emulation of Physics Model",
            "physics_with_ml_da": "Physics Model with ML-Enhanced Data Assimilation",
            "physics_based": "Traditional Physics-Based Model",
        }
        return labels[self.value]


@dataclass
class CompositeScore:
    """Result of the composite scoring computation."""

    overall_score: float
    overall_confidence: float
    suitability_class: SuitabilityClass
    recommended_approach: ApproachType
    dimension_scores: dict  # name -> DimensionScore
    weights_used: dict  # name -> weight
    critical_dimensions: list  # dimensions flagging concerns
    strengths: list  # dimensions showing high suitability
    weaknesses: list  # dimensions showing low suitability


# Default weight profile reflecting the workshop's emphasis areas
DEFAULT_WEIGHTS = {
    "Data Fidelity": 0.18,
    "Problem Learnability": 0.16,
    "Sample Sufficiency": 0.14,
    "Physics Baseline Comparison": 0.12,
    "Computational Justification": 0.10,
    "Physical Consistency": 0.12,
    "Uncertainty Quantification": 0.08,
    "Extrapolation Risk": 0.10,
}


class SuitabilityScorer:
    """
    Core scoring engine that combines dimension scores into a composite
    assessment of AI suitability.

    The scorer supports different weight profiles for different application
    contexts (e.g. operational forecasting vs climate research).
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        classification_thresholds: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        weights : dict, optional
            Custom weights for each dimension. If not provided, uses the
            default weight profile.
        classification_thresholds : dict, optional
            Custom thresholds for suitability classification. Defaults to:
            strong: >= 0.70, hybrid: >= 0.50, cautious: >= 0.35
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.thresholds = classification_thresholds or {
            "strong": 0.70,
            "hybrid": 0.50,
            "cautious": 0.35,
        }
        self._dimensions = self._init_dimensions()

    @staticmethod
    def _init_dimensions() -> dict:
        """Initialize all scoring dimensions."""
        return {
            "Data Fidelity": DataFidelity(),
            "Problem Learnability": ProblemLearnability(),
            "Sample Sufficiency": SampleSufficiency(),
            "Physics Baseline Comparison": PhysicsBaselineComparison(),
            "Computational Justification": ComputationalJustification(),
            "Physical Consistency": PhysicalConsistency(),
            "Uncertainty Quantification": UncertaintyQuantification(),
            "Extrapolation Risk": ExtrapolationRisk(),
        }

    def compute(self, dimension_inputs: dict) -> CompositeScore:
        """
        Compute the composite suitability score from dimension inputs.

        Parameters
        ----------
        dimension_inputs : dict
            Mapping from dimension name to dict of keyword arguments
            for that dimension's evaluate() method.

        Returns
        -------
        CompositeScore
            Full composite assessment including classification and
            recommended approach.
        """
        # Evaluate each dimension
        scores = {}
        for name, dim in self._dimensions.items():
            kwargs = dimension_inputs.get(name, {})
            scores[name] = dim.evaluate(**kwargs)

        # Compute confidence-weighted composite score
        weighted_sum = 0.0
        weight_sum = 0.0
        for name, ds in scores.items():
            w = self.weights.get(name, 0.1)
            weighted_sum += w * ds.score * ds.confidence
            weight_sum += w * ds.confidence

        overall_score = weighted_sum / max(weight_sum, 1e-10)

        # Overall confidence: weighted average of dimension confidences
        conf_weighted = sum(
            self.weights.get(n, 0.1) * ds.confidence
            for n, ds in scores.items()
        )
        conf_total = sum(self.weights.get(n, 0.1) for n in scores)
        overall_confidence = conf_weighted / max(conf_total, 1e-10)

        # Identify strengths, weaknesses, and critical flags
        strengths = [
            name for name, ds in scores.items() if ds.score >= 0.7
        ]
        weaknesses = [
            name for name, ds in scores.items() if ds.score < 0.35
        ]
        critical = [
            name for name, ds in scores.items()
            if ds.score < 0.25 and ds.confidence > 0.5
        ]

        # Classification
        suitability_class = self._classify(
            overall_score, overall_confidence, critical
        )

        # Approach recommendation
        recommended_approach = self._recommend_approach(
            overall_score, scores
        )

        return CompositeScore(
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            suitability_class=suitability_class,
            recommended_approach=recommended_approach,
            dimension_scores=scores,
            weights_used=self.weights.copy(),
            critical_dimensions=critical,
            strengths=strengths,
            weaknesses=weaknesses,
        )

    def _classify(
        self,
        score: float,
        confidence: float,
        critical: list,
    ) -> SuitabilityClass:
        """Classify overall suitability based on score and context."""
        if confidence < 0.35:
            return SuitabilityClass.INSUFFICIENT_EVIDENCE

        if critical:
            # Any critical failure drops classification
            if score >= self.thresholds["hybrid"]:
                return SuitabilityClass.HYBRID_RECOMMENDED
            return SuitabilityClass.PHYSICS_PREFERRED

        if score >= self.thresholds["strong"]:
            return SuitabilityClass.STRONG_CANDIDATE
        elif score >= self.thresholds["hybrid"]:
            return SuitabilityClass.HYBRID_RECOMMENDED
        elif score >= self.thresholds["cautious"]:
            return SuitabilityClass.CAUTIOUS_APPLICATION
        else:
            return SuitabilityClass.PHYSICS_PREFERRED

    def _recommend_approach(
        self,
        overall_score: float,
        scores: dict,
    ) -> ApproachType:
        """
        Recommend a modeling approach based on dimension score profile.

        The recommendation maps to the three thrusts identified at the
        workshop:
        1. ML for parameter/structure improvement → ML_PARAMETERIZATION
        2. Physics-forward hybrid models → HYBRID_PHYSICS_ML
        3. End-to-end ML (foundation models) → END_TO_END_ML
        """
        data_score = scores.get("Data Fidelity", DimensionScore("", 0.5, 0.5, "")).score
        physics_score = scores.get(
            "Physical Consistency", DimensionScore("", 0.5, 0.5, "")
        ).score
        extrap_score = scores.get(
            "Extrapolation Risk", DimensionScore("", 0.5, 0.5, "")
        ).score
        compute_score = scores.get(
            "Computational Justification", DimensionScore("", 0.5, 0.5, "")
        ).score
        baseline_score = scores.get(
            "Physics Baseline Comparison", DimensionScore("", 0.5, 0.5, "")
        ).score

        # End-to-end ML: high data fidelity, good learnability, low extrap risk
        if overall_score >= 0.70 and data_score >= 0.7 and extrap_score >= 0.6:
            return ApproachType.END_TO_END_ML

        # ML emulation: physics models exist and are expensive, data is good
        if compute_score >= 0.6 and data_score >= 0.6 and baseline_score >= 0.4:
            return ApproachType.ML_EMULATION

        # Hybrid: physics constraints important but ML can help
        if physics_score < 0.6 and overall_score >= 0.45:
            return ApproachType.HYBRID_PHYSICS_ML

        # ML parameterization: specific physics deficiencies to address
        if baseline_score >= 0.5 and physics_score >= 0.4:
            return ApproachType.ML_PARAMETERIZATION

        # ML-enhanced DA: data assimilation opportunities
        if data_score >= 0.5 and overall_score >= 0.35:
            return ApproachType.PHYSICS_WITH_ML_DA

        return ApproachType.PHYSICS_BASED

    @staticmethod
    def weight_profiles() -> dict:
        """
        Return predefined weight profiles for different contexts.

        Profiles reflect different priorities:
        - operational: emphasizes speed, reliability, data quality
        - research: emphasizes discovery potential, physical consistency
        - climate: emphasizes extrapolation risk, physical consistency
        """
        return {
            "default": DEFAULT_WEIGHTS.copy(),
            "operational_forecasting": {
                "Data Fidelity": 0.20,
                "Problem Learnability": 0.15,
                "Sample Sufficiency": 0.12,
                "Physics Baseline Comparison": 0.10,
                "Computational Justification": 0.18,
                "Physical Consistency": 0.08,
                "Uncertainty Quantification": 0.10,
                "Extrapolation Risk": 0.07,
            },
            "climate_research": {
                "Data Fidelity": 0.12,
                "Problem Learnability": 0.12,
                "Sample Sufficiency": 0.10,
                "Physics Baseline Comparison": 0.12,
                "Computational Justification": 0.06,
                "Physical Consistency": 0.18,
                "Uncertainty Quantification": 0.12,
                "Extrapolation Risk": 0.18,
            },
            "discovery_science": {
                "Data Fidelity": 0.15,
                "Problem Learnability": 0.20,
                "Sample Sufficiency": 0.12,
                "Physics Baseline Comparison": 0.15,
                "Computational Justification": 0.05,
                "Physical Consistency": 0.15,
                "Uncertainty Quantification": 0.08,
                "Extrapolation Risk": 0.10,
            },
        }

    def format_report(self, result: CompositeScore) -> str:
        """Generate a human-readable assessment report."""
        lines = []
        lines.append("=" * 72)
        lines.append("AI SUITABILITY ASSESSMENT REPORT")
        lines.append("=" * 72)
        lines.append("")

        # Overall score
        bar = self._score_bar(result.overall_score)
        lines.append(
            f"Overall Suitability Score: {result.overall_score:.3f} {bar}"
        )
        lines.append(
            f"Assessment Confidence:     {result.overall_confidence:.3f}"
        )
        lines.append(f"Classification:            {result.suitability_class.label}")
        lines.append(f"Recommended Approach:      {result.recommended_approach.label}")
        lines.append("")

        # Dimension breakdown
        lines.append("-" * 72)
        lines.append("DIMENSION SCORES")
        lines.append("-" * 72)
        for name in sorted(
            result.dimension_scores.keys(),
            key=lambda n: result.dimension_scores[n].score,
            reverse=True,
        ):
            ds = result.dimension_scores[name]
            w = result.weights_used.get(name, 0.0)
            bar = self._score_bar(ds.score)
            lines.append(
                f"  {name:<30s} {ds.score:.3f} {bar}  "
                f"(weight={w:.2f}, conf={ds.confidence:.2f})"
            )
        lines.append("")

        # Strengths and weaknesses
        if result.strengths:
            lines.append("Strengths:")
            for s in result.strengths:
                ds = result.dimension_scores[s]
                lines.append(f"  + {s}: {ds.rationale}")
        if result.weaknesses:
            lines.append("Weaknesses:")
            for w in result.weaknesses:
                ds = result.dimension_scores[w]
                lines.append(f"  - {w}: {ds.rationale}")
        if result.critical_dimensions:
            lines.append("CRITICAL FLAGS:")
            for c in result.critical_dimensions:
                ds = result.dimension_scores[c]
                lines.append(f"  !! {c}: {ds.rationale}")
        lines.append("")

        # Rationale detail
        lines.append("-" * 72)
        lines.append("DETAILED RATIONALE")
        lines.append("-" * 72)
        for name, ds in sorted(result.dimension_scores.items()):
            lines.append(f"  {name}:")
            lines.append(f"    {ds.rationale}")
            if ds.sub_scores:
                for k, v in ds.sub_scores.items():
                    lines.append(f"      {k}: {v:.4f}")
            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    @staticmethod
    def _score_bar(score: float, width: int = 20) -> str:
        """Create a simple ASCII progress bar."""
        filled = int(score * width)
        return "[" + "#" * filled + "." * (width - filled) + "]"
