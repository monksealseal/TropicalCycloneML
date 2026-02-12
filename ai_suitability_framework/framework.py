"""
Main framework orchestrator for AI suitability assessment.

Provides the high-level API for evaluating whether a given earth science
problem is well-suited for AI/ML approaches. Combines quantitative scoring
across eight dimensions with domain-specific heuristic decision gates to
produce an actionable recommendation.

Usage
-----
    from ai_suitability_framework import AISuitabilityFramework, ProblemSpecification

    spec = ProblemSpecification(
        name="Global Weather Forecasting (1-10 day)",
        description="Medium-range deterministic weather prediction",
        target_timescale="weather",
        training_data_type="reanalysis",
        ...
    )
    framework = AISuitabilityFramework()
    result = framework.assess(spec)
    print(result.report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ai_suitability_framework.scoring import (
    SuitabilityScorer,
    CompositeScore,
    SuitabilityClass,
    ApproachType,
)
from ai_suitability_framework.heuristics import (
    EarthScienceHeuristics,
    GateResult,
    GateStatus,
)


class Recommendation(Enum):
    """Final recommendation categories."""

    PROCEED = "proceed"
    PROCEED_WITH_CAUTION = "proceed_with_caution"
    HYBRID_APPROACH = "hybrid_approach"
    RECONSIDER = "reconsider"
    NOT_RECOMMENDED = "not_recommended"

    @property
    def label(self) -> str:
        labels = {
            "proceed": "Proceed with ML Approach",
            "proceed_with_caution": "Proceed with Caution",
            "hybrid_approach": "Adopt Hybrid Physics-ML Approach",
            "reconsider": "Reconsider — Significant Barriers Exist",
            "not_recommended": "ML Not Recommended for This Problem",
        }
        return labels[self.value]


@dataclass
class ProblemSpecification:
    """
    Complete specification of an earth science problem for AI suitability
    assessment.

    Collects all information needed by both the quantitative scoring engine
    and the heuristic decision gates.
    """

    # ---- Identity ----
    name: str
    description: str = ""

    # ---- Problem characteristics (for heuristic gates) ----
    target_timescale: str = "weather"  # weather, subseasonal, seasonal, climate
    training_data_type: str = "reanalysis"  # reanalysis, observations, simulation
    has_well_defined_objective: bool = True
    involves_rare_events: bool = False
    requires_counterfactual: bool = False
    has_analytical_solution: bool = False
    involves_parameterized_physics: bool = False
    requires_conservation_laws: bool = False
    involves_coupled_systems: bool = False
    observational_coverage: str = "moderate"  # dense, moderate, sparse
    purpose: str = "emulation"  # emulation, discovery, operational, research

    # ---- Data Fidelity inputs ----
    reanalysis_quality: float = 0.5
    spatial_temporal_coverage: float = 0.5
    resolution_adequacy: float = 0.5
    known_bias_magnitude: float = 0.3

    # ---- Problem Learnability inputs ----
    signal_to_noise_ratio: float = 1.0
    stationarity: float = 0.7
    effective_dimensionality: float = 100.0
    ood_deployment_risk: float = 0.2
    deterministic_structure: float = 0.5

    # ---- Sample Sufficiency inputs ----
    sample_size: float = 10000
    effective_parameters: float = 1e6
    event_frequency: float = 1.0
    coverage_completeness: float = 0.8
    temporal_redundancy: float = 0.3

    # ---- Physics Baseline inputs ----
    physics_model_skill: float = 0.7
    theoretical_predictability_limit: float = 0.9
    physics_model_cost: float = 0.5
    known_physics_deficiencies: float = 0.3
    physics_community_maturity: float = 0.7

    # ---- Computational Justification inputs ----
    training_cost_gpu_hours: float = 1000.0
    inference_speedup_factor: float = 10.0
    expected_deployment_runs: float = 10000.0
    hardware_alignment: float = 0.7

    # ---- Physical Consistency inputs ----
    conservation_law_criticality: float = 0.5
    symmetry_requirements: float = 0.5
    multiscale_coupling: float = 0.3
    boundary_condition_complexity: float = 0.3
    constraint_enforceability: float = 0.5

    # ---- Uncertainty Quantification inputs ----
    uq_criticality: float = 0.5
    ml_uq_feasibility: float = 0.5
    ensemble_size_need: float = 0.3
    calibration_requirement: float = 0.5

    # ---- Extrapolation Risk inputs ----
    ood_deployment_fraction: float = 0.2
    nonstationarity: float = 0.2
    regime_coverage: float = 0.8
    forcing_novelty: float = 0.1
    tipping_point_proximity: float = 0.1

    def to_dimension_inputs(self) -> dict:
        """Convert specification to dimension-keyed input dictionaries."""
        return {
            "Data Fidelity": {
                "reanalysis_quality": self.reanalysis_quality,
                "spatial_temporal_coverage": self.spatial_temporal_coverage,
                "resolution_adequacy": self.resolution_adequacy,
                "known_bias_magnitude": self.known_bias_magnitude,
            },
            "Problem Learnability": {
                "signal_to_noise_ratio": self.signal_to_noise_ratio,
                "stationarity": self.stationarity,
                "effective_dimensionality": self.effective_dimensionality,
                "ood_deployment_risk": self.ood_deployment_risk,
                "deterministic_structure": self.deterministic_structure,
            },
            "Sample Sufficiency": {
                "sample_size": self.sample_size,
                "effective_parameters": self.effective_parameters,
                "event_frequency": self.event_frequency,
                "coverage_completeness": self.coverage_completeness,
                "temporal_redundancy": self.temporal_redundancy,
            },
            "Physics Baseline Comparison": {
                "physics_model_skill": self.physics_model_skill,
                "theoretical_predictability_limit": self.theoretical_predictability_limit,
                "physics_model_cost": self.physics_model_cost,
                "known_physics_deficiencies": self.known_physics_deficiencies,
                "physics_community_maturity": self.physics_community_maturity,
            },
            "Computational Justification": {
                "training_cost_gpu_hours": self.training_cost_gpu_hours,
                "inference_speedup_factor": self.inference_speedup_factor,
                "expected_deployment_runs": self.expected_deployment_runs,
                "hardware_alignment": self.hardware_alignment,
            },
            "Physical Consistency": {
                "conservation_law_criticality": self.conservation_law_criticality,
                "symmetry_requirements": self.symmetry_requirements,
                "multiscale_coupling": self.multiscale_coupling,
                "boundary_condition_complexity": self.boundary_condition_complexity,
                "constraint_enforceability": self.constraint_enforceability,
            },
            "Uncertainty Quantification": {
                "uq_criticality": self.uq_criticality,
                "ml_uq_feasibility": self.ml_uq_feasibility,
                "ensemble_size_need": self.ensemble_size_need,
                "calibration_requirement": self.calibration_requirement,
            },
            "Extrapolation Risk": {
                "ood_deployment_fraction": self.ood_deployment_fraction,
                "nonstationarity": self.nonstationarity,
                "regime_coverage": self.regime_coverage,
                "forcing_novelty": self.forcing_novelty,
                "tipping_point_proximity": self.tipping_point_proximity,
            },
        }

    def to_heuristic_spec(self) -> dict:
        """Convert specification to heuristic gate input dictionary."""
        return {
            "has_well_defined_objective": self.has_well_defined_objective,
            "training_data_type": self.training_data_type,
            "target_timescale": self.target_timescale,
            "involves_rare_events": self.involves_rare_events,
            "requires_counterfactual": self.requires_counterfactual,
            "has_analytical_solution": self.has_analytical_solution,
            "involves_parameterized_physics": self.involves_parameterized_physics,
            "requires_conservation_laws": self.requires_conservation_laws,
            "involves_coupled_systems": self.involves_coupled_systems,
            "observational_coverage": self.observational_coverage,
            "purpose": self.purpose,
        }


@dataclass
class AssessmentResult:
    """Complete assessment result combining scores and heuristic gates."""

    problem_name: str
    composite_score: CompositeScore
    gate_results: list  # list of GateResult
    gate_summary: dict
    final_recommendation: Recommendation
    report: str  # formatted human-readable report

    @property
    def overall_score(self) -> float:
        return self.composite_score.overall_score

    @property
    def suitability_class(self) -> SuitabilityClass:
        return self.composite_score.suitability_class

    @property
    def recommended_approach(self) -> ApproachType:
        return self.composite_score.recommended_approach


class AISuitabilityFramework:
    """
    Main orchestrator for AI suitability assessment of earth science problems.

    Combines:
    1. Quantitative scoring across 8 dimensions (data fidelity, learnability,
       sample sufficiency, physics baseline, computational justification,
       physical consistency, uncertainty quantification, extrapolation risk)
    2. Domain-specific heuristic decision gates (11 gates covering timescale,
       rare events, conservation laws, counterfactual reasoning, etc.)

    The final recommendation integrates both assessments, with heuristic
    gates able to override quantitative scores when fundamental barriers
    are identified.
    """

    def __init__(
        self,
        weight_profile: str = "default",
        custom_weights: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        weight_profile : str
            One of "default", "operational_forecasting", "climate_research",
            "discovery_science".
        custom_weights : dict, optional
            If provided, overrides the weight profile.
        """
        profiles = SuitabilityScorer.weight_profiles()
        weights = custom_weights or profiles.get(weight_profile, profiles["default"])
        self.scorer = SuitabilityScorer(weights=weights)
        self.heuristics = EarthScienceHeuristics()

    def assess(self, spec: ProblemSpecification) -> AssessmentResult:
        """
        Perform a complete AI suitability assessment.

        Parameters
        ----------
        spec : ProblemSpecification
            Complete problem specification.

        Returns
        -------
        AssessmentResult
            Full assessment including scores, gates, recommendation, and report.
        """
        # Step 1: Evaluate quantitative dimensions
        dimension_inputs = spec.to_dimension_inputs()
        composite = self.scorer.compute(dimension_inputs)

        # Step 2: Evaluate heuristic decision gates
        heuristic_spec = spec.to_heuristic_spec()
        gate_results = EarthScienceHeuristics.evaluate_all(heuristic_spec)
        gate_summary = EarthScienceHeuristics.summarize_gates(gate_results)

        # Step 3: Synthesize final recommendation
        recommendation = self._synthesize_recommendation(
            composite, gate_summary
        )

        # Step 4: Generate report
        report = self._generate_report(
            spec, composite, gate_results, gate_summary, recommendation
        )

        return AssessmentResult(
            problem_name=spec.name,
            composite_score=composite,
            gate_results=gate_results,
            gate_summary=gate_summary,
            final_recommendation=recommendation,
            report=report,
        )

    def _synthesize_recommendation(
        self,
        composite: CompositeScore,
        gate_summary: dict,
    ) -> Recommendation:
        """
        Synthesize final recommendation from scores and gates.

        Gate failures can override a high quantitative score, reflecting
        the principle that fundamental barriers (e.g. counterfactual
        requirements, climate-scale extrapolation) cannot be compensated
        by strong performance in other dimensions.
        """
        score = composite.overall_score
        gate_status = gate_summary["overall_gate_status"]
        n_failures = len(gate_summary["failed"])
        n_warnings = len(gate_summary["warnings"])
        max_severity = gate_summary["max_severity"]

        # Hard gate failures dominate
        if n_failures >= 2 or max_severity >= 0.9:
            return Recommendation.NOT_RECOMMENDED

        if n_failures == 1:
            # Single gate failure: recommend hybrid or reconsider
            if score >= 0.6:
                return Recommendation.HYBRID_APPROACH
            return Recommendation.RECONSIDER

        # No gate failures — score-driven recommendation
        if gate_status == GateStatus.WARN and n_warnings >= 3:
            # Multiple warnings: cap recommendation
            if score >= 0.6:
                return Recommendation.PROCEED_WITH_CAUTION
            return Recommendation.RECONSIDER

        if score >= 0.70:
            return Recommendation.PROCEED
        elif score >= 0.50:
            if n_warnings >= 2:
                return Recommendation.HYBRID_APPROACH
            return Recommendation.PROCEED_WITH_CAUTION
        elif score >= 0.35:
            return Recommendation.HYBRID_APPROACH
        else:
            return Recommendation.NOT_RECOMMENDED

    def _generate_report(
        self,
        spec: ProblemSpecification,
        composite: CompositeScore,
        gate_results: list,
        gate_summary: dict,
        recommendation: Recommendation,
    ) -> str:
        """Generate a comprehensive assessment report."""
        lines = []
        lines.append("=" * 72)
        lines.append("AI SUITABILITY FRAMEWORK — EARTH SCIENCE ASSESSMENT")
        lines.append("=" * 72)
        lines.append("")
        lines.append(f"Problem: {spec.name}")
        if spec.description:
            lines.append(f"Description: {spec.description}")
        lines.append(f"Timescale: {spec.target_timescale}")
        lines.append(f"Data Source: {spec.training_data_type}")
        lines.append(f"Purpose: {spec.purpose}")
        lines.append("")

        # Final recommendation (prominent)
        lines.append("*" * 72)
        lines.append(f"RECOMMENDATION: {recommendation.label}")
        lines.append(
            f"Approach: {composite.recommended_approach.label}"
        )
        lines.append("*" * 72)
        lines.append("")

        # Quantitative scores
        lines.append(self.scorer.format_report(composite))

        # Heuristic gates
        lines.append(EarthScienceHeuristics.format_gate_report(gate_results))

        # Synthesis
        lines.append("-" * 72)
        lines.append("SYNTHESIS")
        lines.append("-" * 72)
        lines.append(self._synthesis_narrative(
            composite, gate_summary, recommendation
        ))
        lines.append("")
        lines.append("=" * 72)

        return "\n".join(lines)

    def _synthesis_narrative(
        self,
        composite: CompositeScore,
        gate_summary: dict,
        recommendation: Recommendation,
    ) -> str:
        """Generate a narrative synthesis of the assessment."""
        parts = []

        score = composite.overall_score
        n_fail = len(gate_summary["failed"])
        n_warn = len(gate_summary["warnings"])

        if recommendation == Recommendation.PROCEED:
            parts.append(
                f"This problem scores {score:.2f} on the AI suitability index, "
                "indicating strong potential for ML approaches. "
                "All heuristic gates pass without critical barriers."
            )
        elif recommendation == Recommendation.PROCEED_WITH_CAUTION:
            parts.append(
                f"This problem scores {score:.2f} on the AI suitability index. "
                f"While generally favorable, {n_warn} heuristic warning(s) "
                "suggest areas requiring careful attention during development."
            )
        elif recommendation == Recommendation.HYBRID_APPROACH:
            parts.append(
                f"This problem scores {score:.2f} on the AI suitability index. "
                "The assessment suggests a hybrid physics-ML approach would be "
                "most appropriate, combining the strengths of both paradigms."
            )
        elif recommendation == Recommendation.RECONSIDER:
            parts.append(
                f"This problem scores {score:.2f} on the AI suitability index. "
                f"With {n_fail} gate failure(s) and {n_warn} warning(s), "
                "significant barriers exist for a purely ML approach. "
                "The problem may benefit from physics-based methods augmented "
                "with targeted ML components."
            )
        else:
            parts.append(
                f"This problem scores {score:.2f} on the AI suitability index. "
                "Multiple fundamental barriers suggest ML is not the appropriate "
                "primary methodology. Physics-based approaches are recommended."
            )

        # Highlight key strengths and weaknesses
        if composite.strengths:
            parts.append(
                "\nKey strengths: "
                + ", ".join(composite.strengths[:3])
                + "."
            )
        if composite.weaknesses:
            parts.append(
                "Key concerns: "
                + ", ".join(composite.weaknesses[:3])
                + "."
            )

        # Gate-specific guidance
        if gate_summary["failed"]:
            fail_names = [g.gate_name for g in gate_summary["failed"]]
            parts.append(
                "\nCritical barriers: "
                + ", ".join(fail_names)
                + ". These represent fundamental challenges that quantitative "
                "scores alone cannot capture."
            )

        return "\n".join(parts)

    @staticmethod
    def compare_problems(
        results: list,
    ) -> str:
        """
        Generate a comparative summary across multiple problem assessments.

        Useful for evaluating which problems in a research portfolio are
        most amenable to ML approaches.

        Parameters
        ----------
        results : list of AssessmentResult
            Assessment results to compare.

        Returns
        -------
        str
            Formatted comparison table.
        """
        lines = []
        lines.append("=" * 72)
        lines.append("COMPARATIVE AI SUITABILITY ASSESSMENT")
        lines.append("=" * 72)
        lines.append("")

        # Header
        lines.append(
            f"{'Problem':<35s} {'Score':>6s} {'Class':<25s} {'Recommendation':<20s}"
        )
        lines.append("-" * 86)

        # Sort by score descending
        sorted_results = sorted(
            results, key=lambda r: r.overall_score, reverse=True
        )

        for r in sorted_results:
            name = r.problem_name[:34]
            score = f"{r.overall_score:.3f}"
            cls = r.suitability_class.label[:24]
            rec = r.final_recommendation.label[:19]
            lines.append(f"{name:<35s} {score:>6s} {cls:<25s} {rec:<20s}")

        lines.append("")
        lines.append("-" * 86)

        # Summary statistics
        scores = [r.overall_score for r in sorted_results]
        if scores:
            lines.append(
                f"Score range: {min(scores):.3f} — {max(scores):.3f}  "
                f"(mean: {sum(scores)/len(scores):.3f})"
            )

        # Count by recommendation
        from collections import Counter
        rec_counts = Counter(r.final_recommendation.label for r in sorted_results)
        lines.append("Distribution of recommendations:")
        for rec_label, count in rec_counts.most_common():
            lines.append(f"  {rec_label}: {count}")
        lines.append("")
        lines.append("=" * 72)

        return "\n".join(lines)
