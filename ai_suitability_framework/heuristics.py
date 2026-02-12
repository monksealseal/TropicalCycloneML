"""
Domain-specific heuristics and decision gates for earth science AI suitability.

Decision gates are hard constraints that can override or qualify the
quantitative suitability score. They capture domain knowledge about when
ML approaches face fundamental barriers that numerical scores alone may
not adequately reflect.

These heuristics are grounded in the following workshop observations:
- Historical training data may be insufficient for extrapolation to OOD
  or counterfactual scenarios (e.g. anthropogenic forcing pathways)
- The benchmark for data-driven ML must be data-constrained physical models
- Physics and ML are both important; patterns in data alone are not a
  replacement for human expertise/domain knowledge
- Rare events (tropical cyclones, blocking) present sample size challenges
- Cloud microphysics and aerosol interactions are dominant uncertainty sources
  in physics models but absent from ML foundation model architectures
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GateStatus(Enum):
    """Outcome of a decision gate evaluation."""

    PASS = "pass"           # No barrier identified
    WARN = "warn"           # Proceed with caution
    FAIL = "fail"           # Fundamental barrier to ML approach
    CONDITIONAL = "conditional"  # Pass only if specific conditions are met


@dataclass
class GateResult:
    """Result of evaluating a single decision gate."""

    gate_name: str
    status: GateStatus
    message: str
    condition: Optional[str] = None  # Required condition if CONDITIONAL
    severity: float = 0.0  # 0=informational, 1=critical


class HeuristicGate:
    """Base class for domain-specific decision gates."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class EarthScienceHeuristics:
    """
    Collection of earth science-specific heuristic rules and decision gates.

    These heuristics encode domain knowledge that supplements the
    quantitative scoring dimensions. They serve as pre-assessment gates
    and post-assessment qualifiers.

    The heuristics are organized into categories:
    1. Data sufficiency gates
    2. Physical consistency gates
    3. Extrapolation safety gates
    4. Problem formulation gates
    5. Emulation vs discovery qualifiers
    """

    @staticmethod
    def evaluate_all(problem_spec: dict) -> list:
        """
        Evaluate all heuristic gates against a problem specification.

        Parameters
        ----------
        problem_spec : dict
            Problem specification containing relevant attributes.
            Expected keys (all optional, with sensible defaults):
            - has_well_defined_objective: bool
            - training_data_type: str ("reanalysis", "observations", "simulation")
            - target_timescale: str ("weather", "subseasonal", "seasonal", "climate")
            - involves_rare_events: bool
            - requires_counterfactual: bool
            - has_analytical_solution: bool
            - involves_parameterized_physics: bool
            - requires_conservation_laws: bool
            - involves_coupled_systems: bool
            - observational_coverage: str ("dense", "moderate", "sparse")
            - purpose: str ("emulation", "discovery", "operational", "research")

        Returns
        -------
        list of GateResult
            Results of all gate evaluations.
        """
        gates = EarthScienceHeuristics()
        results = []

        results.append(gates.objective_function_gate(problem_spec))
        results.append(gates.training_distribution_gate(problem_spec))
        results.append(gates.rare_event_gate(problem_spec))
        results.append(gates.analytical_solution_gate(problem_spec))
        results.append(gates.timescale_gate(problem_spec))
        results.append(gates.counterfactual_gate(problem_spec))
        results.append(gates.parameterization_opportunity_gate(problem_spec))
        results.append(gates.conservation_gate(problem_spec))
        results.append(gates.coupled_system_gate(problem_spec))
        results.append(gates.observational_coverage_gate(problem_spec))
        results.append(gates.emulation_vs_discovery_gate(problem_spec))

        return results

    def objective_function_gate(self, spec: dict) -> GateResult:
        """
        Gate 1: Is there a well-defined objective function?

        ML requires a clear optimization target. Problems where success
        criteria are ambiguous, multi-objective without clear weighting,
        or require human judgment for evaluation are less suited for
        purely automated ML approaches.
        """
        has_objective = spec.get("has_well_defined_objective", True)

        if has_objective:
            return GateResult(
                gate_name="Well-Defined Objective",
                status=GateStatus.PASS,
                message="Problem has a clear, quantifiable objective function",
                severity=0.0,
            )
        return GateResult(
            gate_name="Well-Defined Objective",
            status=GateStatus.WARN,
            message=(
                "Ambiguous or multi-objective optimization target — ML training "
                "requires explicit loss function formulation. Consider whether "
                "the objective can be decomposed into measurable components"
            ),
            severity=0.6,
        )

    def training_distribution_gate(self, spec: dict) -> GateResult:
        """
        Gate 2: Is the training data representative of deployment conditions?

        The workshop emphasized that ML foundation models' success in weather
        prediction stems from ERA5 comprehensively representing the distribution
        of atmospheric states encountered during deployment.
        """
        data_type = spec.get("training_data_type", "reanalysis")
        timescale = spec.get("target_timescale", "weather")

        if data_type == "reanalysis" and timescale == "weather":
            return GateResult(
                gate_name="Training Distribution Representativeness",
                status=GateStatus.PASS,
                message=(
                    "Reanalysis data at weather timescales provides comprehensive "
                    "coverage of the target distribution — this is the regime where "
                    "ML weather emulators have demonstrated greatest success"
                ),
                severity=0.0,
            )
        elif data_type == "reanalysis" and timescale in ("subseasonal", "seasonal"):
            return GateResult(
                gate_name="Training Distribution Representativeness",
                status=GateStatus.WARN,
                message=(
                    "Reanalysis provides good coverage at subseasonal-to-seasonal "
                    "timescales, but predictability is limited by the chaotic nature "
                    "of the atmosphere at these timescales and interactions with "
                    "slowly-varying boundary conditions"
                ),
                severity=0.3,
            )
        elif timescale == "climate":
            return GateResult(
                gate_name="Training Distribution Representativeness",
                status=GateStatus.FAIL,
                message=(
                    "Climate prediction requires extrapolation to conditions not "
                    "present in historical training data. Models trained on weather-"
                    "scale loss functions may not produce credible climate statistics. "
                    "Requires explicit treatment of time-dependent forcing and "
                    "robustness to out-of-distribution conditions"
                ),
                severity=0.9,
            )
        elif data_type == "observations":
            coverage = spec.get("observational_coverage", "moderate")
            if coverage == "sparse":
                return GateResult(
                    gate_name="Training Distribution Representativeness",
                    status=GateStatus.FAIL,
                    message=(
                        "Sparse observational coverage severely limits ML training — "
                        "consider targeted campaigns, simulated observations, or "
                        "ML-enabled data assimilation to fill gaps"
                    ),
                    severity=0.8,
                )
            return GateResult(
                gate_name="Training Distribution Representativeness",
                status=GateStatus.WARN,
                message=(
                    "Observational data may have coverage gaps and inhomogeneities "
                    "that limit training data representativeness"
                ),
                severity=0.4,
            )
        else:
            return GateResult(
                gate_name="Training Distribution Representativeness",
                status=GateStatus.CONDITIONAL,
                message="Training data representativeness needs careful evaluation",
                condition=(
                    "Verify that simulated training data captures the relevant "
                    "physical regimes and does not introduce systematic biases"
                ),
                severity=0.5,
            )

    def rare_event_gate(self, spec: dict) -> GateResult:
        """
        Gate 3: Does the problem involve rare or extreme events?

        Tropical cyclones, blocking events, and other weather extremes are
        high-impact but relatively rare, presenting a challenge for ML due
        to limited sample size. However, ML's cheap inference enables large
        ensembles for better sampling of rare events.
        """
        involves_rare = spec.get("involves_rare_events", False)

        if not involves_rare:
            return GateResult(
                gate_name="Rare Event Handling",
                status=GateStatus.PASS,
                message="Problem does not primarily target rare events",
                severity=0.0,
            )
        return GateResult(
            gate_name="Rare Event Handling",
            status=GateStatus.WARN,
            message=(
                "Rare/extreme events have limited training samples. Current leading "
                "models show ML can match physics-based prediction for tropical "
                "cyclone intensity and tracks, but careful validation on tail "
                "statistics is essential. Consider: (1) guided sampling strategies "
                "via diffusion models, (2) large ML ensembles to better sample "
                "tails, (3) physics-informed augmentation of rare event training data"
            ),
            severity=0.5,
        )

    def analytical_solution_gate(self, spec: dict) -> GateResult:
        """
        Gate 4: Does the problem have known analytical solutions?

        If a problem has well-known analytical or semi-analytical solutions,
        ML is unlikely to provide meaningful improvement and may introduce
        unnecessary complexity and opacity.
        """
        has_analytical = spec.get("has_analytical_solution", False)

        if not has_analytical:
            return GateResult(
                gate_name="Analytical Solution Check",
                status=GateStatus.PASS,
                message="No known analytical solution — ML may provide value",
                severity=0.0,
            )
        return GateResult(
            gate_name="Analytical Solution Check",
            status=GateStatus.FAIL,
            message=(
                "Problem has known analytical or semi-analytical solutions. "
                "ML approaches are unlikely to improve upon these and would "
                "add unnecessary complexity and reduce interpretability. "
                "Use the analytical solution unless computational cost or "
                "generalization to related problems is the driver"
            ),
            severity=0.95,
        )

    def timescale_gate(self, spec: dict) -> GateResult:
        """
        Gate 5: Timescale-dependent assessment.

        Weather prediction (days): Well-established ML success regime.
        Subseasonal-to-seasonal (weeks-months): Emerging, conditional.
        Climate (decades-centuries): Fundamental challenges remain.

        This reflects the workshop's distinction between weather (where skill
        is dominated by initial state estimation) and climate (where biases,
        structural error, and forced responses dominate).
        """
        timescale = spec.get("target_timescale", "weather")

        timescale_assessments = {
            "weather": GateResult(
                gate_name="Timescale Assessment",
                status=GateStatus.PASS,
                message=(
                    "Weather timescale is the strongest regime for ML. Skill is "
                    "dominated by initial state estimation and short-term error "
                    "growth, both well-represented in reanalysis training data. "
                    "ML foundation models have demonstrated skill comparable to "
                    "operational NWP systems"
                ),
                severity=0.0,
            ),
            "subseasonal": GateResult(
                gate_name="Timescale Assessment",
                status=GateStatus.WARN,
                message=(
                    "Subseasonal timescale presents mixed ML opportunities. "
                    "Predictability depends on slowly-varying boundary conditions "
                    "(SST, soil moisture, sea ice) and teleconnections. ML may "
                    "help capture nonlinear interactions but faces challenges from "
                    "the large intrinsic uncertainty at these lead times"
                ),
                severity=0.4,
            ),
            "seasonal": GateResult(
                gate_name="Timescale Assessment",
                status=GateStatus.WARN,
                message=(
                    "Seasonal prediction relies on boundary forcing (ENSO, etc.) "
                    "rather than initial conditions. ML success depends on "
                    "capturing slowly-varying coupled dynamics"
                ),
                severity=0.5,
            ),
            "climate": GateResult(
                gate_name="Timescale Assessment",
                status=GateStatus.FAIL,
                message=(
                    "Climate timescale poses fundamental challenges for ML: "
                    "(1) requires extrapolation to novel forcing scenarios, "
                    "(2) models trained with weather-scale loss functions may not "
                    "produce credible long-term climate statistics, "
                    "(3) need robustness to out-of-distribution conditions. "
                    "Consider nonautonomous dynamical systems framework or "
                    "hybrid physics-ML approaches"
                ),
                severity=0.8,
            ),
        }
        return timescale_assessments.get(
            timescale,
            GateResult(
                gate_name="Timescale Assessment",
                status=GateStatus.CONDITIONAL,
                message="Non-standard timescale — assess on a case-by-case basis",
                severity=0.3,
            ),
        )

    def counterfactual_gate(self, spec: dict) -> GateResult:
        """
        Gate 6: Does the problem require counterfactual reasoning?

        Climate models are used to evaluate responses under different
        anthropogenic forcing pathways. ML approaches face fundamental
        challenges here because historical data may be insufficient for
        extrapolation to counterfactual scenarios.
        """
        requires_cf = spec.get("requires_counterfactual", False)

        if not requires_cf:
            return GateResult(
                gate_name="Counterfactual Reasoning",
                status=GateStatus.PASS,
                message="Problem does not require counterfactual reasoning",
                severity=0.0,
            )
        return GateResult(
            gate_name="Counterfactual Reasoning",
            status=GateStatus.FAIL,
            message=(
                "Counterfactual scenario evaluation (e.g. 'what if emissions were "
                "halved?') requires causal reasoning and extrapolation that purely "
                "data-driven ML cannot reliably provide. Physics-based or hybrid "
                "approaches with explicit causal structure are recommended. "
                "Establishing trust in ML for counterfactuals requires engagement "
                "with nonautonomous dynamical systems theory"
            ),
            severity=0.9,
        )

    def parameterization_opportunity_gate(self, spec: dict) -> GateResult:
        """
        Gate 7: Are there parameterization opportunities?

        Physics models are highly sensitive to parameterized processes
        (cloud microphysics, aerosol interactions, turbulence). ML can
        improve these parameterizations through learned representations,
        potentially addressing dominant uncertainty sources.
        """
        involves_param = spec.get("involves_parameterized_physics", False)

        if not involves_param:
            return GateResult(
                gate_name="Parameterization Opportunity",
                status=GateStatus.PASS,
                message=(
                    "Problem does not directly involve parameterized physics — "
                    "standard ML approaches apply"
                ),
                severity=0.0,
            )
        return GateResult(
            gate_name="Parameterization Opportunity",
            status=GateStatus.PASS,
            message=(
                "Parameterized physics processes present a strong ML opportunity. "
                "ML has demonstrated value in: (1) constraining parameters in "
                "microphysics schemes, (2) learning novel representations of "
                "subgrid processes, (3) scientist-guided ML parameterizations. "
                "This is a well-established hybrid physics-ML pathway"
            ),
            severity=0.0,
        )

    def conservation_gate(self, spec: dict) -> GateResult:
        """
        Gate 8: Are conservation laws critical?

        Many earth system processes require conservation of mass, energy,
        momentum, or other quantities. Violation of these constraints
        produces physically meaningless results.
        """
        requires_conservation = spec.get("requires_conservation_laws", False)

        if not requires_conservation:
            return GateResult(
                gate_name="Conservation Law Requirements",
                status=GateStatus.PASS,
                message="Conservation laws not critical for this application",
                severity=0.0,
            )
        return GateResult(
            gate_name="Conservation Law Requirements",
            status=GateStatus.CONDITIONAL,
            message=(
                "Conservation laws are critical. ML approaches must incorporate "
                "explicit constraints via: (1) conservation layers in network "
                "architecture, (2) physics-informed loss functions, or "
                "(3) post-hoc correction. Unconstrained ML will likely produce "
                "physically inconsistent results over long rollouts"
            ),
            condition=(
                "Architecture must include explicit conservation enforcement "
                "(conservation layers, constrained loss, or correction step)"
            ),
            severity=0.5,
        )

    def coupled_system_gate(self, spec: dict) -> GateResult:
        """
        Gate 9: Does the problem involve coupled systems?

        Coupled earth system components (atmosphere, ocean, land, cryosphere)
        have diverse timescales and interaction mechanisms. Rigorous
        propagation of uncertainty across components remains an open challenge.
        """
        involves_coupled = spec.get("involves_coupled_systems", False)

        if not involves_coupled:
            return GateResult(
                gate_name="Coupled System Complexity",
                status=GateStatus.PASS,
                message="Single-component problem — standard approaches apply",
                severity=0.0,
            )
        return GateResult(
            gate_name="Coupled System Complexity",
            status=GateStatus.WARN,
            message=(
                "Coupled earth system components (atmosphere-ocean-land-cryosphere) "
                "present challenges: (1) diverse timescales requiring different "
                "training strategies, (2) uncertainty propagation across components, "
                "(3) emergent behavior from coupling. ML approaches should consider "
                "component-wise training with coupling interfaces or fully coupled "
                "architectures with appropriate inductive biases"
            ),
            severity=0.5,
        )

    def observational_coverage_gate(self, spec: dict) -> GateResult:
        """
        Gate 10: Observational coverage assessment.

        Observations play multiple roles (ground truth, constraints, features,
        targets). Sparse coverage — particularly in the tropics, deep ocean,
        and polar regions — motivates targeted campaigns and ML-enabled
        data assimilation.
        """
        coverage = spec.get("observational_coverage", "moderate")

        coverage_assessments = {
            "dense": GateResult(
                gate_name="Observational Coverage",
                status=GateStatus.PASS,
                message=(
                    "Dense observational coverage provides strong training signal "
                    "and validation capability"
                ),
                severity=0.0,
            ),
            "moderate": GateResult(
                gate_name="Observational Coverage",
                status=GateStatus.PASS,
                message=(
                    "Moderate observational coverage — sufficient for ML training "
                    "but consider potential biases from unsampled regions"
                ),
                severity=0.2,
            ),
            "sparse": GateResult(
                gate_name="Observational Coverage",
                status=GateStatus.WARN,
                message=(
                    "Sparse observational coverage limits ML training quality and "
                    "validation. Consider: (1) ML-enabled data assimilation to "
                    "fill gaps, (2) simulated observations from physics models, "
                    "(3) targeted observational campaigns informed by information-"
                    "theoretic criteria"
                ),
                severity=0.6,
            ),
        }
        return coverage_assessments.get(
            coverage,
            GateResult(
                gate_name="Observational Coverage",
                status=GateStatus.CONDITIONAL,
                message="Observational coverage needs assessment",
                severity=0.3,
            ),
        )

    def emulation_vs_discovery_gate(self, spec: dict) -> GateResult:
        """
        Gate 11: Emulation vs Discovery qualifier.

        A key workshop distinction: Is ML being used to emulate existing
        models/processes (accelerate, reduce cost) or to discover new
        scientific insights? The answer shapes expectations and evaluation
        criteria.
        """
        purpose = spec.get("purpose", "emulation")

        if purpose == "emulation":
            return GateResult(
                gate_name="Emulation vs Discovery",
                status=GateStatus.PASS,
                message=(
                    "Emulation objective — ML is reproducing known behavior for "
                    "speed/cost benefits. Evaluation criterion: fidelity to the "
                    "emulated model. This is a well-established use case where "
                    "ML has demonstrated clear value"
                ),
                severity=0.0,
            )
        elif purpose == "discovery":
            return GateResult(
                gate_name="Emulation vs Discovery",
                status=GateStatus.WARN,
                message=(
                    "Discovery objective — ML is seeking new scientific insights "
                    "beyond emulating known behavior. This is a promising but "
                    "less-established pathway. Consider: (1) equation discovery "
                    "methods, (2) identification of emergent relationships across "
                    "scales, (3) data-driven atmospheric chemistry. Evaluation "
                    "criteria must go beyond standard ML metrics to assess "
                    "scientific novelty and physical plausibility"
                ),
                severity=0.3,
            )
        elif purpose == "operational":
            return GateResult(
                gate_name="Emulation vs Discovery",
                status=GateStatus.PASS,
                message=(
                    "Operational objective — ML must meet reliability, latency, "
                    "and robustness requirements. Well-suited for weather "
                    "prediction where ML has demonstrated operational viability"
                ),
                severity=0.1,
            )
        else:
            return GateResult(
                gate_name="Emulation vs Discovery",
                status=GateStatus.PASS,
                message=f"Research purpose '{purpose}' — standard evaluation applies",
                severity=0.1,
            )

    @staticmethod
    def summarize_gates(results: list) -> dict:
        """
        Summarize gate results into an actionable overview.

        Returns
        -------
        dict with keys:
            passed: list of passed gates
            warnings: list of warning gates
            failed: list of failed gates
            conditions: list of conditional gates with their conditions
            overall_gate_status: GateStatus (worst status across all gates)
            max_severity: float (highest severity among all gates)
        """
        passed = [r for r in results if r.status == GateStatus.PASS]
        warnings = [r for r in results if r.status == GateStatus.WARN]
        failed = [r for r in results if r.status == GateStatus.FAIL]
        conditional = [r for r in results if r.status == GateStatus.CONDITIONAL]

        if failed:
            overall = GateStatus.FAIL
        elif warnings:
            overall = GateStatus.WARN
        elif conditional:
            overall = GateStatus.CONDITIONAL
        else:
            overall = GateStatus.PASS

        max_severity = max((r.severity for r in results), default=0.0)

        return {
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "conditions": conditional,
            "overall_gate_status": overall,
            "max_severity": max_severity,
            "total_gates": len(results),
            "pass_rate": len(passed) / max(len(results), 1),
        }

    @staticmethod
    def format_gate_report(results: list) -> str:
        """Generate a human-readable gate evaluation report."""
        summary = EarthScienceHeuristics.summarize_gates(results)
        lines = []
        lines.append("-" * 72)
        lines.append("HEURISTIC DECISION GATES")
        lines.append("-" * 72)

        status_symbols = {
            GateStatus.PASS: "PASS",
            GateStatus.WARN: "WARN",
            GateStatus.FAIL: "FAIL",
            GateStatus.CONDITIONAL: "COND",
        }

        for r in results:
            symbol = status_symbols[r.status]
            lines.append(f"  [{symbol}] {r.gate_name}")
            lines.append(f"         {r.message}")
            if r.condition:
                lines.append(f"         Condition: {r.condition}")
            lines.append("")

        lines.append(
            f"Summary: {len(summary['passed'])} passed, "
            f"{len(summary['warnings'])} warnings, "
            f"{len(summary['failed'])} failed, "
            f"{len(summary['conditions'])} conditional"
        )
        lines.append(
            f"Overall Gate Status: {summary['overall_gate_status'].value.upper()}"
        )
        lines.append(f"Maximum Severity: {summary['max_severity']:.2f}")
        lines.append("")

        return "\n".join(lines)
