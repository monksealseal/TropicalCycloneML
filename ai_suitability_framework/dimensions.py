"""
Scoring dimensions for AI suitability assessment.

Each dimension captures a distinct axis along which the suitability of a
machine learning approach can be evaluated for an earth science problem.
Dimension scores are computed as values in [0, 1] where higher values
indicate greater suitability for ML.

The mathematical formulation draws on information-theoretic, statistical,
and domain-specific considerations to produce calibrated scores.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DimensionScore:
    """Result of evaluating a single scoring dimension."""

    name: str
    score: float  # in [0, 1], higher = more suitable for ML
    confidence: float  # in [0, 1], confidence in this score
    rationale: str  # human-readable explanation
    sub_scores: dict = field(default_factory=dict)

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))


class ScoringDimension(ABC):
    """Base class for all scoring dimensions."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, **kwargs) -> DimensionScore:
        ...


class DataFidelity(ScoringDimension):
    """
    Data Fidelity Dimension (D_f).

    Measures how well the available training data represents the true
    underlying physical process. The success of ML weather emulators has
    been closely tied to the quality of ERA5 reanalysis — a product that
    assimilates vast observational networks into a physically consistent
    gridded dataset. Problems with lower-fidelity training data face a
    fundamental ceiling on ML model quality.

    The score is computed as:

        D_f = w1 * R_quality + w2 * S_coverage + w3 * T_resolution
              + w4 * (1 - B_known)

    where:
        R_quality    ∈ [0,1]: Quality of reanalysis or observational product
        S_coverage   ∈ [0,1]: Spatial and temporal coverage completeness
        T_resolution ∈ [0,1]: Resolution adequacy for the target phenomenon
        B_known      ∈ [0,1]: Magnitude of known biases in training data
    """

    name = "Data Fidelity"
    description = (
        "How faithfully the training data represents the true physical process"
    )

    def evaluate(
        self,
        reanalysis_quality: float = 0.5,
        spatial_temporal_coverage: float = 0.5,
        resolution_adequacy: float = 0.5,
        known_bias_magnitude: float = 0.3,
        **kwargs,
    ) -> DimensionScore:
        """
        Parameters
        ----------
        reanalysis_quality : float
            Quality of the gridded data product (0=poor, 1=ERA5-level).
        spatial_temporal_coverage : float
            Fraction of the domain with adequate observational coverage.
        resolution_adequacy : float
            Whether the data resolves the phenomena of interest.
        known_bias_magnitude : float
            Severity of known systematic biases (0=none, 1=severe).
        """
        w = [0.35, 0.25, 0.25, 0.15]
        sub = {
            "reanalysis_quality": reanalysis_quality,
            "spatial_temporal_coverage": spatial_temporal_coverage,
            "resolution_adequacy": resolution_adequacy,
            "bias_penalty": 1.0 - known_bias_magnitude,
        }
        score = (
            w[0] * sub["reanalysis_quality"]
            + w[1] * sub["spatial_temporal_coverage"]
            + w[2] * sub["resolution_adequacy"]
            + w[3] * sub["bias_penalty"]
        )
        # Confidence scales with coverage — sparse data means uncertain score
        confidence = 0.5 + 0.5 * spatial_temporal_coverage

        rationale_parts = []
        if reanalysis_quality >= 0.8:
            rationale_parts.append(
                "High-quality reanalysis (ERA5-level) provides strong training signal"
            )
        elif reanalysis_quality < 0.4:
            rationale_parts.append(
                "Low-quality training data imposes a fundamental ceiling on ML skill"
            )
        if known_bias_magnitude > 0.5:
            rationale_parts.append(
                f"Significant known biases ({known_bias_magnitude:.1%}) "
                "may propagate into learned representations"
            )
        if spatial_temporal_coverage < 0.4:
            rationale_parts.append(
                "Sparse observational coverage limits training data representativeness"
            )

        return DimensionScore(
            name=self.name,
            score=score,
            confidence=confidence,
            rationale="; ".join(rationale_parts) if rationale_parts else
            "Moderate data fidelity — consider data quality improvements",
            sub_scores=sub,
        )


class ProblemLearnability(ScoringDimension):
    """
    Problem Learnability Dimension (P_l).

    Assesses whether the input-output mapping contains sufficient learnable
    structure for a neural network to exploit. This depends on the
    signal-to-noise ratio, stationarity of the governing process, the
    effective dimensionality of the state space, and whether the deployment
    conditions fall within the training distribution.

    The score is computed as:

        P_l = σ(α * SNR_eff) * S_stat * D_eff_factor * OOD_penalty

    where:
        σ(·)            : logistic sigmoid for smooth saturation
        SNR_eff         : effective signal-to-noise ratio (log scale)
        S_stat ∈ [0,1]  : stationarity of the process
        D_eff_factor    : penalty for very high effective dimensionality
        OOD_penalty     : penalty for out-of-distribution deployment risk
    """

    name = "Problem Learnability"
    description = (
        "Whether the input-output mapping has exploitable learned structure"
    )

    @staticmethod
    def _sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
        """Logistic sigmoid: smooth saturation in [0, 1]."""
        z = -k * (x - x0)
        if z > 500:
            return 0.0
        if z < -500:
            return 1.0
        return 1.0 / (1.0 + math.exp(z))

    def evaluate(
        self,
        signal_to_noise_ratio: float = 1.0,
        stationarity: float = 0.7,
        effective_dimensionality: float = 100.0,
        ood_deployment_risk: float = 0.2,
        deterministic_structure: float = 0.5,
        **kwargs,
    ) -> DimensionScore:
        """
        Parameters
        ----------
        signal_to_noise_ratio : float
            Ratio of signal variance to noise variance (linear scale).
        stationarity : float
            Degree to which the governing process is stationary (0=non-stationary,
            1=perfectly stationary). Climate problems score lower than weather.
        effective_dimensionality : float
            Effective degrees of freedom in the state space. Higher values
            require more data and model capacity.
        ood_deployment_risk : float
            Probability that deployment conditions fall outside the training
            distribution (0=in-distribution, 1=entirely OOD).
        deterministic_structure : float
            Degree to which the mapping has deterministic vs stochastic character.
        """
        # SNR contribution: sigmoid on log-SNR, centered at SNR=1 (0 dB)
        log_snr = math.log10(max(signal_to_noise_ratio, 1e-6))
        snr_score = self._sigmoid(log_snr, k=2.0, x0=0.0)

        # Dimensionality penalty: gradual degradation for high-dimensional problems
        # Uses log-scaling: problems with dim > 1000 see significant penalty
        dim_factor = self._sigmoid(
            math.log10(max(effective_dimensionality, 1.0)), k=-1.5, x0=2.5
        )

        # OOD penalty: multiplicative, severe for high OOD risk
        # This reflects the workshop finding that OOD is central for both
        # weather and climate prediction
        ood_penalty = 1.0 - 0.8 * ood_deployment_risk

        # Combine
        score = (
            0.30 * snr_score
            + 0.20 * stationarity
            + 0.15 * dim_factor
            + 0.15 * deterministic_structure
            + 0.20 * (1.0 - ood_deployment_risk)
        ) * ood_penalty

        sub = {
            "snr_score": snr_score,
            "stationarity": stationarity,
            "dimensionality_factor": dim_factor,
            "ood_penalty": ood_penalty,
            "deterministic_structure": deterministic_structure,
        }

        rationale_parts = []
        if ood_deployment_risk > 0.5:
            rationale_parts.append(
                "High out-of-distribution risk — ML models trained on historical "
                "data may not extrapolate to novel conditions (e.g. future climate "
                "forcing scenarios)"
            )
        if stationarity < 0.4:
            rationale_parts.append(
                "Non-stationary process — learned relationships may degrade over time"
            )
        if signal_to_noise_ratio < 0.5:
            rationale_parts.append(
                "Low signal-to-noise ratio limits learnable structure"
            )
        if signal_to_noise_ratio > 5.0 and stationarity > 0.7:
            rationale_parts.append(
                "High SNR with stationary dynamics is favorable for learning"
            )

        confidence = 0.6 + 0.2 * min(stationarity, 1.0 - ood_deployment_risk)

        return DimensionScore(
            name=self.name,
            score=score,
            confidence=confidence,
            rationale="; ".join(rationale_parts) if rationale_parts else
            "Moderate learnability — standard ML approaches may apply with care",
            sub_scores=sub,
        )


class SampleSufficiency(ScoringDimension):
    """
    Sample Sufficiency Dimension (V_a).

    Evaluates whether the available data volume is adequate relative to the
    complexity of the learning task. This is particularly critical for rare
    events like tropical cyclones and blocking events, which are high-impact
    but underrepresented in training sets.

    The score uses a statistical power-inspired formulation:

        V_a = 1 - exp(-N_eff / N_required)

    where:
        N_eff      = N * coverage * (1 - redundancy)
        N_required = C * d^α   (C and α depend on architecture complexity)
    """

    name = "Sample Sufficiency"
    description = (
        "Whether data volume is adequate for the complexity of the task"
    )

    def evaluate(
        self,
        sample_size: float = 10000,
        effective_parameters: float = 1e6,
        event_frequency: float = 1.0,
        coverage_completeness: float = 0.8,
        temporal_redundancy: float = 0.3,
        **kwargs,
    ) -> DimensionScore:
        """
        Parameters
        ----------
        sample_size : float
            Number of training samples available.
        effective_parameters : float
            Approximate number of learnable parameters in the target model.
        event_frequency : float
            Relative frequency of the target event (1.0 = common phenomena,
            0.01 = rare events like major hurricanes).
        coverage_completeness : float
            Fraction of the relevant input space covered by training data.
        temporal_redundancy : float
            Degree of temporal autocorrelation reducing effective sample size.
        """
        # Effective sample size accounting for coverage and redundancy
        n_eff = sample_size * coverage_completeness * (1.0 - temporal_redundancy)
        n_eff = max(n_eff, 1.0)

        # Required samples: heuristic scaling with model complexity
        # Uses the sqrt scaling suggested by statistical learning theory
        n_required = 10.0 * math.sqrt(effective_parameters)

        # Sufficiency via exponential saturation
        ratio = n_eff / max(n_required, 1.0)
        volume_score = 1.0 - math.exp(-ratio)

        # Rare event penalty: severe reduction for infrequent phenomena
        # log-scale: frequency of 0.01 → penalty factor ~0.5
        rare_event_factor = self._rare_event_factor(event_frequency)

        score = volume_score * rare_event_factor

        sub = {
            "effective_sample_size": n_eff,
            "required_sample_size": n_required,
            "sample_ratio": ratio,
            "volume_score": volume_score,
            "rare_event_factor": rare_event_factor,
        }

        rationale_parts = []
        if ratio < 0.3:
            rationale_parts.append(
                f"Severe data deficit: effective samples ({n_eff:.0f}) far below "
                f"requirement ({n_required:.0f})"
            )
        elif ratio < 1.0:
            rationale_parts.append(
                "Data volume approaching but below adequacy threshold"
            )
        if event_frequency < 0.05:
            rationale_parts.append(
                "Rare event — limited training examples constrain ML performance; "
                "consider guided sampling or data augmentation strategies"
            )

        confidence = min(0.9, 0.4 + 0.5 * min(ratio, 1.0))

        return DimensionScore(
            name=self.name,
            score=score,
            confidence=confidence,
            rationale="; ".join(rationale_parts) if rationale_parts else
            "Adequate data volume for the target model complexity",
            sub_scores=sub,
        )

    @staticmethod
    def _rare_event_factor(frequency: float) -> float:
        """Compute penalty factor for rare events using logistic scaling."""
        if frequency >= 1.0:
            return 1.0
        # Map frequency to [0.3, 1.0] range via log-logistic
        log_freq = math.log10(max(frequency, 1e-6))
        # Centered at frequency=0.1 (log=-1), with smooth transition
        return 0.3 + 0.7 / (1.0 + math.exp(-2.0 * (log_freq + 1.0)))


class PhysicsBaselineComparison(ScoringDimension):
    """
    Physics Baseline Comparison Dimension (B_c).

    Evaluates ML suitability relative to existing physics-based models.
    A key workshop finding was that the benchmark for data-driven ML models
    must be data-constrained physical models, not unconstrained ones.

    The score considers:
    - Current physics model skill (higher skill → less room for ML improvement)
    - Theoretical predictability limits (Lyapunov exponents, etc.)
    - Whether marginal improvement justifies the ML investment
    - Maturity of the physics modeling community

    The score formulation rewards problems where:
    1. Physics models have known deficiencies ML could address, OR
    2. Physics models are computationally prohibitive for the use case
    """

    name = "Physics Baseline Comparison"
    description = (
        "ML improvement potential relative to physics-based models"
    )

    def evaluate(
        self,
        physics_model_skill: float = 0.7,
        theoretical_predictability_limit: float = 0.9,
        physics_model_cost: float = 0.5,
        known_physics_deficiencies: float = 0.3,
        physics_community_maturity: float = 0.7,
        **kwargs,
    ) -> DimensionScore:
        """
        Parameters
        ----------
        physics_model_skill : float
            Current best physics-based model skill (0=no skill, 1=perfect).
        theoretical_predictability_limit : float
            Theoretical upper bound on predictability for this problem.
        physics_model_cost : float
            Computational cost of physics models (0=cheap, 1=prohibitive).
        known_physics_deficiencies : float
            Known areas where physics models struggle (e.g. cloud microphysics).
        physics_community_maturity : float
            Maturity of the physics modeling effort (0=nascent, 1=decades of work).
        """
        # Improvement headroom: gap between current skill and theoretical limit
        headroom = max(0.0, theoretical_predictability_limit - physics_model_skill)

        # ML opportunity score: combines headroom with deficiency areas
        opportunity = 0.5 * headroom + 0.5 * known_physics_deficiencies

        # Cost advantage: ML inference is typically cheaper once trained
        cost_advantage = physics_model_cost

        # Mature physics communities may have less low-hanging fruit but
        # also provide better training data (e.g. ERA5 from ECMWF's decades
        # of NWP development)
        maturity_factor = 0.3 + 0.4 * physics_community_maturity + 0.3 * (
            1.0 - physics_community_maturity
        )

        score = (
            0.35 * opportunity
            + 0.30 * cost_advantage
            + 0.20 * known_physics_deficiencies
            + 0.15 * maturity_factor
        )

        sub = {
            "improvement_headroom": headroom,
            "ml_opportunity": opportunity,
            "cost_advantage": cost_advantage,
            "maturity_factor": maturity_factor,
        }

        rationale_parts = []
        if headroom < 0.1 and physics_model_cost < 0.3:
            rationale_parts.append(
                "Physics models already near theoretical limit and computationally "
                "affordable — ML offers limited marginal benefit"
            )
        if physics_model_cost > 0.7:
            rationale_parts.append(
                "High physics model cost creates strong computational justification "
                "for ML emulation (ensemble generation, real-time applications)"
            )
        if known_physics_deficiencies > 0.6:
            rationale_parts.append(
                "Significant known physics deficiencies (e.g. parameterization "
                "schemes) that ML may help address"
            )

        confidence = 0.5 + 0.3 * physics_community_maturity

        return DimensionScore(
            name=self.name,
            score=score,
            confidence=confidence,
            rationale="; ".join(rationale_parts) if rationale_parts else
            "Moderate ML opportunity relative to existing physics baselines",
            sub_scores=sub,
        )


class ComputationalJustification(ScoringDimension):
    """
    Computational Justification Dimension (C_j).

    Evaluates whether the computational economics favor ML over traditional
    approaches. Workshop discussions highlighted that once effective complexity
    is accounted for, ML models do not appear intrinsically more efficient than
    GPU-ported dynamical models — but they do leverage post-Moore's law hardware
    (tensor cores, matrix multiplication units) more effectively.

    Key considerations:
    - Training cost amortization over deployment lifetime
    - Inference speed advantage (critical for ensemble generation)
    - Hardware alignment with current accelerator architectures
    - Operational deployment feasibility
    """

    name = "Computational Justification"
    description = (
        "Whether computational economics favor ML over traditional approaches"
    )

    def evaluate(
        self,
        training_cost_gpu_hours: float = 1000.0,
        inference_speedup_factor: float = 10.0,
        expected_deployment_runs: float = 10000.0,
        hardware_alignment: float = 0.7,
        operational_latency_requirement: Optional[float] = None,
        **kwargs,
    ) -> DimensionScore:
        """
        Parameters
        ----------
        training_cost_gpu_hours : float
            Estimated one-time training cost in GPU-hours.
        inference_speedup_factor : float
            Expected speedup of ML inference vs physics model.
        expected_deployment_runs : float
            Number of inference runs expected over deployment lifetime.
        hardware_alignment : float
            How well the computation maps to modern accelerators (0=poor, 1=ideal).
        operational_latency_requirement : float, optional
            Maximum acceptable latency in seconds. If set, constrains viability.
        """
        # Amortization: training cost spread over deployment runs
        cost_per_run = training_cost_gpu_hours / max(expected_deployment_runs, 1.0)
        amortization_score = 1.0 - math.exp(-expected_deployment_runs / 1000.0)

        # Speedup benefit (log scale, saturating)
        log_speedup = math.log10(max(inference_speedup_factor, 0.1))
        speedup_score = min(1.0, max(0.0, 0.5 + 0.5 * log_speedup / 2.0))

        # Latency constraint check
        latency_factor = 1.0
        if operational_latency_requirement is not None:
            # If ML can meet latency but physics can't, strong advantage
            latency_factor = min(1.0, 0.5 + 0.5 * speedup_score)

        score = (
            0.30 * amortization_score
            + 0.30 * speedup_score
            + 0.20 * hardware_alignment
            + 0.20 * latency_factor
        )

        sub = {
            "amortization_score": amortization_score,
            "speedup_score": speedup_score,
            "hardware_alignment": hardware_alignment,
            "cost_per_run": cost_per_run,
        }

        rationale_parts = []
        if inference_speedup_factor > 100:
            rationale_parts.append(
                "Large inference speedup enables new applications "
                "(large ensembles, real-time forecasting)"
            )
        if amortization_score < 0.3:
            rationale_parts.append(
                "Limited deployment runs make training cost difficult to justify"
            )
        if hardware_alignment > 0.8:
            rationale_parts.append(
                "Computation well-aligned with modern tensor core architectures"
            )

        confidence = 0.7  # Computational costs are relatively well-characterized

        return DimensionScore(
            name=self.name,
            score=score,
            confidence=confidence,
            rationale="; ".join(rationale_parts) if rationale_parts else
            "Moderate computational justification for ML approach",
            sub_scores=sub,
        )


class PhysicalConsistency(ScoringDimension):
    """
    Physical Consistency Dimension (P_c).

    Evaluates how critical physical constraints are and how feasible it is
    for ML to respect them. Workshop discussions emphasized that earth system
    data is fundamentally non-Euclidean (spherical geometry, symmetries,
    conservation laws, multiscale structure) and that decisions about which
    structures to enforce vs learn remain central to model design.

    Approaches to physical constraints include:
    - Conservation layers in neural network architecture
    - Physics-informed loss functions
    - Geometric deep learning (equivariant architectures)
    - Operator-theoretic methods
    """

    name = "Physical Consistency"
    description = (
        "Criticality of physical constraints and ML's ability to respect them"
    )

    def evaluate(
        self,
        conservation_law_criticality: float = 0.5,
        symmetry_requirements: float = 0.5,
        multiscale_coupling: float = 0.3,
        boundary_condition_complexity: float = 0.3,
        constraint_enforceability: float = 0.5,
        **kwargs,
    ) -> DimensionScore:
        """
        Parameters
        ----------
        conservation_law_criticality : float
            How critical conservation laws are (mass, energy, momentum).
            0=not critical, 1=violation causes physically meaningless results.
        symmetry_requirements : float
            Importance of geometric symmetries (spherical, rotational, etc.).
        multiscale_coupling : float
            Degree of coupling between resolved and unresolved scales.
        boundary_condition_complexity : float
            Complexity of boundary conditions that must be satisfied.
        constraint_enforceability : float
            Feasibility of enforcing needed constraints in ML architecture.
            Higher = constraints can be built into architecture.
        """
        # Constraint burden: how demanding the physics constraints are
        constraint_burden = (
            0.35 * conservation_law_criticality
            + 0.25 * symmetry_requirements
            + 0.25 * multiscale_coupling
            + 0.15 * boundary_condition_complexity
        )

        # Score: high when constraints are enforceable relative to their burden
        # If constraints are not critical, ML has an easier path
        # If constraints are critical AND enforceable, still viable
        # If constraints are critical but NOT enforceable, problematic
        if constraint_burden < 0.3:
            # Low constraint burden: ML is relatively unconstrained
            score = 0.8 + 0.2 * constraint_enforceability
        else:
            # Higher burden: enforceability matters more
            score = (
                0.3 * (1.0 - constraint_burden)
                + 0.7 * constraint_enforceability
            )

        sub = {
            "constraint_burden": constraint_burden,
            "conservation_law_criticality": conservation_law_criticality,
            "symmetry_requirements": symmetry_requirements,
            "multiscale_coupling": multiscale_coupling,
            "constraint_enforceability": constraint_enforceability,
        }

        rationale_parts = []
        if conservation_law_criticality > 0.7 and constraint_enforceability < 0.4:
            rationale_parts.append(
                "Critical conservation laws are difficult to enforce in ML "
                "architecture — consider physics-informed or hybrid approaches"
            )
        if symmetry_requirements > 0.7:
            rationale_parts.append(
                "Strong symmetry requirements suggest geometric deep learning "
                "(equivariant networks, HEALPix grids) may be necessary"
            )
        if multiscale_coupling > 0.7:
            rationale_parts.append(
                "Significant multiscale coupling complicates purely data-driven "
                "approaches — ML parameterizations of subgrid processes may help"
            )

        confidence = 0.5 + 0.3 * constraint_enforceability

        return DimensionScore(
            name=self.name,
            score=score,
            confidence=confidence,
            rationale="; ".join(rationale_parts) if rationale_parts else
            "Physical constraints manageable with appropriate architecture choices",
            sub_scores=sub,
        )


class UncertaintyQuantification(ScoringDimension):
    """
    Uncertainty Quantification Dimension (U_q).

    Evaluates ML suitability from the perspective of uncertainty requirements.
    The workshop identified UQ as a critical gap, noting that modern ML
    approaches increasingly focus on predictive distributions rather than
    deterministic trajectories, aligning with ensemble forecasting.

    Key considerations:
    - Is calibrated uncertainty essential for the application?
    - Can ML approaches provide adequate UQ (e.g. diffusion models, ensembles)?
    - How does ML UQ compare to physics-based ensemble approaches?
    """

    name = "Uncertainty Quantification"
    description = (
        "ML's ability to meet the problem's uncertainty quantification needs"
    )

    def evaluate(
        self,
        uq_criticality: float = 0.5,
        ml_uq_feasibility: float = 0.5,
        ensemble_size_need: float = 0.3,
        calibration_requirement: float = 0.5,
        **kwargs,
    ) -> DimensionScore:
        """
        Parameters
        ----------
        uq_criticality : float
            How critical calibrated uncertainty is for this application.
        ml_uq_feasibility : float
            Feasibility of ML-based UQ (diffusion models, MC dropout, ensembles).
        ensemble_size_need : float
            Need for large ensembles (ML advantage: cheap inference).
        calibration_requirement : float
            Stringency of calibration requirements.
        """
        # ML advantage in UQ: cheap ensembles + emerging probabilistic methods
        ensemble_advantage = ensemble_size_need * 0.8  # ML excels at large ensembles

        if uq_criticality < 0.3:
            # UQ not critical: ML faces no barrier here
            score = 0.8
        else:
            # UQ critical: depends on feasibility and whether ML has advantages
            score = (
                0.30 * ml_uq_feasibility
                + 0.30 * ensemble_advantage
                + 0.20 * (1.0 - calibration_requirement * (1.0 - ml_uq_feasibility))
                + 0.20 * ml_uq_feasibility
            )

        sub = {
            "uq_criticality": uq_criticality,
            "ml_uq_feasibility": ml_uq_feasibility,
            "ensemble_advantage": ensemble_advantage,
            "calibration_requirement": calibration_requirement,
        }

        rationale_parts = []
        if ensemble_size_need > 0.7:
            rationale_parts.append(
                "Need for large ensembles favors ML due to cheap inference — "
                "enables better sampling of tail risks and rare events"
            )
        if uq_criticality > 0.7 and ml_uq_feasibility < 0.4:
            rationale_parts.append(
                "Strict UQ requirements with limited ML UQ feasibility — "
                "consider Bayesian approaches or physics-based ensembles"
            )
        if calibration_requirement > 0.7:
            rationale_parts.append(
                "High calibration requirements need careful evaluation of "
                "ML probabilistic outputs (diffusion models, conformal prediction)"
            )

        confidence = 0.4 + 0.4 * ml_uq_feasibility

        return DimensionScore(
            name=self.name,
            score=score,
            confidence=confidence,
            rationale="; ".join(rationale_parts) if rationale_parts else
            "UQ requirements are manageable with current ML approaches",
            sub_scores=sub,
        )


class ExtrapolationRisk(ScoringDimension):
    """
    Extrapolation Risk Dimension (E_r).

    Evaluates the risk that the ML model will need to operate outside
    the training distribution. This was identified as a central challenge
    at the workshop, with important differences between weather and
    climate timescales.

    Weather prediction: largely in-distribution (initial condition estimation,
    short-term error growth well represented in reanalysis training data).

    Climate prediction: fundamentally requires extrapolation to novel forcing
    scenarios, non-stationary conditions, and potential tipping points.

    The score penalizes problems with high extrapolation requirements:

        E_r = (1 - OOD_risk) * (1 - nonstationarity) * regime_coverage
    """

    name = "Extrapolation Risk"
    description = (
        "Risk that deployment requires extrapolation beyond training distribution"
    )

    def evaluate(
        self,
        ood_deployment_fraction: float = 0.2,
        nonstationarity: float = 0.2,
        regime_coverage: float = 0.8,
        forcing_novelty: float = 0.1,
        tipping_point_proximity: float = 0.1,
        **kwargs,
    ) -> DimensionScore:
        """
        Parameters
        ----------
        ood_deployment_fraction : float
            Expected fraction of deployment scenarios that are OOD.
        nonstationarity : float
            Degree of non-stationarity in the system (0=stationary, 1=rapidly
            changing, e.g. under climate change).
        regime_coverage : float
            Fraction of dynamical regimes represented in training data.
        forcing_novelty : float
            Degree to which future forcing scenarios differ from training
            (critical for climate projection).
        tipping_point_proximity : float
            Risk of encountering tipping points or abrupt transitions.
        """
        # Base score: inverse of extrapolation risk
        base = (1.0 - ood_deployment_fraction) * regime_coverage

        # Non-stationarity penalty
        stationarity_factor = 1.0 - 0.5 * nonstationarity

        # Novel forcing penalty (climate-specific)
        forcing_factor = 1.0 - 0.6 * forcing_novelty

        # Tipping point risk: severe penalty for potential regime transitions
        tipping_factor = 1.0 - 0.8 * tipping_point_proximity

        score = base * stationarity_factor * forcing_factor * tipping_factor

        sub = {
            "base_coverage": base,
            "stationarity_factor": stationarity_factor,
            "forcing_factor": forcing_factor,
            "tipping_factor": tipping_factor,
        }

        rationale_parts = []
        if forcing_novelty > 0.5:
            rationale_parts.append(
                "Novel forcing scenarios (e.g. anthropogenic climate pathways) "
                "require extrapolation beyond training distribution — consider "
                "nonautonomous dynamical systems theory or hybrid approaches"
            )
        if tipping_point_proximity > 0.5:
            rationale_parts.append(
                "Proximity to potential tipping points means ML models may "
                "encounter dynamics not represented in historical training data"
            )
        if nonstationarity > 0.6:
            rationale_parts.append(
                "Non-stationary process — learned relationships may not hold "
                "under future conditions"
            )
        if ood_deployment_fraction < 0.1 and regime_coverage > 0.8:
            rationale_parts.append(
                "Deployment conditions well-represented in training data — "
                "analogous to weather prediction where ERA5 provides "
                "comprehensive coverage of dynamical states"
            )

        # Confidence lower when extrapolation risk is high (we're uncertain
        # about how the model will perform OOD)
        confidence = 0.8 - 0.4 * ood_deployment_fraction

        return DimensionScore(
            name=self.name,
            score=score,
            confidence=confidence,
            rationale="; ".join(rationale_parts) if rationale_parts else
            "Moderate extrapolation risk — monitor for distribution shift",
            sub_scores=sub,
        )
