"""
Practical examples demonstrating the AI Suitability Framework on real
earth science problems spanning the full spectrum of ML suitability.

These examples are grounded in the workshop discussion themes:
1. Weather prediction (strong ML candidate — ERA5 success story)
2. Tropical cyclone intensity forecasting (conditional — rare events)
3. Cloud microphysics parameterization (hybrid — ML parameterizations)
4. Climate projection under novel forcing (challenging — OOD extrapolation)
5. Paleoclimate reconstruction (challenging — sparse observations)
6. Tidal prediction (not recommended — analytical solutions exist)
7. Data assimilation for NWP (emerging — ML-enhanced DA)

Usage
-----
    from ai_suitability_framework.examples import run_all_examples
    run_all_examples()
"""

from ai_suitability_framework.framework import (
    AISuitabilityFramework,
    ProblemSpecification,
    AssessmentResult,
)


def global_weather_forecasting() -> ProblemSpecification:
    """
    Global medium-range weather forecasting (1-10 days).

    The canonical success story for ML in earth science. ERA5 reanalysis
    provides high-fidelity training data with comprehensive coverage.
    ML foundation models (Pangu-Weather, GraphCast, FourCastNet) have
    demonstrated skill comparable to operational NWP at these timescales.
    """
    return ProblemSpecification(
        name="Global Weather Forecasting (1-10 day)",
        description=(
            "Medium-range deterministic weather prediction using ML models "
            "trained on ERA5 reanalysis. Skill is dominated by initial state "
            "estimation and short-term error growth."
        ),
        target_timescale="weather",
        training_data_type="reanalysis",
        has_well_defined_objective=True,
        involves_rare_events=False,
        requires_counterfactual=False,
        has_analytical_solution=False,
        involves_parameterized_physics=False,
        requires_conservation_laws=True,
        involves_coupled_systems=False,
        observational_coverage="dense",
        purpose="operational",
        # Data fidelity: ERA5 is the gold standard
        reanalysis_quality=0.92,
        spatial_temporal_coverage=0.90,
        resolution_adequacy=0.85,
        known_bias_magnitude=0.10,
        # Learnability: strong deterministic structure at weather scales
        signal_to_noise_ratio=8.0,
        stationarity=0.85,
        effective_dimensionality=200.0,
        ood_deployment_risk=0.05,
        deterministic_structure=0.80,
        # Samples: decades of 6-hourly global reanalysis
        sample_size=250000,  # ~40 years * 4/day * ~1500 spatial
        effective_parameters=1e8,
        event_frequency=1.0,
        coverage_completeness=0.92,
        temporal_redundancy=0.40,
        # Physics baseline: operational NWP is mature but expensive
        physics_model_skill=0.82,
        theoretical_predictability_limit=0.95,
        physics_model_cost=0.80,
        known_physics_deficiencies=0.25,
        physics_community_maturity=0.95,
        # Computation: massive inference speedup for ensembles
        training_cost_gpu_hours=50000.0,
        inference_speedup_factor=1000.0,
        expected_deployment_runs=1000000.0,
        hardware_alignment=0.90,
        # Physical consistency: conservation matters for multi-day forecasts
        conservation_law_criticality=0.60,
        symmetry_requirements=0.70,
        multiscale_coupling=0.40,
        boundary_condition_complexity=0.30,
        constraint_enforceability=0.65,
        # UQ: ensemble forecasting is core to weather prediction
        uq_criticality=0.70,
        ml_uq_feasibility=0.70,
        ensemble_size_need=0.80,
        calibration_requirement=0.60,
        # Extrapolation: weather is largely in-distribution
        ood_deployment_fraction=0.05,
        nonstationarity=0.10,
        regime_coverage=0.90,
        forcing_novelty=0.05,
        tipping_point_proximity=0.02,
    )


def tropical_cyclone_intensity() -> ProblemSpecification:
    """
    Tropical cyclone intensity prediction.

    High-impact but relatively rare events that challenge ML due to limited
    sample size. Current leading models show ML can match and exceed
    physics-based prediction for TC intensity and tracks. ML's cheap
    inference enables large ensembles for better sampling of rare events.
    """
    return ProblemSpecification(
        name="Tropical Cyclone Intensity Prediction",
        description=(
            "Prediction of tropical cyclone maximum sustained wind speed "
            "and central pressure at 12-120 hour lead times. Rare event "
            "with complex air-sea interaction dynamics."
        ),
        target_timescale="weather",
        training_data_type="reanalysis",
        has_well_defined_objective=True,
        involves_rare_events=True,
        requires_counterfactual=False,
        has_analytical_solution=False,
        involves_parameterized_physics=True,
        requires_conservation_laws=True,
        involves_coupled_systems=True,
        observational_coverage="moderate",
        purpose="operational",
        # Data: good reanalysis but TC-specific data is limited
        reanalysis_quality=0.80,
        spatial_temporal_coverage=0.70,
        resolution_adequacy=0.65,
        known_bias_magnitude=0.25,
        # Learnability: complex but structured dynamics
        signal_to_noise_ratio=3.0,
        stationarity=0.70,
        effective_dimensionality=150.0,
        ood_deployment_risk=0.15,
        deterministic_structure=0.55,
        # Samples: limited by TC occurrence (~80-90 global per year)
        sample_size=5000,
        effective_parameters=5e7,
        event_frequency=0.05,
        coverage_completeness=0.70,
        temporal_redundancy=0.25,
        # Physics: HWRF/HAFS models improving but costly
        physics_model_skill=0.65,
        theoretical_predictability_limit=0.85,
        physics_model_cost=0.85,
        known_physics_deficiencies=0.50,
        physics_community_maturity=0.80,
        # Computation: significant speedup opportunity
        training_cost_gpu_hours=10000.0,
        inference_speedup_factor=500.0,
        expected_deployment_runs=50000.0,
        hardware_alignment=0.80,
        # Physical consistency: air-sea coupling matters
        conservation_law_criticality=0.55,
        symmetry_requirements=0.40,
        multiscale_coupling=0.65,
        boundary_condition_complexity=0.50,
        constraint_enforceability=0.50,
        # UQ: critical for emergency management decisions
        uq_criticality=0.85,
        ml_uq_feasibility=0.60,
        ensemble_size_need=0.90,
        calibration_requirement=0.75,
        # Extrapolation: moderate — climate change may shift TC behavior
        ood_deployment_fraction=0.15,
        nonstationarity=0.25,
        regime_coverage=0.70,
        forcing_novelty=0.15,
        tipping_point_proximity=0.10,
    )


def cloud_microphysics_parameterization() -> ProblemSpecification:
    """
    ML parameterization of cloud microphysics processes.

    Physics models are highly sensitive to cloud microphysics and aerosol
    interactions, making these a dominant source of uncertainty. ML can
    constrain parameters and learn novel representations. This represents
    the hybrid physics-ML pathway.
    """
    return ProblemSpecification(
        name="Cloud Microphysics Parameterization",
        description=(
            "Learning improved parameterizations of cloud microphysical "
            "processes (droplet formation, ice nucleation, precipitation "
            "efficiency) to replace or augment existing schemes in ESMs."
        ),
        target_timescale="weather",
        training_data_type="simulation",
        has_well_defined_objective=True,
        involves_rare_events=False,
        requires_counterfactual=False,
        has_analytical_solution=False,
        involves_parameterized_physics=True,
        requires_conservation_laws=True,
        involves_coupled_systems=False,
        observational_coverage="moderate",
        purpose="research",
        # Data: LES/CRM simulations provide training data
        reanalysis_quality=0.60,
        spatial_temporal_coverage=0.55,
        resolution_adequacy=0.70,
        known_bias_magnitude=0.35,
        # Learnability: complex nonlinear processes with some structure
        signal_to_noise_ratio=2.0,
        stationarity=0.75,
        effective_dimensionality=50.0,
        ood_deployment_risk=0.25,
        deterministic_structure=0.45,
        # Samples: abundant from cloud-resolving simulations
        sample_size=500000,
        effective_parameters=1e5,
        event_frequency=1.0,
        coverage_completeness=0.60,
        temporal_redundancy=0.35,
        # Physics: parameterizations are the dominant uncertainty source
        physics_model_skill=0.45,
        theoretical_predictability_limit=0.80,
        physics_model_cost=0.40,
        known_physics_deficiencies=0.75,
        physics_community_maturity=0.85,
        # Computation: moderate training, fast inference in ESM
        training_cost_gpu_hours=5000.0,
        inference_speedup_factor=50.0,
        expected_deployment_runs=500000.0,
        hardware_alignment=0.65,
        # Physical consistency: conservation is essential
        conservation_law_criticality=0.80,
        symmetry_requirements=0.30,
        multiscale_coupling=0.80,
        boundary_condition_complexity=0.50,
        constraint_enforceability=0.55,
        # UQ: important for propagating into ESM uncertainty
        uq_criticality=0.65,
        ml_uq_feasibility=0.55,
        ensemble_size_need=0.40,
        calibration_requirement=0.60,
        # Extrapolation: moderate — must work across climate regimes
        ood_deployment_fraction=0.20,
        nonstationarity=0.15,
        regime_coverage=0.65,
        forcing_novelty=0.20,
        tipping_point_proximity=0.05,
    )


def climate_projection() -> ProblemSpecification:
    """
    Climate projection under novel anthropogenic forcing scenarios.

    Fundamental challenges for ML: requires extrapolation to conditions
    not present in historical data, counterfactual scenario evaluation,
    and robustness to non-stationary forcing. The workshop identified
    this as requiring nonautonomous dynamical systems theory.
    """
    return ProblemSpecification(
        name="Climate Projection (RCP/SSP Scenarios)",
        description=(
            "Multi-decadal climate projection under anthropogenic forcing "
            "scenarios (SSP1-5). Requires prediction of forced response, "
            "internal variability, and potential tipping points."
        ),
        target_timescale="climate",
        training_data_type="reanalysis",
        has_well_defined_objective=False,  # multi-objective, scenario-dependent
        involves_rare_events=True,
        requires_counterfactual=True,
        has_analytical_solution=False,
        involves_parameterized_physics=True,
        requires_conservation_laws=True,
        involves_coupled_systems=True,
        observational_coverage="moderate",
        purpose="research",
        # Data: ERA5 covers ~80 years of one climate realization
        reanalysis_quality=0.85,
        spatial_temporal_coverage=0.75,
        resolution_adequacy=0.60,
        known_bias_magnitude=0.30,
        # Learnability: non-stationary, OOD is fundamental
        signal_to_noise_ratio=1.5,
        stationarity=0.20,
        effective_dimensionality=500.0,
        ood_deployment_risk=0.80,
        deterministic_structure=0.30,
        # Samples: only one historical climate realization
        sample_size=30000,  # ~80 years of annual-mean data with spatial variation
        effective_parameters=1e8,
        event_frequency=0.10,
        coverage_completeness=0.50,
        temporal_redundancy=0.60,
        # Physics: ESMs are the established tool, decades of development
        physics_model_skill=0.55,
        theoretical_predictability_limit=0.70,
        physics_model_cost=0.90,
        known_physics_deficiencies=0.45,
        physics_community_maturity=0.95,
        # Computation: ESMs are very expensive to run
        training_cost_gpu_hours=100000.0,
        inference_speedup_factor=100.0,
        expected_deployment_runs=10000.0,
        hardware_alignment=0.75,
        # Physical consistency: absolutely critical for climate
        conservation_law_criticality=0.95,
        symmetry_requirements=0.70,
        multiscale_coupling=0.85,
        boundary_condition_complexity=0.70,
        constraint_enforceability=0.35,
        # UQ: essential for policy-relevant projections
        uq_criticality=0.90,
        ml_uq_feasibility=0.30,
        ensemble_size_need=0.70,
        calibration_requirement=0.85,
        # Extrapolation: this IS the central challenge
        ood_deployment_fraction=0.75,
        nonstationarity=0.80,
        regime_coverage=0.30,
        forcing_novelty=0.85,
        tipping_point_proximity=0.30,
    )


def paleoclimate_reconstruction() -> ProblemSpecification:
    """
    Paleoclimate reconstruction from proxy records.

    Sparse, indirect observations (ice cores, tree rings, sediment cores)
    must be inverted to reconstruct past climate states. Significant
    uncertainty in proxy-climate relationships.
    """
    return ProblemSpecification(
        name="Paleoclimate Reconstruction from Proxies",
        description=(
            "Reconstruction of past climate states (temperature, precipitation, "
            "circulation) from sparse proxy records (ice cores, tree rings, "
            "marine sediments, speleothems)."
        ),
        target_timescale="climate",
        training_data_type="observations",
        has_well_defined_objective=True,
        involves_rare_events=False,
        requires_counterfactual=False,
        has_analytical_solution=False,
        involves_parameterized_physics=False,
        requires_conservation_laws=False,
        involves_coupled_systems=True,
        observational_coverage="sparse",
        purpose="research",
        # Data: very sparse proxy records with uncertain calibration
        reanalysis_quality=0.20,
        spatial_temporal_coverage=0.15,
        resolution_adequacy=0.25,
        known_bias_magnitude=0.60,
        # Learnability: weak signal in noisy proxies
        signal_to_noise_ratio=0.5,
        stationarity=0.40,
        effective_dimensionality=80.0,
        ood_deployment_risk=0.50,
        deterministic_structure=0.25,
        # Samples: very limited spatial coverage
        sample_size=500,
        effective_parameters=1e4,
        event_frequency=1.0,
        coverage_completeness=0.15,
        temporal_redundancy=0.10,
        # Physics: paleoclimate models exist but are uncertain
        physics_model_skill=0.40,
        theoretical_predictability_limit=0.60,
        physics_model_cost=0.70,
        known_physics_deficiencies=0.50,
        physics_community_maturity=0.75,
        # Computation: moderate
        training_cost_gpu_hours=500.0,
        inference_speedup_factor=5.0,
        expected_deployment_runs=1000.0,
        hardware_alignment=0.50,
        # Physical consistency: helpful but not absolutely critical
        conservation_law_criticality=0.30,
        symmetry_requirements=0.20,
        multiscale_coupling=0.40,
        boundary_condition_complexity=0.40,
        constraint_enforceability=0.40,
        # UQ: very important given sparse data
        uq_criticality=0.85,
        ml_uq_feasibility=0.35,
        ensemble_size_need=0.50,
        calibration_requirement=0.70,
        # Extrapolation: significant — past climates may be OOD
        ood_deployment_fraction=0.50,
        nonstationarity=0.60,
        regime_coverage=0.30,
        forcing_novelty=0.40,
        tipping_point_proximity=0.20,
    )


def tidal_prediction() -> ProblemSpecification:
    """
    Oceanic tidal prediction.

    Tides are driven by well-understood gravitational forcing with known
    analytical (harmonic) solutions. This represents a case where ML
    is not the right tool — analytical methods are superior.
    """
    return ProblemSpecification(
        name="Oceanic Tidal Prediction",
        description=(
            "Prediction of ocean tide height at coastal locations. "
            "Gravitational forcing is well-understood and harmonic analysis "
            "provides highly accurate predictions."
        ),
        target_timescale="weather",
        training_data_type="observations",
        has_well_defined_objective=True,
        involves_rare_events=False,
        requires_counterfactual=False,
        has_analytical_solution=True,
        involves_parameterized_physics=False,
        requires_conservation_laws=False,
        involves_coupled_systems=False,
        observational_coverage="dense",
        purpose="operational",
        # Data: excellent tide gauge networks
        reanalysis_quality=0.90,
        spatial_temporal_coverage=0.85,
        resolution_adequacy=0.95,
        known_bias_magnitude=0.05,
        # Learnability: highly deterministic, periodic
        signal_to_noise_ratio=50.0,
        stationarity=0.95,
        effective_dimensionality=20.0,
        ood_deployment_risk=0.02,
        deterministic_structure=0.95,
        # Samples: abundant
        sample_size=1000000,
        effective_parameters=1e3,
        event_frequency=1.0,
        coverage_completeness=0.90,
        temporal_redundancy=0.70,
        # Physics: harmonic analysis is near-perfect and fast
        physics_model_skill=0.98,
        theoretical_predictability_limit=0.99,
        physics_model_cost=0.05,
        known_physics_deficiencies=0.02,
        physics_community_maturity=0.99,
        # Computation: analytical methods are instant
        training_cost_gpu_hours=10.0,
        inference_speedup_factor=0.5,  # ML is SLOWER than analytical
        expected_deployment_runs=100000.0,
        hardware_alignment=0.30,
        # Physical consistency: harmonic decomposition is exact
        conservation_law_criticality=0.10,
        symmetry_requirements=0.20,
        multiscale_coupling=0.10,
        boundary_condition_complexity=0.10,
        constraint_enforceability=0.90,
        # UQ: minimal — deterministic predictions are sufficient
        uq_criticality=0.10,
        ml_uq_feasibility=0.90,
        ensemble_size_need=0.05,
        calibration_requirement=0.10,
        # Extrapolation: tides are stationary and well-covered
        ood_deployment_fraction=0.01,
        nonstationarity=0.02,
        regime_coverage=0.98,
        forcing_novelty=0.01,
        tipping_point_proximity=0.0,
    )


def ml_data_assimilation() -> ProblemSpecification:
    """
    ML-enhanced data assimilation for numerical weather prediction.

    An emerging area where ML can enhance traditional Bayesian state
    estimation. Less explored is the integration of DA and ML with
    physics uncertainty quantification.
    """
    return ProblemSpecification(
        name="ML-Enhanced Data Assimilation for NWP",
        description=(
            "Using ML to improve data assimilation: learned observation "
            "operators, bias correction, covariance estimation, and direct "
            "observational input to ML foundation models."
        ),
        target_timescale="weather",
        training_data_type="observations",
        has_well_defined_objective=True,
        involves_rare_events=False,
        requires_counterfactual=False,
        has_analytical_solution=False,
        involves_parameterized_physics=True,
        requires_conservation_laws=True,
        involves_coupled_systems=False,
        observational_coverage="moderate",
        purpose="operational",
        # Data: abundant observations and analysis increments
        reanalysis_quality=0.85,
        spatial_temporal_coverage=0.75,
        resolution_adequacy=0.80,
        known_bias_magnitude=0.20,
        # Learnability: observation-analysis mapping has structure
        signal_to_noise_ratio=4.0,
        stationarity=0.80,
        effective_dimensionality=300.0,
        ood_deployment_risk=0.10,
        deterministic_structure=0.60,
        # Samples: large volume of observation-analysis pairs
        sample_size=200000,
        effective_parameters=5e7,
        event_frequency=1.0,
        coverage_completeness=0.75,
        temporal_redundancy=0.30,
        # Physics: DA is mature but computationally demanding
        physics_model_skill=0.78,
        theoretical_predictability_limit=0.92,
        physics_model_cost=0.75,
        known_physics_deficiencies=0.35,
        physics_community_maturity=0.90,
        # Computation: significant potential speedup
        training_cost_gpu_hours=20000.0,
        inference_speedup_factor=50.0,
        expected_deployment_runs=200000.0,
        hardware_alignment=0.80,
        # Physical consistency: balance constraints important
        conservation_law_criticality=0.65,
        symmetry_requirements=0.50,
        multiscale_coupling=0.45,
        boundary_condition_complexity=0.40,
        constraint_enforceability=0.60,
        # UQ: central to DA
        uq_criticality=0.80,
        ml_uq_feasibility=0.60,
        ensemble_size_need=0.70,
        calibration_requirement=0.70,
        # Extrapolation: mostly in-distribution for weather DA
        ood_deployment_fraction=0.08,
        nonstationarity=0.10,
        regime_coverage=0.85,
        forcing_novelty=0.05,
        tipping_point_proximity=0.02,
    )


# ---- Convenience functions ----

ALL_EXAMPLES = {
    "global_weather": global_weather_forecasting,
    "tropical_cyclone": tropical_cyclone_intensity,
    "cloud_microphysics": cloud_microphysics_parameterization,
    "climate_projection": climate_projection,
    "paleoclimate": paleoclimate_reconstruction,
    "tidal_prediction": tidal_prediction,
    "ml_data_assimilation": ml_data_assimilation,
}


def run_single_example(name: str, weight_profile: str = "default") -> AssessmentResult:
    """Run a single example by name."""
    if name not in ALL_EXAMPLES:
        raise ValueError(
            f"Unknown example '{name}'. Available: {list(ALL_EXAMPLES.keys())}"
        )
    spec = ALL_EXAMPLES[name]()
    framework = AISuitabilityFramework(weight_profile=weight_profile)
    return framework.assess(spec)


def run_all_examples(weight_profile: str = "default") -> list:
    """
    Run all example problems and produce individual reports plus a
    comparative summary.

    Returns
    -------
    list of AssessmentResult
        Results for all example problems, sorted by score descending.
    """
    framework = AISuitabilityFramework(weight_profile=weight_profile)
    results = []

    for name, spec_fn in ALL_EXAMPLES.items():
        spec = spec_fn()
        result = framework.assess(spec)
        results.append(result)

    # Print individual reports
    for result in results:
        print(result.report)
        print("\n\n")

    # Print comparative summary
    comparison = AISuitabilityFramework.compare_problems(results)
    print(comparison)

    return results


if __name__ == "__main__":
    run_all_examples()
