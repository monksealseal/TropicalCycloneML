"""
AI Suitability Framework for Earth Science Problems
====================================================

A mathematical and heuristic framework for predicting the utility of deep
learning architectures given information about training data and objectives
in earth science applications.

Motivated by the observation that high-fidelity reanalysis products (e.g. ERA5)
have enabled remarkably successful ML weather emulators, yet not every earth
science problem shares the properties that make weather forecasting amenable
to data-driven approaches. This framework provides a principled, quantitative
methodology for evaluating whether AI is appropriate for a given earth science
problem before committing resources.

References
----------
Workshop on Mathematics and Machine Learning for Earth System Simulation, 2025.
"""

from ai_suitability_framework.dimensions import (
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
from ai_suitability_framework.scoring import SuitabilityScorer
from ai_suitability_framework.heuristics import (
    HeuristicGate,
    EarthScienceHeuristics,
)
from ai_suitability_framework.framework import (
    AISuitabilityFramework,
    ProblemSpecification,
    AssessmentResult,
    Recommendation,
)

__version__ = "0.1.0"

__all__ = [
    "ScoringDimension",
    "DataFidelity",
    "ProblemLearnability",
    "SampleSufficiency",
    "PhysicsBaselineComparison",
    "ComputationalJustification",
    "PhysicalConsistency",
    "UncertaintyQuantification",
    "ExtrapolationRisk",
    "SuitabilityScorer",
    "HeuristicGate",
    "EarthScienceHeuristics",
    "AISuitabilityFramework",
    "ProblemSpecification",
    "AssessmentResult",
    "Recommendation",
]
