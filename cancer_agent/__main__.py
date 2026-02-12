"""CLI entry point: python -m cancer_agent"""

import argparse
import sys

from cancer_agent.agent import CancerResearchAgent
from cancer_agent.models.trainer import MODEL_CONFIGS
from cancer_agent.data.loader import DATASET_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Autonomous Cancer Research ML Agent - "
            "Runs end-to-end ML analysis on cancer datasets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m cancer_agent\n"
            "  python -m cancer_agent --dataset breast_cancer --models logistic_regression random_forest svm\n"
            "  python -m cancer_agent --scaling minmax --test-size 0.3 --cv-folds 10\n"
            "  python -m cancer_agent --output-dir ./results\n"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=list(DATASET_REGISTRY.keys()),
        help="Cancer dataset to analyze (default: breast_cancer)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to train (default: all available)",
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="standard",
        choices=["standard", "minmax"],
        help="Feature scaling method (default: standard)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cancer_agent_output",
        help="Directory for output files (default: cancer_agent_output)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name in MODEL_CONFIGS:
            cls, _ = MODEL_CONFIGS[name]
            print(f"  {name:<25} ({cls.__name__})")
        return

    if args.list_datasets:
        print("Available datasets:")
        for name, info in DATASET_REGISTRY.items():
            print(f"  {name:<20} {info['description']}")
        return

    agent = CancerResearchAgent(
        dataset=args.dataset,
        models=args.models,
        scaling=args.scaling,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        output_dir=args.output_dir,
    )

    try:
        agent.run()
    except Exception as e:
        print(f"\nAgent failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
