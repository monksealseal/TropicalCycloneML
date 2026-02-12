"""Dataset loading module for cancer research datasets."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris

from cancer_agent.utils import get_logger

log = get_logger(__name__)

# Registry of available datasets
DATASET_REGISTRY = {
    "breast_cancer": {
        "loader": load_breast_cancer,
        "description": "Wisconsin Breast Cancer dataset (569 samples, 30 features)",
        "task": "binary_classification",
        "positive_label": "malignant",
        "negative_label": "benign",
    },
}


class DatasetLoader:
    """Loads and serves cancer-related datasets for analysis."""

    def __init__(self):
        self.datasets = {}

    @staticmethod
    def list_available() -> list[str]:
        """Return names of all available datasets."""
        return list(DATASET_REGISTRY.keys())

    def load(self, name: str = "breast_cancer") -> dict:
        """
        Load a dataset by name.

        Returns a dict with keys:
            - df: pd.DataFrame with features and target
            - feature_names: list of feature column names
            - target_name: name of the target column
            - task: classification task type
            - metadata: extra info about the dataset
        """
        if name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. "
                f"Available: {self.list_available()}"
            )

        entry = DATASET_REGISTRY[name]
        log.info("Loading dataset: %s", name)
        log.info("Description: %s", entry["description"])

        raw = entry["loader"]()
        feature_names = list(raw.feature_names)
        target_name = "target"

        df = pd.DataFrame(raw.data, columns=feature_names)
        df[target_name] = raw.target

        metadata = {
            "name": name,
            "n_samples": len(df),
            "n_features": len(feature_names),
            "task": entry["task"],
            "positive_label": entry.get("positive_label"),
            "negative_label": entry.get("negative_label"),
            "class_distribution": df[target_name].value_counts().to_dict(),
        }

        log.info(
            "Loaded %d samples with %d features",
            metadata["n_samples"],
            metadata["n_features"],
        )

        result = {
            "df": df,
            "feature_names": feature_names,
            "target_name": target_name,
            "task": metadata["task"],
            "metadata": metadata,
        }
        self.datasets[name] = result
        return result

    def load_custom_csv(self, path: str, target_column: str) -> dict:
        """Load a custom CSV dataset for cancer analysis."""
        log.info("Loading custom CSV from: %s", path)
        df = pd.read_csv(path)
        feature_names = [c for c in df.columns if c != target_column]

        metadata = {
            "name": path,
            "n_samples": len(df),
            "n_features": len(feature_names),
            "task": "binary_classification",
            "class_distribution": df[target_column].value_counts().to_dict(),
        }

        log.info(
            "Loaded %d samples with %d features",
            metadata["n_samples"],
            metadata["n_features"],
        )

        return {
            "df": df,
            "feature_names": feature_names,
            "target_name": target_column,
            "task": metadata["task"],
            "metadata": metadata,
        }
