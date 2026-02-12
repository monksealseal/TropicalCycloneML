"""Data preprocessing module for cancer datasets."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from cancer_agent.utils import get_logger

log = get_logger(__name__)


class Preprocessor:
    """Handles data cleaning, scaling, and train/test splitting."""

    def __init__(self, scaling: str = "standard", test_size: float = 0.2,
                 random_state: int = 42):
        self.scaling = scaling
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        self._fitted = False

    def run(self, dataset: dict) -> dict:
        """
        Full preprocessing pipeline.

        Takes a dataset dict from DatasetLoader and returns a processed dict
        with train/test splits and scaled features.
        """
        df = dataset["df"].copy()
        feature_names = dataset["feature_names"]
        target_name = dataset["target_name"]

        log.info("Starting preprocessing pipeline")

        # Step 1: Handle missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            log.info("Found %d missing values, imputing with median", missing_before)
            for col in feature_names:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
        else:
            log.info("No missing values detected")

        # Step 2: Remove duplicate rows
        n_dups = df.duplicated().sum()
        if n_dups > 0:
            log.info("Removing %d duplicate rows", n_dups)
            df = df.drop_duplicates().reset_index(drop=True)

        # Step 3: Train/test split
        X = df[feature_names].values
        y = df[target_name].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        log.info(
            "Split: %d train / %d test (%.0f%% test)",
            len(X_train), len(X_test), self.test_size * 100,
        )

        # Step 4: Feature scaling
        if self.scaling == "standard":
            self.scaler = StandardScaler()
        elif self.scaling == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling}")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self._fitted = True

        log.info("Applied %s scaling", self.scaling)

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
            "target_name": target_name,
            "scaler": self.scaler,
            "metadata": dataset["metadata"],
            "preprocessing_info": {
                "missing_values_imputed": missing_before,
                "duplicates_removed": n_dups,
                "scaling": self.scaling,
                "test_size": self.test_size,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            },
        }
