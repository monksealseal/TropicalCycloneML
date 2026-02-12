"""Model training module with multiple classifier support."""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from cancer_agent.utils import get_logger

log = get_logger(__name__)

# Model configurations: name -> (class, kwargs)
MODEL_CONFIGS = {
    "logistic_regression": (
        LogisticRegression,
        {"max_iter": 2000, "random_state": 42, "C": 1.0},
    ),
    "random_forest": (
        RandomForestClassifier,
        {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier,
        {"n_estimators": 100, "random_state": 42, "learning_rate": 0.1},
    ),
    "svm": (
        SVC,
        {"kernel": "rbf", "probability": True, "random_state": 42},
    ),
    "knn": (
        KNeighborsClassifier,
        {"n_neighbors": 5, "n_jobs": -1},
    ),
    "adaboost": (
        AdaBoostClassifier,
        {"n_estimators": 50, "random_state": 42},
    ),
    "mlp": (
        MLPClassifier,
        {
            "hidden_layer_sizes": (128, 64),
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
        },
    ),
}


class ModelTrainer:
    """Trains and compares multiple ML classifiers for cancer prediction."""

    def __init__(self, models: list[str] | None = None, cv_folds: int = 5):
        """
        Args:
            models: list of model names to train, or None for all.
            cv_folds: number of cross-validation folds.
        """
        if models is None:
            models = list(MODEL_CONFIGS.keys())

        unknown = set(models) - set(MODEL_CONFIGS.keys())
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")

        self.model_names = models
        self.cv_folds = cv_folds
        self.trained_models = {}
        self.cv_results = {}

    @staticmethod
    def list_available_models() -> list[str]:
        """Return all available model names."""
        return list(MODEL_CONFIGS.keys())

    def run(self, processed_data: dict) -> dict:
        """
        Train all selected models and perform cross-validation.

        Returns dict with trained models and CV scores.
        """
        X_train = processed_data["X_train"]
        y_train = processed_data["y_train"]

        log.info(
            "Training %d models with %d-fold CV on %d samples",
            len(self.model_names), self.cv_folds, len(X_train),
        )

        results = []

        for name in self.model_names:
            cls, kwargs = MODEL_CONFIGS[name]
            log.info("Training: %s", name)

            model = cls(**kwargs)

            # Cross-validation
            t0 = time.time()
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.cv_folds, scoring="accuracy", n_jobs=-1,
            )
            cv_time = time.time() - t0

            # Fit on full training set
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            train_acc = model.score(X_train, y_train)

            self.trained_models[name] = model
            self.cv_results[name] = cv_scores

            entry = {
                "name": name,
                "model": model,
                "cv_accuracy_mean": round(cv_scores.mean(), 4),
                "cv_accuracy_std": round(cv_scores.std(), 4),
                "cv_scores": cv_scores.tolist(),
                "train_accuracy": round(train_acc, 4),
                "cv_time_seconds": round(cv_time, 3),
                "train_time_seconds": round(train_time, 3),
            }
            results.append(entry)

            log.info(
                "  %s: CV=%.4f (+/- %.4f), train=%.4f",
                name, entry["cv_accuracy_mean"],
                entry["cv_accuracy_std"], entry["train_accuracy"],
            )

        # Sort by CV accuracy
        results.sort(key=lambda x: x["cv_accuracy_mean"], reverse=True)

        best = results[0]
        log.info(
            "Best model: %s (CV accuracy=%.4f)",
            best["name"], best["cv_accuracy_mean"],
        )

        return {
            "results": results,
            "best_model_name": best["name"],
            "best_model": best["model"],
            "trained_models": self.trained_models,
        }
