"""Model evaluation module with comprehensive metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)

from cancer_agent.utils import get_logger

log = get_logger(__name__)


class ModelEvaluator:
    """Evaluates trained models on the test set with comprehensive metrics."""

    def run(self, training_results: dict, processed_data: dict) -> dict:
        """
        Evaluate all trained models on the held-out test set.

        Returns a dict of model_name -> metrics.
        """
        X_test = processed_data["X_test"]
        y_test = processed_data["y_test"]
        trained_models = training_results["trained_models"]

        log.info("Evaluating %d models on %d test samples", len(trained_models), len(X_test))

        evaluations = {}

        for name, model in trained_models.items():
            y_pred = model.predict(X_test)

            # Probability predictions if available
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }

            if y_prob is not None:
                metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)

            # Feature importance (if available)
            feature_names = processed_data["feature_names"]
            importance = self._get_feature_importance(model, feature_names)
            if importance:
                metrics["top_features"] = importance[:10]

            evaluations[name] = metrics

            log.info(
                "  %s: acc=%.4f, f1=%.4f, auc=%s, mcc=%.4f",
                name,
                metrics["accuracy"],
                metrics["f1"],
                metrics.get("roc_auc", "N/A"),
                metrics["mcc"],
            )

        # Determine best model on test set
        best_name = max(evaluations, key=lambda n: evaluations[n]["f1"])
        log.info(
            "Best model on test set: %s (F1=%.4f)",
            best_name, evaluations[best_name]["f1"],
        )

        return {
            "evaluations": evaluations,
            "best_model_name": best_name,
            "best_metrics": evaluations[best_name],
        }

    def _get_feature_importance(self, model, feature_names: list[str]) -> list[dict]:
        """Extract feature importances from a model if supported."""
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()

        if importances is None:
            return []

        paired = list(zip(feature_names, importances))
        paired.sort(key=lambda x: x[1], reverse=True)

        return [
            {"feature": name, "importance": round(float(imp), 6)}
            for name, imp in paired
        ]
