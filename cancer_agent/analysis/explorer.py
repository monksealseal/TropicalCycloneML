"""Exploratory data analysis module for cancer datasets."""

import numpy as np
import pandas as pd
from scipy import stats

from cancer_agent.utils import get_logger

log = get_logger(__name__)


class DataExplorer:
    """Performs autonomous exploratory data analysis on cancer datasets."""

    def __init__(self):
        self.report = {}

    def run(self, dataset: dict) -> dict:
        """
        Run full exploratory analysis on a dataset.

        Returns a comprehensive EDA report dict.
        """
        df = dataset["df"]
        feature_names = dataset["feature_names"]
        target_name = dataset["target_name"]

        log.info("Running exploratory data analysis on %d samples", len(df))

        self.report = {
            "basic_stats": self._basic_stats(df, feature_names),
            "class_balance": self._class_balance(df, target_name),
            "feature_correlations": self._correlations(df, feature_names),
            "top_discriminative_features": self._discriminative_features(
                df, feature_names, target_name
            ),
            "outlier_summary": self._outlier_analysis(df, feature_names),
        }

        log.info("EDA complete: %d analysis sections generated", len(self.report))
        return self.report

    def _basic_stats(self, df: pd.DataFrame, features: list[str]) -> dict:
        """Compute basic descriptive statistics."""
        desc = df[features].describe()
        return {
            "shape": df.shape,
            "summary": desc.to_dict(),
            "dtypes": df[features].dtypes.astype(str).to_dict(),
        }

    def _class_balance(self, df: pd.DataFrame, target: str) -> dict:
        """Analyze target class distribution."""
        counts = df[target].value_counts()
        proportions = df[target].value_counts(normalize=True)
        imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")

        balance_status = "balanced" if imbalance_ratio < 1.5 else (
            "moderate_imbalance" if imbalance_ratio < 3.0 else "severe_imbalance"
        )

        log.info(
            "Class balance: %s (ratio=%.2f)", balance_status, imbalance_ratio
        )

        return {
            "counts": counts.to_dict(),
            "proportions": proportions.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "status": balance_status,
        }

    def _correlations(self, df: pd.DataFrame, features: list[str]) -> dict:
        """Compute feature correlation matrix and identify highly correlated pairs."""
        corr_matrix = df[features].corr()

        # Find highly correlated feature pairs (|r| > 0.9)
        high_corr_pairs = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.9:
                    high_corr_pairs.append({
                        "feature_1": features[i],
                        "feature_2": features[j],
                        "correlation": round(r, 4),
                    })

        high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        log.info("Found %d highly correlated feature pairs (|r|>0.9)", len(high_corr_pairs))

        return {
            "highly_correlated_pairs": high_corr_pairs,
            "n_highly_correlated": len(high_corr_pairs),
        }

    def _discriminative_features(self, df: pd.DataFrame, features: list[str],
                                  target: str) -> list[dict]:
        """
        Rank features by discriminative power using t-test between classes.
        """
        classes = df[target].unique()
        if len(classes) != 2:
            log.warning("Discriminative analysis requires exactly 2 classes, found %d", len(classes))
            return []

        results = []
        group_0 = df[df[target] == classes[0]]
        group_1 = df[df[target] == classes[1]]

        for feat in features:
            t_stat, p_val = stats.ttest_ind(
                group_0[feat].dropna(), group_1[feat].dropna()
            )
            effect_size = abs(
                group_0[feat].mean() - group_1[feat].mean()
            ) / df[feat].std() if df[feat].std() > 0 else 0.0

            results.append({
                "feature": feat,
                "t_statistic": round(t_stat, 4),
                "p_value": p_val,
                "effect_size_cohens_d": round(effect_size, 4),
            })

        results.sort(key=lambda x: abs(x["t_statistic"]), reverse=True)
        top_n = results[:10]

        log.info("Top discriminative features:")
        for i, r in enumerate(top_n[:5]):
            log.info(
                "  %d. %s (t=%.2f, d=%.2f, p=%.2e)",
                i + 1, r["feature"], r["t_statistic"],
                r["effect_size_cohens_d"], r["p_value"],
            )

        return top_n

    def _outlier_analysis(self, df: pd.DataFrame, features: list[str]) -> dict:
        """Detect outliers using IQR method."""
        outlier_counts = {}
        total_outliers = 0

        for feat in features:
            q1 = df[feat].quantile(0.25)
            q3 = df[feat].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = ((df[feat] < lower) | (df[feat] > upper)).sum()
            if n_outliers > 0:
                outlier_counts[feat] = int(n_outliers)
                total_outliers += n_outliers

        log.info(
            "Outlier analysis: %d total outliers across %d features",
            total_outliers, len(outlier_counts),
        )

        return {
            "features_with_outliers": outlier_counts,
            "total_outlier_values": total_outliers,
            "n_features_with_outliers": len(outlier_counts),
        }
