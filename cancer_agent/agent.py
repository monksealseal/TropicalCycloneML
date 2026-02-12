"""
Autonomous Cancer Research ML Agent.

Orchestrates the full pipeline: data loading -> EDA -> preprocessing ->
model training -> evaluation -> reporting. Runs end-to-end without
human intervention once launched.
"""

import json
import os
import sys
import traceback

from cancer_agent.data import DatasetLoader, Preprocessor
from cancer_agent.analysis import DataExplorer
from cancer_agent.models import ModelTrainer
from cancer_agent.evaluation import ModelEvaluator, Reporter
from cancer_agent.utils import get_logger

log = get_logger("cancer_agent")

DISCLAIMER = (
    "DISCLAIMER: This agent is an ML research tool for analyzing publicly "
    "available cancer datasets. It does NOT provide medical diagnoses, "
    "treatment recommendations, or replace professional medical advice. "
    "All outputs are for research and educational purposes only."
)


class CancerResearchAgent:
    """
    Autonomous agent that runs a complete cancer dataset ML analysis pipeline.

    Stages:
        1. Data Loading    - fetch cancer dataset
        2. EDA             - exploratory data analysis
        3. Preprocessing   - clean, scale, split
        4. Training        - train and cross-validate multiple models
        5. Evaluation      - test-set metrics for all models
        6. Reporting       - compile and print final report
    """

    def __init__(
        self,
        dataset: str = "breast_cancer",
        models: list[str] | None = None,
        scaling: str = "standard",
        test_size: float = 0.2,
        cv_folds: int = 5,
        output_dir: str = "cancer_agent_output",
    ):
        self.dataset_name = dataset
        self.model_names = models
        self.scaling = scaling
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.output_dir = output_dir

        # Pipeline state
        self._raw_data = None
        self._eda_report = None
        self._processed_data = None
        self._training_results = None
        self._evaluation_results = None
        self._report = None

    def run(self) -> dict:
        """
        Execute the full autonomous pipeline.

        Returns the final report dict.
        """
        log.info("=" * 60)
        log.info("AUTONOMOUS CANCER RESEARCH ML AGENT v0.1.0")
        log.info("=" * 60)
        log.info("")
        log.info(DISCLAIMER)
        log.info("")

        stages = [
            ("1/6 Data Loading", self._stage_load),
            ("2/6 Exploratory Analysis", self._stage_eda),
            ("3/6 Preprocessing", self._stage_preprocess),
            ("4/6 Model Training", self._stage_train),
            ("5/6 Evaluation", self._stage_evaluate),
            ("6/6 Report Generation", self._stage_report),
        ]

        for stage_name, stage_fn in stages:
            log.info("")
            log.info("-" * 60)
            log.info("STAGE: %s", stage_name)
            log.info("-" * 60)
            try:
                stage_fn()
            except Exception:
                log.error("Stage '%s' failed:\n%s", stage_name, traceback.format_exc())
                raise

        return self._report

    def _stage_load(self):
        loader = DatasetLoader()
        self._raw_data = loader.load(self.dataset_name)

    def _stage_eda(self):
        explorer = DataExplorer()
        self._eda_report = explorer.run(self._raw_data)

    def _stage_preprocess(self):
        preprocessor = Preprocessor(
            scaling=self.scaling,
            test_size=self.test_size,
        )
        self._processed_data = preprocessor.run(self._raw_data)

    def _stage_train(self):
        trainer = ModelTrainer(
            models=self.model_names,
            cv_folds=self.cv_folds,
        )
        self._training_results = trainer.run(self._processed_data)

    def _stage_evaluate(self):
        evaluator = ModelEvaluator()
        self._evaluation_results = evaluator.run(
            self._training_results, self._processed_data
        )

    def _stage_report(self):
        reporter = Reporter()
        self._report = reporter.generate(
            dataset_metadata=self._processed_data["metadata"],
            eda_report=self._eda_report,
            preprocessing_info=self._processed_data["preprocessing_info"],
            training_results=self._training_results,
            evaluation_results=self._evaluation_results,
        )

        # Print summary
        summary = reporter.print_summary(self._report)
        print("\n" + summary)

        # Save JSON report
        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, "report.json")
        reporter.save_json(self._report, json_path)
        log.info("Full JSON report saved to: %s", json_path)
