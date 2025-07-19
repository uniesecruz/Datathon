import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.experiment import Experiment

class TestExperimentV2(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.llm_name = "mock_llm"
        self.experiment = Experiment(model=self.mock_model, title="Mock Experiment")

    @patch("mlflow.set_experiment")
    @patch("mlflow.models.infer_signature")
    @patch("mlflow.pyfunc.log_model")
    @patch("mlflow.log_params")
    @patch("mlflow.start_run")
    def test_track(self, mock_start_run, mock_log_params, mock_log_model, mock_infer_signature, mock_set_experiment):
        mock_signature = MagicMock()
        mock_infer_signature.return_value = mock_signature
        mock_model_info = MagicMock()
        mock_log_model.return_value = mock_model_info

        result = self.experiment.track(run_name="test_run")

        mock_set_experiment.assert_called_once_with("Mock Experiment")
        mock_infer_signature.assert_called_once()
        mock_log_model.assert_called_once()
        mock_log_params.assert_called_once_with({"model_name": "mock_llm", "task": "rag"})
        self.assertEqual(result, mock_model_info)

    @patch("mlflow.metrics.latency")
    @patch("mlflow.metrics.rouge1")
    @patch("mlflow.metrics.bleu")
    @patch("mlflow.evaluate")
    @patch("mlflow.start_run")
    def test_evaluate(self, mock_start_run, mock_evaluate, mock_bleu, mock_rouge1, mock_latency):
        mock_results = MagicMock()
        metrics = {"accuracy": 0.95, "latency": 200}
        mock_results.metrics = metrics
        mock_evaluate.return_value = mock_results
        test_df = pd.DataFrame({"pergunta": ["a"], "resposta_esperada": ["b"]})
        result = self.experiment.evaluate(model_uri="runs:/12345/model", test_df=test_df)
        mock_start_run.assert_called_once_with(run_id="12345")
        mock_evaluate.assert_called_once_with(
            "runs:/12345/model",
            test_df,
            evaluators="default",
            model_type="text-generation",
            targets="resposta_esperada",
            extra_metrics=[mock_latency(), mock_rouge1(), mock_bleu()]
        )
        self.assertEqual(result, metrics)

    @patch("mlflow.search_runs")
    def test_search_finished_experiments(self, mock_search_runs):
        mock_runs = MagicMock()
        mock_search_runs.return_value = mock_runs
        result = self.experiment.search_finished_experiments(run_name="test_run")
        mock_search_runs.assert_called_once_with(
            experiment_names=["Mock Experiment"],
            filter_string="attributes.run_name = 'test_run' and attributes.status = 'FINISHED'"
        )
        self.assertEqual(result, mock_runs)

if __name__ == "__main__":
    unittest.main()
