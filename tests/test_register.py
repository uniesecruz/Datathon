import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.register import Register

class TestRegisterV2(unittest.TestCase):
    def setUp(self):
        self.reg = Register(title='RAG_Recrutamento')

    @patch('src.register.mlflow')
    @patch('src.register.RAGRunnable')
    def test_log_rag_model(self, mock_ragrunnable, mock_mlflow):
        # Setup mocks
        mock_run = MagicMock()
        mock_run.info.run_id = 'runid1'
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_model_info = MagicMock()
        mock_model_info.model_uri = 'runs:/runid1/rag_agent'
        mock_mlflow.pyfunc.log_model.return_value = mock_model_info
        mock_ragrunnable.return_value.predict.return_value = ['mocked resposta']
        # Patch metrics
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.log_param = MagicMock()
        mock_mlflow.log_text = MagicMock()
        # Patch rouge and bleu
        with patch('src.register.rouge_scorer.RougeScorer') as mock_rouge, \
             patch('src.register.SmoothingFunction') as mock_smooth, \
             patch('src.register.sentence_bleu', return_value=0.5):
            mock_rouge.return_value.score.return_value = {'rouge1': MagicMock(fmeasure=0.9)}
            mock_smooth.return_value.method4 = MagicMock()
            run_id, model_uri = self.reg.log_rag_model()
        self.assertEqual(run_id, 'runid1')
        self.assertEqual(model_uri, 'runs:/runid1/rag_agent')
        mock_mlflow.log_metric.assert_any_call('response_length', unittest.mock.ANY)
        mock_mlflow.log_metric.assert_any_call('manual_rouge1', unittest.mock.ANY)
        mock_mlflow.log_metric.assert_any_call('manual_bleu', unittest.mock.ANY)

    @patch('src.register.mlflow')
    @patch('src.register.bentoml')
    def test_register_model(self, mock_bentoml, mock_mlflow):
        mock_result = MagicMock()
        mock_mlflow.register_model.return_value = mock_result
        mock_bentoml.mlflow.import_model = MagicMock()
        result = self.reg.register_model(run_id='runid1')
        self.assertEqual(result, mock_result)
        mock_mlflow.register_model.assert_called_once()
        mock_bentoml.mlflow.import_model.assert_called_once()

if __name__ == "__main__":
    unittest.main()
