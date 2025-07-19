import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import src.model_selector as model_selector

class TestModelSelectorV2(unittest.TestCase):
    def setUp(self):
        self.test_csv_path = 'fake_test.csv'
        self.df = pd.DataFrame({
            'pergunta': ['p1', 'p2', 'p3', 'p4', 'p5'],
            'resposta_esperada': ['r1', 'r2', 'r3', 'r4', 'r5']
        })

    @patch('src.model_selector.pd.read_csv')
    def test_prepare_test_df_with_resposta_esperada(self, mock_read_csv):
        mock_read_csv.return_value = self.df.copy()
        test_df = model_selector.prepare_test_df(self.test_csv_path, n=3)
        self.assertIn('pergunta', test_df.columns)
        self.assertIn('resposta_esperada', test_df.columns)
        self.assertEqual(len(test_df), 3)
        self.assertEqual(test_df['pergunta'].tolist(), ['p1', 'p2', 'p3'])
        self.assertEqual(test_df['resposta_esperada'].tolist(), ['r1', 'r2', 'r3'])

    @patch('src.model_selector.pd.read_csv')
    def test_prepare_test_df_without_resposta_esperada(self, mock_read_csv):
        df = pd.DataFrame({'pergunta': ['p1', 'p2', 'p3']})
        mock_read_csv.return_value = df
        test_df = model_selector.prepare_test_df(self.test_csv_path, n=2)
        self.assertIn('pergunta', test_df.columns)
        self.assertNotIn('resposta_esperada', test_df.columns)
        self.assertEqual(len(test_df), 2)
        self.assertEqual(test_df['pergunta'].tolist(), ['p1', 'p2'])

    @patch('src.model_selector.Register')
    @patch('src.model_selector.mlflow')
    @patch('src.model_selector.Experiment')
    @patch('src.model_selector.RAGRunnable')
    def test_main_flow(self, mock_rag, mock_experiment, mock_mlflow, mock_register):
        # Setup mocks
        mock_model = MagicMock()
        mock_rag.return_value = mock_model
        mock_exp = MagicMock()
        mock_experiment.return_value = mock_exp
        mock_exp.track.return_value = MagicMock(model_uri='runs:/runid1/rag_agent')
        mock_model.predict.return_value = ['resp1', 'resp2', 'resp3', 'resp4', 'resp5']
        mock_register_instance = MagicMock()
        mock_register.return_value = mock_register_instance
        # Patch np.argmax to always return 0
        with patch('src.model_selector.np.argmax', return_value=0):
            with patch('src.model_selector.prepare_test_df', return_value=self.df.copy()):
                with patch('src.model_selector.rouge_scorer.RougeScorer') as mock_rouge:
                    mock_scorer = MagicMock()
                    mock_scorer.score.return_value = {'rouge1': MagicMock(fmeasure=0.9)}
                    mock_rouge.return_value = mock_scorer
                    # Run main
                    import sys
                    sys.modules['__main__'].__file__ = 'src/model_selector.py'  # Trick for __main__ check
                    exec(open('c:/Users/win/Desktop/Projetos/Datathon/llm/src/model_selector.py').read(), globals())
        # Check if register_model was called
        mock_register_instance.register_model.assert_called()

if __name__ == "__main__":
    unittest.main()
