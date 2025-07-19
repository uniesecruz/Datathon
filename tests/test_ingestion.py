import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src import ingestion

class TestIngestionV2(unittest.TestCase):
    def setUp(self):
        self.sample_csv = 'fake_path.csv'
        self.sample_df = pd.DataFrame({
            'cv_pt': ['CV1', 'CV2'],
            'perfil_vaga.principais_atividades': ['Ativ1', 'Ativ2'],
            'perfil_vaga.competencia_tecnicas_e_comportamentais': ['Comp1', 'Comp2']
        })

    @patch('src.ingestion.pd.read_csv')
    def test_load_and_prepare_data(self, mock_read_csv):
        mock_read_csv.return_value = self.sample_df.copy()
        df = ingestion.load_and_prepare_data(self.sample_csv)
        self.assertIn('contexto', df.columns)
        self.assertEqual(df['contexto'][0], 'CV1 Ativ1 Comp1')
        self.assertEqual(df['contexto'][1], 'CV2 Ativ2 Comp2')

    @patch('src.ingestion.SentenceTransformer')
    def test_embed_contexts(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1,2,3],[4,5,6]])
        mock_sentence_transformer.return_value = mock_model
        df = self.sample_df.copy()
        df['contexto'] = ['contexto1', 'contexto2']
        embeddings = ingestion.embed_contexts(df, model_name='mock-model')
        mock_sentence_transformer.assert_called_once_with('mock-model')
        mock_model.encode.assert_called_once_with(['contexto1', 'contexto2'], show_progress_bar=True)
        np.testing.assert_array_equal(embeddings, np.array([[1,2,3],[4,5,6]]))

if __name__ == "__main__":
    unittest.main()
