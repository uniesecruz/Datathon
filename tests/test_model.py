import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from src.model import RAGAgent, RAGRunnable

class TestRAGAgent(unittest.TestCase):
    @patch('src.model.AutoTokenizer')
    @patch('src.model.AutoModelForCausalLM')
    def test_init_and_generate(self, mock_model, mock_tokenizer):
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock(generate=MagicMock(return_value=[[1,2,3]]))
        embeddings = np.random.rand(2, 3)
        contexts = ['ctx1', 'ctx2']
        agent = RAGAgent('distilgpt2', embeddings, contexts)
        # Patch retrieve to avoid sentence_transformers
        agent.retrieve = MagicMock(return_value=['ctx1'])
        agent.tokenizer = MagicMock()
        agent.tokenizer.return_tensors = 'pt'
        agent.tokenizer.__call__ = MagicMock(return_value={'input_ids': np.array([[1,2,3]])})
        agent.model = MagicMock()
        agent.model.generate = MagicMock(return_value=np.array([[1,2,3]]))
        agent.tokenizer.decode = MagicMock(return_value='mocked output')
        result = agent.generate('pergunta')
        self.assertEqual(result, 'mocked output')

class TestRAGRunnable(unittest.TestCase):
    @patch('src.model.RAGAgent')
    @patch('src.model.np.load')
    @patch('src.model.pd.read_csv')
    def test_init(self, mock_read_csv, mock_np_load, mock_rag_agent):
        mock_np_load.return_value = np.random.rand(2, 3)
        mock_read_csv.return_value = pd.DataFrame({'contexto': ['ctx1', 'ctx2']})
        rag_agent_instance = MagicMock()
        mock_rag_agent.return_value = rag_agent_instance
        runnable = RAGRunnable()
        self.assertIs(runnable.agent, rag_agent_instance)

    @patch('src.model.RAGAgent')
    def test_predict(self, mock_rag_agent):
        agent_instance = MagicMock()
        agent_instance.generate.side_effect = lambda q: f"resp:{q}"
        mock_rag_agent.return_value = agent_instance
        runnable = RAGRunnable()
        # DataFrame input
        df = pd.DataFrame({'pergunta': ['a', 'b']})
        out = runnable.predict(None, df)
        self.assertEqual(out, ['resp:a', 'resp:b'])
        # List input
        out2 = runnable.predict(None, ['c', 'd'])
        self.assertEqual(out2, ['resp:c', 'resp:d'])
        # String input
        out3 = runnable.predict(None, 'e')
        self.assertEqual(out3, ['resp:e'])

if __name__ == "__main__":
    unittest.main()
