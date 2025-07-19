import mlflow
import bentoml 
import pandas as pd
from model import RAGRunnable

class Register:
    """Classe para registro de modelos no MLflow e BentoML."""
    def __init__(self, title: str):
        self.title = title

    def log_rag_model(self):
        import numpy as np
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        input_example = pd.DataFrame({'pergunta': ['Quais competências técnicas são mais valorizadas para a vaga Java?']})
        mlflow.set_experiment(self.title)
        pergunta = input_example['pergunta'][0]
        resposta = None
        with mlflow.start_run() as run:
            # Gera resposta usando o modelo, se possível
            try:
                resposta = RAGRunnable().predict(input_example)[0]
            except Exception:
                resposta = "Resposta de exemplo gerada pelo modelo."
            model_info = mlflow.pyfunc.log_model(
                artifact_path='rag_agent',
                python_model=RAGRunnable(),
                code_paths=['model.py'],
                pip_requirements=[
                    'transformers==4.46.3',
                    'torch==2.7.1',
                    'sentence-transformers==2.7.0',
                    'pandas',
                    'numpy'
                ],
                input_example=input_example
            )
            run_id = run.info.run_id
            # Loga pergunta e resposta
            mlflow.log_param('pergunta', pergunta)
            mlflow.log_text(resposta, 'resposta.txt')
            # Métricas adicionais
            response_length = len(resposta)
            mlflow.log_metric('response_length', response_length)
            word_count = len(resposta.split())
            mlflow.log_metric('response_word_count', word_count)
            sentence_count = resposta.count('.')
            mlflow.log_metric('response_sentence_count', sentence_count)
            alpha_ratio = sum(c.isalpha() for c in resposta) / max(1, len(resposta))
            mlflow.log_metric('response_alpha_ratio', alpha_ratio)
            avg_word_length = sum(len(w) for w in resposta.split()) / max(1, word_count)
            mlflow.log_metric('response_avg_word_length', avg_word_length)
            # ROUGE-1 e BLEU se possível
            referencia = pergunta  # Exemplo: pode ser ajustado para referência real
            try:
                scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
                rouge1 = scorer.score(referencia, resposta)['rouge1'].fmeasure
                mlflow.log_metric('manual_rouge1', rouge1)
                smoothie = SmoothingFunction().method4
                bleu = sentence_bleu([referencia.split()], resposta.split(), smoothing_function=smoothie)
                mlflow.log_metric('manual_bleu', bleu)
            except Exception:
                pass
        return run_id, model_info.model_uri

    def register_model(self, run_id: str):
        """Registra o modelo no MLflow Model Registry e no BentoML."""
        model_uri = f"runs:/{run_id}/rag_agent"
        result = mlflow.register_model(
            model_uri,
            name=self.title,
            tags={"status": "demo", "owner": "Sergio"}
        )
        bentoml.mlflow.import_model(self.title, model_uri)
        return result

if __name__ == "__main__":
    reg = Register(title='RAG_Recrutamento')
    run_id, model_uri = reg.log_rag_model()
    print(f"Modelo logado em run_id: {run_id}, uri: {model_uri}")
    reg.register_model(run_id)
