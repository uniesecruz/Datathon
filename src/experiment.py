
import mlflow
import mlflow.models
from typing import Dict
import numpy as np
import pandas as pd
from src.model import RAGAgent
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

class Experiment:
    """Classe para experimentos e tracking com MLflow."""
    def __init__(self, model: object, title: str):
        self.model = model
        self.title = title

    def track(self, run_name: str, **kwargs):
        mlflow.set_experiment(self.title)
        signature = mlflow.models.infer_signature(
            model_input=pd.DataFrame({'pergunta': ['What are the three primary colors?']}),
            model_output=pd.DataFrame({'resposta': ['The three primary colors are red, yellow, and blue.']})
        )
        with mlflow.start_run(run_name=run_name):
            model_info = mlflow.pyfunc.log_model(
                python_model=self.model,
                signature=signature,
                artifact_path="rag_agent",
                pip_requirements=["transformers==4.46.3", "torch==2.7.1", "sentence-transformers==2.7.0", "pandas", "numpy"],
                code_paths=["model.py"],
                **kwargs,
            )
            mlflow.log_params({"model_name": getattr(self.model, 'llm_name', 'rag_agent'), "task": "rag"})
        return model_info

    def evaluate(self, model_uri: str, test_df: pd.DataFrame) -> Dict:
        run_id = model_uri.split("/")[1] if "/" in model_uri else None
        extra_metrics = [mlflow.metrics.latency(), mlflow.metrics.rouge1(), mlflow.metrics.bleu()]
        with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
            results = mlflow.evaluate(
                model_uri,
                test_df,
                evaluators="default",
                model_type="text-generation",
                targets="resposta_esperada",
                extra_metrics=extra_metrics,
            )
        return results.metrics

    def search_finished_experiments(self, run_name: str, **kwargs) -> pd.DataFrame:
        filters = (
            f"attributes.run_name = '{run_name}'" + " and attributes.status = 'FINISHED'"
        )
        return mlflow.search_runs(
            experiment_names=[self.title], filter_string=filters, **kwargs
        )

def run_experiment(llm_name, pergunta, embeddings_path, contextos_path):
    embeddings = np.load(embeddings_path)
    df = pd.read_csv(contextos_path)
    agent = RAGAgent(
        llm_name=llm_name,
        retriever_embeddings=embeddings,
        retriever_contexts=df['contexto'].tolist()
    )
    resposta = agent.generate(pergunta)
    return resposta

# Função de avaliação para modelos MLflow (esta função ainda não está sendo chamada no seu main)
def evaluate_model(model_uri: str, test_df: pd.DataFrame) -> Dict:
    """Avalia o modelo MLflow usando o DataFrame de teste.

    Args:
        model_uri (str): URI do modelo MLflow para avaliação.
        test_df (pd.DataFrame): DataFrame de teste com coluna 'pergunta' e (opcionalmente) 'resposta_esperada'.

    Returns:
        Dict: Métricas de avaliação.
    """
    run_id = model_uri.split("/")[1] if "/" in model_uri else None
    # Adicione métricas de similaridade e qualidade de texto
    extra_metrics = [
        mlflow.metrics.latency(),
        mlflow.metrics.rouge1(),
        mlflow.metrics.bleu(),
    ]
    with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
        results = mlflow.evaluate(
            model_uri,
            test_df,
            evaluators="default",
            model_type="text-generation",
            targets="resposta_esperada",  # Ajuste para o nome da coluna de referência
            extra_metrics=extra_metrics,
        )
    return results.metrics

if __name__ == "__main__":
    # Exemplo de inferência de assinatura para registro de modelo
    import mlflow.models
    signature = mlflow.models.infer_signature(
        model_input=pd.DataFrame({'pergunta': ['What are the three primary colors?']}),
        model_output=pd.DataFrame({'resposta': ['The three primary colors are red, yellow, and blue.']})
    )
    print(f"Exemplo de signature inferida para registro de modelo:\n{signature}")
    mlflow.set_experiment('RAG_Recrutamento')
    pergunta = 'Qual o nome do melhor candidato para trabalhar com analise de dados?'

    # Prepare um DataFrame de teste simples para a avaliação
    # IMPORTANTE: Em um cenário real, 'resposta_esperada' viria de um dataset de teste validado.
    test_df_for_evaluation = pd.DataFrame({
        'pergunta': [pergunta],
        'resposta_esperada': ['Conhecimentos em Python, frameworks como Django ou Flask, experiência com bancos de dados relacionais e versionamento de código com Git são altamente valorizados para a vaga Python.'],
        'predictions': [None]  # Será preenchido após gerar a resposta
    })

    with mlflow.start_run() as current_run: # Capture a execução atual
        run_id = current_run.info.run_id

        agent = RAGAgent( # Instancie o agente aqui para poder logá-lo
            llm_name='distilgpt2',
            retriever_embeddings=np.load('../data/processed/embeddings.npy'),
            retriever_contexts=pd.read_csv('../data/processed/contextos_completos.csv')['contexto'].tolist()
        )
        resposta = agent.generate(pergunta)
        # Preenche a coluna de predições no DataFrame de teste
        test_df_for_evaluation.at[0, 'predictions'] = resposta

        mlflow.log_param('pergunta', pergunta)
        mlflow.log_text(resposta, 'resposta.txt')

        # --- SUAS MÉTRICAS ADICIONAIS ---
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
        # ---------------------------

        # Salva a resposta em um arquivo temporário e loga como artefato
        temp_file_path = "resposta_gerada.txt"
        with open(temp_file_path, "w") as f:
            f.write(resposta)
        mlflow.log_artifact(temp_file_path, "saidas")
        os.remove(temp_file_path)

        # Para avaliação automática, é necessário um modelo registrado no MLflow.
        # Aqui, calculamos métricas manualmente para as predições e referência.


        referencia = test_df_for_evaluation.at[0, 'resposta_esperada']
        predicao = test_df_for_evaluation.at[0, 'predictions']

        # ROUGE-1
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge1 = scorer.score(referencia, predicao)['rouge1'].fmeasure
        mlflow.log_metric('manual_rouge1', rouge1)

        # BLEU
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([referencia.split()], predicao.split(), smoothing_function=smoothie)
        mlflow.log_metric('manual_bleu', bleu)

        print(f"Métricas de avaliação registradas: ROUGE-1={rouge1:.4f}, BLEU={bleu:.4f}")
        # Busca experimentos finalizados com o mesmo nome de execução
        experiment = Experiment(agent, 'RAG_Recrutamento')
        finished_experiments = experiment.search_finished_experiments(run_name=current_run.data.tags.get('mlflow.runName', current_run.info.run_id))
        print("Experimentos finalizados encontrados:")
        print(finished_experiments)