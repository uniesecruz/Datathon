
import os
import pandas as pd
import numpy as np
from src.model import RAGRunnable
from src.experiment import Experiment
from src.register import Register
import mlflow

title = "rag_recrutamento"
key_metric = "manual_rouge1"

def prepare_test_df(test_csv_path, n=5):
    df = pd.read_csv(test_csv_path)
    perguntas = df['pergunta'].dropna().tolist() if 'pergunta' in df.columns else [
        "Quais competências técnicas são mais valorizadas para a vaga Java?"
    ]
    # Se houver coluna resposta_esperada, inclua no test_df
    if 'resposta_esperada' in df.columns:
        test_df = pd.DataFrame({
            'pergunta': perguntas[:n],
            'resposta_esperada': df['resposta_esperada'].dropna().tolist()[:n]
        })
    else:
        test_df = pd.DataFrame({'pergunta': perguntas[:n]})
    return test_df

if __name__ == "__main__":
    test_csv_path = os.path.join('..', 'data', 'processed', 'cv_atividades_competencias.csv')
    contextos_path = os.path.join('..', 'data', 'processed', 'contextos_completos.csv')
    embeddings_path = os.path.join('..', 'data', 'processed', 'embeddings.npy')
    test_df = prepare_test_df(test_csv_path, n=5)

    models = {
        "distilgpt2": RAGRunnable(),
        # Adicione outros modelos se desejar, ex: "gpt2": RAGRunnable(llm_name="gpt2"),
    }
    results = []
    model_uris = []

    for run_name, model in models.items():
        print(f"Rodando experimento para: {run_name}")
        exp = Experiment(model=model, title=title)
        model_info = exp.track(run_name=run_name)
        respostas = model.predict(None, test_df)
        test_df['predictions'] = respostas

        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        if 'resposta_esperada' in test_df.columns:
            scores = [
                scorer.score(ref, pred)['rouge1'].fmeasure
                for ref, pred in zip(test_df['resposta_esperada'], test_df['predictions'])
            ]
            avg_rouge1 = np.mean(scores)
        else:
            avg_rouge1 = 0.0
        print(f"{run_name} - Média ROUGE-1: {avg_rouge1:.4f}")
        results.append(avg_rouge1)
        # Salva o model_uri e o run_id para registro posterior
        model_uris.append(model_info.model_uri)

        # Loga a métrica no mesmo run do modelo
        run_id = model_info.model_uri.split('/')[1] if '/' in model_info.model_uri else None
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric(key_metric, avg_rouge1)

    # Seleciona o melhor modelo
    best_run_idx = int(np.argmax(results))
    best_run_name = list(models)[best_run_idx]
    print(f"Melhor modelo: {best_run_name} (ROUGE-1={results[best_run_idx]:.4f})")

    # Registra o melhor modelo no MLflow Model Registry e BentoML
    reg = Register(title=title)
    # Extrai o run_id do model_uri do melhor modelo
    best_model_uri = model_uris[best_run_idx]
    best_run_id = best_model_uri.split('/')[1] if '/' in best_model_uri else None
    if best_run_id:
        reg.register_model(run_id=best_run_id)
        print(f"Modelo registrado no MLflow Model Registry e BentoML: {best_run_name}")
    else:
        print("Nenhum run_id encontrado para registro.")
