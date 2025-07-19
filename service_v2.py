"""Service para servir o modelo RAG de recrutamento via BentoML e MLflow."""

from __future__ import annotations
from typing import List, Dict
import bentoml
import mlflow
import pandas as pd
import tempfile

# Configure o URI de tracking do MLflow.
mlflow.set_tracking_uri("file:///C:/Users/win/Desktop/Projetos/Datathon/llm/src/mlruns")

EXAMPLE_INPUT = [
    "Quais competências técnicas são mais valorizadas para a vaga Python?"
]

@bentoml.service(
    resources={"cpu": "4"},
    traffic={"timeout": 30},
    monitoring={"enabled": True},
    metrics={
        "enabled": True,
        "namespace": "rag_recrutamento_service",
    },
    cors={"enabled": True, "allow_origins": ["*"]}
)
class RAGRecrutamentoService:


    def __init__(self):
        """
        Inicializa o serviço, carrega o modelo e busca os metadados do MLflow UMA VEZ.
        """
        self.bento_model = bentoml.models.get("RAG_Recrutamento:latest")
        self.model = bentoml.mlflow.load_model(self.bento_model)
        self.mlflow_client = mlflow.tracking.MlflowClient()

        MLFLOW_REGISTERED_MODEL_NAME = "RAG_Recrutamento"
        try:
            latest_versions = self.mlflow_client.get_latest_versions(MLFLOW_REGISTERED_MODEL_NAME)
            if latest_versions:
                latest_mlflow_version = latest_versions[0]
                run_id = latest_mlflow_version.run_id
                run_info = self.mlflow_client.get_run(run_id)
                self.model_metadata = {
                    "model_name": MLFLOW_REGISTERED_MODEL_NAME,
                    "bento_model_tag": str(self.bento_model.tag),
                    "mlflow_model_version": latest_mlflow_version.version,
                    "mlflow_run_id": run_id,
                    "metrics": run_info.data.metrics,
                    "parameters": run_info.data.params,
                    "tags": run_info.data.tags
                }
            else:
                self.model_metadata = {
                    "error": f"Modelo '{MLFLOW_REGISTERED_MODEL_NAME}' encontrado, mas não possui versões no MLflow."
                }
        except Exception as e:
            self.model_metadata = {
                "error": f"Falha ao carregar metadados do MLflow: {str(e)}",
                "searched_model_name": MLFLOW_REGISTERED_MODEL_NAME
            }


    @bentoml.api()
    def inserir_pergunta(self, perguntas: List[str]) -> Dict:
        """
        Endpoint que recebe perguntas, retorna a resposta e os metadados do modelo.
        """
        if not perguntas or not isinstance(perguntas, list):
            return {"error": "Input deve ser uma lista de perguntas não vazia."}
        
        df = pd.DataFrame({"pergunta": perguntas})
        respostas = self.model.predict(df)
        respostas_list = respostas.tolist() if hasattr(respostas, 'tolist') else list(respostas)

        # Loga esta interação específica como um novo run no MLflow
        with mlflow.start_run(run_name="log_prediction_request") as log_run:
            mlflow.log_param("num_perguntas", len(perguntas))
            mlflow.log_param("perguntas", perguntas)
            # Logar respostas como artefato é uma boa prática para textos longos
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding='utf-8') as f:
                f.write("\n".join(map(str, respostas_list)))
                mlflow.log_artifact(f.name, artifact_path="respostas")
        
        # Combina a resposta da predição com os metadados do modelo
        return {
            "respostas": respostas_list,
            "prediction_log_run_id": log_run.info.run_id,
            "model_metadata": self.model_metadata
        }

    @bentoml.api(batchable=True)
    def responder(self, perguntas: List[str] = EXAMPLE_INPUT) -> List[str]:
        """
        Endpoint padrão de inferência que retorna apenas a lista de respostas.
        """
        if not perguntas or not isinstance(perguntas, list):
            return []
        with bentoml.monitor("rag_recrutamento") as mon:
            mon.log(perguntas, name="request", role="input", data_type="list")
            df = pd.DataFrame({"pergunta": perguntas})
            respostas = self.model.predict(df)
            respostas_list = respostas.tolist() if hasattr(respostas, 'tolist') else list(respostas)
            mon.log(respostas_list, name="response", role="prediction", data_type="list")
            return respostas_list

    @bentoml.api(input_spec=None, route="/info" )
    def info(self) -> Dict:
        """
        Endpoint que retorna os metadados sobre o modelo em produção.
        """
        return self.model_metadata