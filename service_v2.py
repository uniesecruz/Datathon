"""Service para servir o modelo RAG de recrutamento via BentoML e MLflow."""

from __future__ import annotations
from typing import List
import bentoml
import mlflow

# Configure o URI de tracking do MLflow. 
# Se estiver rodando localmente, ele geralmente fica em uma pasta ./mlruns
# Se estiver usando um servidor, coloque o endereço aqui.
mlflow.set_tracking_uri("file:./mlruns")

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
)
class RAGRecrutamentoService:
    """Serviço BentoML para o modelo RAG de recrutamento."""

    bento_model = bentoml.models.get("RAG_Recrutamento:latest")

    def __init__(self):
        """Inicializa o serviço e carrega o modelo."""
        self.model = bentoml.mlflow.load_model(self.bento_model)
        self.mlflow_client = mlflow.tracking.MlflowClient()

    @bentoml.api(batchable=True)
    def responder(self, perguntas: List[str] = EXAMPLE_INPUT) -> List[str]:
        """
        Recebe uma lista de perguntas via POST e retorna as respostas geradas pelo modelo RAG.
        """
        import pandas as pd
        with bentoml.monitor("rag_recrutamento") as mon:
            mon.log(perguntas, name="request", role="input", data_type="list")
            df = pd.DataFrame({"pergunta": perguntas})
            respostas = self.model.predict(df)
            mon.log(respostas, name="response", role="prediction", data_type="list")
            return respostas

    @bentoml.api()
    def info(self):
        """
        Endpoint GET que retorna metadados sobre o modelo em produção,
        incluindo nome, versão, e métricas do run do MLflow.
        """
        try:
            model_name = self.bento_model.tag.name
            model_version = self.bento_model.tag.version

            # Busca a versão mais recente do modelo registrado com o nome correspondente
            latest_version_info = self.mlflow_client.get_latest_versions(model_name, stages=["None"])
            if not latest_version_info:
                return {"error": f"Nenhuma versão do modelo '{model_name}' encontrada no MLflow."}

            # Pega o ID do run associado a esta versão do modelo
            run_id = latest_version_info[0].run_id
            run_info = self.mlflow_client.get_run(run_id)

            return {
                "model_name": model_name,
                "model_version": model_version,
                "mlflow_run_id": run_id,
                "source_algorithm": run_info.data.tags.get('mlflow.source.name', 'N/A'),
                "metrics": run_info.data.metrics,
                "parameters": run_info.data.params
            }
        except Exception as e:
            return {"error": str(e)}