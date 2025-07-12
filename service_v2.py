"""Service para servir o modelo RAG de recrutamento via BentoML e MLflow."""

from __future__ import annotations
from typing import List
import bentoml

EXAMPLE_INPUT = [
    "Quais competências técnicas são mais valorizadas para a vaga Python?"
]

@bentoml.service(
    resources={"cpu": "4"},
    traffic={"timeout": 10},
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
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(batchable=True)
    def responder(self, perguntas: List[str] = EXAMPLE_INPUT) -> List[str]:
        """Gera respostas para perguntas de recrutamento usando o modelo RAG.

        Args:
            perguntas (list[str]): Perguntas para o modelo responder.

        Returns:
            list[str]: Respostas geradas.
        """
        with bentoml.monitor("rag_recrutamento") as mon:
            mon.log(perguntas, name="request", role="input", data_type="list")
            respostas = self.model.predict(perguntas)
            mon.log(respostas, name="response", role="prediction", data_type="list")
            return respostas
