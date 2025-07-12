import mlflow
import bentoml 
import pandas as pd
from model import RAGRunnable

class Register:
    """Classe para registro de modelos no MLflow e BentoML."""
    def __init__(self, title: str):
        self.title = title

    def log_rag_model(self):
        input_example = pd.DataFrame({'pergunta': ['Quais competências técnicas são mais valorizadas para a vaga Java?']})
        mlflow.set_experiment(self.title)
        with mlflow.start_run() as run:
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
