# 01_model.py
# Módulo do agente RAG com HuggingFace para Recrutamento

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import util
import numpy as np
import mlflow.pyfunc
import pandas as pd
import torch
import os

device = 0 if torch.cuda.is_available() else -1

class RAGAgent:
    def __init__(self, llm_name, retriever_embeddings, retriever_contexts):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_name)
        if device == 0:
            self.model = self.model.to('cuda')
        self.retriever_embeddings = retriever_embeddings
        self.retriever_contexts = retriever_contexts

    def retrieve(self, query, top_k=3):
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        query_emb = embedder.encode([query])[0]
        hits = util.semantic_search(query_emb, self.retriever_embeddings, top_k=top_k)[0]
        return [self.retriever_contexts[hit['corpus_id']] for hit in hits]

    def generate(self, query):
        retrieved = self.retrieve(query)
        prompt = query + '\nContexto:\n' + '\n'.join(retrieved)
        # Para distilgpt2: max_length total (entrada+saida) = 1024
        max_new_tokens = 256
        max_model_length = 1024
        max_input_length = max_model_length - max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_input_length)
        if device == 0:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class RAGRunnable(mlflow.pyfunc.PythonModel):
    def __init__(self):
        # Inicializa o agente ao instanciar a classe (para uso direto)
        embeddings_path = os.path.join('..', 'data', 'processed', 'embeddings.npy')
        contextos_path = os.path.join('..', 'data', 'processed', 'contextos_completos.csv')
        embeddings = np.load(embeddings_path)
        df = pd.read_csv(contextos_path)
        self.agent = RAGAgent(
            llm_name='distilgpt2',
            retriever_embeddings=embeddings,
            retriever_contexts=df['contexto'].tolist()
        )

    def load_context(self, context):
        # Mantém para compatibilidade com MLflow
        pass

    def predict(self, context, model_input):
        # model_input pode ser um DataFrame com coluna 'pergunta', lista ou string
        if isinstance(model_input, pd.DataFrame) and 'pergunta' in model_input.columns:
            perguntas = model_input['pergunta'].tolist()
        elif isinstance(model_input, list):
            perguntas = model_input
        else:
            perguntas = [str(model_input)]
        return [self.agent.generate(q) for q in perguntas]
