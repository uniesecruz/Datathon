# 02_ingestion.py
# Ingestão, pré-processamento e indexação dos dados do CSV para o retriever RAG
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    # Concatenar campos relevantes para contexto
    df['contexto'] = df['cv_pt'].fillna('') + ' ' + \
        df['perfil_vaga.principais_atividades'].fillna('') + ' ' + \
        df['perfil_vaga.competencia_tecnicas_e_comportamentais'].fillna('')
    return df

def embed_contexts(df, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['contexto'].tolist(), show_progress_bar=True)
    return embeddings

if __name__ == "__main__":
    csv_path = os.path.join('..', 'data', 'processed', 'cv_atividades_competencias.csv')
    df = load_and_prepare_data(csv_path)
    embeddings = embed_contexts(df)
    np.save(os.path.join('..', 'data', 'processed', 'embeddings.npy'), embeddings)
    df.to_csv(os.path.join('..', 'data', 'processed', 'contextos_completos.csv'), index=False)
