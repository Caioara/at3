"""Funções reutilizáveis para o protótipo Ouvidoria Inteligente."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MODEL_OPTIONS = {
    "Multilíngue MiniLM (recomendado)": "paraphrase-multilingual-MiniLM-L12-v2",
    "BGE português": "BAAI/bge-small-pt-v1.5",
}

def load_manifestacoes(path="manifestacoes.json"):
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)

def embed_texts(texts, model_name=None):
    """Gera embeddings densos; sem rede, usa Hashing TF-IDF como fallback estável."""
    texts = list(texts)
    if model_name:
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name).encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            pass
    vectorizer = HashingVectorizer(n_features=768, alternate_sign=False, norm="l2", ngram_range=(1, 2), strip_accents="unicode")
    return vectorizer.transform(texts).toarray()

def cosine_matrix(embeddings):
    return cosine_similarity(embeddings)

def detectar_duplicatas(textos, limiar=0.85, model_name=None):
    """Retorna pares i<j cuja similaridade de cosseno alcança o limiar."""
    matriz = cosine_matrix(embed_texts(textos, model_name))
    return [(i, j, float(matriz[i, j])) for i in range(len(textos)) for j in range(i + 1, len(textos)) if matriz[i, j] >= limiar]

def split_text(text, chunk_size=300, overlap=50):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]).split_text(text)
    except ImportError:
        step = max(1, chunk_size - overlap)
        return [text[i:i + chunk_size] for i in range(0, len(text), step)]

def reduce_embeddings(embeddings, method="PCA"):
    n = len(embeddings)
    if n < 2: return np.c_[np.zeros(n), np.zeros(n)]
    if method == "t-SNE":
        return TSNE(n_components=2, perplexity=min(10, n - 1), init="pca", learning_rate="auto", random_state=42).fit_transform(embeddings)
    return PCA(n_components=2, random_state=42).fit_transform(embeddings)
