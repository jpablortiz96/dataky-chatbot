import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ---------------------------
# Configuración
# ---------------------------
PAGES_PATH = "data/pages.jsonl"
INDEX_PATH = "data/index.joblib"

# Ajustes típicos para MVP:
MAX_FEATURES = 50000        # vocabulario máximo (controla memoria)
NGRAM_RANGE = (1, 2)        # 1-gramas y 2-gramas (mejor para frases)
MIN_DF = 2                  # ignora términos que aparecen en 1 solo doc (ruido)
MAX_DF = 0.95               # ignora términos demasiado comunes (stop-like)
LOWERCASE = True


@dataclass
class Doc:
    url: str
    title: str
    text: str


# ---------------------------
# Carga de datos
# ---------------------------
def load_pages_jsonl(path: str) -> List[Doc]:
    """
    Lee un archivo JSONL (1 JSON por línea) y devuelve una lista de Docs.
    Esto es eficiente para datasets grandes y fácil de depurar.
    """
    docs: List[Doc] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(Doc(url=obj["url"], title=obj.get("title", ""), text=obj.get("text", "")))
    return docs


def build_corpus(docs: List[Doc]) -> List[str]:
    """
    Construye el texto final por documento.
    Nota: concatenar title + text suele mejorar el retrieval porque
    el título contiene keywords importantes.
    """
    corpus: List[str] = []
    for d in docs:
        combined = (d.title + "\n" + d.text).strip()
        corpus.append(combined)
    return corpus


# ---------------------------
# Indexación TF-IDF
# ---------------------------
def train_tfidf(corpus: List[str]) -> Tuple[TfidfVectorizer, "scipy.sparse.csr_matrix"]:
    """
    Entrena el vectorizador TF-IDF y transforma el corpus.
    Devuelve:
      - vectorizer: objeto con el vocabulario y parámetros
      - X: matriz sparse (n_docs x n_features)
    """
    vectorizer = TfidfVectorizer(
        lowercase=LOWERCASE,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        # stop_words=None  # opcional: en español se puede añadir lista custom
    )

    X = vectorizer.fit_transform(corpus)

    # Normalizamos para mejorar similitud coseno.
    # Con TF-IDF normalizado, cosine similarity se comporta mejor y es estable.
    X = normalize(X)

    return vectorizer, X


def save_index(index_path: str, vectorizer: TfidfVectorizer, X, docs: List[Doc]) -> None:
    """
    Guarda todo en un solo archivo joblib:
      - vectorizer (vocabulario)
      - X (matriz TF-IDF)
      - docs (metadata: url/title/text)
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    payload = {
        "vectorizer": vectorizer,
        "X": X,
        # guardamos metadata, NO todo el texto completo si quieres ahorrar
        "docs": [{"url": d.url, "title": d.title, "text": d.text} for d in docs],
    }

    joblib.dump(payload, index_path)


# ---------------------------
# Main
# ---------------------------
def main():
    if not os.path.exists(PAGES_PATH):
        raise FileNotFoundError(
            f"No encontré {PAGES_PATH}. Primero ejecuta el módulo 2.3 para generar pages.jsonl"
        )

    docs = load_pages_jsonl(PAGES_PATH)
    if not docs:
        raise RuntimeError("No hay documentos en pages.jsonl")

    print(f"📄 Documentos cargados: {len(docs)}")

    corpus = build_corpus(docs)
    print(f"🧠 Corpus listo: {len(corpus)} textos")

    vectorizer, X = train_tfidf(corpus)

    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"✅ Vectorizer entrenado | vocab_size={vocab_size:,} | shape={X.shape}")

    save_index(INDEX_PATH, vectorizer, X, docs)
    print(f"💾 Índice guardado en: {INDEX_PATH}")


if __name__ == "__main__":
    main()