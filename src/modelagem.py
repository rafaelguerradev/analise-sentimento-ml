"""
Treinamento do pipeline de análise de sentimento (TF-IDF + Regressão Logística).
Base de dados Olist - Com balanceamento de classes e sem validação cruzada.
"""

import re
import string
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "olist_order_reviews_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentimento_pipeline.joblib"


# ---------------------------------------------------------------------------
# Passo 1: Carregar e Balancear os dados
# ---------------------------------------------------------------------------
def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    
    # 1. Juntar título e mensagem para enriquecer o contexto
    df['texto'] = df['review_comment_title'].fillna('') + " " + df['review_comment_message'].fillna('')
    df['texto'] = df['texto'].str.strip()
    
    # 2. Remover linhas que não possuem texto (Isso responde sua dúvida: sim, já são tiradas!)
    df = df[df['texto'] != '']
    
    # 3. Remover as avaliações neutras (nota 3)
    df = df[df['review_score'] != 3]
    
    # 4. Mapear o review_score para as duas classes
    def mapear_sentimento(nota):
        if nota <= 2:
            return "negativo"
        else:
            return "positivo"
            
    df['sentimento'] = df['review_score'].apply(mapear_sentimento)
    df = df[['texto', 'sentimento']].reset_index(drop=True)
    
    print(f"Total de linhas com texto (antes do balanceamento): {len(df)}")
    
    # 5. BALANCEAMENTO DAS CLASSES (Undersampling)
    # Pega o tamanho da menor classe para igualar
    min_class_size = df['sentimento'].value_counts().min()
    
    df_neg = df[df['sentimento'] == 'negativo'].sample(min_class_size, random_state=42)
    df_pos = df[df['sentimento'] == 'positivo'].sample(min_class_size, random_state=42)
    
    # Junta as duas e embaralha (frac=1)
    df_balanceado = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total de linhas APÓS balanceamento perfeito: {len(df_balanceado)}")
    return df_balanceado


# ---------------------------------------------------------------------------
# Passo 2: Limpeza de texto (preserva ! e ?)
# ---------------------------------------------------------------------------
def limpar_texto(texto: str) -> str:
    texto = str(texto).lower().strip()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    pontuacao_manter = string.punctuation.replace("!", "").replace("?", "")
    texto = texto.translate(str.maketrans("", "", pontuacao_manter))
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def construir_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def main() -> None:
    df = carregar_dados()
    print("\nDistribuição das classes (balanceadas):")
    print(df["sentimento"].value_counts())

    df["texto_limpo"] = df["texto"].apply(limpar_texto)

    # Divisão Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        df["texto_limpo"], df["sentimento"],
        test_size=0.2, random_state=42, stratify=df["sentimento"],
    )
    print(f"\nConjunto de Treino: {len(X_train)} amostras")
    print(f"Conjunto de Teste: {len(X_test)} amostras")

    pipeline = construir_pipeline()

    # Baseline (piso de comparação)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    acc_baseline = accuracy_score(y_test, baseline.predict(X_test))
    print(f"\nAcurácia baseline (chute mais frequente): {acc_baseline:.4f}")

    # Treino final (Sem validação cruzada)
    print("\nTreinando o modelo principal...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nMétricas de Avaliação no Conjunto de Teste (dados inéditos):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Matriz de Confusão:")
    # Importante: removido o "neutro" daqui para não dar erro
    print(confusion_matrix(y_test, y_pred, labels=["negativo", "positivo"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()