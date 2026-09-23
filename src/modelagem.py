"""
Treinamento do pipeline de análise de sentimento (TF-IDF + Regressão Logística).

Dataset: 1.500 frases anotadas manualmente (negativo/neutro/positivo),
já balanceadas entre as classes.
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "olist_order_reviews_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentimento_pipeline.joblib"


# ---------------------------------------------------------------------------
# Passo 1: Carregar dados
# ---------------------------------------------------------------------------
def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    
    # 1. Juntar título e mensagem para enriquecer o contexto
    df['texto'] = df['review_comment_title'].fillna('') + " " + df['review_comment_message'].fillna('')
    df['texto'] = df['texto'].str.strip()
    
    # 2. Remover linhas que não possuem texto
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
    
    # Manter apenas as colunas necessárias
    df = df[['texto', 'sentimento']].reset_index(drop=True)
    
    print(f"Linhas carregadas com sucesso (sem notas 3): {len(df)}")
    return df


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
    print("\nDistribuição das classes (já balanceadas):")
    print(df["sentimento"].value_counts())

    df["texto_limpo"] = df["texto"].apply(limpar_texto)

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

    # Validação cruzada (5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nRodando validação cruzada no treino...")
    resultados_cv = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"]
    )
    print(f"Acurácia média (CV): {resultados_cv['test_accuracy'].mean():.4f}")
    print(f"F1-macro médio (CV): {resultados_cv['test_f1_macro'].mean():.4f}")

    # Treino final e avaliação
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nMétricas de Avaliação no Conjunto de Teste (dados inéditos):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred, labels=["negativo", "neutro", "positivo"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()