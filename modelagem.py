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

BASE_DIR = Path(__file__).resolve().parent
# Atualizado para o nome do seu novo arquivo de frases
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentimento_pipeline.joblib"

# ---------------------------------------------------------------------------
# Passo 1: Carregar dados
# ---------------------------------------------------------------------------
def carregar_dados() -> pd.DataFrame:
    # Lendo o CSV. Espera-se que tenha as colunas 'texto' e 'sentimento'
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["texto", "sentimento"]).reset_index(drop=True)
    print(f"Linhas carregadas com sucesso: {len(df)}")
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

def main() -> None:
    # Carregamento e distribuição
    df = carregar_dados()
    print("\nDistribuição das classes (já balanceadas):")
    print(df["sentimento"].value_counts())

    # Aplicação da limpeza
    df["texto_limpo"] = df["texto"].apply(limpar_texto)

    # Split treino/teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        df["texto_limpo"], df["sentimento"],
        test_size=0.2, random_state=42, stratify=df["sentimento"]
    )
    print(f"\nConjunto de Treino: {len(X_train)} amostras")
    print(f"Conjunto de Teste: {len(X_test)} amostras")

    # Construção do Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
        # Removido o class_weight="balanced" pois o dataset já é simétrico
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # Baseline (Piso de comparação estatística)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    acc_baseline = accuracy_score(y_test, baseline.predict(X_test))
    print(f"\nAcurácia baseline (chute aleatório): {acc_baseline:.4f}")

    # Validação Cruzada (5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nRodando validação cruzada no Treino...")
    resultados_cv = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"]
    )
    print(f"Acurácia média (CV): {resultados_cv['test_accuracy'].mean():.4f}")
    print(f"F1-macro médio (CV): {resultados_cv['test_f1_macro'].mean():.4f}")

    # Treinamento Final e Avaliação
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nMétricas de Avaliação no Conjunto de Teste (Dados Inéditos):")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred, labels=["negativo", "neutro", "positivo"]))

    # Exportação do modelo
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    main()