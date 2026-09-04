import re # Biblioteca com funcções úteis para limpeza do texto
import string # Biblioteca com funcções úteis para limpeza do texto

from pathlib import Path # Biblioteca para pegar o caminho de arquivos
import joblib # Biblioteca para salvar e ler arquivos
import pandas as pd # Biblioteca para abrir/manipular dados/tabelas

# A biblioteca Sklearn é uma biblioteca específica para ML com diversas funções úteis
from sklearn.feature_extraction.text import TfidfVectorizer # Transforma textos em números (vetorização)
from sklearn.linear_model import LogisticRegression # Tipo de modelo de IA para classificação
from sklearn.metrics import classification_report, accuracy_score # Avaliam o nível de acerto do modelo
from sklearn.model_selection import train_test_split # Separa os dados dois grupos, treino e teste
from sklearn.pipeline import Pipeline # Garante que, além da exitência de etapas, sejam executadas na ordem correta 

# Caminhos baseados nos nomes originais do seu projeto
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "modelo_sentimento.pkl" # Mantido o nome original e na pasta raiz

# Função com o objetivo de fazer um passo FUNDAMENTAL do PRÉ-PROCESSAMENTO dos dados, a limpeza.
def limpar_texto(texto: str) -> str:
    texto = texto.lower().strip() # Deixa tudo minúsculo 
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto) # Remove links da internet (importante para ESSE caso)
    texto = re.sub(r"\d+", " ", texto) # Remove números (importante para ESSE caso)
    texto = texto.translate(str.maketrans("", "", string.punctuation)) # Remove pontuação 
    texto = re.sub(r"\s+", " ", texto) # Troca espaços seguidos por apenas 1 e depois remove espaços do começo/final
    return texto.strip() # Retorna o texto limpo

# Função principal onde tudo acontece. Aqui começa o PROCESSAMENTO
def main() -> None:
    df = pd.read_csv(DATA_PATH) # Utilização da função read_csv do pandas para ler os textos e rótulos
    df["texto_limpo"] = df["texto"].astype(str).apply(limpar_texto) # Cria uma nova coluna com o texto tratado

    # Separação focada na sua coluna "sentimento"
    X_train, X_test, y_train, y_test = train_test_split( 
        df["texto_limpo"],
        df["sentimento"], # <-- Adaptado para o seu nome de coluna
        test_size=0.25, # 25% dos dados serão para teste
        random_state=42, 
        stratify=df["sentimento"], # <-- Adaptado para o seu nome de coluna
    )

    # Montagem do pipeline de NLP avançado
    pipeline = Pipeline(
        steps=[
            (
                "tfidf", 
                TfidfVectorizer( 
                    ngram_range=(1, 2), # Trabalha com palavras simples e compostas (ex: "não gostei")
                    min_df=1, 
                    max_df=0.95, # Ignora palavras que apareçam em mais de 95% dos textos
                    sublinear_tf=True, # Penaliza repetições excessivas de uma mesma palavra no texto
                ),
            ),
            (
                "clf", 
                LogisticRegression( # Você optou por voltar à Regressão Logística (excelente para probabilidades)
                    max_iter=1000, 
                    class_weight="balanced", # Evita o viés com classes desbalanceadas
                    random_state=42,
                ),
            ),
        ]
    )

    print("🧠 Treinando o modelo...")
    pipeline.fit(X_train, y_train) # Treinamento do modelo
    
    print("\n🎯 Realizando inferência nos dados de teste...")
    y_pred = pipeline.predict(X_test) # IA adivinha o sentimento

    # Medição de precisão e geração do relatório
    print("Acurácia:", round(accuracy_score(y_test, y_pred), 4)) 
    print("\nRelatório:\n")
    print(classification_report(y_test, y_pred)) 

    # Salvamento do modelo mantendo o seu padrão de arquitetura
    joblib.dump(pipeline, MODEL_PATH) 
    print(f"\n💾 Modelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    main()