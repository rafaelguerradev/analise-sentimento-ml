import re
import string
from pathlib import Path

import joblib
import streamlit as st

# 1. Configurar caminhos
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentimento_pipeline.joblib"

# 2. Replicar a MESMA função de limpeza usada no treinamento
def limpar_texto(texto: str) -> str:
    texto = str(texto).lower().strip()
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    pontuacao_manter = string.punctuation.replace("!", "").replace("?", "")
    texto = texto.translate(str.maketrans("", "", pontuacao_manter))
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

# 3. Carregar o modelo de forma otimizada (em cache)
@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)

try:
    pipeline = carregar_modelo()
except FileNotFoundError:
    st.error("Erro: Modelo não encontrado. Rode o 'modelagem.py' primeiro.")
    st.stop()

# 4. Construção da Interface Web
st.title("Análise de Sentimentos com IA")
st.write("Digite uma avaliação ou comentário abaixo e o modelo tentará prever a emoção.")

# Caixa de texto para o usuário
frase_usuario = st.text_area("Texto para análise:")

# Botão de ação
if st.button("Analisar Sentimento"):
    if frase_usuario.strip() == "":
        st.warning("Por favor, digite alguma frase para analisar.")
    else:
        # A. Limpar o texto
        frase_limpa = limpar_texto(frase_usuario)
        
        # B. Fazer a previsão matemática
        predicao = pipeline.predict([frase_limpa])[0]
        
        # C. Extrair a porcentagem de confiança (probabilidade)
        probabilidades = pipeline.predict_proba([frase_limpa])[0]
        confianca = max(probabilidades) * 100
        
        # D. Exibir o resultado visual
        st.subheader("Resultado:")
        if predicao == "positivo":
            st.success(f"**Positivo** (Confiança: {confianca:.1f}%)")
        elif predicao == "negativo":
            st.error(f"**Negativo** (Confiança: {confianca:.1f}%)")
        else:
            st.info(f"**Neutro** (Confiança: {confianca:.1f}%)")
            
        with st.expander("Ver texto processado pela IA"):
            st.write(f"_{frase_limpa}_")