"""
Interface web para o modelo de análise de sentimento.
Uso: streamlit run app/app.py
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Importamos não só o limpar_texto, mas também a função de carregar para o gráfico ficar correto
from modelagem import limpar_texto, carregar_dados 

MODEL_PATH = BASE_DIR / "models" / "sentimento_pipeline.joblib"

# Atualizado para ter a cor neutra caso o modelo fique em dúvida
CORES = {"negativo": "#c0392b", "neutro": "#7f8c8d", "positivo": "#27ae60"}


@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def buscar_dataset_processado():
    # Usamos a mesma função do modelagem.py para evitar o KeyError
    return carregar_dados()


def main() -> None:
    st.set_page_config(page_title="Análise de Sentimento", page_icon="💬")
    st.title("💬 Análise de Sentimento")
    st.caption("Classificação Olist (Se a confiança for menor que 60%, considera Neutro)")

    try:
        pipeline = carregar_modelo()
    except FileNotFoundError:
        st.error(
            f"Modelo não encontrado em `{MODEL_PATH}`. "
            "Rode `python src/modelagem.py` primeiro para treinar e salvar o modelo."
        )
        return

    texto = st.text_area("Digite um texto para analisar:", height=120,
                         placeholder="Ex: o atendimento foi ótimo, super recomendo!")

    if st.button("Analisar", type="primary") and texto.strip():
        texto_limpo = limpar_texto(texto)
        
        # Extrai as probabilidades da predição
        probabilidades = pipeline.predict_proba([texto_limpo])[0]
        classes = pipeline.classes_ # ["negativo", "positivo"]
        
        # Encontra qual foi a maior probabilidade (ex: 0.55)
        prob_max = max(probabilidades)
        
        # REGRA DOS 40%-60%:
        # Se a maior probabilidade não atingir 60% (0.60), significa que 
        # a distribuição está muito próxima (ex: 55% positivo e 45% negativo).
        if prob_max < 0.60:
            predicao_final = "neutro"
        else:
            # Caso contrário, pega a classe com maior probabilidade
            idx_maior = probabilidades.argmax()
            predicao_final = classes[idx_maior]

        cor = CORES.get(predicao_final, "#333333")
        st.markdown(
            f"### Resultado: {predicao_final.upper()}",
            unsafe_allow_html=True,
        )
        
        st.write(f"**Confiança do modelo:** {prob_max * 100:.1f}%")

        # Plota o gráfico de barras com as porcentagens
        df_probs = pd.DataFrame({"classe": classes, "probabilidade": probabilidades})
        df_probs = df_probs.set_index("classe")
        st.bar_chart(df_probs)

    with st.sidebar:
        st.subheader("Sobre o dataset de treino")
        try:
            # Agora não dará mais KeyError, pois usa a base já tratada
            df = buscar_dataset_processado()
            st.write(f"{len(df)} frases usadas no treinamento")
            st.bar_chart(df["sentimento"].value_counts())
        except Exception as e:
            st.info(f"Não foi possível carregar as estatísticas: {e}")


if __name__ == "__main__":
    main()