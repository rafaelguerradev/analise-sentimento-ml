"""
Interface web para o modelo de análise de sentimento.

Uso:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from modelagem import limpar_texto  # noqa: E402

MODEL_PATH = BASE_DIR / "models" / "sentimento_pipeline.joblib"
DATA_PATH = BASE_DIR / "data" / "olist_order_reviews_dataset.csv"

CORES = {"negativo": "#c0392b", "positivo": "#27ae60"}


@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def carregar_dataset():
    return pd.read_csv(DATA_PATH)


def main() -> None:
    st.set_page_config(page_title="Análise de Sentimento", page_icon="💬")
    st.title("💬 Análise de Sentimento")
    st.caption("Classificação de texto em negativo ou positivo (TF-IDF + Regressão Logística)")

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
        predicao = pipeline.predict([texto_limpo])[0]
        probabilidades = pipeline.predict_proba([texto_limpo])[0]
        classes = pipeline.classes_

        cor = CORES.get(predicao, "#333333")
        st.markdown(
            f"### Resultado: <span style='color:{cor}'>{predicao.upper()}</span>",
            unsafe_allow_html=True,
        )

        df_probs = pd.DataFrame({"classe": classes, "probabilidade": probabilidades})
        df_probs = df_probs.set_index("classe")
        st.bar_chart(df_probs)

    with st.sidebar:
        st.subheader("Sobre o dataset de treino")
        try:
            df = carregar_dataset()
            st.write(f"{len(df)} frases anotadas")
            st.bar_chart(df["sentimento"].value_counts())
        except FileNotFoundError:
            st.info("Dataset não encontrado em `data/dataset.csv`.")


if __name__ == "__main__":
    main()