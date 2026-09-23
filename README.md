# Análise de Sentimento — TF-IDF + Regressão Logística

Classificação de texto em português em três categorias — **negativo**,
**neutro** e **positivo** — usando um pipeline de NLP clássico (TF-IDF +
Regressão Logística), com interface web para testar o modelo interativamente.

## Estrutura do repositório

```
analise-sentimento-ml/
├── data/
│   └── dataset.csv           # 1.500 frases anotadas (negativo/neutro/positivo)
├── models/
│   └── sentimento_pipeline.joblib   # Modelo treinado (gerado por src/modelagem.py)
├── src/
│   └── modelagem.py           # Pré-processamento, treino, validação e exportação
├── app/
│   └── app.py                 # Interface web (Streamlit) para testar o modelo
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

1. **Dataset:** 1.500 frases anotadas manualmente, já balanceadas entre as
   três classes.
2. **Pré-processamento:** normalização de texto (minúsculas, remoção de URLs
   e números), preservando `!` e `?` como sinal de sentimento.
3. **Vetorização:** TF-IDF com n-gramas (1,2), `sublinear_tf=True`.
4. **Modelo:** Regressão Logística.
5. **Avaliação:** baseline de comparação (`DummyClassifier`), validação
   cruzada estratificada (5 folds, acurácia e F1-macro) e avaliação final em
   conjunto de teste (20%) nunca visto pelo modelo.
6. **Interface:** app Streamlit que carrega o modelo salvo e classifica texto
   digitado pelo usuário em tempo real, mostrando a probabilidade por classe.

## Como rodar

```bash
pip install -r requirements.txt

# Treinar o modelo (gera models/sentimento_pipeline.joblib)
python src/modelagem.py

# Rodar a interface web
streamlit run app/app.py
```

## Tecnologias

- **Linguagem:** Python
- **NLP / ML:** Scikit-learn (TF-IDF, Regressão Logística)
- **Interface:** Streamlit
- **Manipulação de dados:** Pandas

## Notas

- O projeto inicialmente usava o *Brazilian E-Commerce Public Dataset*
  (Olist), mas foi migrado para um dataset próprio de 1.500 frases anotadas
  manualmente, permitindo maior controle sobre o balanceamento das classes.