# Análise de Sentimento — TF-IDF + Regressão Logística

Classificação de texto em português utilizando um pipeline de NLP clássico (TF-IDF + Regressão Logística) treinado com dados reais de e-commerce. A classificação baseia-se em duas categorias principais (**negativo** e **positivo**), mas a interface web aplica uma regra de negócio que classifica o texto como **neutro** caso o modelo apresente incerteza.

## Estrutura do repositório

```
analise-sentimento-ml/
├── data/
│   └── olist_order_reviews_dataset.csv  # Base da Olist com notas de 1 a 5
├── models/
│   └── sentimento_pipeline.joblib       # Modelo treinado (gerado por src/modelagem.py)
├── src/
│   └── modelagem.py                     # Pré-processamento, balanceamento, treino e exportação
├── app/
│   └── app.py                           # Interface web (Streamlit) para testar o modelo
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

1. **Dataset e Balanceamento:** Utiliza o *Brazilian E-Commerce Public Dataset* (Olist). Avaliações sem texto escrito e avaliações de nota 3 são removidas. As notas restantes são agrupadas (1-2 = negativo, 4-5 = positivo) e as classes são perfeitamente balanceadas via técnica de *undersampling*.
2. **Pré-processamento:** Normalização de texto (minúsculas, remoção de URLs e números), preservando pontuações como `!` e `?` por serem fortes indicadores de sentimento.
3. **Vetorização:** TF-IDF com n-gramas (1,2), `sublinear_tf=True`.
4. **Modelo:** Regressão Logística treinada para classificação binária.
5. **Avaliação:** Baseline de comparação (`DummyClassifier`) e avaliação direta em um conjunto de teste (20%) contendo apenas dados nunca vistos pelo modelo.
6. **Interface e Regra do Neutro:** App Streamlit que carrega o modelo salvo e analisa textos digitados em tempo real. Implementa uma **regra de confiança**: se a maior probabilidade predita pelo modelo (seja para positivo ou negativo) for inferior a 60% (margem de dúvida entre 40% e 60%), a interface classifica a resposta automaticamente como "Neutro".

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

- **Por que a nota 3 foi removida do treino?** Textos atrelados à nota 3 costumam ser muito inconsistentes e prejudicam a capacidade do modelo de traçar uma fronteira clara de decisão. Treiná-lo apenas nos extremos (binário) e inferir o "neutro" dinamicamente pelo grau de confiança do modelo (`predict_proba`) gerou resultados consideravelmente superiores e mais estáveis.