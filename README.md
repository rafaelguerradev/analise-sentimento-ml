# Analisador de Sentimentos com Machine Learning e NLP

Pipeline completo de Processamento de Linguagem Natural (NLP) e Machine Learning para classificar avaliações de texto em três categorias: **Positivo**, **Neutro** e **Negativo**. O projeto engloba desde a curadoria e limpeza de dados até o treinamento estatístico, validação cruzada e uma interface web interativa em tempo real.

---

## 🛠️ Tecnologias Utilizadas

* **Python** (Linguagem principal)
* **Pandas** (Manipulação e limpeza estruturada de dados)
* **Scikit-Learn** (Vetorização TF-IDF, Regressão Logística, Validação Cruzada e Métricas)
* **Joblib** (Serialização e exportação do pipeline treinado)
* **Streamlit** (Interface web interativa)

---

## 📂 Estrutura do Projeto

```text
analise-sentimento-ml/
│
├── models/
│   └── sentimento_pipeline.joblib  # Modelo treinado congelado
│
├── dataset.csv                     # Base de dados rotulada (texto, sentimento)
├── modelagem.py                    # Script de ETL, treinamento, validação e exportação
├── app.py                          # Interface web local (Streamlit)
├── requirements.txt                # Dependências e bibliotecas do projeto
└── README.md                       # Documentação do projeto

## ⚙️ Como Executar o Projeto Localmente

Siga os passos abaixo para configurar o ambiente e rodar o projeto na sua máquina:

1. Clonar ou abrir a pasta do projeto
Abra o terminal na pasta raiz do repositório.

2. Criar e ativar o ambiente virtual (Recomendado)
Bash
python -m venv venv
# No Windows (PowerShell):
.\venv\Scripts\Activate
# No Linux/macOS:
source venv/bin/activate
3. Instalar as dependências
Bash
pip install -r requirements.txt
4. Treinar o modelo
Execute o script de modelagem para processar o dataset, rodar a validação cruzada (5 folds) e gerar o arquivo binário do modelo na pasta models/:

Bash
python modelagem.py
5. Iniciar a interface web
Com o modelo treinado e salvo, execute o Streamlit para testar as previsões de texto em tempo real:

Bash
streamlit run app.py