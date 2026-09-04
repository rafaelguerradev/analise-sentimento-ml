import pandas as pd

# Carrega os dados
df = pd.read_csv("olist_order_reviews_dataset.csv")

# Filtra apenas notas 3 que não estão vazias
neutras = df[(df["review_score"] == 3) & (df["review_comment_message"].notna())]

print(f"Total de avaliações nota 3 com texto: {len(neutras)}\n")
print("Exemplos:\n")

# Exibe as 15 primeiras
for texto in neutras["review_comment_message"].head(15):
    print(f"- {texto}\n")