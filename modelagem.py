import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("dataset.csv")
print(df.head())

print((df["sentimento"] == "neutro").sum())