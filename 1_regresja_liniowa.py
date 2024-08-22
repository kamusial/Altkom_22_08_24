import pandas as pd                 # biblioteka do czytania danych
import matplotlib.pyplot as plt     # biblioteka do wykresów
import seaborn as sns               # biblioteka do wykresów
from sklearn.linear_model import LinearRegression    # regresja liniowa

df = pd.read_csv('weight-height.csv', delimiter=';')
print(df)
df.Height *= 2.54     # df.Height = df.Height * 2.54
df.Weight /= 2.2
print('obrobione dane')
print(df.head(10))    # wyswietl 10 pierwszych wierszy



