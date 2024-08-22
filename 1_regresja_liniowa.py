import pandas as pd                 # biblioteka do czytania danych
import matplotlib.pyplot as plt     # biblioteka do wykresów
import seaborn as sns               # biblioteka do wykresów
from sklearn.linear_model import LinearRegression    # regresja liniowa

# czytanie danych
df = pd.read_csv('weight-height.csv', delimiter=';')
print(df)

# obrobka danych
df.Height *= 2.54     # df.Height = df.Height * 2.54
df.Weight /= 2.2
print('obrobione dane')
print(df.head(10))    # wyswietl 10 pierwszych wierszy

# wykres przy pomocy plt
plt.hist(df.Weight)    # przygotowanie wykresu
plt.show()             # wyswietlenie wykresu

# wykres przy pomocy sns
sns.histplot(df.Weight)    # przygotowanie wykresu
plt.show()             # wyswietlenie wykresu

# problem - mezczyzni i kobiety razem
# wykres przy pomocy sns
sns.histplot(df.query('Gender=="Female"').Weight)    # przygotowanie wykresu, same kobiety
# plt.show()
sns.histplot(df.query('Gender=="Male"').Weight)    # przygotowanie wykresu, sami panowie
plt.show()

# zamiana gender na dane numeryczne
df = pd.get_dummies(df)   # nadpisz dane
del (df['Gender_Male'])   # usuwanie kolumny
df = df.rename(columns={'Gender_Female': 'Gender'})   # zmiana nazwy kolumny
# False - facet,  True - kobieta
print(df)

# przygotowanie modelu
model = LinearRegression()   # wybor modelu
model.fit(df[['Height', 'Gender']],   df['Weight'])   # dane do modelu
print(f'Wspolczynnik kierunkowy: {model.coef_}\nWyraz wolny: {model.intercept_}')


