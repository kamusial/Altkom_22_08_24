import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('otodom.csv')
print(df)
print(df.describe())    # opis danych
print(df.describe().T.to_string())    # ladny opis

# print(df.iloc[13:33, 2:5])    # wyciecie kawalka danych
# print(df.iloc[:, 1:].corr())    # korelacja bez pierwszej kolumny
sns.heatmap(df.iloc[:, 1:].corr(), annot=True)
plt.show()

sns.histplot(df.price)
plt.show()

q1 = df.describe().T.loc['price', '25%']
q3 = df.describe().T.loc['price', '75%']
print(f'cena na koncu q1 to: {q1}')
print(f'cena na koncu q3 to: {q3}')

df = df[(df.price >= 0) & (df.price <= q3) & df.year <= 2023]
sns.histplot(df.price)
plt.show()

X = df.iloc[:, 2:]      # bez id, bez ceny
y = df.price
# podzial na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)    # nauka na danych trenchermen
# sprawdzenie modelu
print(f'Dokladnosc modelu {model.score(X_test, y_test)}')

print(pd.DataFrame(model.coef_, X.columns))


