import pandas as pd
import numpy as np          # obliczenia numeryczne
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix    # macierz pomylek

df = pd.read_csv('diabetes.csv')
print(f'Ile danych: {df.shape}')
print(df.describe().T.to_string())
print(df.isna().sum())    # ile pustych pol

# wszędzie, gdzie są zera lub brak wartości - przypiszmy średnią (bez zer)
for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age']:
    df[col].replace(0, np.NaN, inplace=True)   # usuwamy zera
    mean_ = df[col].mean()   # liczymy średnią
    df[col].replace(np.NaN, mean_, inplace=True)   # wpisujemy średnią

print(df.isna().sum())    # ile pustych pol

df.to_csv('cukrzyca.csv', index=False)     # zapisanie wyczyszczonego pliku do csv

X = df.iloc[:, :-1]      # wszystko, bez ostatniej kolumny
y = df.outcome           # ostatnia kolumna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict((X_test)))))

print('Ile chorych, a ile zdrowych?')
print(df.outcome.value_counts())

print('zmiana danych')
df1 = df.query('outcome==0').sample(n=500)
df2 = df.query('outcome==1').sample(n=500)
df3 = pd.concat([df1, df2])   # 1000 pacjentów, 500 zdrowych i 500 chorych

X = df3.iloc[:, :-1]
y = df3.outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict((X_test)))))