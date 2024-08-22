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

