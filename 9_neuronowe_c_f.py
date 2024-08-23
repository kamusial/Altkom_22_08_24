import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential     # model sekwencyjny
from tensorflow.keras.layers import Dense           # siec gesta

model = Sequential()
model.add(Dense(1, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(4, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mse')

df = pd.read_csv('f-c.csv', sep=';',  usecols=[1, 2])     # omin kolumne 0
print(df)

result = model.fit(df.F, df.C, epochs=3000, verbose=2)
df1 = pd.DataFrame(result.history)
print(df1)
df1.plot()
plt.show()

C_pred = model.predict(df.F)
plt.scatter(df.F, df.C)
plt.plot(df.F, C_pred, c='r')
plt.show()


