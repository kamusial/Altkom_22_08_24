import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('iris.csv')
species = {
    'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2
}
df['class_value'] = df['class'].map(species)
print(df)

print('Klasyfikator Decission Tree')
X = df.iloc[:, 0:4]   # 4 pierwsze kolumny
y = df.class_value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# criterion{“gini”, “entropy”, “log_loss”
model = DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_split=2)     #(max_depth=8, min_samples_split=4)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
print(pd.DataFrame(model.feature_importances_, X.columns))

# # granice decyzyjne
# from mlxtend.plotting import plot_decision_regions
# plot_decision_regions(X.values, y.values, model)
# plt.show()