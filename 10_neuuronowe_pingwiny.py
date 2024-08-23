import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
penguins = sns.load_dataset('penguins')

print(penguins.to_string())
sns.pairplot(penguins, hue='species')
plt.show()

penguins_filtered = penguins.drop(columns=['island', 'sex']).dropna()
penguins_features = penguins_filtered.drop(columns=['species'])    # X'y
penguins_target = pd.get_dummies(penguins_filtered['species'])     # y'ki
print(penguins_features)