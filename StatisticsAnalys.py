import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Dataset = pd.read_csv("Stars.csv")
Dataset.info()
#Dataset['Color'] = pd.factorize(Dataset['Color'])[0]
Dataset[['Color', 'Spectral_Class']] = Dataset[['Color', 'Spectral_Class']].apply(lambda x: pd.factorize(x)[0])
sns.heatmap(Dataset.corr(),  vmin=-1, vmax=+1, annot=True, cmap='coolwarm')
plt.show()
