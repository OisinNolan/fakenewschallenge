import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

S = np.load('model/basic_nli/S.npy')
y = np.load('model/basic_nli/y.npy')

D = np.hstack((S, y.reshape(-1, 1)))

df = pd.DataFrame(data=D, columns=['contradiction', 'entailment', 'neutral', 'label'])
sns.pairplot(df, hue='label')
plt.show()