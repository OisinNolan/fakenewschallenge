from torch.utils.data import DataLoader
from dataset import FakeNewsDataset
from torchmodel import RelatedNet
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

train_data = FakeNewsDataset('combined_stances_train.csv', 'combined_bodies_train.csv')
train_dataloader = DataLoader(train_data)

model = RelatedNet()

related = []
pred = []

idx = 1
for (head, body), y in train_dataloader:
    if idx > 250:
        continue
    idx = idx + 1
    y_pred = model(head[0], body[0]) # for some reason head and body are tuples instead of strings 🤔
    related.append(y_pred)
    pred.append(1 if y < 3 else 0)

related = np.array(related).reshape(-1, 1)
pred = np.array(pred).reshape(-1, 1)

data = np.hstack((related, pred))

df = pd.DataFrame(data, columns=['similarity', 'related'])
sns.displot(df, x='similarity', hue='related')
plt.show()

# print(int(y < 3 and y_pred[0] == 1)) # 1 if related and predicted related, 0 otherwise