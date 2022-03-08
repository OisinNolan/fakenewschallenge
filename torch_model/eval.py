from torch.utils.data import DataLoader, Subset
from dataset import FakeNewsDataset, STANCE_MAP
from torchmodel import RelatedNet
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

N_subset = 1000

train_data = FakeNewsDataset('train_stances.csv', 'train_bodies.csv')
train_data_subset = Subset(train_data, list(range(N_subset)))
train_dataloader = DataLoader(train_data_subset)

model = RelatedNet()

data = np.zeros((N_subset,2))
correct = 0

for ((head, body), stance), idx in (zip(train_dataloader, range(N_subset))):
    y_true = [1,0] if stance < 3 else [0,1]
    y_pred = model(head[0], body[0])
    
    if (y_true == y_pred):
        correct += 1

    print(f"{correct}/{(idx+1)} = {correct / (idx+1)*100}%") # Live output

#print(correct / N_subset)