# See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

from re import I
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

STANCE_MAP = {
    'agree':0,
    'disagree':1,
    'discuss':2,
    'unrelated':3,
}

class FakeNewsDataset(Dataset):
    def __init__(self, stances_file=None, bodies_file=None, related_only=False):
        self.stances = pd.read_csv(stances_file) if stances_file else None
        self.bodies = pd.read_csv(bodies_file) if bodies_file else None
        if related_only:
            self.stances.drop(self.stances[self.stances['Stance'] == 'unrelated'].index, inplace=True)

    def set_df(self, stances_df, bodies_df):
        self.stances = stances_df
        self.bodies = bodies_df

    def __len__(self):
        return len(self.stances)
    
    def __getitem__(self, idx):
        headline, body_id, stance = self.stances.iloc[idx]
        select = self.bodies['Body ID'] == body_id
        body = self.bodies[select]['articleBody'].values[0]
        return (headline, body), STANCE_MAP[stance]


# train_data = FakeNewsDataset('../data/combined_stances_train.csv', '../data/combined_bodies_train.csv', related_only=True)
# train_dataloader = DataLoader(train_data, batch_size=64)

# for batch, ((head, body), y) in enumerate(train_dataloader):
    
