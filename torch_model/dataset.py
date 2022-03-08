# See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import torch
from torch.utils.data import Dataset
import pandas as pd

DATA_LOCATION = "../data"

STANCE_MAP = {
    'agree':0,
    'disagree':1,
    'discuss':2,
    'unrelated':3,
}

class FakeNewsDataset(Dataset):
    def __init__(self, stances_file=None, bodies_file=None):
        self.stances = pd.read_csv(f"{DATA_LOCATION}/{stances_file}") if stances_file else None
        self.bodies = pd.read_csv(f"{DATA_LOCATION}/{bodies_file}") if bodies_file else None

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

if __name__ == "__main__":
    training_data = FakeNewsDataset(f'{DATA_LOCATION}/train_stances.csv', f'{DATA_LOCATION}/train_bodies.csv')
    training_data.__getitem__(4)