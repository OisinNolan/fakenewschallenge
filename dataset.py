# See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

from torch.utils.data import Dataset
import pandas as pd
import pickle

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

class FakeNewsEncodedDataset(Dataset):
    def __init__(self, stances_file, bodies_file):
        self.stances = []
        self.sim_bodies = {}
        self.nli_bodies = {}
        
        with open(stances_file, "rb") as sf:
            completed_read = False
            while not completed_read:
                try:
                    stance = pickle.load(sf)
                    self.stances.append(stance) # TODO: memory inefficient?
                except EOFError:
                    completed_read = True
        
        with open(bodies_file, "rb") as bf:
            completed_read = False
            while not completed_read:
                try:
                    body_id, sim_embeddings, nli_embeddings = pickle.load(bf)
                    self.sim_bodies[body_id] = sim_embeddings
                    self.nli_bodies[body_id] = nli_embeddings
                except EOFError:
                    completed_read = True

    def __len__(self):
        return len(self.stances) # TODO Is this correct?

    def __getitem__(self, idx):
        body_id, sim_stance_embedding, nli_stance_embedding, stance = self.stances[idx]
        sim_body_embedding = self.sim_bodies[body_id]
        nli_body_embedding = self.nli_bodies[body_id]
        return (
            sim_stance_embedding,
            nli_stance_embedding,
            sim_body_embedding,
            nli_body_embedding,
        ), STANCE_MAP[stance]