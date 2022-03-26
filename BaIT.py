from dataset import FakeNewsEncodedDataset
from model import AgreemNet
import torch
import torch.nn.functional as F
import torchinfo
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIM_DIM = 384
NLI_DIM = 768

trained_model = AgreemNet(
    hdim_1=50,
    hdim_2=500,
    dropout=0,
    num_heads=12,
)

trained_model.to(DEVICE)
trained_model.load_state_dict(
    torch.load("trained_models/AgreemNet.pth", map_location=DEVICE)
)
trained_model.eval()

torchinfo.summary(trained_model)

##
dataset = FakeNewsEncodedDataset(
    stances_file="data/test_stances.csv.stance.dat",
    bodies_file="data/test_bodies.csv.body.dat",
    no_unrelated=True,
)

train_dataloader = DataLoader(dataset, batch_size=16)
class_correct = np.zeros(3)
class_total = np.zeros(3)

with tqdm(train_dataloader) as pbar:
    for (embeddings, stance) in pbar:
        embeddings = [_embedding.to(DEVICE) for _embedding in embeddings]
        stance = stance.to(DEVICE)
        logits = trained_model(*embeddings)
        pred_stance = torch.max(logits,dim=1).indices
        for ss, pp in zip(stance, pred_stance):
            class_correct[ss] += 1 if (ss == pp) else 0
            class_total[ss] += 1

        class_avgs = [class_correct[ii] / class_total[ii] for ii in range(3)]
        overall_score = sum(class_correct) / sum(class_total) * 100

        pbar.set_postfix(class_averages=f"{[np.round(ca * 100,1) for ca in class_avgs]}", overall_score=f"{overall_score:>.1f}%")

print("Class correct:", class_correct)
print("Class total:", class_total)
print("Class avgs:", class_avgs)
print(f"Overall score: {overall_score:>.1f}%")