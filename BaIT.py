from dataset import FakeNewsEncodedDataset, STANCE_MAP, RELATED_STANCE_MAP
from model import AgreemNet, RelatedNet, TopKNet
import torch
import torch.nn.functional as F
import torchinfo
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIM_DIM = 384
NLI_DIM = 768
CLASSES = 4

# ---------- AGREEMNET ---------- #
trained_agreemnet = AgreemNet(
    hdim_1=60,
    hdim_2=20,
    num_heads=11,
)

trained_agreemnet.to(DEVICE)
trained_agreemnet.load_state_dict(
    torch.load("trained_models/AgreemNet.pth", map_location=DEVICE)
)
trained_agreemnet.eval()
torchinfo.summary(trained_agreemnet)
# ----------_----------_---------- #

# ---------- TOPKNET ---------- #
trained_topknet = TopKNet(
    hdim_1=60,
    hdim_2=60,
    kk=3,
)

trained_topknet.to(DEVICE)
trained_topknet.load_state_dict(
    torch.load("trained_models/TopKNet.pth", map_location=DEVICE)
)
trained_topknet.eval()
torchinfo.summary(trained_topknet)
# ----------_-------_---------- #

# ---------- RELATEDNET ---------- #
trained_relatednet = RelatedNet(
    hdim_1=60,
    hdim_2=40,
    kk=6,
)

trained_relatednet.to(DEVICE)
trained_relatednet.load_state_dict(
    torch.load("trained_models/RelatedNet.pth", map_location=DEVICE)
)
trained_relatednet.eval()
torchinfo.summary(trained_topknet)
# ----------_----------_---------- #

# Load in the *test* dataset
dataset = FakeNewsEncodedDataset(
    stances_files=["data/test_stances.csv.stance.dat"],
    bodies_file="data/test_bodies.csv.body.dat",
)
dataloader = DataLoader(dataset, batch_size=1) # Can only do Batch Size of 1

# Scores
class_correct_related = 0
class_correct = np.zeros(CLASSES)
class_total = np.zeros(CLASSES)

# MODEL = "BAIT_TOPK"
MODEL = "BAIT_AGREEMNET"

with tqdm(dataloader) as pbar:
    for (embeddings, stance) in pbar:
        pred_stance = -1
        
        embeddings = [_embedding.to(DEVICE) for _embedding in embeddings]
        stance = stance.to(DEVICE)

        relatednet_logits = trained_relatednet(*embeddings)
        relatednet_pred = torch.max(relatednet_logits,dim=1).indices

        if (relatednet_pred == RELATED_STANCE_MAP["unrelated"]):
            pred_stance = STANCE_MAP["unrelated"]
        else:
            if (MODEL == "BAIT_TOPK"):
                # TOPKNET
                topknet_logits = trained_topknet(*embeddings)
                topknet_pred = torch.max(topknet_logits,dim=1).indices
                pred_stance = topknet_pred
            elif (MODEL == "BAIT_AGREEMNET"):
                # AGREEMNET
                agreemnet_logits = trained_agreemnet(*embeddings)
                agreemnet_pred = torch.max(agreemnet_logits,dim=1).indices
                pred_stance = agreemnet_pred

        class_correct_related += (
            0 if pred_stance == STANCE_MAP["unrelated"] and stance != STANCE_MAP["unrelated"]
            else 1
        )

        class_correct[stance] += 1 if (stance == pred_stance) else 0
        class_total[stance] += 1

        class_avgs = [class_correct[ii] / class_total[ii] for ii in range(CLASSES)]
        overall_score = sum(class_correct) / sum(class_total) * 100

        pbar.set_postfix(class_averages=f"{[np.round(ca * 100,1) for ca in class_avgs]}", overall_score=f"{overall_score:>.1f}%")

print("Class correct:\t\t", class_correct)
print("Class total:\t\t", class_total)
print(f"Class avgs:\t\t {[np.round(ca * 100,1) for ca in class_avgs]}")
print(f"Overall score:\t\t {overall_score:>.1f}%")
print(f'Related task accuracy:\t\t {(class_correct_related / 25418)}')