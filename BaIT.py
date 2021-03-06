from dataset import FakeNewsEncodedDataset, STANCE_MAP, RELATED_STANCE_MAP
from model import AgreemNet, RelatedNet, TopKNet
import torch
import torch.nn.functional as F
import torchinfo
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIM_DIM = 384
NLI_DIM = 768
CLASSES = 4
parser = ArgumentParser()
parser.add_argument("-a", "--AgreemNet", type=str, default="AgreemNet",)
parser.add_argument("-t", "--TopKNet", type=str, default="TopKNet",)
parser.add_argument("-r", "--RelatedNet", type=str, default="RelatedNet",)
model_names = parser.parse_args()

def run_bait(BAIT="BAIT_AGREEMNET", show_model_summarys=False): # Should pass all config as params
    '''
    BAIT = BAIT_AGREEMNET or BAIT_TOPK
    '''
    # ---------- AGREEMNET ---------- #
    trained_agreemnet = AgreemNet(
        hdim_1=60,
        hdim_2=20,
        num_heads=11,
    )

    trained_agreemnet.to(DEVICE)
    trained_agreemnet.load_state_dict(
        torch.load(f"trained_models/{model_names.AgreemNet}.pth", map_location=DEVICE)
    )
    trained_agreemnet.eval()
    if (show_model_summarys):
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
        torch.load(f"trained_models/{model_names.TopKNet}.pth", map_location=DEVICE)
    )
    trained_topknet.eval()
    if (show_model_summarys):
        torchinfo.summary(trained_topknet)
    # ----------_-------_---------- #

    # ---------- RELATEDNET ---------- #
    trained_relatednet = RelatedNet(
        hdim_1=600,
        hdim_2=600,
        kk=4,
    )

    trained_relatednet.to(DEVICE)
    trained_relatednet.load_state_dict(
        torch.load(f"trained_models/{model_names.RelatedNet}.pth", map_location=DEVICE)
    )
    trained_relatednet.eval()
    if (show_model_summarys):
        torchinfo.summary(trained_relatednet)
    # ----------_----------_---------- #

    # Load in the *test* dataset
    dataset = FakeNewsEncodedDataset(
        stances_files=["data/test_stances.csv.stance.dat"],
        bodies_file="data/test_bodies.csv.body.dat",
    )
    dataloader = DataLoader(dataset, batch_size=1) # Can only do Batch Size of 1

    with tqdm(dataloader) as pbar:
        stance_preds = np.zeros(len(pbar))
        stance_trues = np.zeros(len(pbar))

        for ii, (embeddings, stance) in enumerate(pbar):
            pred_stance = -1
            
            embeddings = [_embedding.to(DEVICE) for _embedding in embeddings]
            stance = stance.to(DEVICE)

            relatednet_logits = trained_relatednet(*embeddings)
            relatednet_pred = torch.max(relatednet_logits,dim=1).indices

            if (relatednet_pred == RELATED_STANCE_MAP["unrelated"]):
                pred_stance = STANCE_MAP["unrelated"]
            else:
                if (BAIT == "BAIT_TOPK"):
                    # TOPKNET
                    topknet_logits = trained_topknet(*embeddings)
                    topknet_pred = torch.max(topknet_logits,dim=1).indices
                    pred_stance = topknet_pred
                elif (BAIT == "BAIT_AGREEMNET"):
                    # AGREEMNET
                    agreemnet_logits = trained_agreemnet(*embeddings)
                    agreemnet_pred = torch.max(agreemnet_logits,dim=1).indices
                    pred_stance = agreemnet_pred

            stance_preds[ii] = pred_stance
            stance_trues[ii] = stance
    
    return stance_preds, stance_trues

def FNC_score_unnorm(stance_preds, stance_trues):
    score = 0
    for pred, stance in zip(stance_preds, stance_trues):
        if pred == stance:
            score += 0.25
            if stance != STANCE_MAP["unrelated"]:
                score += 0.50
        if stance < STANCE_MAP["unrelated"] and pred < STANCE_MAP["unrelated"]:
            score += 0.25
    return score

def raw_score(stance_preds, stance_trues):
    class_correct = np.zeros(4)
    class_total = np.zeros(4)
    for pred, stance in zip(stance_preds, stance_trues):
        class_total[int(stance)] += 1
        if (pred == stance):
            class_correct[int(stance)] += 1
    return class_correct, class_total

def main():
    # RelatedNet + AgreemNet
    stance_preds, stance_trues = run_bait("BAIT_AGREEMNET")
    print("BAIT_AGREEMNET - Confusion Matrix:")
    print(confusion_matrix(stance_trues, stance_preds))
    
    print(f"Unnorm FNC score:\t\t{FNC_score_unnorm(stance_trues, stance_preds)}")
    print("-"*50)
    
    # RelatedNet + AgreemNet
    stance_preds, stance_trues = run_bait("BAIT_TOPK")
    print("BAIT_TOPK - Confusion Matrix:")
    print(confusion_matrix(stance_trues, stance_preds))

    print(f"Unnorm FNC score:\t\t{FNC_score_unnorm(stance_trues, stance_preds)}")
    print("-"*50)


if __name__ == "__main__":
    main()
