from modulefinder import Module
from dataset import FakeNewsEncodedDataset, STANCE_MAP
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import AgreemDeep, AgreemFlat, AgreemNet, NUM_CLASSES
from util import pad_tokenize 
import torch
from torch import batch_norm, nn
from tqdm import tqdm
import math
import wandb
from time import time
from copy import deepcopy
import numpy as np
from typing import Dict

EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EVAL_FREQ = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_CUTOFF = 0.7
WANDB_PROJ = "yo-am-i-going-crazy"
STANCE_MAP_INV = dict((v,k) for k, v in STANCE_MAP.items())

wandb.init(project=WANDB_PROJ, entity="mlpbros")

wandb.config.update({
   "learning_rate": LEARNING_RATE,
   "epochs": EPOCHS,
   "batch_size": BATCH_SIZE,
})

def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader], loss_fn, optimizer, num_epochs)-> nn.Module:
    since = time()

    best_model = deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):   
        with tqdm(dataloaders["train"], bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            # TRAIN
            for batch_idx, (embeddings, stance) in enumerate(pbar):
                model.train()
                pbar.set_description(f"Epoch {epoch}/{num_epochs-1} | " + "Training...")
                
                embeddings = [_embedding.to(DEVICE) for _embedding in embeddings]
                stance = stance.to(DEVICE)

                with torch.set_grad_enabled(True):
                    pred = model(*embeddings)
                    loss = loss_fn(pred, stance) # TODO: use weight for unbalanced??

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    wandb.log({f"train-loss": loss / BATCH_SIZE})

                # EVAL
                if (batch_idx % EVAL_FREQ == 0):
                    model.eval()
                    pbar.set_description(f"Epoch {epoch}/{num_epochs-1} | " + "Evaluating...")

                    cum_loss = 0
                    class_correct = np.zeros(NUM_CLASSES)
                    class_total = np.zeros(NUM_CLASSES)

                    for (embeddings, stance) in dataloaders["val"]:
                        embeddings = [_embedding.to(DEVICE) for _embedding in embeddings]
                        stance = stance.to(DEVICE)

                        with torch.set_grad_enabled(False):
                            pred = model(*embeddings)
                            loss = loss_fn(pred, stance) # TODO: use weight for unbalanced??

                        cum_loss += loss.item()
                        pred_stance = torch.max(pred,dim=1).indices
                        for ss, pp in zip(stance, pred_stance):
                            class_correct[ss] += 1 if (ss == pp) else 0
                            class_total[ss] += 1

                    val_loss = cum_loss / len(dataloaders["val"].dataset)
                    class_avgs = [class_correct[ii] / class_total[ii] for ii in range(NUM_CLASSES)]

                    pbar.set_postfix(class_averages=f"{[np.round(ca * 100,1) for ca in class_avgs]}")

                    wandb.log({f"val-loss": val_loss})
                    wandb.log({
                        f'{STANCE_MAP_INV[ii]}': class_avgs[ii] for ii in range(NUM_CLASSES)
                    })
                    
                    # deep copy the model
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = deepcopy(model.state_dict())

    time_elapsed = time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Loss: {best_loss:4f}")

    model.load_state_dict(best_model)
    return model

def save_model(model: nn.Module, type: str, name: str):
    path = f"./saved_models/{name}.pth"
    
    torch.save(model.state_dict(), path)
    run = wandb.init(project=WANDB_PROJ)

    artifact = wandb.Artifact(name=name, type=type)
    artifact.add_file(path)
    
    run.log_artifact(artifact)
    
    run.finish()

def main():
    ##
    dataset = FakeNewsEncodedDataset(
        stances_file="data/train_stances.csv.stance.dat",
        bodies_file="data/train_bodies.csv.body.dat"
    )
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    cutoff = math.floor(dataset_size * VAL_CUTOFF)

    train_sampler = SubsetRandomSampler(dataset_indices[:cutoff])
    val_sampler = SubsetRandomSampler(dataset_indices[cutoff:])

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    #model = AgreemFlat()
    model = AgreemDeep().to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trained_model = train_model(
        model=model,
        dataloaders={
            "train": train_dataloader,
            "val": val_dataloader,
        },
        loss_fn=loss_fn,
        optimizer=optimizer, 
        num_epochs=EPOCHS,
    )

    print("* Saving model *")
    save_model(
        model=trained_model,
        type="AgreemDeep",
        name="test-model"
    )

if __name__ == "__main__":
    main()