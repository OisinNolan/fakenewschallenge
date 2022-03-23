from dataset import FakeNewsEncodedDataset, STANCE_MAP
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import AgreemDeep, AgreemFlat, AgreemNet
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
from args import create_parser

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STANCE_MAP_INV = dict((v,k) for k, v in STANCE_MAP.items())
VAL_CUTOFF = 0.7
EVAL_FREQ = 20
WANDB_ENTITY = "mlpbros"

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

                    wandb.log({f"train-loss": loss / dataloaders["train"].batch_size})

                # EVAL
                if (batch_idx % EVAL_FREQ == 0):
                    model.eval()
                    pbar.set_description(f"Epoch {epoch}/{num_epochs-1} | " + "Evaluating...")

                    cum_loss = 0
                    class_correct = np.zeros(model.num_classes)
                    class_total = np.zeros(model.num_classes)

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
                    class_avgs = [class_correct[ii] / class_total[ii] for ii in range(model.num_classes)]

                    pbar.set_postfix(class_averages=f"{[np.round(ca * 100,1) for ca in class_avgs]}")

                    wandb.log({f"val-loss": val_loss})
                    wandb.log({
                        f'{STANCE_MAP_INV[ii]}': class_avgs[ii] for ii in range(model.num_classes)
                    })
                    
                    # deep copy the model
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model = deepcopy(model.state_dict())

    time_elapsed = time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    model.load_state_dict(best_model)
    return model

def save_model(model: nn.Module, name: str):
    path = f"./saved_models/{name}.pth"
    torch.save(model.state_dict(), path)

    run = wandb.init(project=WANDB_PROJ)

    artifact = wandb.Artifact(name=name, type="model")
    artifact.add_file(path)
    
    run.log_artifact(artifact)
    
    run.finish()

def main():
    args = create_parser().parse_args()

    wandb.init(project=args.project_name, entity=WANDB_ENTITY)
    wandb.config.update(args)
    config = wandb.config
    print(f"Config: {config}")

    ##
    dataset = FakeNewsEncodedDataset(
        stances_file="data/train_stances.csv.stance.dat",
        bodies_file="data/train_bodies.csv.body.dat",
        no_unrelated=True
    )
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    cutoff = math.floor(dataset_size * VAL_CUTOFF)

    train_sampler = SubsetRandomSampler(dataset_indices[:cutoff])
    val_sampler = SubsetRandomSampler(dataset_indices[cutoff:])

    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=val_sampler)

    model = ...
    if (config.model == "AgreemFlat"):
        ...
    elif (config.model == "AgreemDeep"):
        model = AgreemDeep(
            kk=config.top_k,
            hdim_1=config.hidden_dims[0],
            hdim_2=config.hidden_dims[1],
        ).to(DEVICE)
    elif (config.model == "AgreemNet"):
        model = AgreemNet(
            hdim=config.hidden_dims[0],
        ).to(DEVICE)
    else:
        assert False # Shouldn't get here

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    trained_model = train_model(
        model=model,
        dataloaders={
            "train": train_dataloader,
            "val": val_dataloader,
        },
        loss_fn=loss_fn,
        optimizer=optimizer, 
        num_epochs=config.epochs,
    )

    save_model(model=trained_model, name=config.model)

if __name__ == "__main__":
    main()