from dataset import FakeNewsEncodedDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import AgreemFlat, AgreemNet
from util import pad_tokenize 
import torch
from torch import batch_norm, nn
from tqdm import tqdm
import math
import wandb

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EVAL_FREQ = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_CUTOFF = 0.7

wandb.init(project="yo-am-i-going-crazy", entity="mlpbros")

wandb.config = {
   "learning_rate": LEARNING_RATE,
   "epochs": EPOCHS,
   "batch_size": BATCH_SIZE,
}

def train(dataloader, model, loss_fn, optimizer):
    for batch, (embeddings, stance) in tqdm(enumerate(dataloader)):
        model.train()
        pred = model(*embeddings)
        loss = loss_fn(pred, stance) # TODO: use weight for unbalanced??

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch * BATCH_SIZE)
        tqdm.write(f"{current:>5d} | loss: {loss:>7f}")
        wandb.log({"training-loss": loss})

        if (batch % EVAL_FREQ == 0):
            model.eval()
            val(val_dataloader, model, loss_fn)

def val(dataloader, model, loss_fn):
    print("*"*20,"VALIDATION","*"*20)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }
    class_size = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    with torch.no_grad():
        for batch, (embeddings, stance) in tqdm(enumerate(dataloader)):
            pred = model(*embeddings)
            test_loss += loss_fn(pred, stance).item()
            for pred_i, stance_i in zip(pred, stance):
                correct[int(stance_i)] += int(pred_i.argmax() == stance_i)
                class_size[int(stance_i)] += 1

    test_loss /= BATCH_SIZE
    print("Per-class avg:")
    print(f"{[correct[i] / class_size[i] for i in class_size.keys()]}")
    print(f"Avg loss: {test_loss:>8f}\n")

    wandb.log({"val-loss": test_loss})
    wandb.log({
       f'class-{i}':correct[i] / class_size[i] for i in class_size.keys()
    })
    print("*"*50)

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

model = AgreemFlat()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for ii in range(EPOCHS):
    train(
        dataloader=train_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )