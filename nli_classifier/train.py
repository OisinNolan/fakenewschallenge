from dataset import FakeNewsDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import FakeNewsDataset
from model import AgreemNet
from util import pad_tokenize 
import torch
from torch import nn
from tqdm import tqdm
import math
import wandb

wandb.init(project="first-tests", entity="mlpbros")

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.05
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
}

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, ((H, B), y) in tqdm(enumerate(dataloader)):
        B_pad = pad_tokenize(B)
        # Compute prediction and loss
        pred = model(H, B_pad).to(DEVICE)
        loss = loss_fn(pred, y.to(DEVICE))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 1 == 0:
        loss, current = loss.item(), (batch * BATCH_SIZE)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        wandb.log({"training-loss": loss})

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = {
        0: 0,
        1: 0,
        2: 0
    }
    class_size = {
        0: 0,
        1: 0,
        2: 0
    }

    with torch.no_grad():
        for (H, B), y in tqdm(dataloader):
            y = y.to(DEVICE)
            B_pad = pad_tokenize(B)
            pred = model(H, B_pad)
            test_loss += loss_fn(pred, y).item()
            for pred_i, y_i in zip(pred, y):
                correct[int(y_i)] += int(pred_i.argmax() == y_i)
                class_size[int(y_i)] += 1

    test_loss /= num_batches
    print("Per-class avg:")
    print(f"{[correct[i] / class_size[i] for i in class_size.keys()]}")
    print(f"Avg loss: {test_loss:>8f}\n")
    
    wandb.log({"test-loss": test_loss})
    wandb.log({
        f'class-{i}':correct[i] / class_size[i] for i in class_size.keys()
    })

dataset = FakeNewsDataset('../data/combined_stances_train.csv', '../data/combined_bodies_train.csv', related_only=True)
SUBSET_SIZE = len(dataset)

# Partition dataset into train and val sets
dataset_size = SUBSET_SIZE
dataset_indices = list(range(dataset_size))
cutoff = math.floor(dataset_size * 0.7)
train_sampler = SubsetRandomSampler(dataset_indices[:cutoff])
val_sampler = SubsetRandomSampler(dataset_indices[cutoff:])

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

model = AgreemNet()
model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loop(train_dataloader, model, loss_fn, optimizer)

    # Eval
    model.eval()
    test_loop(val_dataloader, model, loss_fn)
