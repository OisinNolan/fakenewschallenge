from dataset import FakeNewsDataset
from torch.utils.data import DataLoader
from dataset import FakeNewsDataset
from model import AgreemNet
from util import pad_tokenize 
import torch
from torch import nn


BATCH_SIZE = 10
LEARNING_RATE = 0.05

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, ((H, B), y) in enumerate(dataloader):
        B_pad = pad_tokenize(B)
        # Compute prediction and loss
        pred = model(H, B_pad)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 1 == 0:
        loss, current = loss.item(), (batch * BATCH_SIZE)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train_data = FakeNewsDataset('../data/combined_stances_train.csv', '../data/combined_bodies_train.csv', related_only=True)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

model = AgreemNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

train_loop(train_dataloader, model, loss_fn, optimizer)