from dataset import FakeNewsDataset
from torch.utils.data import DataLoader
from dataset import FakeNewsDataset
from model import AgreemNet 
from nltk.tokenize import sent_tokenize
import torch
from torch import nn

BATCH_SIZE = 1
LEARNING_RATE = 0.05
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, ((head, body), y) in enumerate(dataloader):
        body_sents = sent_tokenize(body[0])
        # Compute prediction and loss
        pred = model(head[0], body_sents).view(1, 3)
        y.to(DEVICE)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train_data = FakeNewsDataset('../data/combined_stances_train.csv', '../data/combined_bodies_train.csv', related_only=True)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

model = AgreemNet()
model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

train_loop(train_dataloader, model, loss_fn, optimizer)