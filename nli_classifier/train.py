from dataset import FakeNewsDataset
from torch.utils.data import DataLoader
from dataset import FakeNewsDataset
from model import AgreemNet 

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, ((head, body), y) in enumerate(dataloader):
        print(batch, head, body, y)
        # # Compute prediction and loss
        # pred = model(X)
        # loss = loss_fn(pred, y)

        # # Backpropagation
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train_data = FakeNewsDataset('combined_stances_train.csv', 'combined_bodies_train.csv', related_only=True)
train_dataloader = DataLoader(train_data)

model = AgreemNet()

train_loop(train_dataloader)