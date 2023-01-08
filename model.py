import pandas as pd
import torch
from torch import nn


class DenseNet(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        """
        :param inputs: length of input vector
        :param outputs: number of classes
        """
        super(DenseNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inputs, 125),
            nn.LeakyReLU(),
            nn.Linear(125, 125),
            nn.LeakyReLU(),
            nn.Linear(125, outputs),
            nn.Softmax(dim=0)
        )

    def forward(self, inputs_):
        return self.net(inputs_.float())


def train(dataloader, model, optimizer, loss_fn):
    model.train()
    cnt = 1
    truepos, falsepos, falseneg = 0, 0, 0

    for q_vec, doc_vec, target in dataloader:
        X = torch.cat((q_vec, doc_vec), -1)
        pred = model(X)
        loss = loss_fn(pred, target.to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        truepos += torch.sum((torch.round(pred) == target == 1).float()).float()
        falsepos += torch.sum((torch.round(pred) != target == 1).float()).float()
        falseneg += torch.sum((torch.round(pred) != target == 0).float()).float()
        if cnt == len(dataloader):
            loss = loss.item()
            precision = truepos / (truepos + falsepos + 0.0001)
            recall = truepos / (truepos + falseneg + 0.0001)
            f1 = 2 * (precision * recall) / (precision + recall + 0.0001)
            print(f"Train loss: {loss:>7f}")
            print(f"F1 score: {f1:>7f} \n")
        cnt += 1

    return loss


def test(dataloader, model, loss_fn):
    size = 0
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for qlen, qvec, idxdoc, target in dataloader:
            X = torch.cat((qvec, idxdoc), -1)
            pred = model(X)
            test_loss += loss_fn(pred, target.to(torch.float32)).item()
            correct += torch.sum(pred == target)

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:> 8f} \n")