import pandas as pd
import sklearn.metrics
import torch
from sklearn.metrics import f1_score
from torch import nn


class DenseNet(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        """
        :param inputs: length of input vector
        :param outputs: number of classes
        """
        super(DenseNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inputs, inputs),
            nn.LeakyReLU(),
            nn.Linear(inputs, inputs),
            nn.LeakyReLU(),
            nn.Linear(inputs, inputs),
            nn.LeakyReLU(),
            nn.Linear(inputs, outputs),
            nn.Softmax(dim=0)
        )

    def forward(self, inputs_):
        return self.net(inputs_.float())


def train(dataloader, model, optimizer, loss_fn):
    model.train()
    total_truepos, total_falsepos, total_falseneg = 0, 0, 0

    for q_vec, doc_vec, target in dataloader:
        X = torch.cat((q_vec, doc_vec), -1)
        pred = model(X)
        loss = loss_fn(pred, target.to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_is_relevant = torch.round(pred)
        truepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 1).float()).float()
        falsepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 0).float()).float()
        falseneg = torch.sum(torch.logical_and(pred_is_relevant == 0, target == 1).float()).float()

        f1 = calc_f1(falseneg, falsepos, truepos)
        loss = loss.item()
        print(f"Batch loss: {loss:>7f}")
        print(f"Batch F1 score: {f1:>7f} \n")
        total_truepos += truepos
        total_falsepos += falsepos
        total_falseneg += falseneg

    f1 = calc_f1(total_falseneg, total_falsepos, total_truepos)
    print(f"Train loss: {loss:>7f}")
    print(f"F1 score: {f1:>7f} \n")

    return loss


def calc_f1(falseneg: float, falsepos: float, truepos: float) -> float:
    precision = truepos / (truepos + falsepos) if truepos + falsepos > 0 else 0
    recall = truepos / (truepos + falseneg) if truepos + falseneg > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall + 0.0001)
    return f1


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
