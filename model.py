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
            nn.Linear(inputs, inputs),
            nn.LeakyReLU(),
            nn.Linear(inputs, 150),
            nn.LeakyReLU(),
            nn.Linear(150, 150),
            nn.LeakyReLU(),
            nn.Linear(150, outputs),
            nn.Sigmoid()
        )

    def forward(self, inputs_):
        return self.net(inputs_.float())


def train(dataloader, model, optimizer, loss_fn):
    model.train()
    total_truepos, total_falsepos, total_falseneg = 0, 0, 0
    total_loss = 0
    for q_vec, doc_vec, target in dataloader:
        X = torch.cat((q_vec, doc_vec), -1)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, target.to(torch.float32))
        loss.backward()
        optimizer.step()

        pred_is_relevant = torch.round(pred)
        total_truepos += torch.sum(torch.logical_and(pred_is_relevant == 1, target == 1).float()).float()
        total_falsepos += torch.sum(torch.logical_and(pred_is_relevant == 1, target == 0).float()).float()
        total_falseneg += torch.sum(torch.logical_and(pred_is_relevant == 0, target == 1).float()).float()
        total_loss += loss.item()

    return float(total_loss / len(dataloader)), float(calc_f1(total_falseneg, total_falsepos, total_truepos))


def calc_f1(falseneg: float, falsepos: float, truepos: float) -> float:
    precision = truepos / (truepos + falsepos) if truepos + falsepos > 0 else 0
    recall = truepos / (truepos + falseneg) if truepos + falseneg > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall + 0.0001)
    return f1


def test(dataloader, model, loss_fn):
    model.eval()
    total_truepos, total_falsepos, total_falseneg = 0, 0, 0
    total_loss = 0
    with torch.no_grad():
        for q_vec, doc_vec, target in dataloader:
            X = torch.cat((q_vec, doc_vec), -1)
            pred = model(X)
            loss = loss_fn(pred, target.to(torch.float32))
            pred_is_relevant = torch.round(pred)
            loss = loss.item()
            total_truepos += torch.sum(torch.logical_and(pred_is_relevant == 1, target == 1).float()).float()
            total_falsepos += torch.sum(torch.logical_and(pred_is_relevant == 1, target == 0).float()).float()
            total_falseneg += torch.sum(torch.logical_and(pred_is_relevant == 0, target == 1).float()).float()
            total_loss += loss

    return float(total_loss / len(dataloader)), float(calc_f1(total_falseneg, total_falsepos, total_truepos))

