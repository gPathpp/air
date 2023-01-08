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
        truepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 1).float()).float()
        falsepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 0).float()).float()
        falseneg = torch.sum(torch.logical_and(pred_is_relevant == 0, target == 1).float()).float()

        f1 = calc_f1(falseneg, falsepos, truepos)
        loss = loss.item()
        print(f"Train Batch loss: {loss:>7f}")
        print(f"Train Batch F1 score: {f1:>7f} \n")
        total_truepos += truepos
        total_falsepos += falsepos
        total_falseneg += falseneg
        total_loss += loss

    total_f1 = calc_f1(total_falseneg, total_falsepos, total_truepos)
    total_loss = total_loss / len(dataloader)
    print(f"Epoch loss: {total_loss:>7f}")
    print(f"F1 score: {total_f1:>7f} \n")
    return float(total_loss), float(total_f1)


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
            truepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 1).float()).float()
            falsepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 0).float()).float()
            falseneg = torch.sum(torch.logical_and(pred_is_relevant == 0, target == 1).float()).float()

            f1 = calc_f1(falseneg, falsepos, truepos)
            loss = loss.item()
            print(f"Test Batch loss: {loss:>7f}")
            print(f"Test Batch F1 score: {f1:>7f} \n")
            total_truepos += truepos
            total_falsepos += falsepos
            total_falseneg += falseneg
            total_loss += loss

        total_f1 = calc_f1(total_falseneg, total_falsepos, total_truepos)
        total_loss = total_loss / len(dataloader)
        print(f"Test Epoch loss: {total_loss:>7f}")
        print(f"Test F1 score: {total_f1:>7f} \n")
        return float(total_loss), float(total_f1)

