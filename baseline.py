import torch
import numpy as np
from numpy.linalg import norm
from model import calc_f1
from torch import nn


def cosine_sim(q, d):
    return torch.tensor(np.dot(q, d)/(norm(q) * norm(d)))


def train_baseline(dataloader, loss_fn):
    total_truepos, total_falsepos, total_falseneg = 0, 0, 0
    total_loss = 0
    for q_vec, doc_vec, target in dataloader:
        pred = nn.functional.cosine_similarity(q_vec, doc_vec)
        pred = pred.reshape([len(target), 1])
        loss = loss_fn(pred, target.to(torch.float32))
        for i in range(len(pred)):
            if pred[i] > 1 or pred[i] < 0:
                print(pred[i])

        loss.backward()

        pred_is_relevant = torch.round(pred)
        truepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 1).float()).float()
        falsepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 0).float()).float()
        falseneg = torch.sum(torch.logical_and(pred_is_relevant == 0, target == 1).float()).float()

        f1 = calc_f1(falseneg, falsepos, truepos)
        loss = loss.item()
        print(f"[COS] Train Batch loss: {loss:>7f}")
        print(f"[COS] Train Batch F1 score: {f1:>7f} \n")
        total_truepos += truepos
        total_falsepos += falsepos
        total_falseneg += falseneg
        total_loss += loss

    total_f1 = calc_f1(total_falseneg, total_falsepos, total_truepos)
    total_loss = total_loss / len(dataloader)
    print(f"[COS] Epoch loss: {total_loss:>7f}")
    print(f"[COS] F1 score: {total_f1:>7f} \n")
    return float(total_loss), float(total_f1)


def test_baseline(dataloader, loss_fn):
    total_truepos, total_falsepos, total_falseneg = 0, 0, 0
    total_loss = 0
    for q_vec, doc_vec, target in dataloader:
        pred = nn.functional.cosine_similarity(q_vec, doc_vec)
        pred = pred.reshape([len(target), 1])
        loss = loss_fn(pred, target.to(torch.float32))

        pred_is_relevant = torch.round(pred)
        truepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 1).float()).float()
        falsepos = torch.sum(torch.logical_and(pred_is_relevant == 1, target == 0).float()).float()
        falseneg = torch.sum(torch.logical_and(pred_is_relevant == 0, target == 1).float()).float()

        f1 = calc_f1(falseneg, falsepos, truepos)
        loss = loss.item()
        print(f"[COS] Test Batch loss: {loss:>7f}")
        print(f"[COS] Test Batch F1 score: {f1:>7f} \n")
        total_truepos += truepos
        total_falsepos += falsepos
        total_falseneg += falseneg
        total_loss += loss

    total_f1 = calc_f1(total_falseneg, total_falsepos, total_truepos)
    total_loss = total_loss / len(dataloader)
    print(f"[COS] Test Epoch loss: {total_loss:>7f}")
    print(f"[COS] Test F1 score: {total_f1:>7f} \n")
    return float(total_loss), float(total_f1)
