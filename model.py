from typing import List

import numpy as np
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from torch.distributions.constraints import one_hot


class DenseNet(nn.Module):
    def __init__(self):
        """
        :param inputs: length of input vector
        :param outputs: number of classes
        """
        super(DenseNet, self).__init__()
        self.sentence_transformer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        hidden_layer_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        channels = 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(2, 1), stride=1),
            nn.BatchNorm2d(channels),
            nn.Flatten(),
            nn.Linear(channels * hidden_layer_dim, hidden_layer_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.Linear(hidden_layer_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, song_text: List[str], query: List[str]):
        doc = self.sentence_transformer.encode(song_text)
        query = self.sentence_transformer.encode(query)
        inputs_ = torch.stack((doc, query), dim=1).float()
        return self.net(torch.unsqueeze(inputs_, dim=1))


def train(dataloader, model, loss_fn, optimizer):
    tp, rel, ans = 0, 0, 0  # True Positive, Relevant docs, Answered Docs
    b_loss, epoch_loss = 0, 0
    # f1 = F1Score(task="multiclass", num_classes=2)
    for step, (doc, query, target) in enumerate(dataloader):
        # ------------------------------------------------------------------------
        optimizer.zero_grad()
        out = model.forward(doc, query)
        batch_loss = loss_fn(out, torch.squeeze(one_hot(target, 2)).float())
        batch_loss.backward()
        b_loss += batch_loss.item()
        epoch_loss += batch_loss.item()
        optimizer.step()
        # ------------------------------------------------------------------------
        tp += torch.sum(torch.logical_and(
            torch.argmax(out, dim=1).bool(), torch.squeeze(target).bool()
        ).float())
        rel += torch.sum(target)
        ans += torch.sum(torch.argmax(out, dim=1))
        # ------------------------------------------------------------------------
        # As requested every 1000 batches:
        if (step + 1) % 1000 == 0:
            recall, precision = tp / rel, tp / ans
            print(f"  Step {step + 1}:")
            print(f"  F1-score: {2 * precision * recall / (precision + recall)}")
            print(f"  Loss: {b_loss / 1000}")
            tp, rel, ans = 0, 0, 0
            b_loss = 0
    return epoch_loss / len(dataloader)
