from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from model import DenseNet

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def replace_artist_with_id(data: DataFrame) -> Dict[int, str]:
    artist_dict = {artist_id: artist for artist_id, artist in enumerate(set(data.artist))}
    id_dict = {artist_dict[artist_id]: artist_id for artist_id in artist_dict.keys()}
    data.artist = data.artist.apply(lambda artist: id_dict[artist])
    return artist_dict

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    for current_token, target in dataloader:
        optimizer.zero_grad()
        out = torch.softmax(model.forward(current_token), dim=1)
        batch_loss = loss_fn(out, one_hot(target, model.outputs).float())
        batch_loss.backward()
        epoch_loss += batch_loss.item()
        optimizer.step()
        accuracy = torch.sum(torch.argmax(out, dim=1) == target) / dataloader.batch_size
        print(f"Batch-Loss: {batch_loss.item()}")
        print(f"Batch-Accuracy: {accuracy}")
    print(f"Train epoch loss: {epoch_loss / len(dataloader)}")


if __name__ == "__main__":
    data = pd.read_csv("spotify_millsongdata.csv")
    artist_dict = replace_artist_with_id(data)
    model = DenseNet(outputs=len(artist_dict))
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1, weight_decay=0.02)
    loss_fn = torch.nn.CrossEntropyLoss()
    targets = data.artist.to_numpy()
    texts = data.text.to_numpy()
    train_dataloader = DataLoader(list(zip(texts, targets)), shuffle=True, sampler=None, batch_size=3000)
    train(train_dataloader, model, loss_fn, optimizer)
