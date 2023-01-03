from typing import List

import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.nn import CosineSimilarity
from torch.nn.functional import one_hot

from model import DenseNet, train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


def get_substrings_with_length_n(txt: str, n=5) -> List[str]:
    pass  # TODO


if __name__ == "__main__":
    data = pd.read_csv("spotify_millsongdata.csv")
    artist_map = {artist_id: artist for artist_id, artist in enumerate(set(data.artist))}
    model = DenseNet()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.02)
    data = list(zip(data.text.to_numpy(), data.text.to_numpy()))
    loss_fn = torch.nn.CrossEntropyLoss()


# TODO mark docs with similartiy > 0.5 as relevant and calculate F1 score
