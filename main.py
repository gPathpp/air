from pathlib import Path

import numpy as np
import torch
import csv
import nltk
from model import DenseNet
from model import train
from model import test
from preprocessing import preprocess_data
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
columns = ['len_of_text', 'text', 'document', 'relevance']


if __name__ == "__main__":
    # inputs = 100
    # model = DenseNet(inputs, outputs=100)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.02)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # model.eval()  # set into eval mode TODO remove
    # print(model.forward(torch.from_numpy(np.random.random(inputs))))

    # train_loader, test_loader = preprocess_data(batch_size=10_000)
    sentence_transformer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    model = DenseNet(384 * 2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    train_loader, test_loader = preprocess_data(batch_size=2500)

    print("Training started")
    for i in range(5):
        print(f"Epoch {i + 1}\n-------------------------------")
        train(train_loader, model, optimizer, loss_fn)
       # test(test_loader, model, loss_fn)
