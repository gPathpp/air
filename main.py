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


def create_dataset(data_dict):
    dataset = []
    for row in data_dict:
        entry = []
        for column in columns:
            if column == 'text':
                entry.append(torch.tensor(sentence_transformer.encode(row[column])))
            elif column == 'relevance':
                if row[column]:
                    entry.append(torch.tensor([1]))
                else:
                    entry.append(torch.tensor([0]))
            elif column == 'len_of_text':
                entry.append(torch.tensor(float(row[column])))
            elif column == 'document':
                entry.append(torch.tensor([int(row[column])]))
        dataset.append(entry)
    return dataset


if __name__ == "__main__":
    # inputs = 100
    # model = DenseNet(inputs, outputs=100)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.02)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # model.eval()  # set into eval mode TODO remove
    # print(model.forward(torch.from_numpy(np.random.random(inputs))))

    # train_loader, test_loader = preprocess_data(batch_size=10_000)


    sentence_transformer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    model = DenseNet(385, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    train_csv = open('train_queries_len_5.csv', mode='r')
    train_dict = csv.DictReader(train_csv)

    test_csv = open('test_queries_len_5.csv', mode='r')
    test_dict = csv.DictReader(test_csv)

    train_dataset, test_dataset = create_dataset(train_dict), create_dataset(test_dict)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

    print("Training started")
    for i in range(5):
        print(f"Epoch {i + 1}\n-------------------------------")
        train(train_dataloader, model, optimizer, loss_fn)
        test(test_dataloader, model, loss_fn)
