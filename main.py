import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import DenseNet
from model import test
from model import train
from preprocessing import preprocess_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
columns = ['len_of_text', 'text', 'document', 'relevance']

if __name__ == "__main__":
    train_loss_dict, test_loss_dict = {}, {}
    train_f1_dict, test_f1_dict = {}, {}
    cos_train_f1_dict, cos_test_f1_dict = {}, {}
    for q_len in [5, 15, 30, 50, 75, 100]:
        model = DenseNet(384 * 2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.BCELoss()
        train_loader, test_loader = preprocess_data(batch_size=10000, query_length=q_len)
        # ---------------------------------------------------------------------------------------
        train_loss_dict[q_len] = []
        test_loss_dict[q_len] = []
        train_f1_dict[q_len] = []
        test_f1_dict[q_len] = []
        cos_train_f1_dict[q_len] = []
        cos_test_f1_dict[q_len] = []
        # ---------------------------------------------------------------------------------------
        for i in tqdm(range(100), desc=f"Train model for query_length {q_len}. Epoch"):
            train_loss, train_f1 = train(train_loader, model, optimizer, loss_fn)
            test_loss, test_f1 = test(test_loader, model, loss_fn)
            train_loss_dict[q_len].append(train_loss)
            test_loss_dict[q_len].append(test_loss)
            train_f1_dict[q_len].append(train_f1)
            test_f1_dict[q_len].append(test_f1)
            # TODO Covariance
            #   cos_train_f1_dict[query_length] = []
            #   cos_test_f1_dict[query_length] = []
        print(f"Min test-loss value: {np.min(test_loss_dict[q_len])} at Epoch {np.argmin(test_loss_dict[q_len])}")
        print(f"Max test-F1-score: {np.max(test_f1_dict[q_len])} at Epoch: {np.argmax(test_f1_dict[q_len])}")
    pd.DataFrame(data=train_loss_dict).to_csv("train_loss.csv")
    pd.DataFrame(data=test_loss_dict).to_csv("test_loss.csv")
    pd.DataFrame(data=train_f1_dict).to_csv("train_f1.csv")
    pd.DataFrame(data=test_f1_dict).to_csv("test_f1.csv")
