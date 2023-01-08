import pandas as pd
import torch

from model import DenseNet
from model import test
from model import train
from preprocessing import preprocess_data

from baseline import train_baseline, test_baseline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
columns = ['len_of_text', 'text', 'document', 'relevance']

if __name__ == "__main__":
    train_loss_dict, test_loss_dict = {}, {}
    train_f1_dict, test_f1_dict = {}, {}
    cos_train_f1_dict, cos_test_f1_dict = {}, {}
    cos_train_loss_dict, cos_test_loss_dict = {}, {}
    for query_length in [20, 30, 50, 75, 100]:
        model = DenseNet(384 * 2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCELoss()
        train_loader, test_loader = preprocess_data(batch_size=10000, query_length=query_length)
        # ---------------------------------------------------------------------------------------
        train_loss_dict[query_length] = []
        test_loss_dict[query_length] = []
        train_f1_dict[query_length] = []
        test_f1_dict[query_length] = []

        cos_train_f1_dict[query_length] = []
        cos_test_f1_dict[query_length] = []
        cos_train_loss_dict[query_length] = []
        cos_test_loss_dict[query_length] = []
        # ---------------------------------------------------------------------------------------
        print("Training started")
        for i in range(100):
            print(f"Epoch {i + 1}\n-------------------------------")
            train_loss, train_f1 = train(train_loader, model, optimizer, loss_fn)
            test_loss, test_f1 = test(test_loader, model, loss_fn)
            train_loss_dict[query_length].append(train_loss)
            test_loss_dict[query_length].append(test_loss)
            train_f1_dict[query_length].append(train_f1)
            test_f1_dict[query_length].append(test_f1)
            # TODO Covariance
            #   cos_train_f1_dict[query_length] = []
            #   cos_test_f1_dict[query_length] = []

            baseline_train_loss, baseline_train_f1 = train_baseline(train_loader, loss_fn)
            baseline_test_loss, baseline_test_f1 = test_baseline(test_loader, loss_fn)

            cos_train_f1_dict[query_length].append(baseline_train_f1)
            cos_test_f1_dict[query_length].append(baseline_test_f1)
            cos_train_loss_dict[query_length].append(baseline_train_loss)
            cos_test_loss_dict[query_length].append(baseline_test_loss)

    pd.DataFrame(data=train_loss_dict).to_csv("train_loss.csv")
    pd.DataFrame(data=test_loss_dict).to_csv("test_loss.csv")
    pd.DataFrame(data=train_f1_dict).to_csv("train_f1.csv")
    pd.DataFrame(data=test_f1_dict).to_csv("test_f1.csv")

    pd.DataFrame(data=cos_train_f1_dict).to_csv("cos_train_f1.csv")
    pd.DataFrame(data=cos_test_f1_dict).to_csv("cos_test_f1.csv")
    pd.DataFrame(data=cos_train_loss_dict).to_csv("cos_train_loss.csv")
    pd.DataFrame(data=cos_test_loss_dict).to_csv("cos_test_loss.csv")
