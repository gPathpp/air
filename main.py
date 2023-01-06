import numpy as np
import torch

from model import DenseNet
from preprocessing import preprocess_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # inputs = 100
    # model = DenseNet(inputs, outputs=100)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.02)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # model.eval()  # set into eval mode TODO remove
    # print(model.forward(torch.from_numpy(np.random.random(inputs))))
    train_loader, test_loader = preprocess_data(batch_size=10_000)
    # TODO train
