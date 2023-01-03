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
            nn.Linear(inputs, 125),
            nn.LeakyReLU(),
            nn.Linear(125, 125),
            nn.LeakyReLU(),
            nn.Linear(125, outputs),
            nn.Softmax(dim=0)
        )

    def forward(self, inputs_):
        return self.net(inputs_.float())
