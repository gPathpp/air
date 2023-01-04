import torch
from sentence_transformers import SentenceTransformer
from torch import nn


class DenseNet(nn.Module):
    def __init__(self, outputs: int):
        """
        :param inputs: length of input vector
        :param outputs: number of classes
        """
        super(DenseNet, self).__init__()
        self.sentence_transformer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        self.outputs = outputs
        hidden_dim = 1000
        self.net = nn.Sequential(
            nn.Linear(self.sentence_transformer.get_sentence_embedding_dimension(), hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, outputs),
            nn.Softmax(dim=0)
        )

    def forward(self, inputs_):
        in_ = self.sentence_transformer.encode(inputs_)
        return self.net(torch.from_numpy(in_))
