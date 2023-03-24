import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.fc(x)
        out = self.logsoftmax(out)
        return out
