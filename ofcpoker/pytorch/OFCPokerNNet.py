import torch
import torch.nn as nn
import torch.nn.functional as F


class OFCPokerNet(nn.Module):

    def __init__(self, input_dim, action_space, num_layers=2, hidden_size=128):
        super(OFCPokerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        self.value = nn.Linear(hidden_size, 1)
        self.policy = nn.Linear(hidden_size, action_space)

    def forward(x):
        out = F.relu(self.fc1(x))
        for layer in self.layers:
            out = F.relu(layer(out))
        pi = self.policy(out)
        v = self.value(out)
        return F.log_softmax(pi, dim=1), F.tanh(v)
