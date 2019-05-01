import torch
import torch.nn as nn
import torch.nn.functional as F


class OFCPokerNet(nn.Module):

    def __init__(self, embedding_dim, action_space, drop_prob, num_layers=2, hidden_size=128):
        super(OFCPokerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.three_hand_embed = nn.Linear(53, self.embedding_dim)
        self.five_hand_embed = nn.Linear(53, self.embedding_dim)
        self.cur_card_embed = nn.Linear(53, self.hidden_size)

        input_dim = 6 * self.embedding_dim + self.hidden_size
        self.fc1 = nn.Linear(input_dim, self.hidden_size)
        self.layers = [nn.Linear(hidden_size, self.hidden_size) for _ in range(num_layers)]
        self.value = nn.Linear(self.hidden_size, 1)
        self.policy = nn.Linear(self.hidden_size, action_space)
        self.dropout = nn.Dropout(p=drop_prob)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.layers = [layer.to(*args, **kwargs) for layer in self.layers]
        return self

    def forward(self, front, mid, back, cur):
        front_embed = self.three_hand_embed(front).view(-1, self.embedding_dim * 2)
        mid_embed = self.five_hand_embed(mid).view(-1, self.embedding_dim * 2)
        back_embed = self.five_hand_embed(back).view(-1, self.embedding_dim * 2)
        card_embed = self.cur_card_embed(cur)
        x = torch.cat((front_embed, mid_embed, back_embed, card_embed), 1)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        for layer in self.layers:
            out = F.relu(layer(out))
            out = self.dropout(out)
        pi = self.policy(out)
        v = self.value(out)
        return F.log_softmax(pi, dim=1), torch.tanh(v)
