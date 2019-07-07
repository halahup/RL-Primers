import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Actor, self).__init__()

        self.fc_1 = nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True)
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.fc_3 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.fc_4 = nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        logits = self.fc_4(x)
        return logits


class Critic(nn.Module):
    def __init__(self, observation_space_size: int, hidden_size: int):
        super(Critic, self).__init__()

        self.fc_1 = nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True)
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.fc_3 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.fc_4 = nn.Linear(in_features=hidden_size, out_features=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.relu(self.fc_3(x))
        logit = self.fc_4(x)
        return logit
