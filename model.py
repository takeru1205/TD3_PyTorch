import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 Network
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Q2 Network
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, x, action):
        # Q1 Value
        q1 = F.relu(self.fc1(torch.cat([x, action], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Q2 Value
        q2 = F.relu(self.fc4(torch.cat([x, action], dim=1)))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    # to calculate deterministic policy gradient
    def Q1(self, x, action):
        q1 = F.relu(self.fc1(torch.cat([x, action], dim=1)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1
