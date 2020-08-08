import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, action_dim, width=40, height=40):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_width * conv_height * 32

        self.fc1 = nn.Linear(linear_input_size + action_dim, 200)
        self.fc2 = nn.Linear(200, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class Critic(nn.Module):
    def __init__(self, action_dim, width=40, height=40):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        self.conv4 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_width * conv_height * 32

        # Q1 Network
        self.fc1 = nn.Linear(linear_input_size + action_dim, 200)
        self.fc2 = nn.Linear(200, 1)

        # Q2 Network
        self.fc3 = nn.Linear(linear_input_size + action_dim, 200)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x, action):
        # Q1 Value
        q1 = F.relu(self.conv1(x))
        q1 = F.relu(self.conv2(q1))
        q1 = F.relu(self.conv3(q1))
        q1 = q1.view(q1.size(0), -1)
        q1 = F.relu(self.fc1(torch.cat([q1, action], dim=1)))
        q1 = self.fc2(q1)

        # Q2 Value
        q2 = F.relu(self.conv1(x))
        q2 = F.relu(self.conv2(q1))
        q2 = F.relu(self.conv3(q1))
        q2 = q2.view(q2.size(0), -1)
        q2 = F.relu(self.fc3(torch.cat([q2, action], dim=1)))
        q2 = self.fc4(q2)

        return q1, q2

    # to calculate deterministic policy gradient
    def Q1(self, x, action):
        q1 = F.relu(self.conv1(x))
        q1 = F.relu(self.conv2(q1))
        q1 = F.relu(self.conv3(q1))
        q1 = q1.view(q1.size(0), -1)
        q1 = F.relu(self.fc1(torch.cat([q1, action], dim=1)))
        q1 = self.fc2(q1)
        return q1
