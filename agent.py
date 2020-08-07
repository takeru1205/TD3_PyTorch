# TD3 agent
import torch
import torch.nn as nn
import torch.optim as optim
from memory import ReplayMemory
from model import Actor, Critic

from const import *


class TD3(object):
    def __init__(self, env, writer=None):
        """
        Twin Delayed Deep Deterministic Policy Gradient Algorithm(TD3)
        """
        self.env = env
        self.writer = writer

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        # Randomly initialize network parameter
        self.actor = Actor(state_dim, action_dim).to('cuda')
        self.critic = Critic(state_dim, action_dim).to('cuda')

        # Initialize target network parameter
        self.target_actor = Actor(state_dim, action_dim).to('cuda')
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Critic(state_dim, action_dim).to('cuda')
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Replay memory
        self.memory = ReplayMemory(state_dim, action_dim)

        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.tau = tau

        # network parameter optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

    def get_action(self):
        # TODO: select action with exploration noise(Gaussian Noise)
        raise NotImplementedError

    def store_transition(self, state, action, state_, reward, done):
        self.memory.store_transition(state, action, state_, reward, done)

    def update(self):
        raise NotImplementedError

    def save_model(self, path='models/'):
        torch.save(self.actor.state_dict(), path + 'actor')
        torch.save(self.critic.state_dict(), path + 'critic')
        torch.save(self.target_actor.state_dict(), path + 'target_actor')
        torch.save(self.target_critic.state_dict(), path + 'target_critic')

    def load_model(self, path='models/'):
        self.actor.load_state_dict(torch.load(path + 'actor'))
        self.critic.load_state_dict(torch.load(path + 'critic'))
        self.target_actor.load_state_dict(torch.load(path + 'target_actor'))
        self.target_critic.load_state_dict(torch.load(path + 'target_critic'))




