# TD3 agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        self.tau = tau

        # network parameter optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state, initial_act=False):
        if initial_act:
            return self.env.action_space.sample()
        action = self.actor(torch.from_numpy(state).to('cuda', torch.float))
        action = np.random.normal(0, 0.1) + action.detach().cpu().numpy()
        return np.clip(action, -1, 1)

    def store_transition(self, state, action, state_, reward, done):
        self.memory.store_transition(state, action, state_, reward, done)

    def soft_update(self, target_net, net):
        """Target parameters soft update"""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def update(self, time_step, batch_size=64):
        states, actions, states_, rewards, terminals = self.memory.sample(batch_size)

        # Update Critic
        with torch.no_grad():
            noise = (
                    torch.randn_like(actions) * policy_noise
            ).clamp(-noise_clip, noise_clip)

            actions_ = (
                    self.target_actor(states_) + noise
            ).clamp(-1, 1)

            target_q1, target_q2 = self.target_critic(states_, actions_)
            y = rewards.unsqueeze(1) + terminals.unsqueeze(1) * gamma * torch.min(target_q1, target_q2)
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.writer and time_step % 1000 == 0:
            self.writer.add_scalar("loss/critic", critic_loss.item(), time_step)

        # Delayed Policy Update
        if time_step % policy_freq == 0:
            # Update Actor
            actor_loss = -1 * self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if self.writer and time_step % 1000 == 0:
                self.writer.add_scalar("loss/actor", actor_loss.item(), time_step)

            # target parameter soft update
            self.soft_update(self.target_actor, self.actor)  # update target actor network
            self.soft_update(self.target_critic, self.critic)  # update target critic network

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
