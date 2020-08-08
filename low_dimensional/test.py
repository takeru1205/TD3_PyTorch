import gym
import numpy as np
import torch
from agent import TD3

env = gym.make('Pendulum-v0')

# seed
np.random.seed(42)
env.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

agent = TD3(env)
agent.load_model()

epoch = 150
initial_act = 1000
all_timestep = 0

state = env.reset()
cumulative_reward = 0
for i in range(200):
    action = agent.get_action(state)
    state_, reward, done, _ = env.step(action * env.action_space.high[0])
    env.render()
    agent.store_transition(state, action, state_, reward, done)

    state = state_
    cumulative_reward += reward

    all_timestep += 1
print('Cumulative Reward : {}'.format(cumulative_reward))

env.close()
