import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import TD3

env = gym.make('Pendulum-v0')

# seed
np.random.seed(42)
env.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

writer = SummaryWriter(log_dir='logs/')

agent = TD3(env, writer=writer)

epoch = 150
initial_act = 1000
all_timestep = 0

for e in range(epoch):
    state = env.reset()
    cumulative_reward = 0
    for i in range(200):
        if all_timestep < initial_act:
            action = agent.get_action(state, initial_act=True)
            state_, reward, done, _ = env.step(action)
        else:
            action = agent.get_action(state)
            state_, reward, done, _ = env.step(action * env.action_space.high[0])
        env.render()
        agent.store_transition(state, action, state_, reward, done)

        state = state_
        cumulative_reward += reward

        agent.update(all_timestep)
        all_timestep += 1
    print('Epoch : {} / {}, Cumulative Reward : {}'.format(e, epoch, cumulative_reward))
    writer.add_scalar("reward", cumulative_reward, e)

agent.save_model()
env.close()
