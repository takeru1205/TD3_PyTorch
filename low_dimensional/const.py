# constant parameters

gamma = 0.99  # discount factor
weight_decay = 1e-2  # L2 weight decay
actor_lr = 3e-4  # learning rate for actor network
critic_lr = 3e-4  # learning rate for critic network
tau = 0.005  # soft target update coefficient
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2  # Delayed policy update
