import torch
from nn import NN

class PPO:
    def __init__(self, env) -> None:
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.actor = NN(self.obs_dim, self.act_dim)
        self.critic = NN(self.obs_dim, 1)
        self._initi_hyperparameters()

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
    
    def rollout(self):
        batch_obs = []      # batch observations
        batch_acts = []     # batch actions
        batch_log_probs = []        # log probs of each action
        batch_rews = []     # batch rewards
        batch_rtgs = []     # batch rewards-to-go
        batch_lens = []     # episodic lengths in batch

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:
            
    
