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

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:
            #step 3 of the algorithm
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_rtgs, batch_lens = self.rollout()

            #Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            #step 5 of the algorithm, calculate advantage
            A_k = batch_rtgs - V.detach()

            #Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            
    
