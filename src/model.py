import torch
import torch.nn as nn
import numpy as np
from nn import NN

class PPO:
    def __init__(self, env) -> None:
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.actor = NN(self.obs_dim, self.act_dim)
        self.critic = NN(self.obs_dim, 1)
        self._initi_hyperparameters()

        # Initialize actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        # Initialize critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # Create covariance matrix for get_action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.clip = 0.2     # Recommended by the paper
        self.lr = 0.005
    
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

            #Increment timesteps ran this batch so far
            t_so_far += np.sum(batch_lens)

            #Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            #step 5 of the algorithm, calculate advantage
            A_k = batch_rtgs - V.detach()

            #Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                #Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                #Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                #Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                #Calculate actor and critic losses
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

    
