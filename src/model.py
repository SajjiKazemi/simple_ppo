import gymnasium as gym
import time

import torch
import torch.nn as nn
import numpy as np
from nn import NN


class PPO:
    """
    This a base implementation for Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(self, policy_class, env, **hyperparameters) -> None:
        """Initializes PPO algorithm, using the hyperparameters provided.

        Args:
            policy_class(torch.nn): The class of the policy network for our actor/critic networks.
            env(gym env): The environment to train on.
            hyperparameters(Dict): The hyperparameters for training (timesteps_per_batch, gamma, etc).
        """
        #Making sure the environment is compatible with our code
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)

        # Initialize hyperparameters
        self._init_hyperparameters(hyperparameters)

        # Define and set the environment
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Define actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)

        # Initialize actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr)

        # Initialize critic optimizer
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr)

        # Create covariance matrix for get_action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # Set a logger method to print summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }
 
    def _init_hyperparameters(self, hyperparameters):
        """ Initializes default hyperparameters.

        Args:
            hyperparameters(Dict): Dictionary of hyperparameters to initialize.

        """
        # Initialize default values for hyperparameters
        self.timesteps_per_batch = 4800     # timesteps per batch
        self.max_timesteps_per_episode = 1600       # max timesteps per episode
        self.n_updates_per_iteration = 5        # number of times to update actor/critic per iteration
        self.clip = 0.2     # Recommended by the paper
        self.lr = 0.005     
        self.gamma = 0.95   # Discount factor to be applied when calculating Rewards-To-Go

        # Miscellaneous parameters
        self.render = True     # If we should render during rollout
        self.render_every_i = 50       # Only render every n iterations
        self.save_freq = 10    # How often we save in number of iterations
        self.seed = None       # Sets the seed of our program, used for reproducibility of results

        # Update values for hyperparameters that were provided
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}", flush=True)

    def rollout(self):
        """Saving the data from running the environment into a batch.

        Returns:
            batch_obs: the observations collected this batch. Shape: (number of timesteps, dimension of observation)
            batch_acts: the actions collected this batch. Shape: (number of timesteps, dimension of action)
            batch_log_probs: the log probabilities of actions taken this batch. Shape: (number of timesteps)
            batch_rtgs: the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
            batch_lens: the lengths of each episode this batch. Shape: (number of episodes)
            batch_rews: the rewards received by the agent for each episode this batch. Shape: (number of episodes)
        """
        batch_obs = []      # batch observations
        batch_acts = []     # batch actions
        batch_log_probs = []        # log probs of each action
        batch_rews = []     # batch rewards
        batch_rtgs = []     # batch rewards-to-go
        batch_lens = []     # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0
        while t < self.timesteps_per_batch:

            # Reward this episode
            ep_rews = []

            obs = self.env.reset()[0]
            done = False

            # Keep going until we have enough timesteps in this batch
            for ep_t in range(self.max_timesteps_per_episode):
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                # Increment timesteps ran this batch so far
                t += 1

                # Collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, terminated, truncated = self.env.step(action)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            # plus 1 because timestep starts at 0
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # Compute rewards-to-go as the 4th step of the algorithm
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_rtgs, batch_lens

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def get_action(self, obs):
        """Queries an action from the actor network, should be called from rollout.

        Args:
            obs (torch.tensor): the observation at the current timestep

        Returns:
            action (narray): the action to take, as a numpy array
            log_prob (narray): the log probability of the selected action in the distribution
        """
        mean = self.actor(obs)

        # Create the multivariate normal distribution
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        """Computes the Rewards-To-Go of each timestep in a batch given the rewards.

        Args:
            batch_rews (list): the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

        Returns:
            batch_rtgs (list): the rewards-to-go in a batch, Shape: (number of timesteps in batch)
        """
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def learn(self, total_timesteps):
        """The main core of the PPO method, where the actor and critic are trained.

        Args:
            total_timesteps (int): the total number of timesteps
        """
        print(f"Beginning training with {total_timesteps} timesteps.", flush=True)
        print(f"and {self.timesteps_per_batch} timesteps per batch.", flush=True)
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:
            # step 3 of the algorithm
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_rtgs, batch_lens = self.rollout()

            # Increment timesteps ran this batch so far
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            # step 5 of the algorithm, calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses
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

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())
            
            # Print the summary stats
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), f"ppo_actor.pth")
                torch.save(self.critic.state_dict(), f"ppo_critic.pth")

    def _log_summary(self):
        """
        Prints a summary of training so far.
        """
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

