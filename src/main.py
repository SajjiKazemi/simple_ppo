import gymnasium as gym
import sys
import torch

from arguments import get_args
from model import PPO
from nn import NN
from eval_policy import eval_policy


def train(env, hyperparameters ,actor_model, critic_model):
    """The core of the training happens here.

    Args:
        env (gym.env): the environment to train on
        hyperparameters (Dict): a dict of hyperparameters defined in main
        actor_model (torch.nn): the actor model
        critic_model (torch.nn): the critic model
    
    Returns:
        None
    """
    print(f"Training", flush=True)
    model = PPO(policy_class=NN, env=env, **hyperparameters)

    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Loaded in {actor_model} and {critic_model}", flush=True)
    elif actor_model != '' or critic_model != '':
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)
    
    model.learn(total_timesteps=20000)

def test(env, actor_model):
    print(f"Testing", flush=True)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = NN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(actor_model))
    eval_policy(policy, env, render=True)

def main(args):
    """The main function to train the model.

    Args:
        args (Dict): the arguments parsed from command line
    """
    hyperparameters = {
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 200,
        'gamma': 0.99,
        'clip': 0.2,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'render': True,
        'render_every_i': 10
    }

    env = gym.make('Pendulum-v1', render_mode=args.render_mode)
    if args.mode == 'train':
        train(env, hyperparameters, args.actor_model, args.critic_model)
    else:
        test(env, args.actor_model)


if __name__ == '__main__':
    args = get_args()
    main(args)
        