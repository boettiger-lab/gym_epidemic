import gym
import gym_epidemic
import numpy as np

from stable_baselines3.sac import MlpPolicy as policy_sac
from stable_baselines3.ppo import MlpPolicy as policy_ppo
from stable_baselines3 import SAC, PPO
from tuning_utils_sb3 import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", default="fs", type=str, help="Input intervention type.")
parser.add_argument("-a", default="sac", type=str, help="Input algorithm.")
args = parser.parse_args()

algos = {"sac":[SAC, policy_sac], "ppo":[PPO, policy_ppo]}
def create_env(n_envs=0, eval_env=None):
    env = gym.make("sir-v0", intervention=f'{args.i}', random_params=True, random_obs=True)
    return env

def create_model(*_args, **kwargs):
    """
    Helper to create a model with different hyperparameters
    """
    return algos[args.a][0](env=create_env(), policy=algos[args.a][1], verbose=1, **kwargs)

hyperparam_optimization(args.a, create_model, create_env, n_trials=100,
                                             n_timesteps=int(2e6),
                                             n_jobs=1, seed=0, log_interval=int(1e2), 
                                             sampler_method='tpe', pruner_method='median',
                                             verbose=1)
