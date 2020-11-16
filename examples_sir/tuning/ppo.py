import gym
import gym_epidemic
import numpy as np

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from tuning_utils_sb3 import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", default="fs", type=str, help="Input intervention type.")
args = parser.parse_args()

def create_env(n_envs=0, eval_env=None):
    env = gym.make("sir-v0", intervention=f'{args.i}', random_params=True, random_obs=True)
    return env

def create_model(*_args, **kwargs):
    """
    Helper to create a model with different hyperparameters
    """
    return PPO(env=create_env(), policy=MlpPolicy, verbose=1, **kwargs)

hyperparam_optimization("ppo", create_model, create_env, n_trials=100,
                                             n_timesteps=int(2e6),
                                             n_jobs=1, seed=0, log_interval=int(1e6), 
                                             sampler_method='tpe', pruner_method='median',
                                             verbose=2)
