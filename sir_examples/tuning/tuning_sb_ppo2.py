import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
import sir_gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from tuning_utils import *

def create_env():
    env = gym.make("sir-v2")
    return env

def create_model(*_args, **kwargs):
    """
    Helper to create a model with different hyperparameters
    """
    return PPO2(env=create_env(), policy=MlpPolicy, verbose=1, **kwargs)

hyperparam_optimization("ppo2", create_model, create_env, n_trials=10,
                                             n_timesteps=int(5e4),
                                             n_jobs=1, seed=0,
                                             sampler_method='tpe', pruner_method='median',
                                             verbose=1)
