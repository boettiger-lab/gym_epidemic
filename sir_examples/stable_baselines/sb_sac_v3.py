import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
from gym import spaces
import sir_gym
import numpy as np

import torch as th

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import SAC

env = gym.make('sir-v3', intervention='fs', random_params=True, random_obs=False)

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[400, 400, 400, 400, 400])
model = SAC("MlpPolicy", env, verbose=2, batch_size=512, buffer_size=100000,  policy_kwargs=policy_kwargs, learning_starts=10000, ent_coef=0.01, tensorboard_log="./sac_v3_tb/")
model.learn(total_timesteps=int(1e5), log_interval=1000)
model.save("sir_sac_v3_fs.zip")
