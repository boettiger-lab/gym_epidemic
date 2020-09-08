import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
from gym import spaces
import sir_multistep_gym
import numpy as np

import torch as th

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise 
output_string = ""
for i in range(1):
    env = gym.make('sir-v0', intervention='fc')
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[64, 64], log_std_init=-2.65)
    model = SAC("MlpPolicy", env, verbose=2, batch_size=64, learning_rate=2e-5, train_freq=8, tau=.02, buffer_size=int(1e5), policy_kwargs=policy_kwargs, ent_coef=0.01, POI_R0s=[2,3,4], epsilon=1, burn_in=int(0))
    model.learn(total_timesteps=int(2e6), log_interval=int(1e2))
    model.save(f"sb3_sac_o")

    del model
print(output_string)
