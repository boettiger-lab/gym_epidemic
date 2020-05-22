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

class ConstantSchedule():
    def __init__(self, value):
        self._value = value

    def value(self, step):
        return self._value
    
    def __call__(self, value):
        return self._value

env = gym.make('sir-v3', intervention='fs', random_params=True, random_obs=False)

policy = SACPolicy(net_arch=[400,400,400,400,400],
                    observation_space = spaces.Box(low=0, high=10**2, shape=(4,), dtype=np.float64),
                    action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
                    lr_schedule = ConstantSchedule(0.0001))

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[400, 400, 400, 400, 400])
model = SAC("MlpPolicy", env, verbose=1, batch_size=512, buffer_size=100000,  policy_kwargs=policy_kwargs, learning_starts=10000, ent_coef=0.01)
model.learn(total_timesteps=int(1e5), log_interval=1000)
model.save("sir_sac_v3_fs.zip")
