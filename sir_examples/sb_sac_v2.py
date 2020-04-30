import os
import sys
sys.path.append(os.path.realpath('..'))

import gym
import sir_gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

env = gym.make('sir-v2', intervention='o')

model = SAC(MlpPolicy, env, verbose=1, ent_coef='auto_0.99', learning_rate=0.0001)
model.learn(total_timesteps=int(5e4), log_interval=1000)
model.save("sir_sac_v2.zip")
