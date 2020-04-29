import os
import sys
sys.path.append(os.path.realpath('..'))

import gym
import sir_gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

env = gym.make('sir-v2')

model = SAC(MlpPolicy, env, verbose=1, ent_coef='auto_0.9')
model.learn(total_timesteps=30000, log_interval=1000)
model.save("sac_pendulum")
