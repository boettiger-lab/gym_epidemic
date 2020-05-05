import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
import sir_gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

env = gym.make('sir-v3', intervention='fs')

model = SAC(MlpPolicy, env, verbose=1, ent_coef='auto_0.9', learning_rate=0.0001)
model.learn(total_timesteps=int(1e5), log_interval=1000)
model.save("sir_sac_v3_fs.zip")
