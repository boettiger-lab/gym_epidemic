import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
import sir_gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines import SAC

class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[400, 400, 400, 400, 400],
                                           layer_norm=False,
                                           feature_extraction="mlp")

env = gym.make('sir-v3', intervention='fs', random_params=True, random_obs=True)

model = SAC(CustomSACPolicy, env, verbose=1, batch_size=512, buffer_size=100000, learning_starts=10000, ent_coef=0.01, learning_rate=0.0001)
model.learn(total_timesteps=int(1e5), log_interval=1000)
model.save("sir_sac_v3_fs.zip")
