import os
import sys
sys.path.append(os.path.realpath('..'))

import gym
import sir_gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

env = gym.make('sir-v2')
model = PPO2(MlpPolicy, env, verbose=2, ent_coef=0.01, cliprange=0.3)
model.learn(total_timesteps=int(1e4))
model.save("sir_2")