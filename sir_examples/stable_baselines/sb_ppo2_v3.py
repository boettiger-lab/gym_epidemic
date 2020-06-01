import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
import sir_gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

class CustomPPOPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(*args, **kwargs,
                                           layers=[400, 300],
                                           feature_extraction="mlp")




env = gym.make('sir-v3', intervention='fs')
model = PPO2(CustomPPOPolicy, env, verbose=2, ent_coef=0.00001, cliprange=0.1, n_steps=1024, noptepochs=20)
model.learn(total_timesteps=int(3e4), log_interval=1000)
model.save("sir_ppo2_v3")
