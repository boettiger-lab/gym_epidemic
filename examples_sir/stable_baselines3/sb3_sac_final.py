import gym
from gym import spaces
import gym_epidemic
import numpy as np

import torch as th

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise 
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    for i in range(5):
        env = gym.make('sir-v0', intervention='fs', random_params=True, random_obs=True)
        env = Monitor(env, "./")
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[64, 64], log_std_init=-2.65)
        model = SAC("MlpPolicy", env, verbose=2, batch_size=16, learning_rate=5e-5, train_freq=16, tau=.02, buffer_size=int(1e4), policy_kwargs=policy_kwargs, ent_coef=0.001)
        model.learn(total_timesteps=int(2e6), log_interval=int(1e4))

        del model
