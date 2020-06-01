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
output_string = ""
for i in range(5):
    env = gym.make('sir-v3', intervention='fs', random_params=True, random_obs=True)
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256])
    model = SAC("MlpPolicy", env, verbose=2, batch_size=256, buffer_size=int(1e6), policy_kwargs=policy_kwargs, learning_starts=int(1e4), ent_coef="auto")
    model.learn(total_timesteps=int(1e7), log_interval=int(1e7))
    model.save(f"sb3_sac_auto_ent_{i}")

    rewards = []
    for i in range(10):
        for R0 in np.linspace(2, 10, 50):
            env = gym.make('sir-v3', intervention='fs')
            env.covid_sir.random_obs = False 
            env.covid_sir.random_params = False
            env.covid_sir.R0 = R0
            obs = env.reset()
            action, states = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            rewards.append(reward)
    
    output_string += f"For iteration {i}, avg reward over range of R0: {np.mean(rewards)} \n"
    del model
print(output_string)
