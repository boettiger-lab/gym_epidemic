import gym
import os
import sys
sys.path.append(os.path.realpath('../..'))
import sir_gym
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import time
t0 = time.time()

env = gym.make('sir-v3', intervention='fs')

model = SAC.load("sb3_sac_final_5e7")
y1 = []
y2 = []
for i in range(100):
    env = gym.make('sir-v3', intervention='fs', random_obs=True, random_params=True)
    obs = env.reset()
    action, states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    _, _, t_i, sigma, f_t = env.compare_peak()
    y1.append(t_i)
    y2.append((action * 360)[0])
print(((np.array(y2) - np.array(y1))**2).mean(axis=None))

model = SAC.load("sb3_sac_final")
y1 = []
y2 = []
for i in range(100):
    env = gym.make('sir-v3', intervention='fs', random_obs=True, random_params=True)
    obs = env.reset()
    action, states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    _, _, t_i, sigma, f_t = env.compare_peak()
    y1.append(t_i)
    y2.append((action[0] * 360))
print(((np.array(y2) - np.array(y1))**2).mean(axis=None))
t1 = time.time()
print(t1 - t0)
