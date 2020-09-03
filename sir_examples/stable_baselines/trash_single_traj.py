import os
import sys
sys.path.append(os.path.realpath('../..'))
import gym
import sir_gym
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

model = SAC.load("sb3_sac_final")
env = gym.make('sir-v3', intervention='fs')
obs = env.reset()
action, states = model.predict(obs)
obs, reward, dones, info = env.step(action)
x, y, t_i, sigma, f_t = env.compare_peak()
plt.plot(x[:, 0], x[:, 2], label="Optimal")
plt.plot(y[:, 0], y[:, 2], label="SAC")
plt.xlabel("t")
plt.ylabel("I")
plt.legend()
plt.savefig("trash.png")
