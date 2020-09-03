import gym
import os
import sys
sys.path.append(os.path.realpath('../..'))
import sir_gym
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

env = gym.make('sir-v3', intervention='fs')

model = SAC.load("sb3_meta")
y1 = [[] for i in range(30)]
y2 = [[] for i in range(30)]
for i in range(100):
    for j, R0 in enumerate(np.linspace(2, 4, 30)):
        env = gym.make('sir-v3', intervention='fs')
        env.covid_sir.R0 = R0
        obs = env.reset()
        action, states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        _, _, t_i = env.compare_peak()
        y1[j].append(t_i)
        y2[j].append(action * 360)

plt.scatter(np.linspace(2, 4, 30), np.array(y1).mean(axis=1), color='b', label="Analytical Solution")
plt.scatter(np.linspace(2, 4, 30), np.array(y2).mean(axis=1), color='r', label="SAC Agent")
plt.xlabel("R0")
plt.ylabel("Intervention Time")
plt.title("Average Intervention Times of a SAC Agent")
plt.legend()
plt.savefig("trash.png")
