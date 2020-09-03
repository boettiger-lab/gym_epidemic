import gym
import os
import sys
sys.path.append(os.path.realpath('../..'))
import sir_gym
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt


model = SAC.load("sb3_sac_o")
y1 = [[] for i in range(10)]
y2 = [[] for i in range(10)]
y3 = [[] for i in range(10)]
y4 = [[] for i in range(10)]
y5 = [[] for i in range(10)]
y6 = [[] for i in range(10)]
for i in range(10):
    for j, R0 in enumerate(np.linspace(2, 4, 10)):
        env = gym.make('sir-v3', intervention='o')
        env.covid_sir.R0 = R0
        obs = env.reset()
        action, states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        _, _, t_i, sigma, f_t = env.compare_peak()
        y1[j].append(t_i)
        y2[j].append(action[0] * 360)
        y3[j].append(sigma)
        y4[j].append(action[1])
        y5[j].append(f_t)
        y6[j].append(action[2])

#plt.scatter(np.linspace(2, 4, 10), np.array(y1).mean(axis=1), color='b', label="Analytical Solution")
#plt.scatter(np.linspace(2, 4, 10), np.array(y2).mean(axis=1), color='r', label="SAC Agent")
#plt.scatter(np.linspace(2, 4, 10), np.array(y3).mean(axis=1), color='g', label="Analytical Solution")
#plt.scatter(np.linspace(2, 4, 10), np.array(y4).mean(axis=1), color='y', label='SAC Agent')
plt.scatter(np.linspace(2, 4, 10), np.array(y5).mean(axis=1), color='orange', label='Analaytical Sol')
plt.scatter(np.linspace(2, 4, 10), np.array(y6).mean(axis=1), color='indigo', label='SAC Agent')
plt.xlabel("R0")
plt.ylabel("F_t")
plt.title("Average F_t of a SAC Agent")
plt.legend()
plt.savefig("trash.png")
