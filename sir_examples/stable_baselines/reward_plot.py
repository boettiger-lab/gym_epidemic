import gym
import os
import sys
sys.path.append(os.path.realpath('../..'))
import sir_gym
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

env = gym.make('sir-v3', intervention='fs')

model = SAC.load("sb3_sac_final")
y1 = []
y2 = []
rewards =[]
for i in np.linspace(0.01, 0.99, 500):
    env = gym.make('sir-v3', intervention='fs')
    obs = env.reset()
    obs, reward, dones, info = env.step(np.array([i]))
    _, _, t_i = env.compare_peak()
    y1.append(t_i)
    y2.append(i * 360)
    rewards.append(10*reward - 7.4)

plt.plot(np.linspace(0.01, 0.99, 500), rewards, color='b')
plt.xlabel("Action")
plt.ylabel("Reward")
plt.title("Reward Structure of One Shot Intervention")
plt.savefig("trash.png")
