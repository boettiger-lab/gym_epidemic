import gym
import os
import sys
sys.path.append(os.path.realpath('../..'))
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import sir_gym
import numpy as np

model = SAC.load("models/sb3_sac_auto_ent_0")    
y1 = []
y2 = []
rewards = []
for R0 in np.linspace(2, 10, 50):
    env = gym.make('sir-v3', intervention='fs')
    env.covid_sir.random_obs = False 
    env.covid_sir.random_params = False
    env.covid_sir.R0 = R0
    obs = env.reset()
    action, states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    _, _, t_i = env.compare_peak()
    y1.append(t_i)
    y2.append(action * 360)
    rewards.append(reward)
print(np.mean(rewards))
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(np.linspace(2, 10, 50), y1, color='b')
ax1.scatter(np.linspace(2, 10, 50), y2, color='r')

env = gym.make("sir-v3", intervention='fs')
obs = env.reset()
print(obs)
env.covid_sir.R0 = 10
action, states = model.predict(obs)
obs, reward, dones, info = env.step(action)
x, y, t = env.compare_peak()
ax2.plot(x[:,0], x[:,2], color='blue')
ax2.plot(y[:,0], y[:,2], color='red')
plt.savefig("trash_random_obs_params.png")
print(np.exp(-np.max(x[:,2])), np.exp(-np.max(y[:,2])), reward)
