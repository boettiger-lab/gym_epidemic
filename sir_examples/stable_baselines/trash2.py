import gym
import os
import sys
sys.path.append(os.path.realpath('../..'))
import sir_gym
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

env = gym.make('sir-v3', intervention='fs')

model = SAC.load("sb3_tuned_randomparams_0")
y1 = []
y2 = []
trajs = {}
for R0 in [2, 3, 4]:
    env = gym.make('sir-v3', intervention='fs')
    env.covid_sir.R0 = R0
    I = 9e-7 
    R = 1e-6
    S = 1 - I - R
    env.covid_sir.inits = np.array([S, I, R])
    obs = env.reset()
    action, states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    anal, emp, t_i = env.compare_peak()
    y1.append(t_i)
    y2.append(action * 360)
    trajs[R0] = [anal, emp]
y = []
for R0 in np.linspace(2, 4.1, 100):
    env = gym.make("sir-v3", intervention='fs')
    env.covid_sir.R0 = R0
    I = 9e-7
    R = 1e-6
    S = 1 - I - R
    env.covid_sir.inits = np.array([S, I, R])
    obs = env.reset()
    action, states = model.predict(obs)
    obs, reward, dones, _ = env.step(action)
    _, __, t_i = env.compare_peak()
    y.append(t_i)

cs = CubicSpline(np.linspace(2, 4, 3), np.concatenate(y2, axis=0))
fig, axs = plt.subplots(2, 2)

axs[0, 0].scatter([2, 3, 4], y1, color='b')
axs[0, 0].scatter([2, 3, 4], y2, color='r')
axs[0, 0].plot(np.linspace(1.5, 4.5, 50), cs(np.linspace(1.5, 4.5, 50)), color='r')
axs[0, 0].plot(np.linspace(2, 4.1, 100), y, color='b')
axs[0, 1].plot(trajs[2][0][:, 0], trajs[2][0][:, 2], color='b')
axs[0, 1].plot(trajs[2][1][:, 0], trajs[2][1][:, 2], color='r')
axs[1, 0].plot(trajs[3][0][:, 0], trajs[3][0][:, 2], color='b')
axs[1, 0].plot(trajs[3][1][:, 0], trajs[3][1][:, 2], color='r')
# axs[1, 1].plot(trajs[4][0][:, 0], trajs[4][0][:, 2], color='b')
# axs[1, 1].plot(trajs[4][1][:, 0], trajs[4][1][:, 2], color='r')

env.covid_sir.R0 = 2.5
obs = env.reset()
import pdb;pdb.set_trace()
obs, reward, dones, _ = env.step(np.array([cs(2.5)/360]))
anal, emp, t_i = env.compare_peak()
axs[1,1].plot(anal[:, 0], anal[:, 2], color='b')
axs[1, 1].plot(emp[:, 0], emp[:, 2], color='r')

plt.savefig("trash.png")
