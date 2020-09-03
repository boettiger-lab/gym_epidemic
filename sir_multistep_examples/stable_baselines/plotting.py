import gym
import os
import sys
sys.path.append(os.path.realpath('../..'))
import sir_multistep_gym
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import time
t0 = time.time()

env = gym.make('sir-v0', intervention='fc')

model = SAC.load("sb3_sac_o")
env = gym.make('sir-v0', intervention='fc')
obs = env.reset()

while True:
    action, states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    if dones:
        break
traj = np.reshape(env.trajectory, (-1, 4))
plt.plot(traj[:, 0], traj[:, 2])
plt.savefig("trash.png")
