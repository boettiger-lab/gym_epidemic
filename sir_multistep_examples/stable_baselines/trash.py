import gym
import os
import sys
sys.path.append(os.path.realpath('../..'))
import sir_multistep_gym
import numpy as np
import matplotlib.pyplot as plt
import time
t0 = time.time()

env = gym.make('sir-v0', intervention='fc', plotting=True)
obs = env.reset()
flag = False
x = 1
rewards = 0
while True:
    if obs[1] > 0.07:
        action = np.array([0, x])
        flag = True
    else:
        action = np.array([1, 1])
    if flag:
        action = np.array([0, x])
    obs, reward, dones, info = env.step(action)
    if dones:
        break
    rewards += reward
trajs = np.reshape(np.array(env.trajectory), (-1, 4))
pois = []
for i in trajs:
    if i[0] % 7 == 0:
        pois.append(i)
    if i[2] > 0.29:
        break
import pdb; pdb.set_trace()
np.savetxt("trash_pois.csv", pois, delimiter=",", header="t,S,I,R")
np.savetxt("trash.csv", np.column_stack((trajs[:, 0], trajs[:, 2])), delimiter=",", header="t,I")
print(rewards)
