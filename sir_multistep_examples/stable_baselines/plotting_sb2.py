import gym
import os
import sys
sys.path.append(os.path.realpath('../..'))
import sir_multistep_gym
import numpy as np
from stable_baselines import PPO2
import matplotlib.pyplot as plt
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
import time

t0 = time.time()
pois = np.loadtxt("trash_pois.csv", delimiter=",")

def make_env(env_id, rank, seed=0, inits_loc=-10, plotting=True):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, inits=pois[inits_loc, 1:], plotting=plotting)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__=='__main__':
    env = SubprocVecEnv([make_env("sir-v0", i) for i in range(25)])
    model = PPO2.load("sb2_ppo2_curriculum")
    m_traj = []
    for i in range(10):
        obs = env.reset()

        while True:
            action, states = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            if dones.all():
                break
        trajs = np.reshape(np.array(env.get_attr("trajectory")), (25, -1, 4))
        np.savetxt(f"trash{i}.csv", np.column_stack((trajs[0][:, 0], np.mean(trajs[:, :, 2], axis=0))), delimiter=",", header="t,I")
