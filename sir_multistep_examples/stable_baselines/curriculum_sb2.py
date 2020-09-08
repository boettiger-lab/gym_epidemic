import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
from gym import spaces
import numpy as np
import sir_multistep_gym
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
import time
a = time.time()

pois = np.loadtxt("trash_pois.csv", delimiter=",")

def make_env(env_id, rank, seed=0, inits_loc=-1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, inits=pois[inits_loc, 1:])
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=1000, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[64, 'lstm', dict(vf=[64, 64], pi=[64, 64])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

if __name__=="__main__":
    env = SubprocVecEnv([make_env("sir-v0", i) for i in range(25)])
    model = PPO2(CustomLSTMPolicy, env, verbose=2, ent_coef=0.08, nminibatches=25)
    for i in range(-2, -17, -1):
        model.learn(total_timesteps=int(2e5), log_interval=int(1e1))
        env = SubprocVecEnv([make_env("sir-v0", j, inits_loc=i) for j in range(25)])
        model.set_env(env)
    model.save("sb2_ppo2_curriculum_15")
    b = time.time()
    print(b-a)

