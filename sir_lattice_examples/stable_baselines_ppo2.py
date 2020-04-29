import os
import sys
sys.path.append(os.path.realpath('..'))

import gym
import sir_lattice_gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpLnLstmPolicy, MlpLstmPolicy, LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

class CustomLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[400, 'lstm', dict(vf=[400, 400], pi=[400])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

env = make_vec_env('sir_lattice-v0', n_envs=1)
model = PPO2(CustomLstmPolicy, env, verbose=2, nminibatches=1, ent_coef=0.01, learning_rate=0.001)
model.learn(total_timesteps=int(5e4))
model.save("sir_lattice_0")
