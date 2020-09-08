import os
import sys
sys.path.append(os.path.realpath('../..'))

import gym
from gym import spaces
import numpy as np
import sir_multistep_gym
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=400, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[64, 'lstm', dict(vf=[64, 64], pi=[64, 64])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

for i in range(1):
    env = make_vec_env('sir-v0', n_envs=50)
    model = PPO2(CustomLSTMPolicy, env, verbose=2, ent_coef=0.5, nminibatches=50)
    model.learn(total_timesteps=int(2e5), log_interval=int(1e2))
    model.save(f"sb2_ppo2_recurrent")

