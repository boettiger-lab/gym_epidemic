import gym
import gym_epidemic
import numpy as np

import torch as th

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import argparse

ALGOS = {"sac": SAC, "ppo":PPO}

hyper_fs = {'batch_size': 64, 'buffer_size': int(1e6), 'ent_coef': 0.005, 
            'gamma': 0.995, 'learning_starts': 0, 
            'log_std_init': 0.18854, 'learning_rate': 0.008247999, 
            'tau': 0.01, 'train_freq': 8}
hyper_fc = {'batch_size': 1000, 'buffer_size': int(1e4), 'ent_coef': 1e-3, 
            'gamma': 0.9999, 'learning_starts': 1000, 
            'log_std_init': -2.133345, 'learning_rate': 1e-4, 
            'tau': 0.005, 'train_freq': 512}
hyper = {'fs': hyper_fs, 'fc': hyper_fc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algo", default="sac", type=str,
                        help="Specify which of the algorithms to use")
    parser.add_argument("-i", default="fs", type=str,
                        help="Select Intervention type")
    parser.add_argument("-v", "--verbose", default=2, type=int,
                        help="Verbosity option when training")
    parser.add_argument("-n", "--n_timesteps", default=int(2e6), type=int,
                        help="Number of timesteps used for training")
    parser.add_argument("-e", "--env", default="sir-v0", type=str,
                        help="Select which environment to use")
    parser.add_argument("-l", "--log", default=1000000, type=int,
                        help="Specify the logging interval when trainig")
    parser.add_argument("-r", "--reps", default=5, type=int,
                        help="Specify the number of reps for training.")
    args = parser.parse_args()
    for i in range(args.reps):
        env = gym.make(args.env, intervention=args.i, random_params=True, random_obs=True)
        env = Monitor(env, "./")
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256], 
                             log_std_init=hyper[args.i]['log_std_init'])
        model = ALGOS[args.algo]("MlpPolicy", 
                                 env, 
                                 verbose=args.verbose,
                                 gamma=hyper[args.i]['gamma'],
                                 learning_rate=hyper[args.i]['learning_rate'],
                                 batch_size=hyper[args.i]['batch_size'],
                                 ent_coef=hyper[args.i]['ent_coef'],
                                 tau=hyper[args.i]['tau'],
                                 train_freq=hyper[args.i]['train_freq'],
                                 policy_kwargs=policy_kwargs)

        model.learn(total_timesteps=args.n_timesteps, log_interval=args.log)
        model.save(f"models/{args.algo}_{args.i}_{i}")
        del model
