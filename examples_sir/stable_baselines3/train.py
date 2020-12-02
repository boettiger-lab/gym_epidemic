import gym
import gym_epidemic
import gym_fishing
import numpy as np

import torch as th

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import argparse

ALGOS = {"sac": SAC, "ppo":PPO}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algo", default="sac", type=str,
                        help="Specify which of the algorithms to use")
    parser.add_argument("-i", "--intervention", default="fs", type=str,
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
        env = gym.make(args.env, intervention=args.intervention, random_params=True, random_obs=True)
        env = Monitor(env, "./")
        policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[64, 64], log_std_init=-0.9)
        model = ALGOS[args.algo]("MlpPolicy", 
                                 env, 
                                 verbose=args.verbose,
                                 gamma=0.99,
                                 learning_rate=0.00005,
                                 batch_size=8,
                                 n_steps=16,
                                 ent_coef=0.06,
                                 clip_range=0.3,
                                 n_epochs=20,
                                 gae_lambda=0.95,
                                 max_grad_norm=0.6,
                                 vf_coef=0.1,
                                 sde_sample_freq=64,
                                 policy_kwargs=policy_kwargs)

        model.learn(total_timesteps=args.n_timesteps, log_interval=args.log)
        model.save(f"{args.algo}_{args.intervention}_{i}")
        del model
