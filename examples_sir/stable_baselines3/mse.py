import gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import gym_epidemic
import numpy as np
import argparse 
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, default="fs", help="Argument for what method of intervention to examine")
    parser.add_argument("-s", type=int, default=0, help="Set seed")
    args = parser.parse_args()

    np.random.seed(args.s)
    y1 = []
    y2 = []
    for i in range(5):
        model = SAC.load(f"models/sb3_sac_sir_{args.i}_{i}")    
        rewards = []
        # Plotting a snippet of Intervention time
        for k in range(2000):
            env = gym.make('sir-v0', intervention=args.i)
            env.covid_sir.random_obs = True 
            env.covid_sir.random_params = True
            obs = env.reset()
            action, states = model.predict(obs)
            obs, reward, dones, info = env.step(action)
            _x, _y, t_i, sigma, f = env.compare_peak()
            # Adding the corresponding actions for each environment
            y1.append([t_i, action[0] * 360])
            if args.i == "fc":
                y2.append([sigma, action[1]])
            rewards.append(reward)

    y1, y2 = np.array(y1), np.array(y2)
    print("t_i mse and std:")
    print(((y1[:, 0] - y1[:, 1])**2).mean())
    print(((y1[:, 0] - y1[:, 1])**2).std())
    if args.i == "fc":
        print("\n\nsigma mse and std:")
        print(((y2[:, 0] - y2[:, 1])**2).mean())
        print(((y2[:, 0] - y2[:, 1])**2).std())


