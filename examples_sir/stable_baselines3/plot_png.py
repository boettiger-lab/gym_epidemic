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
    args = parser.parse_args()

    y1 = [[] for i in range(10)]
    y2 = [[] for i in range(10)]
    y3 = [[] for i in range(10)]
    y4 = [[] for i in range(10)]
    y5 = [[] for i in range(10)]
    y6 = [[] for i in range(10)]
    for i in range(5):
        model = SAC.load(f"models/sb3_sac_sir_{args.i}_{i}")    
        rewards = []
        R_space = np.linspace(2, 4, 10)
        for j, R0 in enumerate(R_space):
            for k in range(10):
                env = gym.make('sir-v0', intervention=args.i)
                env.covid_sir.random_obs = False 
                env.covid_sir.random_params = False
                env.covid_sir.R0 = R0
                obs = env.reset()
                action, states = model.predict(obs)
                obs, reward, dones, info = env.step(action)
                if k == 0:
                    _x, _y, t_i, sigma, f = env.compare_peak()
                # Adding the corresponding actions for each environment
                y1[j].append(t_i)
                y2[j].append(action[0] * 360)
                if args.i in ["fc", "o"]:
                    y3[j].append(sigma)
                    y4[j].append(action[1])
                    if args.i == "o":
                        y5[j].append(f)
                        y6[j].append(action[2])
                rewards.append(reward)

    # Splitting up plotting by intervention method
    if args.i == "fs":
        fig, axs = plt.subplots(1, 1, figsize=(15,10))
        axs.scatter(R_space, np.mean(y1, axis=1), s=70, color='b', label="Optimal")
        axs.scatter(R_space, np.mean(y2, axis=1), s=70, color='r', label="SAC Agent")
        axs.set_xlabel("R0", fontsize=24)
        axs.set_ylabel("Intervention Time", fontsize=24)
        p_bar_low = np.mean(y2, axis=1) - np.percentile(y2, 25, axis=1)
        p_bar_high = np.percentile(y2, 75, axis=1) - np.mean(y2, axis=1)
        axs.errorbar(R_space, np.mean(y2, axis=1), yerr=np.vstack([p_bar_low, p_bar_high]), fmt='none', ecolor='r', elinewidth=3)
        axs.legend(prop={'size': 24})
        plt.savefig("fs_intervention.png")

    elif args.i == "fc":
        fig, axs = plt.subplots(1, 1, figsize=(15,10))
        axs.scatter(R_space, np.mean(y1, axis=1), s=70, color='b', label="Optimal")
        axs.scatter(R_space, np.mean(y2, axis=1), s=70, color='r', label="SAC Agent")
        axs.set_xlabel("R0", fontsize=24)
        axs.set_ylabel("Intervention Time", fontsize=24)
        p_bar_low = np.mean(y2, axis=1) - np.percentile(y2, 25, axis=1)
        p_bar_high = np.percentile(y2, 75, axis=1) - np.mean(y2, axis=1)
        axs.errorbar(R_space, np.mean(y2, axis=1), yerr=np.vstack([p_bar_low, p_bar_high]), fmt='none', ecolor='r', elinewidth=3)
        axs.legend(prop={'size': 24})
        plt.savefig("fc_intervention.png")
        

        fig, axs = plt.subplots(1, 1, figsize=(15,10))
        axs.scatter(R_space, np.mean(y3, axis=1), s=70, color='b', label="Optimal")
        axs.scatter(R_space, np.mean(y4, axis=1), s=70, color='r', label="SAC Agent")
        axs.set_xlabel("R0", fontsize=24)
        axs.set_ylabel("Sigma", fontsize=24)
        p_bar_low = np.mean(y4, axis=1) - np.percentile(y4, 25, axis=1)
        p_bar_high = np.percentile(y4, 75, axis=1) - np.mean(y4, axis=1)
        axs.errorbar(R_space, np.mean(y4, axis=1), yerr=np.vstack([p_bar_low, p_bar_high]), fmt='none', ecolor='r', elinewidth=3)
        axs.legend(prop={'size': 24})
        plt.savefig("fc_sigma.png")



