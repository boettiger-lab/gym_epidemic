import gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import sir_gym
import numpy as np
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=0, help="Argument for what training run to examine")
parser.add_argument("-i", type=str, default="fs", help="Argument for what method of intervention to examine")
args = parser.parse_args()

model = SAC.load(f"trash/sb3_sac_sir_reward{args.n}")    

# Making lists so I can average over 10 trajectories
y1 = [[] for i in range(10)]
y2 = [[] for i in range(10)]
y3 = [[] for i in range(10)]
y4 = [[] for i in range(10)]
y5 = [[] for i in range(10)]
y6 = [[] for i in range(10)]
rewards = []
R_space = np.linspace(2, 4, 30)
# Plotting a snippet of Intervention time
for i in range(10):
    for R0 in R_space:
        env = gym.make('sir-v3', intervention=args.i)
        env.covid_sir.random_obs = False 
        env.covid_sir.random_params = False
        env.covid_sir.R0 = R0
        obs = env.reset()
        action, states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        _, _, t_i, sigma, f = env.compare_peak()
        # Adding the corresponding actions for each environment
        y1[i].append(t_i)
        y2[i].append(action * 360)
        if args.i in ["fc", "o"]:
            y3[i].append(sigma)
            y4[i].append(action[1])
            if args.i == "o":
                y5[i].append(f)
                y6[i].append(action[2])
        rewards.append(reward)

# Splitting up plotting by intervention method
if args.i == "fs":
    fig, axs = plt.subplots(1, 2, figsize=(15,10))
    axs[0].scatter(R_space, np.mean(y1, axis=0), color='b')
    axs[0].scatter(R_space, np.mean(y2, axis=0), color='r')
    axs[0].set_title("Mean Intervention Time vs. R0")
    axs[1].scatter(R_space, np.std(y2, axis=0), color='r')
    axs[1].set_title("Intervention Time Std vs. R0")

elif args.i == "fc":
    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    axs[0, 0].scatter(R_space, np.mean(y1, axis=0), color='b')
    axs[0, 0].scatter(R_space, np.mean(y2, axis=0), color='r')
    axs[0, 0].set_title("Mean Intervention Time vs. R0")
    axs[0, 1].scatter(R_space, np.std(y2, axis=0), color='r')
    axs[0, 1].set_title("Intervention Time Std vs. R0")
    axs[1, 0].scatter(R_space, np.mean(y3, axis=0), color='b')
    axs[1, 0].scatter(R_space, np.mean(y4, axis=0), color='r')
    axs[1, 0].set_title("Mean Sigma vs. R0")
    axs[1, 1].scatter(R_space, np.std(y4, axis=0), color='r')
    axs[1, 1].set_title("Sigma Std vs. R0")

elif args.i == "o":
    fig, axs= plt.subplot(3, 2, figsize=(15, 10))
    axs[0, 0].scatter(R_space, np.mean(y1, axis=0), color='b')
    axs[0, 0].scatter(R_space, np.mean(y2, axis=0), color='r')
    axs[0, 0].set_title("Mean Intervention Time vs. R0")
    axs[0, 1].scatter(R_space, np.std(y2, axis=0), color='r')
    axs[0, 1].set_title("Intervention Time Std vs. R0")
    axs[1, 0].scatter(R_space, np.mean(y3, axis=0), color='b')
    axs[1, 0].scatter(R_space, np.mean(y4, axis=0), color='r')
    axs[1, 0].set_title("Mean Sigma vs. R0")
    axs[1, 1].scatter(R_space, np.std(y4, axis=0), color='r')
    axs[1, 1].set_title("Sigma Std vs. R0")
    axs[2, 0].scatter(R_space, np.mean(y5, axis=0), color='b')
    axs[2, 0].scatter(R_space, np.mean(y6, axis=0), color='r')
    axs[2, 0].set_title("Mean F vs. R0")
    axs[2, 1].scatter(R_space, np.std(y6, axis=0), color='r')
    axs[2, 1].set_title("F Std vs. R0")

plt.savefig("trash.png")



