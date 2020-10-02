import gym
import gym_epidemic
import numpy as np
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

if __name__ == "__main__":
    y1 = []
    y2 = []
    rewards =[]
    action_space = np.linspace(0.01, 0.99, 100)
    t_i, sigma = np.meshgrid(action_space, action_space)
    for t_i in action_space:
        for sigma in action_space:
            env = gym.make('sir-v0', intervention='fc')
            obs = env.reset()
            obs, reward, dones, info = env.step(np.array([t_i, sigma]))
            y1.append(t_i)
            y2.append(sigma)
            rewards.append(reward)

    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(360*np.array(y1).reshape(100, -1), np.array(y2).reshape(100, -1), np.array(rewards).reshape(100, -1), levels=25)
    fig.colorbar(cp)
    ax.set_xlabel("Intervention Time")
    ax.set_ylabel("Sigma")
    plt.savefig("reward_plot_2d_fc.png")
