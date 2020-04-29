import math
import numpy as np
import gym
from copy import deepcopy
import random
from gym import spaces, logger, error, utils
from gym.utils import seeding
import bokeh.plotting
import bokeh.io
bokeh.io.output_notebook()

class SIREnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.N = 1
        self.S = .9
        self.I = .1
        self.beta = 0.3
        self.gamma = 0.1
        self.t = 0
        self.history = [[self.S, self.I]]
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(low=0, high=1, shape=(2,))

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        if action == 1:
            reduction = 0.2
        else:
            reduction = 1
        
        p = 1 - np.exp(-reduction * self.beta * self.I)
        dN_SI = np.random.normal(self.S * p , self.S * p * (1 - p))
        dN_SI = dN_SI * (dN_SI > 0)
        p = 1 - np.exp(-self.gamma)
        dN_IR = np.random.normal(self.I * p , self.I * p * (1 - p))
        dN_IR = dN_IR * (dN_IR > 0)

        self.S -= dN_SI
        self.I += dN_SI - dN_IR

        self.t += 1
        self.history.append([self.S, self.I])

        done = bool(self.t < 199)

        return np.array([self.S, self.I]), -self.I, done, {}

    def reset(self):
        self.S = 0.9
        self.I = 0.1
        self.t = 0
        self.history = [[self.S, self.I]]

        return np.array([self.S, self.I])

    def render(self, mode='human'):
        plot = bokeh.plotting.figure(plot_width=600,
                               plot_height=400,
                               x_axis_label='time')
        colors = ['blue', 'red', 'orange']
        labels = ['Susceptible', 'Infected', 'Recovered']
        for i in range(2):
            plot.line(range(len(self.history)), [item[i] for item in self.history], color=colors[i], legend=labels[i])
        plot.line(range(len(self.history)), [self.N - sum(item) for item in self.history], color=colors[2], legend=labels[2])
        return bokeh.plotting.show(plot)

    def close(self):
        pass







