import math
import numpy as np
import gym
from copy import deepcopy
import random
from gym import spaces, logger, error, utils
from gym.utils import seeding

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
        pass

    def close(self):
        pass







