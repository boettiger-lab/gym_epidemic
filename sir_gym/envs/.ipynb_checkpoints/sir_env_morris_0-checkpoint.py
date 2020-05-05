import math
import numpy as np
import gym
from copy import deepcopy
import random
from gym import spaces, logger, error, utils
from gym.utils import seeding

from sir_gym.envs.InterventionSIR import *
from sir_gym.envs.parameters import *


class SIREnvMorris(gym.Env):
    metadata = {'render.modes':['human']}
    # I stopped working on this ENV; Probably should delete.
    def __init__(self):
        self.covid_sir = InterventionSIR(b_func = Intervention(), R0 = R0_default, gamma = gamma_default, inits = inits_default)
        self.covid_sir.reset()
        
        self.action_space = spaces.Box(low=0.1, high=1, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float64)
        

    def step(self, action):
        self.covid_sir.b_func = make_fixed_b_func(150, 0, action[0])
        t_sim_max = 360
        self.covid_sir.integrate(t_sim_max)
        return self.covid_sir.state, -self.covid_sir.get_I_max(True), True, {}

    def reset(self):
        self.covid_sir.reset()
        return self.covid_sir.state
        

    def render(self, mode='human'):
        pass
        

    def close(self):
        pass








