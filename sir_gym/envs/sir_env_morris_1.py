import math
import numpy as np
import gym
from copy import deepcopy
import random
from gym import spaces, logger, error, utils
from gym.utils import seeding

from sir_gym.envs.InterventionSIR import *
from sir_gym.envs.parameters import *


class SIREnvMorris1(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.covid_sir = InterventionSIR(b_func = Intervention(), R0 = R0_default, gamma = gamma_default, inits = inits_default)
        self.covid_sir.reset()
        
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float64)
        

    def step(self, action):
        # From the action space, action[0] will be the duration, action[1] will be the start time,
        # action[2] will be b(t)
        t_sim_max = 360
        tau = action[0] * t_sim_max
        t_1 = action[1] * t_sim_max
        self.covid_sir.b_func = make_fixed_b_func(tau, t_1, action[2])
        self.covid_sir.integrate(t_sim_max)
        return self.covid_sir.state, np.exp(-self.covid_sir.get_I_max(True)* (1 + 10 * (action[0] > 7 / 45))), True, {}
    
    def reset(self):
        self.covid_sir.reset()
        return self.covid_sir.state
        

    def render(self, mode='human'):
        pass
        

    def close(self):
        pass








