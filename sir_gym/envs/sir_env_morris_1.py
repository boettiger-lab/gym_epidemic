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

    def __init__(self, tau=56, intervention='fc', t_sim_max = 360):
        self.covid_sir = InterventionSIR(b_func = Intervention(),\
                                         R0 = R0_default, \
                                         gamma = gamma_default,\
                                         inits = inits_default)
        self.covid_sir.reset()
        self.t_sim_max = t_sim_max
        self.intervention = intervention
        self.tau = tau
        # Here I allow for the different intervention types discussed in the Morris et al. paper
        assert self.intervention in ['o', 'fc', 'fs'], "Invalid intervention input"
        if self.intervention == 'o':
            self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float64)
        elif self.intervention == 'fc':
            self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float64)
        elif self.intervention == 'fs':
            self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float64)
        

    def step(self, action):
        
        if self.intervention == 'fc':
            # From the action space, action[0] will be the start time, 
            # action[1] will be reduction in transmissibility
            t_1 = action[0] * self.t_sim_max
            self.covid_sir.b_func = make_fixed_b_func(self.tau, t_1, action[1])
            self.covid_sir.integrate(self.t_sim_max)
        
        elif self.intervention == 'fs':
            # From the action space, action[0] will be the start time
            t_1 = action[0] * self.t_sim_max
            self.covid_sir.b_func = make_fixed_b_func(self.tau, t_1, 0)
            self.covid_sir.integrate(self.t_sim_max)
        
        elif self.intervention == 'o':
            # From the action space, action[0] will be the start time,
            # action[1] will be the fraction of time spent in phase 1,
            # action[2] will be the reduction in transmissibility in phase 1
            # action[3] will be the reduction in transmissibility in phase 2
            t_1 = action[0] * self.t_sim_max
            self.covid_sir.b_func = make_2phase_b_func(self.tau,
                                                        t_1,
                                                        action[1],
                                                        action[2],
                                                        action[3])
            self.covid_sir.integrate(self.t_sim_max)
        
        return self.covid_sir.state, np.exp(-self.covid_sir.get_I_max(True)), True, {}
    
    def reset(self):
        self.covid_sir.reset()
        return self.covid_sir.state
        

    def render(self, mode='human'):
        pass
        

    def close(self):
        pass








