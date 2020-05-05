import math
import numpy as np
import gym
from copy import deepcopy
import random
from gym import spaces, logger, error, utils
from gym.utils import seeding
import os

from sir_gym.envs.InterventionSIR import *
from sir_gym.envs.parameters import *


class SIREnvMorris1(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, tau=56, intervention='fc', t_sim_max = 360):
        self.covid_sir = InterventionSIR(b_func = Intervention(),
                                         R0 = R0_default,
                                         gamma = gamma_default,
                                         inits = inits_default)
        self.covid_sir.reset()
        self.t_sim_max = t_sim_max
        self.intervention = intervention
        self.tau = tau
        # Here I allow for the different intervention types discussed in the Morris et al. paper
        # o - optimal intervention/maintain then suppress
        # fc - fixed control
        # fs - fixed suppression
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
            assert action in self.action_space, "Error: Invalid action"
            t_1 = action[0] * self.t_sim_max
            self.covid_sir.b_func = make_fixed_b_func(self.tau, t_1, action[1])
            self.covid_sir.integrate(self.t_sim_max)
        
        elif self.intervention == 'fs':
            # From the action space, action[0] will be the start time
            assert action in self.action_space, "Error: Invalid action"
            t_1 = action[0] * self.t_sim_max
            self.covid_sir.b_func = make_fixed_b_func(self.tau, t_1, 0)
            self.covid_sir.integrate(self.t_sim_max)
        
        elif self.intervention == 'o':
            # From the action space, action[0] will be the start time,
            # action[1] will be the fraction of time spent in phase 1,
            # action[2] will be the reduction in transmissibility in phase 1
            # action[3] will be the reduction in transmissibility in phase 2
            assert action in self.action_space, "Error: Invalid action"
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
    
    def compare_peak(self):
        """
        This returns 2 numpy arrays: first one being the analytical result, second being that from the env
        Both arrays are of the form [t, S, I, R]
        """
        name_map = {"o":"mc-time", "fc":"fixed", "fs":"full-suppression"}
        # TODO: Make sure you get path so this can run regardless of directory
        # I am presuming to be in the examples directory
        print(os.getcwd())
        a_result = np.loadtxt(f"../sir_gym/envs/analytical_results/{name_map[self.intervention]}_{self.tau}.csv",
                             delimiter=',')
        return a_result, np.column_stack((self.covid_sir.time_ts, self.covid_sir.state_ts))
        
        

    def render(self, mode='human'):
        pass
        

    def close(self):
        pass








