import math
import numpy as np
import gym
from copy import deepcopy
import random
from gym import spaces, logger, error, utils
from gym.utils import seeding
import os

from sir_gym.envs.analytical_results import Intervention as I
from sir_gym.envs.InterventionSIR import *
from sir_gym.envs.parameters import *
import sir_gym.envs.optimize_interventions as oi


class SIREnvMorris2(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, tau=56, intervention='fc', t_sim_max = 360):
        self.covid_sir = InterventionSIR(b_func = Intervention(),
                                         R0 = R0_default,
                                         gamma = gamma_default,
                                         inits = inits_default)
        self.covid_sir.random = True
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

        
        elif self.intervention == 'fs':
            # From the action space, action[0] will be the start time
            assert action in self.action_space, "Error: Invalid action"
            t_1 = action[0] * self.t_sim_max
            self.covid_sir.b_func = make_fixed_b_func(self.tau, t_1, 0)

        
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
        I = np.random.beta(1,10**4)
        R = np.random.beta(1,10**8)
        S = 1 - I - R
        return np.array([S, I, R])
    
    def compare_peak(self):
        """
        This returns 2 numpy arrays: first one being the analytical result,
        second being that from the env.
        Both arrays have sub-arrays of the form [t, S, I, R].
        """
        y = np.column_stack((self.covid_sir.time_ts, self.covid_sir.state_ts))
        
        
        covid_sir = InterventionSIR(
            b_func = I(),
            R0 = R0_default,
            gamma = gamma_default,
            inits = self.covid_sir.inits)
        
        covid_sir.reset()
        
        covid_sir.b_func.tau = self.tau
        S_i_expected = 0
        
        if self.intervention == "o":
            covid_sir.b_func.strategy = "mc-time"
            S_i_expected, f = oi.calc_Sf_opt(
                covid_sir.R0,
                covid_sir.gamma * self.tau)
            I_i_expected = covid_sir.I_of_S(S_i_expected)
            covid_sir.b_func.S_i_expected = S_i_expected
            covid_sir.b_func.I_i_expected = I_i_expected
            covid_sir.b_func.f = f
    
        elif self.intervention == "fc":
            covid_sir.b_func.strategy = "fixed"
            S_i_expected, sigma = oi.calc_Sb_opt(
                    covid_sir.R0,
                    covid_sir.gamma,
                    tau)
            covid_sir.b_func.sigma = sigma
                
        elif self.intervention == "fs":
            covid_sir.b_func.strategy = "full-suppression"
            S_i_expected = oi.calc_S_var_opt(
                    covid_sir.R0,
                    covid_sir.gamma * self.tau,
                    0)
            covid_sir.b_func.sigma = 0
            
        t_i_opt = covid_sir.t_of_S(S_i_expected)[0]
        covid_sir.b_func.t_i = t_i_opt
            
        covid_sir.integrate(self.t_sim_max)
        anal = np.column_stack((covid_sir.time_ts, covid_sir.state_ts))
        return anal, y
        
        

    def render(self, mode='human'):
        pass
        

    def close(self):
        pass








