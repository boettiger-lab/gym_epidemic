import math
import numpy as np
import gym
from copy import deepcopy
import random
from gym import spaces, logger, error, utils
from gym.utils import seeding
import os

from gym_epidemic.envs.sir_single.optimal_intervention import Intervention as I
from gym_epidemic.envs.sir_single.InterventionSIR import *
from gym_epidemic.envs.sir_single.parameters import *
import gym_epidemic.envs.sir_single.optimize_interventions as oi


class EnvSIRMorris(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, tau=56, intervention='fs', t_sim_max = 360, random_obs = False, random_params = False, reward_scale=.1):
        self.covid_sir = InterventionSIR(b_func = Intervention(),
                                         R0 = R0_default,
                                         gamma = gamma_default,
                                         inits = inits_default)
        self.covid_sir.random_params = random_params
        self.covid_sir.random_obs = random_obs
        self.covid_sir.reset()
        self.t_sim_max = t_sim_max
        self.intervention = intervention
        self.tau = tau
        self.reward_scale = reward_scale
        # Here I allow for the different intervention types discussed in the Morris et al. paper
        # o - optimal intervention/maintain then suppress - note that this is not the true analog as
        #     we do not allow for a dynamic transmission reduction
        # fc - fixed control
        # fs - fixed suppression
        assert self.intervention in ['o', 'fc', 'fs'], f"{self.intervention} Invalid intervention input"
        if self.intervention == 'fc':
            self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float64)
        elif self.intervention == 'fs':
            self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64)
        
        self.observation_space = spaces.Box(low=0, high=10**2, shape=(3,), dtype=np.float64)
        

    def step(self, action):
        
        if self.intervention == 'fc':
            # From the action space, action[0]*360 will be the start time, 
            # action[1] will be reduction in transmissibility
            assert action in self.action_space, f"Error: {action} Invalid action"
            t_1 = action[0] * self.t_sim_max
            self.covid_sir.b_func = make_fixed_b_func(self.tau, t_1, action[1])

        
        elif self.intervention == 'fs':
            # From the action space, action[0]*360 will be the start time
            # Occasionally, with stable baselines I've noticed that it selects an action
            # that is a very small negative number, not sure why this is, but in this case,
            # I clip the action.
            action = np.clip(action, 0, 1)
            assert action in self.action_space, f"Error: {action} Invalid action"
            t_1 = action[0] * self.t_sim_max
            self.covid_sir.b_func = make_fixed_b_func(self.tau, t_1, 0)

        
        self.covid_sir.integrate(self.t_sim_max)
        state = np.concatenate((self.covid_sir.state[:2], np.array([self.covid_sir.R0])))
        return state, np.clip(self.reward_scale / self.covid_sir.get_I_max(True), 0, 2), True, {}
    
    def reset(self):
        self.covid_sir.reset()
        return np.concatenate((self.covid_sir.state[:2], np.array([self.covid_sir.R0])))
    
    def compare_peak(self):
        """
        This returns 2 numpy arrays: first one being the analytical result,
        second being that from the env.
        Both arrays have sub-arrays of the form [t, S, I, R].
        """
        y = np.column_stack((self.covid_sir.time_ts, self.covid_sir.state_ts))
        
        
        covid_sir = InterventionSIR(
            b_func = I(),
            R0 = self.covid_sir.R0,
            gamma = self.covid_sir.gamma,
            inits = self.covid_sir.inits)
        
        covid_sir.reset()
        
        covid_sir.b_func.tau = self.tau
        S_i_expected = 0
    
        if self.intervention == "fc":
            covid_sir.b_func.strategy = "fixed"
            S_i_expected, sigma = oi.calc_Sb_opt(
                    covid_sir.R0,
                    covid_sir.gamma,
                    self.tau)
            covid_sir.b_func.sigma = sigma
            f = None
                
        elif self.intervention == "fs":
            covid_sir.b_func.strategy = "full-suppression"
            S_i_expected = oi.calc_S_var_opt(
                    covid_sir.R0,
                    covid_sir.gamma * self.tau,
                    0)
            covid_sir.b_func.sigma = 0
            sigma = None
            f = None
        t_i_opt = covid_sir.t_of_S(S_i_expected)[0]
        covid_sir.b_func.t_i = t_i_opt
            
        covid_sir.integrate(self.t_sim_max)
        anal = np.column_stack((covid_sir.time_ts, covid_sir.state_ts))
        return anal, y, t_i_opt, sigma, f
        
        

    def render(self, mode='human'):
        pass
        

    def close(self):
        pass








