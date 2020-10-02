import math
import numpy as np
import gym
from copy import deepcopy
import random
from gym import spaces, logger, error, utils
from gym.utils import seeding
import os

from gym_epidemic.envs.sir_multi.InterventionSIR import *
from gym_epidemic.envs.sir_multi.parameters import *
import gym_epidemic.envs.sir_multi.optimize_interventions as oi


class EnvSIRMultiMorris(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, tau=7, intervention='fc', t_sim_max = 7, budget=8,
            plotting=False, inits=inits_default):
        #Instatiating the covid sir object that handles solving
        self.inits = inits
        self.covid_sir = InterventionSIR(b_func = Intervention(),
                                         R0 = R0_default,
                                         gamma = gamma_default,
                                         inits = self.inits)
        self.covid_sir.reset()
        self.t_sim_max = t_sim_max
        self.intervention = intervention
        self.tau = tau
        self.weeks_intervened = 0
        self.budget = budget
        self.trajectory = []
        self.t = 0
        self.plotting = plotting
        
        assert self.intervention in ['fc'], f"{self.intervention} Invalid intervention input"
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=10**2, shape=(3,), dtype=np.float64)


    def step(self, action):
        done = False
        # Agent performs action on weeks basis for duration of 1 year
        if self.t >= 51:
            done = True
        if self.weeks_intervened > self.budget:
            action[1] = 1

        if self.intervention == 'fc':
            # From the action space, action[0]*360 will notify intention to intervene, 
            # action[1] will be reduction in transmissibility
            assert action in self.action_space, f"Error: {action} Invalid action"
            # If action[0] < 0.5 then we intervene, otherwise we don't reduce transmissibility
            if action[0] < 0.5:
                self.weeks_intervened += 1
            else:
                action[1] = 1
            self.covid_sir.b_func = make_fixed_b_func(self.tau, self.covid_sir.time, action[1])

        self.covid_sir.integrate((self.t + 1) * 7)
        addendum = np.column_stack((self.covid_sir.time_ts, self.covid_sir.state_ts))
        obs = np.concatenate((self.covid_sir.state[:2], np.array([self.covid_sir.R0])))
        # Reward here is set to minimize peak in infectious
        reward = -max(self.covid_sir.state_ts[:, 1])
        # We do a soft reset here, which means that we don't change time or state, just clear state_ts
        # and time_ts for memory purposes
        self.covid_sir.reset(hard=False)
        # To avoid memory problems when training, we do not store the entire trajectory in self.trajectory
        # as with multiple workers.
        if self.plotting:
            self.trajectory.append(addendum)
        self.t += 1
        return obs, reward, done, {}
    
    def reset(self):
        # Here we do a hard reset, so set time=0 and reset to initial values 
        self.covid_sir.reset()
        self.weeks_intervened = 0
        self.t = 0
        return np.concatenate((self.covid_sir.state[:2], np.array([self.covid_sir.R0])))
    

    def render(self, mode='human'):
        pass
        

    def close(self):
        pass








