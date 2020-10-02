#!/usr/bin/env python3

####################################################
# filename: InterventionSIR.py
# author: Dylan Morris <dhmorris@princeton.edu>
#
# description: specifies the SIR models
# with and without or peak-minimizing
# intervention according to our
# model of intervention functions b(t)
####################################################

import numpy as np 
from scipy.integrate import odeint

class InterventionSIR():
    """
    class for handling our SIR
    model with interventions
    """

    def __init__(self,
                 b_func = None,
                 R0 = None,
                 gamma = None,
                 inits = None
    ):
        if b_func is None:
            b_func = Intervention()
        self.b_func = b_func
        self.R0 = R0
        self.gamma = gamma
        self.inits = inits
        self.random_obs = False
        self.random_params = False
        self.reset()
        
    def reset(self):
        if self.random_obs:
            I = np.random.uniform(0, 1e-6)
            R = np.random.uniform(0, 1e-6) # np.random.uniform(0,0)
            S = 1 - I - R
            self.inits = np.array([S, I, R])
        if self.random_params:
            self.R0 = np.random.uniform(2, 4)
            #self.gamma = np.random.beta(1,10)
        self.state = self.inits
        self.time = 0
        self.time_ts = np.array([])
        self.state_ts = np.array([[], [], []]).reshape((-1, 3))

    def deriv(self, state, time):
        S, I, R = state
        beta = self.R0 * self.gamma
        b = self.b_func(time, beta, self.gamma, S, I)
        
        dS = -b * beta * S * I
        dI = b * beta * S * I - self.gamma * I
        dR = self.gamma * I

        return np.array([dS, dI, dR])

    def deriv_null(self, state, time):
        S, I, R = state
        beta = self.R0 * self.gamma
        dS = -beta * S * I
        dI = beta * S * I - self.gamma * I
        dR = self.gamma * I
        return np.array([dS, dI, dR])
    
    def integrate(self, final_time, fineness = 10000):
        times = np.linspace(self.time,
                            final_time,
                            fineness)
        results = odeint(self.deriv, self.state, times)
        self.state = results[-1]
        self.time = final_time
        self.time_ts = np.concatenate([self.time_ts,
                                       times])
        self.state_ts = np.concatenate([self.state_ts,
                                        results])

        return (times, results)

    def integrate_null(self, final_time, fineness = 10000):
        times = np.linspace(self.time,
                            final_time,
                            fineness)
        results = odeint(self.deriv_null, self.state, times)
        return (times, results)
    
    def I_max_SI(self, S_x, I_x):
        """
        get the maximum value of I(t) 
        in the window from t s.t. S = S_x,
        I = I_x to t = infinity
        """
        return (S_x + I_x - 
                (1/self.R0) * np.log(S_x) - 
                (1/self.R0) + 
                (1/self.R0) * np.log(1/self.R0))

    def I_of_S(self, S):
        S0, I0, Rec0 = self.inits
        return (I0 + S0 - (1/self.R0) * np.log(S0) -
                S + (1/self.R0) * np.log(S))

    def t_of_S(self, S_target):
        S0, I0, Rec0 = self.inits
        if np.isnan(S_target):
            raise ValueError("Cannot find time "
                             "for non-numeric/nan S\n\n"
                             "check that S is being "
                             "calculated correctly")
        def deriv(t, S_val):
            I = self.I_of_S(S_val)
            return -1 / (self.R0 * self.gamma * S_val * I)
        return odeint(deriv, 0, np.linspace(S0, S_target, 2))[-1]

    def get_I_max(self,
                  allow_boundary_max = False):
        last_timestep_error = (
            "Max at last timestep. "
            "You likely need to "
            "increase integration "
            "max time. If this was expected, "
            "set allow_boundary_max = True")
        fist_timestep_error = (
            "Max at first timestep. "
            "Your model may be misspecified. "
            "If this was expected, "
            "set allow_boundary_max = True")
        ## check that we didn't get a boundary soln
        wheremax = np.argmax(self.state_ts[:, 1])
        if not allow_boundary_max:
            if wheremax == self.state_ts[:, 1].size: 
                raise ValueError(last_timestep_error)
            elif wheremax == 0:
                raise ValueError(first_timestep_error)
        return self.state_ts[wheremax, 1]
    
    def get_t_peak(self):
        return self.t_of_S(1 / self.R0)

    def __repr__(self):
        return ("InterventionSIR with R0 = {}, "
                "gamma = {}, and an intervention "
                "function {}".format(
                    self.R0,
                    self.gamma,
                    self.b_func))



class Intervention():
    """
    class for defining intervention
    functions b(t)
    """

    def __init__(self,
                 tau = None,
                 t_i = None,
                 sigma = None,
                 sigma_1 = None,
                 f = None,
                 S_i_expected = None,
                 I_i_expected = None,
                 strategy = None):
        self.tau = tau
        self.t_i = t_i
        self.sigma = sigma
        self.sigma_1 = sigma_1
        self.f = f
        self.S_i_expected = S_i_expected
        self.I_i_expected = I_i_expected
        self.strategy = strategy
    
        self.repertoire = {
            "fixed": self.fixed_b,
            #"mc-time": self.maintain_contain_time,
            "mc-state": self.maintain_contain_state,
            "full-suppression": self.fixed_b}

    def __call__(self,
                 time,
                 beta,
                 gamma,
                 S,
                 I):
        return self.repertoire[self.strategy](
            time,
            beta,
            gamma,
            S,
            I)

    def fixed_b(self,
                time,
                beta,
                gamma,
                S,
                I):
        """
        Fixed intervention of strictness
        sigma
        """
        if time >= self.t_i and time < self.t_i + self.tau:
            result = self.sigma
        else:
            result = 1
        return result


    def maintain_contain_state(self,
                               time,
                               beta,
                               gamma,
                               S,
                               I):
        """
        Variable maintain/contain
        intervention tuned by 
        current state of the system
        (S(t), I(t))
        """
        if time >= self.t_i and time < self.t_i + self.tau * self.f:
            result = self.sigma
        elif (time >= self.t_i + self.tau * self.f and
              time < self.t_i + self.tau):
            result = self.sigma_1
        else:
            result = 1
        return result
        


def make_fixed_b_func(tau, t_i, sigma):
    """
    create a function to execute
    the fixed intervention
    with parameters t_i, tau and sigma 
    """
    return Intervention(
        tau = tau,
        t_i = t_i,
        sigma = sigma,
        strategy = "fixed")

def make_2phase_b_func(tau, t_i, f, sigma, sigma_1):
    """
    create a function to execute a 2 phase fixed intervention
    with parameters, t_i, tau, f, sigma and sigma_1
    """
    return Intervention(
        tau = tau,
        t_i = t_i,
        f = f,
        sigma = sigma,
        sigma_1 = sigma_1,
        strategy ='mc-state')


