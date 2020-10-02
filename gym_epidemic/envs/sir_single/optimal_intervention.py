from gym_epidemic.envs.sir_single.InterventionSIR import InterventionSIR
import gym_epidemic.envs.sir_single.parameters as params
import gym_epidemic.envs.sir_single.optimize_interventions as oi
import numpy as np

####################################################
# filename: optimal_intervention.py 
#
# description: defines the Intervention class which
# is modified from InterventionSIR.py. There is a 
# slight modification that allows for the maintain-
# suppress method to find the optimal solution.
####################################################

class Intervention():
    """
    class for defining intervention
    functions b(t)
    """

    def __init__(self,
                 tau = None,
                 t_i = None,
                 sigma = None,
                 f = None,
                 S_i_expected = None,
                 I_i_expected = None,
                 strategy = None):
        self.tau = tau
        self.t_i = t_i
        self.sigma = sigma
        self.f = f
        self.S_i_expected = S_i_expected
        self.I_i_expected = I_i_expected
        self.strategy = strategy
    
        self.repertoire = {
            "fixed": self.fixed_b,
            "mc-time": self.maintain_contain_time,
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

    def maintain_contain_time(self,
                              time,
                              beta,
                              gamma,
                              S,
                              I):
        """
        Variable maintain/contain
        intervention tuned by 
        current time
        """
        if time >= self.t_i and time < self.t_i + self.tau * self.f:
            S_expected = (self.S_i_expected -
                          gamma * (time - self.t_i) *
                          self.I_i_expected)
            result = gamma / (beta * S_expected)
        elif (time >= self.t_i + self.tau * self.f and
              time < self.t_i + self.tau):
            result = 0
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
            result = gamma / (beta * S)
        elif (time >= self.t_i + self.tau * self.f and
              time < self.t_i + self.tau):
            result = 0
        else:
            result = 1
        return result

