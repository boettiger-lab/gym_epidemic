from sir_gym.envs.InterventionSIR import InterventionSIR
import sir_gym.envs.parameters as params
import sir_gym.envs.optimize_interventions as oi
import numpy as np

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

if __name__ == "main":
    covid_sir = InterventionSIR(
            b_func = Intervention(),
            R0 = params.R0_default,
            gamma = params.gamma_default,
            inits = params.inits_default)
    
    covid_sir.reset()
    
    taus = params.taus_figure_interventions
    
    t_sim_max = 360
    null_time, null_result = covid_sir.integrate_null(t_sim_max)
    
    for name in ["mc-time", "fixed", "full-suppression"]:
        solids = [] # save solid b lines for plotting later
    
        ## set intervention strategy
        covid_sir.b_func.strategy = name
    
        ## iterate over intervention durations
        for i_tau, tau in enumerate(taus):
            covid_sir.b_func.tau = tau
            S_i_expected = 0
            print("optimizing strategy for {} "
                  "with tau = {}".format(name, tau))
            if name == "mc-time":
                S_i_expected, f = oi.calc_Sf_opt(
                    covid_sir.R0,
                    covid_sir.gamma * tau)
                I_i_expected = covid_sir.I_of_S(S_i_expected)
                covid_sir.b_func.S_i_expected = S_i_expected
                covid_sir.b_func.I_i_expected = I_i_expected
                covid_sir.b_func.f = f
    
            elif name == "fixed":
                S_i_expected, sigma = oi.calc_Sb_opt(
                    covid_sir.R0,
                    covid_sir.gamma,
                    tau)
                covid_sir.b_func.sigma = sigma
                
            elif name == "full-suppression":
                S_i_expected = oi.calc_S_var_opt(
                    covid_sir.R0,
                    covid_sir.gamma * tau,
                    0)
                covid_sir.b_func.sigma = 0
            
            t_i_opt = covid_sir.t_of_S(S_i_expected)[0]
            covid_sir.b_func.t_i = t_i_opt
            
            covid_sir.reset()
            covid_sir.integrate(t_sim_max)
            output = np.column_stack((covid_sir.time_ts, covid_sir.state_ts))
            np.savetxt(f"./analytical_results/{name}_{tau}.csv", output, delimiter=",")
        
