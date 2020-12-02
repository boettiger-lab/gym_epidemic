def get_action(action, env):
    action = (action + 1) / 2
    action[0] *= env.t_sim_max
    return action
