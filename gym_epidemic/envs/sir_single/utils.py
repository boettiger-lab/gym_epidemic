def get_action(env, action):
    action = (action + 1) / 2
    action[0] *= env.t_sim_max
    return action
