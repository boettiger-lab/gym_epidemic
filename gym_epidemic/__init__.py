from gym.envs.registration import register

register(
    id='sir-v0',
    entry_point='gym_epidemic.envs.sir_single:EnvSIRMorris',
)

register(
    id='sir_multi-v0',
    entry_point='gym_epidemic.envs.sir_multi:EnvSIRMultiMorris',
)

