from gym.envs.registration import register

register(
    id='sir-v0',
    entry_point='gym_sir.envs.single:EnvSIRMorris',
)

register(
    id='sir_multi-v0',
    entry_point='gym_sir.envs.multi:EnvSIRMultiMorris',
)

