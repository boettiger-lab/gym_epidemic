from gym.envs.registration import register

register(
    id='sir_multi-v0',
    entry_point='gym_sir_multistep.envs:EnvSIRMultiMorris',
)

