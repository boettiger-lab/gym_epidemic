from gym.envs.registration import register

register(
    id='sir-v0',
    entry_point='sir_gym.envs:SIREnv',
)

register(
    id='sir-v1',
    entry_point='sir_gym.envs:SIREnvMorris',
)

register(
    id='sir-v2',
    entry_point='sir_gym.envs:SIREnvMorris1',
)
