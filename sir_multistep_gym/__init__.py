from gym.envs.registration import register

register(
    id='sir-v0',
    entry_point='sir_multistep_gym.envs:SIREnvMorris2',
)

