from gym.envs.registration import register

register(
    id='sir_lattice-v0',
    entry_point='sir_lattice_gym.envs:SIRLatticeEnv',
)


