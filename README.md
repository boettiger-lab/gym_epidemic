## Environments

We have the following environments:

- `sir_gym` is based on the [Morris et al. paper](https://github.com/dylanhmorris/optimal-sir-intervention). There are different modes of intervention (specifically, full suppression, fixed control and maintain then suppress), which have different action spaces; but generally, the agent inputs a real-valued vector that specifies the non-pharmaceutical intervention. The observation space is a real-valued vector that reports S, I and R0. The current working gym is `sir_env_morris_2.py`. In `sir_examples/stable_baselines`, `sb_sac_v3.py` is working example training script using stable-baselines3.
- `sir_lattice_gym` has a MultiDiscrete action space (i.e. an n-dimensional vector). For each individual, the agent can decide to test this person -- 1 denotes a test, 0 otherwise. The agent takes in a MultiDiscrete observation: 0 for a susceptible, 1 for an infectious and 2 for recovered.
- `sir_multistep_gym` is also based on [Morris et al. paper](https://github.com/dylanhmorris/optimal-sir-intervention). But here, instead of allowing for one-shot intervention as in `sir_gym`, the agent has the ability to intervene on a weekly basis. The agent can set the factor to reduce beta. The agent has an intervention budget of 8 weeks.

We have done some exploration of these RL Environments using Stable Baselines with SAC and PPO2. Some hyperparameter tuning has been performed using Optuna. 
