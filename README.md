## Environments

We have the following environments:

- `sir_gym` is based on the [Morris et al. paper](https://github.com/dylanhmorris/optimal-sir-intervention). There are different modes of intervention (specifically, full suppression, fixed control and maintain then suppress), which have different action spaces; but generally, the agent inputs a real-valued vector that specifies the non-pharmaceutical intervention. The observation space is a real-valued vector that reports S, I and R0.
- `sir_lattice` has a MultiDiscrete action space (i.e. an n-dimensional vector). For each individual, the agent can decide to test this person -- 1 denotes a test, 0 otherwise. The agent takes in a MultiDiscrete observation: 0 for a susceptible, 1 for an infectious and 2 for recovered.

We have done some exploration of these RL Environments using Stable Baselines with SAC and PPO2. Some hyperparameter tuning has been performed using Optuna. 
