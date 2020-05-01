## Environments

We have the following environments:

- SIR Lattice has a MultiDiscrete action space, where the agent decides what person/node to test. 0 for no test, 1 for test. The agent takes in a MultiDiscrete observation: 0 for a susceptible, 1 for an infectious and 2 for recovered.
- SIR is based on the [Morris et al. paper](https://github.com/dylanhmorris/optimal-sir-intervention). There are a few environments that have various action spaces but predominantly the agent is controlling beta, the parameter describing transmissibility. Observation space is a Box with an entry for S and I.

We have done some exploration of these RL Environments using Stable Baselines with SAC and PPO2. Some hyperparameter tuning has been performed using Optuna.
