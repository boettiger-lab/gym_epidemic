# Epidemic Control Gym

## Repository Info
This repo contains RL environments that model epidemic control problems. Additionally in `examples_sir`, there are models and scripts to recreate the results in "One-Shot Epidemic Control with Soft Actor Critic". 

We have the following environments:

- `sir-v0` is based on the [Morris et al. paper](https://github.com/dylanhmorris/optimal-sir-intervention). There are different modes of intervention (specifically, full suppression and fixed control), which have different action spaces; but generally, the agent inputs a real-valued vector that specifies the non-pharmaceutical intervention -- e.g. when to intervene and how severely the agent should reduce transmission. The observation space is a real-valued vector that reports S, I and R0.
- `sir_multi-v0` is also based on the  [Morris et al. paper](https://github.com/dylanhmorris/optimal-sir-intervention). But here, instead of allowing for one-shot intervention as in `sir-v0`, the agent has the ability to intervene on a weekly basis. The agent can decide how severely to reduce transmission. The agent has an intervention budget of 8 weeks.

## Installation Instructions

The environments can be installed and run from the command line via:
`pip install gym_epidemic`

If you are cloning this repo from github; from the current directory run:
`pip install -e .`

