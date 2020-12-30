# jacks_car_rental_env

This repository contains a Gym Environment that implements Jack's Car Rental Problem from the book "Reinforcement Learning" by Sutton & Barto.

The environment is suitable for Dynamic Programming, as it exposes the full one-step-ahead MDP dynamics.

See for more information the accompanying blog post at https://gsverhoeven.github.io

## Installation
```bash
git clone https://github.com/gsverhoeven/jacks_car_rental_env
cd jacks_car_rental_env
pip install .
```

In your gym environment:

```python
import gym_jcr
env = gym.make("JacksCarRentalEnv-v0") 
```

