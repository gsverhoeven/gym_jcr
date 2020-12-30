from gym.envs.registration import register

from .jcr_env import JacksCarRentalEnv

environments = [['JacksCarRentalEnv', 'v0']]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='gym_jcr:{}'.format(environment[0]),
    )

