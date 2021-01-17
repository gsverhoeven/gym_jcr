import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

from scipy.stats import poisson
from .jcr_mdp import *

MAX_CARS = 20 # maximum # of cars in each location
MAX_MOVE_OF_CARS = 5 # maximum # of cars to move during night

Ptrans = create_P_matrix()

R = create_R_matrix()

class JacksCarRentalEnv(discrete.DiscreteEnv):
    """
    Jack’s Car Rental 
    
    Jack manages two locations for a nationwide car rental company. 
    Each day, some number of customers arrive at each location to rent cars. 
    If Jack has a car available, he rents it out and is credited $10 by 
    the national company. If he is out of cars at that location, then the 
    business is lost. Cars become available for renting the day after they 
    are returned. To help ensure that cars are available where they are 
    needed, Jack can move them between the two locations overnight, at a cost 
    of $2 per car moved. We assume that the number of cars requested and 
    returned at each location are Poisson random variables. Suppose Lambda is
    3 and 4 for rental requests at the first and second locations and 
    3 and 2 for returns. 
    
    To simplify the problem slightly, we assume that there can be no more than
    20 cars at each location (any additional cars are returned to the 
    nationwide company, and thus disappear from the problem) and a maximum 
    of five cars can be moved from one location to the other in one night. 
    
    We take the discount rate to be gamma = 0.9 and formulate this as a 
    continuing finite MDP, where the time steps are days, the state is the 
    number of cars at each location at the end of the day, and the actions 
    are the net numbers of cars moved between the two locations overnight.
     
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        print("")

        # n.b. not all actions are availabe in all states 
        # i.e. we cannot  move 5 cars if there are less than 5 cars at location
        actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)
        nA = len(actions)
        
        # can be MAX_CARS at each location A, B
        nS = (MAX_CARS + 1)**2
        
        # obtain one-step dynamics for dynamic programming setting
        P = {
            s: {
                a: [
                    (Ptrans[next, s, a], next, R[s, a], False)
                    for next in range(nS)
                ]
                for a in range(nA)
            }
            for s in range(nS)
        }
       
        # isd initial state dist 
        isd = np.full(nS, 1 / nS)
              
        # This allows easier plotting of policy and value functions:
        # e.g. V.reshape(env.observation_shape) will give a 21x21 matrix
        observation_shape = (MAX_CARS + 1, MAX_CARS + 1)

        super(JacksCarRentalEnv, self).__init__(nS, nA, P, isd)

        # The following three elements enable this enviroment to be interfaced with
        # the dynamic programming algorithms of the doctrina library:
        # https://github.com/rhalbersma/doctrina/blob/master/src/doctrina/algorithms/dp.py
        # https://github.com/rhalbersma/doctrina/blob/master/exercises/dp-gym_jcr.ipynb

        # Equation (3.4) in Sutton & Barto (p.49):
        # p(s'|s, a) = probability of transition to state s', from state s taking action a.
        self.transition = Ptrans.transpose(1, 2, 0)
        assert np.isclose(self.transition.sum(axis=2), 1).all()
        
        # Equation (3.5) in Sutton & Barto (p.49):
        # r(s, a) = expected immediate reward from state s after action a.        
        self.reward = R

