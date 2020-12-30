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
        
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
               
        # prob, next_state, reward, done
        for s in range(nS):
            # need a state vec to extract correct probs from Ptrans
            state_vec = np.zeros(nS)
            state_vec[s] = 1
            for a in range(nA):
                prob_vec = np.dot(Ptrans[:,:,a], state_vec)
                li = P[s][a]
                # add rewards for all transitions
                for ns in range(nS):
                    li.append((prob_vec[ns], ns, R[s][a], False))

        
        # obtain one-step dynamics for dynamic programming setting
        self.P = P
        
        # isd initial state dist 
        isd = np.ones(nS)/nS
        
        self.P = P
       
        super(JacksCarRentalEnv, self).__init__(nS, nA, P, isd)


