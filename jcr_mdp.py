### Code License (MIT)
#
#Copyright 2018  Christian Herta, Patrick Baumann
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# IN THE SOFTWARE.

# External Modules
import numpy as np
from scipy.stats import poisson

REQUEST_RATE = (3., 4.)      
RETURN_RATE  = (3., 2.)

GAMMA = 0.9
RENTAL_INCOME = 10
TRANSFER_COST = 2
TRANSFER_MAX  = 5
MAX_CAPACITY  = 20
#FREE_PARKING_CAP = 10
#PARKING_COST = 4

# location indicies
A = 0
B = 1

MAX_PMF = 30

action_space = np.arange(-TRANSFER_MAX, TRANSFER_MAX+1)

def get_most_probable_location(state):
     i = state.argmax()
     return (i//(MAX_CAPACITY+1), i%(MAX_CAPACITY+1))

def get_state_vector(a, b):
    s = np.zeros((MAX_CAPACITY+1)**2)
    s[a*(MAX_CAPACITY+1)+b] = 1
    return s

def get_request_transitions_for_one_location(loc):
    """
    Construct transition matrix P_{to, from} for one location only for requests.
    The matrix has form (21, 21). 
  """
    assert(loc==A or loc==B)
    # transition matrix P_{to, from} for one location only requests
    transition_matrix = np.zeros([MAX_CAPACITY+1, MAX_CAPACITY+1])
    
    request_pmf = poisson.pmf(np.arange(MAX_PMF), REQUEST_RATE[loc])
    np.testing.assert_almost_equal(request_pmf[-1], 0., decimal=12)
    for i in range(MAX_CAPACITY+1):  
        for j in range(MAX_CAPACITY+1):  
            if j==0:
                transition_matrix[i,j] = request_pmf[i:].sum()
            elif j<=i:    
                transition_matrix[i,j] = request_pmf[i-j]             
    return transition_matrix.T

def full_transition_matrix_A(transition_one_loc):
    block_size = MAX_CAPACITY+1 # for convenience
    transition_matrix = np.zeros([block_size**2, block_size**2])
    for i in range(block_size):
        transition_matrix[i:block_size**2: block_size,
                          i:block_size**2: block_size] = transition_one_loc
    return transition_matrix

def full_transition_matrix_B(transition_one_loc):
    block_size = MAX_CAPACITY+1 # for convenience
    transition_matrix = np.zeros([block_size**2, block_size**2])
    for i in range(block_size):
        transition_matrix[i*block_size:(i*block_size)+block_size,
                          i*block_size:(i*block_size)+block_size] = transition_one_loc
    return transition_matrix

def get_return_transition_matrix_one_location(loc):
    """
    Construct transition matrix P_{to, from} for one location only for returns
  
    """
    assert(loc==0 or loc==1)
    transition_matrix = np.zeros([MAX_CAPACITY+1, MAX_CAPACITY+1])

    return_pmf = poisson.pmf(np.arange(MAX_PMF), RETURN_RATE[loc])
    np.testing.assert_almost_equal(return_pmf[-1], 0., decimal=12)
    for i in range(MAX_CAPACITY+1):  
        for j in range(MAX_CAPACITY+1):  
            if j==MAX_CAPACITY:
                transition_matrix[i,j] = return_pmf[j-i:].sum()
            elif j>=i and j<MAX_CAPACITY:    
                transition_matrix[i,j] = return_pmf[j-i]     
    return transition_matrix.T

def get_moves(a, b, action):
    if action > 0: # from A to B
        return min(a, action)
    else:
        return max(-b, action)

def get_nightly_moves():
    transition_matrix = np.zeros([(MAX_CAPACITY+1)**2, (MAX_CAPACITY+1)**2, 
                                  action_space.shape[0]])
    for a in range(MAX_CAPACITY+1):
        for b in range(MAX_CAPACITY+1):
            for i, action in enumerate(action_space):
                old_state_index = a*(MAX_CAPACITY+1)+b
                moves = get_moves(a, b, action)
                new_a = min(a - moves, MAX_CAPACITY)
                new_b = min(b + moves, MAX_CAPACITY)
                new_state_index = new_a *(MAX_CAPACITY+1) + new_b
                transition_matrix[new_state_index, old_state_index, i] = 1.
    return transition_matrix

def create_P_matrix():   
    P_request_A_one_loc = get_request_transitions_for_one_location(A)
    P_request_A = full_transition_matrix_A(P_request_A_one_loc)
    
    P_request_B_one_loc = get_request_transitions_for_one_location(B)
    P_request_B = full_transition_matrix_B(P_request_B_one_loc)
    
    P_request = np.dot(P_request_A, P_request_B)
    
    P_return_A_one_loc = get_return_transition_matrix_one_location(A)
    P_return_A = full_transition_matrix_A(P_return_A_one_loc)
    
    P_return_B_one_loc = get_return_transition_matrix_one_location(B)
    P_return_B = full_transition_matrix_B(P_return_B_one_loc)
    
    P_return = np.dot(P_return_B, P_return_A)
    
    P_return_request = np.dot(P_return, P_request)
    
    P_move = get_nightly_moves() 
    # this is the transpose of the state transition probability kernel
    P = np.ndarray(((MAX_CAPACITY+1)**2, (MAX_CAPACITY+1)**2, action_space.shape[0]))
    for i in range(action_space.shape[0]): # TODO: without a loop?
        P[:,:,i] = np.dot(P_return_request, P_move[:,:,i]) 
    return P

def create_R_matrix():
    # rename from get_reward()
    poisson_mask = np.zeros((2, MAX_CAPACITY+1, MAX_CAPACITY+1))
    po = (poisson.pmf(np.arange(MAX_CAPACITY+1), REQUEST_RATE[A]),
          poisson.pmf(np.arange(MAX_CAPACITY+1), REQUEST_RATE[B]))
    for loc in (A,B):
        for i in range(MAX_CAPACITY+1):
            poisson_mask[loc, i, :i] = po[loc][:i]
            poisson_mask[loc, i, i] = po[loc][i:].sum()
    # the poisson mask contains the probability distribution for renting x cars (x column) 
    # in each row j, with j the number of cars available at the location

    reward = np.zeros([MAX_CAPACITY+1, MAX_CAPACITY+1, 2*TRANSFER_MAX+1])
    for a in range(MAX_CAPACITY+1):
        for b in range(MAX_CAPACITY+1):
            for action in range(-TRANSFER_MAX, TRANSFER_MAX+1):
                moved_cars = min(action, a) if action>=0 else max(action, -b)
                a_ = a - moved_cars
                a_ = min(MAX_CAPACITY, max(0, a_))
                b_ = b + moved_cars
                b_ = min(MAX_CAPACITY, max(0, b_))
                reward_a = np.dot(poisson_mask[A, a_], np.arange(MAX_CAPACITY+1)) 
                reward_b = np.dot(poisson_mask[B, b_], np.arange(MAX_CAPACITY+1))     
                reward[a, b, action+TRANSFER_MAX] = ( 
                            (reward_a + reward_b) * RENTAL_INCOME -
                            np.abs(action) * TRANSFER_COST )
                #if a==20 and b==20 and action==0:
                #    print (a_,b_, action)
                #    print (reward_a, reward_b)
                #    print (reward[a, b, action+TRANSFER_MAX])
    reward = reward.reshape(441,11)    
    return reward


# Then we solve it with the modifications in Exercise 4.7:

# Write a program for policy iteration and re-solve Jack’s car rental problem
# with the following changes. One of Jack’s employees at the first location 
# rides a bus home each night and lives near the second location. She is happy
# to shuttle one car to the second location for free. Each additional car still
# costs $2, as do all cars moved in the other direction. In addition, Jack has 
# limited parking space at each location. If more than 10 cars are kept overnight 
# at a location (after any moving of cars), then an additional cost of $4 must 
# be incurred to use a second parking lot (independent of how many cars are kept
# there). These sorts of nonlinearities and arbitrary dynamics often occur in 
# real problems and cannot easily be handled by optimization methods other than
# dynamic programming. To check your program, first replicate the results given 
# for the original problem.