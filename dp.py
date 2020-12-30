import numpy as np
import copy

def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    
    for a in range(env.nA):
        for ns in range(len(env.P[s][a])):
            prob, next_state, reward, done = env.P[s][a][ns]
            q[a] += prob * (reward + gamma * V[next_state])
    return q

def policy_evaluation(env, policy, gamma=1, theta=1e-8):

    V = np.zeros(env.nS)
    delta = np.ones(env.nS)

    while np.max(delta) > theta:
        print("e.", end = '')
        for s in range(env.nS):
            V_s = 0
            for a in range(env.nA):
                for ns in range(len(env.P[s][a])):
                    prob, next_state, reward, done = env.P[s][a][ns]
                    V_s +=  policy[s][a] * prob * (reward + gamma * V[next_state])
            delta[s] = np.abs(V[s] - V_s)
            V[s] = V_s
        #print(V)

    return V

def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    
    # policy improvement: process of making a new policy that improves on an original policy,
    # by making it greedy with respect to the value function of the original policy
    for s in range(env.nS):
        q_s = q_from_v(env, V, s, gamma)
        # all actions that have max q
        sel_vec = (q_s == np.max(q_s))
        # give all max q actions equal prob of being selected
        policy[s, sel_vec] = 1/sum(sel_vec)
        
    return policy

def policy_iteration(env, gamma=1, theta=1e-8):
    policy = np.ones([env.nS, env.nA]) / env.nA    
    policy_stable = False
    
    while policy_stable == False:
        print("E.")
        # policy evaluation
        V = policy_evaluation(env, policy, gamma, theta)
        print("I.")
        # policy improvement
        improved_policy = policy_improvement(env, V, gamma)

        # check if policy is stable
        if np.array_equal(policy, improved_policy):
            policy_stable = True
        else:
            policy = copy.copy(improved_policy)
    
    return policy, V

def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    q = np.zeros(env.nA)
    i = 0
    
    while True:
        i += 1
        V_old = copy.copy(V)
        #print("V.", end = '')
        for s in range(env.nS):
            for a in range(env.nA):
                q[a] = 0
                for ns in range(len(env.P[s][a])): # ns is next state
                    prob, next_state, reward, done = env.P[s][a][ns]
                    q[a] +=  prob * (reward + gamma * V[next_state])
            V[s] = max(q)
        """
        Note to self:  delta = max(delta, V_old - V[s]) is the exact version of the algo
        this only updates delta if the current diff is bigger than the prev ones
        
        here i went for a vector wise eval of all deltas in one sweep
        """
        
        delta = max(abs(V - V_old))
        print('Iteration {}: delta is {}'.format(i, delta))
        if delta < theta:
            break
            
    policy = policy_improvement(env, V, gamma)
    
    return policy, V

