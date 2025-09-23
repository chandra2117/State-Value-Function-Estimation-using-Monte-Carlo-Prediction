# State-Value-Function-Estimation-using-Monte-Carlo-Prediction

## Name : CHANDRAPRIYADHARSHINI C
## Reg.No : 212223240019

## AIM:
To implement and analyze a Monte Carlo (MC) prediction method for estimating the value function in a reinforcement learning environment.

## Problem statement:
Develop an algorithm to estimate the state value function for the Frozen Lake Problem using Monte Carlo Prediction. Compare the results with the dynamic programming results.

## Steps:
1. Define the environment and policy.  
2. Generate multiple episodes by following the policy.  
3. For each state, calculate the return (sum of discounted rewards) after the first occurrence.  
4. Average the returns across multiple episodes to estimate the value function.  
5. Repeat the process until the values converge.  
6. Display the estimated value function.

## Program:
```
# Name : CHANDRAPRIYADHARSHINI C
# Registor. no : 212223240019

#import libraries
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);

#define the policy
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

#define the state value function
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

#define the probability success
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

#define mean return
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

#create a environment
env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)

P

init_state

#Policy environment
pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

#check the probability
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,mean_return(env, pi_frozenlake)))

#define the policy evaluation
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[s] - v))
            V[s] = v

        if delta < theta:
            break

    return V

#evalute the environment as policy
V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)

from tqdm import tqdm

#define the decay schedule
def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2,log_base=10):
  decay_steps=int(max_steps * decay_ratio)
  rem_steps=max_steps-decay_steps

  values=np.logspace(log_start,0,decay_steps,base=log_base,endpoint=True)[::-1]
  values=(values-values.min())/(values.max()-values.min())
  values=(init_value-min_value)*values+min_value
  values=np.pad(values,(0,rem_steps),'edge')
  return values

#generate the trajectory(collection of experience)
from itertools import count
def generate_trajectory (pi, env, max_steps=20):
  done, trajectory = False, []
  while not done:
    state = env.reset()
    for t in count():
      action = pi (state)
      next_state, reward, done,_=env.step(action)
      experience=(state, action, reward,next_state,done)
      trajectory.append(experience)
      if done:
        break
      if t>=max_steps-1:
        trajectory=[]
        break
      state=next_state
  return np.array(trajectory,object)

#define the Monte control Prediction
def mc_prediction(pi,env,gamma=1.0,init_alpha=0.5,min_alpha=0.01,
                  alpha_decay_ratio=0.3,
                  n_episodes=50000,max_steps=100,first_visit=True):
  nS=env.observation_space.n
  discounts=np.logspace(0,max_steps,num=max_steps,base=gamma,endpoint=False)
  alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
  V=np.zeros(nS)
  V_track=np.zeros((n_episodes,nS))
  for e in tqdm(range(n_episodes),leave=False):
    trajectory=generate_trajectory(pi,env,max_steps)
    visited=np.zeros(nS,dtype=np.bool)
    for t, (state, _, reward, _, _) in enumerate(trajectory):
      if visited[state] and first_visit:
        continue
      visited[state]=True
      n_steps=len(trajectory[t:])
      G=np.sum(discounts[:n_steps]*trajectory[t:,2])
      V[state]=V[state]+alphas[e]*(G-V[state])
    V_track[e]=V
  return V.copy(),V_track

v_mc, v2_mc = mc_prediction(pi_frozenlake, env)

print("Name :  CHANDRAPRIYADHARSHINI C")
print("Reg.No: 212223240019")
print()
print_state_value_function(v_mc, P, n_cols=4, prec=5)

print(v2_mc)

#compare v-mc and V1
print("Name :  CHANDRAPRIYADHARSHINI C")
print("Reg.No: 212223240019")
print()
if np.sum(v_mc > V1) == 11:
    print("\nThe first policy is the better policy.")
elif np.sum(V1 > v_mc) == 11:
    print("\nYour policy is the better policy.")
else:
    print("\nBoth policies have their merits.")

```

## Output:

<img width="727" height="197" alt="image" src="https://github.com/user-attachments/assets/8102abeb-b951-4266-ad55-b882f9af2395" />

<img width="780" height="132" alt="image" src="https://github.com/user-attachments/assets/dd282b5c-cef3-4197-a2cc-6d6f3985f716" />


## Result:
The program successfully implements Monte Carlo prediction.
