
import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import time
import brian2 as b2
from brian2tools import *

from operator import itemgetter

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

def action_to_str(action):
  return '{:+05.3f}'.format(action)

def to_str(actions):
  return [action_to_str(actions[0]), action_to_str(actions[2]) + " " + action_to_str(actions[3]), action_to_str(actions[1])]

def disp_norm(x):

  x = np.asarray(x)
  x = x.reshape(16,4)
  max_val = np.max( np.absolute(x) )
  x = x / max_val
  x = np.round(x, 3)

  for i in range(4):
    grid_str = ["", "", ""]
    for j in range(4):
      state = (4 - i - 1) * 4 + j
      state_str = to_str( x[state] )

      grid_str[0] += "         " + state_str[0]
      grid_str[1] += "   " + state_str[1]
      grid_str[2] += "         " + state_str[2]

    print (grid_str[0])
    print (grid_str[1])
    print (grid_str[2])
    print ("")

def disp(x):

  x = np.asarray(x)
  x = x.reshape(16,4)
  x = np.round(x, 3)

  for i in range(4):
    grid_str = ["", "", ""]
    for j in range(4):
      state = (4 - i - 1) * 4 + j
      state_str = to_str( x[state] )

      grid_str[0] += "         " + state_str[0]
      grid_str[1] += "   " + state_str[1]
      grid_str[2] += "         " + state_str[2]

    print (grid_str[0])
    print (grid_str[1])
    print (grid_str[2])
    print ("")

def num_to_state(state):
    ret = np.zeros(16)
    for i in range(16):
        if state == i:
            ret[i] = 1
    # return np.reshape(ret, 16)
    return np.reshape(ret, [1, 16])

def state_to_num(state):
    for i in range(16):
        if state[0][i]:
            return i

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

'''
C D E F
8 9 A B
4 5 6 7
0 1 2 3
'''

def manhatten(coord, dest):
    return abs(dest[0] - coord[0]) + abs(dest[1] - coord[1])

class Env():
    def __init__(self):
        self.state = [0, 0]
        self.steps = 0

        self.nrows = 4
        self.ncols = 4

        self.wins = [[3,3]]
        self.fails = [[2,2]]

        self.left = []
        self.right = []
        self.up = []
        self.down = []

        for ii in range(self.nrows):
            for jj in range(self.ncols):
                if ( ii == self.nrows-1 ):
                    self.up.append([ii, jj])

                if ( ii == 0 ):
                    self.down.append([ii, jj])

                if ( jj == self.ncols-1 ):
                    self.right.append([ii, jj])

                if ( jj == 0 ):
                    self.left.append([ii, jj])

        self.moves = np.zeros(shape=(self.nrows, self.ncols, 4))
        for ii in range(self.nrows):
            for jj in range(self.ncols):
                for kk in range(4):

                    next_state = self.next_state(kk, [ii, jj])
                    diff = manhatten([ii, jj], [3,3]) - manhatten(next_state, [3, 3])

                    # print (ii, jj, kk, diff)

                    if (diff > 0):
                        self.moves[ii][jj][kk] = 10
                    elif (diff < 0):
                        self.moves[ii][jj][kk] = -10
                    else:
                        self.moves[ii][jj][kk] = -5

    def reset(self):
        self.state = [0, 0]
        self.steps = 0
        return self.state_to_num(self.state)

    def state_to_num(self, state):
        return state[0] * self.ncols + state[1]

    def num_to_state(self, state):
        return [state/self.ncols, state%self.ncols]

    def next_state(self, action, current_state):
        next_state = current_state
        if (action == 0) and (current_state not in self.up):
            next_state = [current_state[0] + 1, current_state[1]]
        elif (action == 1) and (current_state not in self.down):
            next_state = [current_state[0] - 1, current_state[1]]
        # left = 2, right = 3 ...backwards but w.e.
        elif (action == 2) and (current_state not in self.left):
            next_state = [current_state[0], current_state[1] - 1]
        elif (action == 3) and (current_state not in self.right):
            next_state = [current_state[0], current_state[1] + 1]
        return next_state

    def step(self, action):
        self.steps = self.steps + 1

        next_state = self.next_state(action, self.state)

        if next_state in self.wins:
            reward = 100
            done = True
        elif next_state in self.fails:
            reward = -100
            done = True
        else:
            reward = self.moves[self.state[0]][self.state[1]][action]
            if (self.steps >= 20):
                done = True
            else:
                done = False

        self.state = next_state
        return self.state_to_num(self.state), reward, done

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

class Solver():
    def __init__(self, n_episodes=1000, max_env_steps=None, gamma=1.0, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.99, alpha=0.1, alpha_decay=0.04, batch_size=32, quiet=False):
        self.memory = deque(maxlen=64)
        self.hist = {}

        self.env = Env()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.quiet = quiet
        self.decay_step = 10
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        self.model = Sequential()
        self.model.add(Dense(4, input_dim=16, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            ret = random.randint(0, 3)
        else:
            ret = np.argmax(self.model.predict(state))

        return ret

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state))
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def run(self):
        scores = deque(maxlen=100)
        sum_avg_delta = 0

        for e in range(1000):
            state = self.env.reset()
            state = num_to_state(state)

            reward_sum = 0
            done = False
            step = 0
            prev = self.model.get_weights()[0]
            sum_avg_delta = sum_avg_delta + np.average(np.absolute(prev))

            if (np.mean(scores) > 0.5) and (e > 100):
                break

            while not done:
                step = step + 1
                action = self.choose_action(state, self.get_epsilon(e))

                next_state, reward, done = self.env.step(action)
                next_state = num_to_state(next_state)
                reward_sum = reward_sum + reward
                self.remember(state, action, reward, next_state, done)

                state_num = state_to_num(state)
                if (state_num, action) in self.hist:
                    self.hist[(state_num, action)] = self.hist[(state_num, action)] + 1
                else:
                    self.hist[(state_num, action)] = 1

                itr = str(e) + " "
                itr = itr + str(step) + "/" + str(20) + " "
                itr = itr + str(state_to_num(state)) + " " + str(action) + " " + str(state_to_num(next_state)) + " "
                itr = itr + str(reward) + " "
                print (itr)

                state = next_state

                if done:
                    self.replay(self.batch_size)

                    if (reward > 0) and (self.epsilon > self.epsilon_min):
                        self.epsilon *= 0.7

                    scores.append(reward_sum > 0)
                    mean_score = np.mean(scores)

                    print (e, step, mean_score, self.epsilon, sum_avg_delta)

                    disp (self.model.get_weights()[0] - prev)
                    disp (self.model.get_weights()[0])

                    # print (self.hist)
                    for key in sorted(self.hist, key=itemgetter(0)):
                        print (key, self.hist[key])

                    break
        

if __name__ == '__main__':
    agent = Solver()
    agent.run()
