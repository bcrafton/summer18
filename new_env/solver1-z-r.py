
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

'''
C D E F
8 9 A B
4 5 6 7
0 1 2 3
'''

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

class Env():
    def __init__(self):
        self.state = 0
        self.steps = 0

        self.nrows = 4
        self.ncols = 4

        self.grid = np.ones(self.nrows * self.ncols) * -5
        self.wins = [15]
        self.fails = [10]

        self.left = []
        self.right = []
        self.up = []
        self.down = []

        for i in self.wins:
            self.grid[i] = 100

        for i in self.fails:
            self.grid[i] = -100

        for i in range(self.nrows * self.ncols):
            if ( i % self.ncols == 3 ):
                self.right.append(i)

            if ( i % self.ncols == 0 ):
                self.left.append(i)

            if ( math.floor(i / self.ncols) == (self.nrows-1) ):
                self.up.append(i)

            if ( math.floor(i / self.ncols) == 0 ):
                self.down.append(i)

    def reset(self):
        self.state = 0
        self.steps = 0

    def step(self, action):
        self.steps = self.steps + 1

        next_state = self.state
        if (action == 0) and (self.state not in self.up):
            next_state = self.state + 4
        elif (action == 1) and (self.state not in self.down):
            next_state = self.state - 4
        elif (action == 2) and (self.state not in self.left):
            next_state = self.state - 1
        elif (action == 3) and (self.state not in self.right):
            next_state = self.state + 1

        if next_state in self.wins:
            reward = 100
            done = True
        elif next_state in self.fails:
            reward = -100
            done = True
        else:
            reward = -5
            if (self.steps >= 20):
                done = True
            else:
                done = False

        self.state = next_state
        return next_state, reward, done

class Solver():
    def __init__(self, n_episodes=1000, max_env_steps=None, gamma=1.0, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.99, alpha=0.04, alpha_decay=0.04, batch_size=32, quiet=False):
        self.memory = deque(maxlen=64)

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

        self.weights = np.random.normal(10.0, 3.0, size=(16,4))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            ret = random.randint(0, 3)
        else:
            ret = np.argmax(state.dot(self.weights))

        return ret

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.env.reset()
            state = num_to_state(state)

            reward_sum = 0
            step = 0
            done = False

            elig = np.zeros((16, 4))

            prev = self.weights

            while not done:
                step = step + 1
                action = self.choose_action(state, self.get_epsilon(e))

                next_state, reward, done = self.env.step(action)
                next_state = num_to_state(next_state)
                reward_sum = reward_sum + reward

                print (state_to_num(state), state_to_num(next_state), action)

                for i in range(16):
                    for j in range(4):
                        if (i == state_to_num(state)) and (action == j):
                            elig[i][j] = 16
                        else:
                            elig[i][j] = max(elig[i][j] - 1, 0)

                gradient = elig * reward * (1/10)
                gradient = np.array(gradient)
                # print (gradient)
                self.weights = self.weights + gradient

                state = next_state
                if done:

                    print (self.weights)
                    print (self.weights - prev)

                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay ** (e / self.decay_step)

                    scores.append(reward_sum > 0)
                    mean_score = np.mean(scores)

                    print (step, reward_sum, mean_score)
                    break

        np.save("weights", self.model.get_weights())
        

if __name__ == '__main__':
    agent = Solver()
    agent.run()
