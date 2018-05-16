
import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import time
import tkinter as tk
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 4  # grid height
WIDTH = 4  # grid width

'''
C D E F
8 9 A B
4 5 6 7
0 1 2 3
'''

class Env():
    def __init__(self):
        self.state = 0

        self.grid = np.ones(16) * -5
        self.wins = [15]
        self.fails = [10]

        for i in self.wins:
            self.grid[i] = 100

        for i in self.fails:
            self.grid[i] = -100

    def reset(self):
        self.state = 0

    def step(self, action):

        next_state = self.state
        if (action == 0) and (self.state not in [12, 13, 14, 15]):
            next_state = self.state + 4
        elif (action == 1) and (self.state not in [0, 1, 2, 3]):
            next_state = self.state - 4
        elif (action == 2) and (self.state not in [0, 4, 8, 11]):
            next_state = self.state - 1
        elif (action == 3) and (self.state not in [3, 7, 11, 15]):
            next_state = self.state + 1

        if next_state in self.wins:
            reward = 100
            done = True
        elif next_state in self.fails:
            reward = -100
            done = True
        else:
            reward = -5
            done = False

        self.state = next_state
        return next_state, reward, done

class Solver():
    def __init__(self, n_episodes=10000, max_env_steps=None, gamma=1.0, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.99, alpha=0.04, alpha_decay=0.04, batch_size=32, quiet=False):
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

    def preprocess_state(self, state):
        ret = np.zeros(16)
        for i in range(16):
            if state == i:
                ret[i] = 1
        return np.reshape(ret, [1, 16])

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

        for e in range(self.n_episodes):
            state = self.env.reset()
            state = self.preprocess_state(state)

            reward_sum = 0
            step = 0
            done = False
            while not done:
                step = step + 1
                action = self.choose_action(state, self.get_epsilon(e))

                next_state, reward, done = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                reward_sum = reward_sum + reward
                self.remember(state, action, reward, next_state, done)

                state = next_state

                table = self.model.get_weights()
                table = table[0]

                if done or step >= 20:
                    self.replay(self.batch_size)

                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay ** (e / self.decay_step)

                    scores.append(reward_sum > 0)
                    mean_score = np.mean(scores)

                    print (step, reward_sum, mean_score)
                    break
        

if __name__ == '__main__':
    agent = Solver()
    agent.run()
