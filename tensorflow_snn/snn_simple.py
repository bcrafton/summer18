
import numpy as np
import matplotlib.pyplot as plt
from ez_env import Env

import random
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.constraints import non_neg
import keras
from operator import itemgetter

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
        self.model.add(Dense(4, input_dim=16, activation='linear', use_bias=False, kernel_constraint=non_neg()))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        init_weights = [np.ones(shape=(16, 4)) * 1000.0, np.zeros(shape=(4))]
        self.model.set_weights(init_weights)

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

    def spike_weights(self):
        weights = self.model.get_weights()[0]
        weights = weights - np.min(weights) + 1
        weights = weights / np.absolute(np.average(weights))
        return weights

    def run(self):
        T = 1000
        dt = 0.5
        time_steps = int(T / dt)
        num_inputs = 16
        num_actions = 4 

        scores = deque(maxlen=100)

        for e in range(1000):

            state = self.env.reset()
            state = num_to_state(state)

            reward_sum = 0
            done = False
            step = 0
            prev = self.model.get_weights()[0]

            if (np.mean(scores) > 0.5) and (e > 100):
                break

            # Wsyn = np.ones(shape=(num_inputs, num_actions))
            # Wsyn = self.model.get_weights()[0] / np.average(self.model.get_weights()[0])
            Wsyn = self.spike_weights()
            while not done:
                ################
                v = np.zeros(num_actions)
                fire_counts = np.zeros(4)
                fired = []
                rates = state * 0.01
                for t in range(time_steps):
                    v[fired] = 0
                  
                    f = np.random.rand(1, 16) < rates * dt
                    Isyn = np.dot(f, Wsyn)
                    
                    dv = 0.04 * v * v
                    v = v + (dv + Isyn) * dt
                    v = v[0]

                    fired = np.where(v > 35)
                    fired = fired[0]
                    fire_counts[fired] = fire_counts[fired] + 1
                    ################

                print (fire_counts)

                step = step + 1

                if (np.random.random() <= self.epsilon):
                    action = random.randint(0, 3)
                else:
                    action = np.argmax(fire_counts)

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

                    if (reward_sum > 1000) and (self.epsilon > self.epsilon_min):
                        self.epsilon *= 0.7

                    scores.append(reward_sum > 1000)
                    mean_score = np.mean(scores)

                    print (e, step, mean_score, self.epsilon)

                    # disp (self.model.get_weights()[0] - prev)
                    # disp (self.model.get_weights()[0])

                    # print (self.hist)
                    for key in sorted(self.hist, key=itemgetter(0)):
                        print (key, self.hist[key])

                    break
        

if __name__ == '__main__':
    agent = Solver()
    agent.run()


