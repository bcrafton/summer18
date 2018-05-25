
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

        #############
        Wsyn = np.random.normal(1.0, 0.25, size=(4,16))
        Wsyn = np.transpose(Wsyn)

        # Wsyn = np.random.normal(1.0, 0.25, size=(16,4))

        Wsyn = np.absolute(Wsyn)
        #############

        scores = deque(maxlen=100)
        for e in range(5000):

            state = self.env.reset()
            state = num_to_state(state)

            reward_sum = 0
            done = False
            step = 0
            prev = self.model.get_weights()[0]

            if (np.mean(scores) > 0.5) and (e > 100):
                break

            while not done:
                step = step + 1

                ################
                Isyn = np.zeros(4)
                g = np.zeros(shape=(1, 16))
                v = np.zeros(4)
                u = np.zeros(4)

                input_fires = np.zeros(shape=(2000,16))
                input_fired_counts = np.zeros(16)

                output_fired = np.zeros(4)
                output_not_fired = np.zeros(4)
                output_fires = np.zeros(shape=(2000,4))
                output_fired_counts = np.zeros(4)

                # what is freq / ms
                # Hz / 1000
                # (25.0 / 1000.0) = 25Hz
                rates = state * (25.0 / 1000.0)

                for t in range(time_steps):
                    '''
                    Isyn = Isyn - 0.5
                    neg_idx = np.where(Isyn < 0)
                    Isyn[neg_idx] = 0
                    '''
                  
                    input_fired = np.random.rand(1, 16) < rates * dt
                    input_fires[t] = input_fired
                    input_fired_counts = input_fired_counts + input_fired

                    '''
                    Isyn = Isyn + np.dot(input_fired, Wsyn) * 15 # average weight = 0.5, want 7 for current.
                    '''

                    # this is slightly different than the matlab code.
                    # them: 1000x100 * 100x1 (1000x1)
                    # us:   1x16     * 16x4  (1x4)
                    # 100, 1000 = in, out
                    # 16, 4     = in, out
                    g = g + input_fired
                    Isyn = np.dot(g, Wsyn)
                    Isyn = Isyn - (np.dot(g, Wsyn) * v)
                    g = (1 - dt/10) * g

                    dv = (0.04 * v + 5) * v + 140 - u
                    v = v + (dv + Isyn) * dt
                    du = 0.02 * (0.2 * v - u)
                    u = u + dt * du

                    output_fired = v > 35
                    output_not_fired = v < 35
                    output_fires[t] = output_fired
                    output_fired_counts = output_fired_counts + output_fired

                    v = v * output_not_fired
                    v = v + output_fired * -65
                    u = u + output_fired * 8

                output_fires_post = np.zeros(shape=(2000,4))
                output_fires_pre = np.zeros(shape=(2000,4))

                for i in range(4):
                    flag = 0
                    for j in range(2000):
                        if (output_fires[j][i]):
                            flag = 10
                        if flag:
                            output_fires_post[j][i] = 1
                            flag = flag - 1

                    flag = 0
                    for j in reversed(range(2000)):
                        if (output_fires[j][i]):
                            flag = 10
                        if flag:
                            output_fires_pre[j][i] = 1
                            flag = flag - 1

                
                input_fires = np.transpose(input_fires)
                output_fires_post = np.transpose(output_fires_post)
                output_fires_pre = np.transpose(output_fires_pre)

                post = np.zeros(shape=(16,4))
                pre = np.zeros(shape=(16,4))
                for i in range(4):
                    for j in range(16):
                        post[j][i] = np.count_nonzero(np.logical_and(input_fires[j], output_fires_post[i]))
                        pre[j][i] = np.count_nonzero(np.logical_and(input_fires[j], output_fires_pre[i]))

                elig = pre - post
                neg_idx = np.where(elig < 0)
                elig[neg_idx] = 0
                elig = elig * np.random.normal(1.0, 0.4, size=(16,4))

                if (np.random.random() <= self.epsilon):
                    action = random.randint(0, 3)
                else:
                    action = np.random.choice(np.flatnonzero(output_fired_counts == output_fired_counts.max()))

                next_state, reward, done = self.env.step(action)
                next_state = num_to_state(next_state)
                reward_sum = reward_sum + reward
                self.remember(state, action, reward, next_state, done)

                gradient = elig * reward_sum * (1 / 1000)
                Wsyn = (9 * Wsyn + gradient) / (9 + elig)

                itr = str(e) + " "
                itr = itr + str(step) + "/" + str(20) + " "
                itr = itr + str(state_to_num(state)) + " " + str(action) + " " + str(state_to_num(next_state)) + " "
                itr = itr + str(reward) + " "
                # print (itr)

                ################
                print(output_fired_counts)
                '''
                print("-------------")
                print(input_fired_counts)
                print(output_fired_counts)
                print (Wsyn[state_to_num(state)])
                print (itr)
                '''

                '''
                if np.average(output_fired_counts) < 4:
                    print("-------------")
                    print(output_fired_counts)
                    print (Wsyn[state_to_num(state)])
                    print (itr)
                if np.average(output_fired_counts) > 50:
                    print("-------------")
                    print(output_fired_counts)
                    print (Wsyn[state_to_num(state)])
                    print (itr)
                '''
                ################

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


