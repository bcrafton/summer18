# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

SPIKE = 1

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_reward(state, next_state):
    state = state.flatten()
    next_state = next_state.flatten()
    
    pos1 = state[0] - state[1]
    pos2 = next_state[0] - next_state[1]
    
    ang1 = state[4] - state[5]
    ang2 = next_state[4] - next_state[5]
    
    if (pos1 > 0):
      diff_pos = pos1 - pos2
    else:
      diff_pos = pos2 - pos1
    
    if (ang1 > 0):
      diff_ang = ang1 - ang2
    else:
      diff_ang = ang2 - ang1
    
    # reward = np.clip(100 * (diff_pos + diff_ang), -1, 1)
    # reward = np.clip(0.5 + (diff_pos + diff_ang), 0, 1)
    
    if (diff_pos + diff_ang > 0):
      reward = 1
    else:
      reward = 0
      
    return reward
  
class DQNCartPoleSolver():
    def __init__(self, n_episodes=100000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        self.weights1 = np.absolute(np.random.normal(0.5, 0.2, size=(8, 24)))
        self.weights2 = np.absolute(np.random.normal(0.5, 0.2, size=(24, 48)))
        self.weights3 = np.absolute(np.random.normal(0.5, 0.2, size=(48, 2)))
        
        # Init model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=8, activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        
    def preprocess_state(self, state):
        ret = np.zeros(8)
        
        ## 0, 1 < 0
        if state[0] >= 0:
          ret[0] = np.absolute(state[0])
          ret[1] = 0
        else:
          ret[0] = 0
          ret[1] = np.absolute(state[0])
          
        ## 2, 3 < 1
        if state[1] >= 0:
          ret[2] = np.absolute(state[1])
          ret[3] = 0
        else:
          ret[2] = 0
          ret[3] = np.absolute(state[1])

        ## 4, 5 < 2
        if state[2] >= 0:
          ret[4] = np.absolute(state[2])
          ret[5] = 0
        else:
          ret[4] = 0
          ret[5] = np.absolute(state[2])
        
        ## 6, 7 < 3
        if state[3] >= 0:
          ret[6] = np.absolute(state[3])
          ret[7] = 0
        else:
          ret[6] = 0
          ret[7] = np.absolute(state[3])
        
        return ret.reshape(1,8)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))
        
    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            
            if SPIKE:
                prev1 = self.weights1
                prev2 = self.weights2
                prev3 = self.weights3
            
            while not done:
            
                if SPIKE:

                    out1 = np.dot(state, self.weights1)
                    xw1 = np.dot(state, self.weights1)
                    xw1 = xw1 / np.max(xw1)
                    xw1 = np.power(xw1, 3)
                    sig = sigmoid(xw1)
                    err = sig - 0.5 * xw1
                    elig1 = np.ones(shape=np.shape(err)) * np.max(err) - err
                    # print (np.min(elig1), np.max(elig1))

                    xw2 = np.dot(xw1, self.weights2)
                    xw2 = xw2 / np.max(xw2)
                    xw2 = np.power(xw2, 3)
                    sig = sigmoid(xw2)
                    err = sig - 0.5 * xw2
                    elig2 = np.ones(shape=np.shape(err)) * np.max(err) - err
                    
                    xw3 = np.dot(xw2, self.weights3)
                    xw3 = xw3 / np.max(xw3)
                    xw3 = np.power(xw3, 3)
                    sig = sigmoid(xw3)
                    err = sig - 0.5 * xw3
                    elig3 = np.ones(shape=np.shape(err)) * np.max(err) - err

                    action = np.argmax(xw3)
                    
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.preprocess_state(next_state)
                    
                    reward = get_reward(state, next_state)
                    state = next_state
                    
                    self.weights1 = (self.weights1 * 0.9 + 0.1 * reward * elig1)
                    self.weights2 = (self.weights2 * 0.9 + 0.1 * reward * elig2)
                    self.weights3 = (self.weights3 * 0.9 + 0.1 * reward * elig3)
                    
                    col_norm = np.average(self.weights1, axis = 0)
                    col_norm = 0.5 / col_norm
                    for j in range(24):
                      self.weights1[:, j] *= col_norm[j]
                    
                    col_norm = np.average(self.weights2, axis = 0)
                    col_norm = 0.5 / col_norm
                    for j in range(48):
                      self.weights2[:, j] *= col_norm[j]
                    
                    col_norm = np.average(self.weights3, axis = 0)
                    col_norm = 0.5 / col_norm
                    for j in range(2):
                      self.weights3[:, j] *= col_norm[j]
                
                    i += 1
                if not SPIKE: 
                    action = self.choose_action(state, self.epsilon)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.preprocess_state(next_state)
                    
                    reward = get_reward(state, next_state)
                    
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            self.epsilon *= 0.99
            if not SPIKE: 
                self.replay(self.batch_size) 
            
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
                
                if SPIKE:
                    assert(np.min(self.weights1) >= 0)
                    assert(np.min(self.weights2) >= 0)
                    assert(np.min(self.weights3) >= 0)
                    
                    print ( np.sum(np.absolute(self.weights1)), np.sum(np.absolute(self.weights1 - prev1)) / np.sum(np.absolute(self.weights1)), np.std(self.weights1) )
                    print ( np.sum(np.absolute(self.weights2)), np.sum(np.absolute(self.weights2 - prev2)) / np.sum(np.absolute(self.weights2)), np.std(self.weights2) )
                    print ( np.sum(np.absolute(self.weights3)), np.sum(np.absolute(self.weights3 - prev3)) / np.sum(np.absolute(self.weights3)), np.std(self.weights3) )
                    
                    prev1 = self.weights1
                    prev2 = self.weights2
                    prev3 = self.weights3
                
if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
