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
NUM_SCORES = 100

def update_elig(elig, grad):
  elig = np.copy(elig)
  grad = np.copy(grad)
  
  if True:
      elig = elig * 0.9
      elig = elig + grad
  elif False:
      elig = np.power(elig, 1.25)
      elig = elig + grad
      elig = elig / np.max(elig)

  return elig

def sigmoid(x):
  return 1 / (1 + np.exp(-x)) - 0.5

def get_reward(state, next_state, done):
    if done:
       return -1

    state = state.flatten()
    next_state = next_state.flatten()
    reward = 0
    
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
      
    '''
    from_zero = np.absolute(ang2)
    if (from_zero < 10):
      reward += 1 - (from_zero / 10)
    
    reward += 100 * (diff_ang)
    reward = np.clip(reward, -1, 1)
    '''
    
    reward = 0.15 - np.absolute(ang2)
    
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

        self.weights1 = np.absolute(np.random.normal(0.1, 0.01, size=(8, 24)))
        self.weights2 = np.absolute(np.random.normal(0.1, 0.01, size=(24, 48)))
        self.weights3 = np.absolute(np.random.normal(0.1, 0.01, size=(48, 2)))

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
        scores = deque(maxlen=NUM_SCORES)
        lr = 0.01

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            
            if SPIKE:
                prev1 = self.weights1
                prev2 = self.weights2
                prev3 = self.weights3

            elig1 = np.zeros(shape=(8, 24))
            elig2 = np.zeros(shape=(24, 48))
            elig3 = np.zeros(shape=(48, 2))
            
            rewards = []
            eligs = []
            actions = []
            
            '''
            if lr > 0.001:
                lr = lr * 0.999
            else:
                lr = 0.001
            '''
            
            while not done:
                i += 1
                
                if SPIKE:
                    xw1 = np.dot(state, self.weights1)
                    xw1 = xw1 / np.max(xw1)
                    xw1 = np.power(xw1, 2)                    
                    elig_grad1 = np.dot(np.transpose(state), xw1) * np.power(self.weights1, 2)
                    elig1 = update_elig(elig1, elig_grad1)

                    xw2 = np.dot(xw1, self.weights2)
                    xw2 = xw2 / np.max(xw2)
                    xw2 = np.power(xw2, 2)                    
                    elig_grad2 = np.dot(np.transpose(xw1), xw2) * np.power(self.weights2, 2)
                    elig2 = update_elig(elig2, elig_grad2)
                    
                    xw3 = np.dot(xw2, self.weights3)
                    xw3 = xw3 / np.max(xw3)
                    xw3 = np.power(xw3, 2)                    
                    elig_grad3 = np.dot(np.transpose(xw2), xw3) * np.power(self.weights3, 2)
                    elig3 = update_elig(elig3, elig_grad3)

                    action = np.argmax(xw3)
                    next_state, reward, done, _ = self.env.step(action)
                    # print (next_state)
                    next_state = self.preprocess_state(next_state)
                    reward = get_reward(state, next_state, done)
                    state = next_state
                    
                    actions.append(action)
                    rewards.append(reward)
                    
                    # fluct1 = np.absolute(np.random.normal(0.1, 0.04, size=(8, 24)))
                    # fluct2 = np.absolute(np.random.normal(0.1, 0.04, size=(24, 48)))
                    # fluct3 = np.absolute(np.random.normal(0.1, 0.04, size=(48, 2)))
                    
                    # err = (elig1 / np.average(self.weights1)) - self.weights1
                    # self.weights1 += lr * err * reward
                    err = elig1 / np.average(elig1) * np.average(self.weights1) - self.weights1
                    self.weights1 = np.clip(self.weights1 + lr * err * reward, 0.05, 5)
                    
                    # err = (elig2 / np.average(self.weights2)) - self.weights2
                    # self.weights2 += lr * err * reward
                    err = elig2 / np.average(elig2) * np.average(self.weights2) - self.weights2
                    self.weights2 = np.clip(self.weights2 + lr * err * reward, 0.05, 5)
                    
                    # err = (elig3 / np.average(self.weights3)) - self.weights3
                    # self.weights3 += lr * err * reward
                    err = elig3 / np.average(elig3) * np.average(self.weights3) - self.weights3
                    grad = lr * err * reward
                    self.weights3 = np.clip(self.weights3 + grad, 0.05, 5)
                    
                    # print (np.max(err), np.min(err))
                    
                if not SPIKE: 
                    action = self.choose_action(state, self.epsilon)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.preprocess_state(next_state)
                    reward = get_reward(state, next_state)
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                   
            # print ("end") 
            
            # print (np.average(np.asarray(rewards)))
            # print (np.average(np.asarray(eligs)))            

            # print(actions)
            # print(rewards)
            
            # making sure we were getting good elig, good reward very important.
            
            # def most val print statement:
            # print ( np.average(np.asarray(rewards)), np.count_nonzero(np.asarray(actions)), len(actions), lr)
            
            self.epsilon *= 0.99
            
            scores.append(i)
            mean_score = np.mean(scores)
            
            if (mean_score > 150):
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
                
                assert(np.min(self.weights1) >= 0)
                assert(np.min(self.weights2) >= 0)
                assert(np.min(self.weights3) >= 0)
                
                sum1 = np.sum(np.absolute(self.weights1))
                diff1 = np.sum(np.absolute(self.weights1 - prev1))
                avg1 = np.sum(np.absolute(self.weights1 - prev1)) / np.sum(prev1)
                std1 = np.std(self.weights1)
                
                sum2 = np.sum(np.absolute(self.weights2))
                diff2 = np.sum(np.absolute(self.weights2 - prev2))
                avg2 = np.sum(np.absolute(self.weights2 - prev2)) / np.sum(prev2)
                std2 = np.std(self.weights2)
                
                sum3 = np.sum(np.absolute(self.weights3))
                diff3 = np.sum(np.absolute(self.weights3 - prev3))
                avg3 = np.sum(np.absolute(self.weights3 - prev3)) / np.sum(prev3)
                std3 = np.std(self.weights3)
                
                print ('{:05.3f}'.format(sum1), '{:05.3f}'.format(diff1), '{:05.3f}'.format(avg1), '{:05.3f}'.format(std1))
                print ('{:05.3f}'.format(sum2), '{:05.3f}'.format(diff2), '{:05.3f}'.format(avg2), '{:05.3f}'.format(std2))
                print ('{:05.3f}'.format(sum3), '{:05.3f}'.format(diff3), '{:05.3f}'.format(avg3), '{:05.3f}'.format(std3))
                
                prev1 = self.weights1
                prev2 = self.weights2
                prev3 = self.weights3
                
                break 
            elif (mean_score > 100):
                lr = 0.0005
            elif (mean_score > 25):
                lr = 0.001
            if (mean_score < 25):
                lr = 0.01
            
            if SPIKE:
                '''
                elig1 = elig1 / np.max(elig1)
                elig1 = np.power(elig1, 2)
                
                elig2 = elig2 / np.max(elig2)
                elig2 = np.power(elig2, 2)
                
                elig3 = elig3 / np.max(elig3)
                elig3 = np.power(elig3, 2)
            
                self.weights1 = (self.weights1 + lr * reward * elig1) / (1 + lr * elig1)
                self.weights2 = (self.weights2 + lr * reward * elig2) / (1 + lr * elig2)
                self.weights3 = (self.weights3 + lr * reward * elig3) / (1 + lr * elig3)
                '''
                
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
                  
            if not SPIKE: 
                self.replay(self.batch_size) 
            
            if (((e+1) % self.n_episodes == 0) and e > 0):
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))
             
            if (((e+1) % self.n_episodes == 0) and e > 0):
                if SPIKE:
                    assert(np.min(self.weights1) >= 0)
                    assert(np.min(self.weights2) >= 0)
                    assert(np.min(self.weights3) >= 0)
                    
                    sum1 = np.sum(np.absolute(self.weights1))
                    diff1 = np.sum(np.absolute(self.weights1 - prev1))
                    avg1 = np.sum(np.absolute(self.weights1 - prev1)) / np.sum(prev1)
                    std1 = np.std(self.weights1)
                    
                    sum2 = np.sum(np.absolute(self.weights2))
                    diff2 = np.sum(np.absolute(self.weights2 - prev2))
                    avg2 = np.sum(np.absolute(self.weights2 - prev2)) / np.sum(prev2)
                    std2 = np.std(self.weights2)
                    
                    sum3 = np.sum(np.absolute(self.weights3))
                    diff3 = np.sum(np.absolute(self.weights3 - prev3))
                    avg3 = np.sum(np.absolute(self.weights3 - prev3)) / np.sum(prev3)
                    std3 = np.std(self.weights3)
                    
                    print ('{:05.3f}'.format(sum1), '{:05.3f}'.format(diff1), '{:05.3f}'.format(avg1), '{:05.3f}'.format(std1))
                    print ('{:05.3f}'.format(sum2), '{:05.3f}'.format(diff2), '{:05.3f}'.format(avg2), '{:05.3f}'.format(std2))
                    print ('{:05.3f}'.format(sum3), '{:05.3f}'.format(diff3), '{:05.3f}'.format(avg3), '{:05.3f}'.format(std3))
                    
                    prev1 = self.weights1
                    prev2 = self.weights2
                    prev3 = self.weights3
                
if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
