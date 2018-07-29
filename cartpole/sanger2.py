# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

NUM_SCORES = 100

LAYER0 = 8
LAYER1 = 48
LAYER2 = 256
LAYER3 = 2

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

  print (np.average(elig))
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
    
    reward = 0.15 - np.absolute(ang2)
    
    return reward
  
class DQNCartPoleSolver():
    def __init__(self, n_episodes=5000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=0.5, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
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

        self.weights1 = np.absolute(np.random.normal(0.5, 0.01, size=(LAYER0, LAYER1)))
        self.weights2 = np.absolute(np.random.normal(0.5, 0.01, size=(LAYER1, LAYER2)))
        self.weights3 = np.absolute(np.random.normal(0.5, 0.01, size=(LAYER2, LAYER3)))

        col_norm = np.average(self.weights1, axis = 0)
        col_norm = 0.5 / col_norm
        for j in range(LAYER1):
          self.weights1[:, j] *= col_norm[j]
        
        col_norm = np.average(self.weights2, axis = 0)
        col_norm = 0.5 / col_norm
        for j in range(LAYER2):
          self.weights2[:, j] *= col_norm[j]
        
        col_norm = np.average(self.weights3, axis = 0)
        col_norm = 0.5 / col_norm
        for j in range(LAYER3):
          self.weights3[:, j] *= col_norm[j]

    def preprocess_state(self, state):
        ret = np.zeros(LAYER0)
        
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
        
        return ret.reshape(1,LAYER0)
        
    def run(self):
        scores = deque(maxlen=NUM_SCORES)
        lr = 0.01

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            
            prev1 = self.weights1
            prev2 = self.weights2
            prev3 = self.weights3

            elig1 = np.zeros(shape=(LAYER0, LAYER1))
            elig2 = np.zeros(shape=(LAYER1, LAYER2))
            elig3 = np.zeros(shape=(LAYER2, LAYER3))
            
            rewards = []
            eligs = []
            actions = []
            
            while not done:
                i += 1
                
                y1 = np.dot(state, self.weights1)
                y1 = y1 / np.max(y1)
                wy1 = np.dot(self.weights1, np.transpose(y1))
                d1 = state - np.transpose(wy1) / np.average(wy1) * np.average(state)
                elig_grad1 = np.dot(np.transpose(y1), d1)
                elig1 = update_elig(elig1, np.transpose(elig_grad1))

                y2 = np.dot(y1, self.weights2)
                y2 = y2 / np.max(y2)
                wy2 = np.dot(self.weights2, np.transpose(y2))
                d2 = y1 - np.transpose(wy2) / np.average(wy2) * np.average(y1)
                elig_grad2 = np.dot(np.transpose(y2), d2)
                elig2 = update_elig(elig2, np.transpose(elig_grad2))

                y3 = np.dot(y2, self.weights3)
                y3 = y3 / np.max(y3)
                wy3 = np.dot(self.weights3, np.transpose(y3))
                d3 = y2 - np.transpose(wy3) / np.average(wy3) * np.average(y2)
                elig_grad3 = np.dot(np.transpose(y3), d3)
                elig3 = update_elig(elig3, np.transpose(elig_grad3))

                action = 0
                if (np.random.random() <= self.epsilon):
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(y3)
                    
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                # reward = get_reward(state, next_state, done)
                state = next_state
                
                actions.append(action)
                rewards.append(reward)
                
                grad = lr * elig1
                self.weights1 = np.clip(self.weights1 + grad, 0.05, 5)
                
                grad = lr * elig2
                self.weights2 = np.clip(self.weights2 + grad, 0.05, 5)
                
                grad = lr * elig3 * reward
                self.weights3 = np.clip(self.weights3 + grad, 0.05, 5)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= 0.999
            
            scores.append(i)
            mean_score = np.mean(scores)
            
            if (mean_score > 190):
                np.save("weights1", self.weights1)
                np.save("weights2", self.weights2)
                np.save("weights3", self.weights3)
                break
              
            if (mean_score > 100):
                lr = 0.0005
            elif (mean_score > 25):
                lr = 0.001
            if (mean_score < 25):
                lr = 0.01
            
            col_norm = np.average(self.weights1, axis = 0)
            col_norm = 0.5 / col_norm
            for j in range(LAYER1):
              self.weights1[:, j] *= col_norm[j]
            
            col_norm = np.average(self.weights2, axis = 0)
            col_norm = 0.5 / col_norm
            for j in range(LAYER2):
              self.weights2[:, j] *= col_norm[j]
            
            col_norm = np.average(self.weights3, axis = 0)
            col_norm = 0.5 / col_norm
            for j in range(LAYER3):
              self.weights3[:, j] *= col_norm[j]
            
            if (((e+1) % 100 == 0) and e > 0):
                print (e, mean_score, self.epsilon)
             
            if (((e+1) % 500 == 0) and e > 0):
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
                
                '''
                print ('{:05.3f}'.format(sum1), '{:05.3f}'.format(diff1), '{:05.3f}'.format(avg1), '{:05.3f}'.format(std1))
                print ('{:05.3f}'.format(sum2), '{:05.3f}'.format(diff2), '{:05.3f}'.format(avg2), '{:05.3f}'.format(std2))
                print ('{:05.3f}'.format(sum3), '{:05.3f}'.format(diff3), '{:05.3f}'.format(avg3), '{:05.3f}'.format(std3))
                '''
                
                prev1 = self.weights1
                prev2 = self.weights2
                prev3 = self.weights3
                
if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
