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

E_TAU = 10 # 10 steps ? 

def update_elig(elig, grad):
  de = -elig / E_TAU + grad
  elig = np.clip(elig + de, 0.0, 5.0)
  return elig
  
class DQNCartPoleSolver():
    def __init__(self, n_episodes=5000, n_win_ticks=195, max_env_steps=200, gamma=1.0, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.999, alpha=0.01, alpha_decay=0.01):
        self.env = gym.make('CartPole-v0')
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.max_env_steps = max_env_steps

        self.weights1 = np.absolute(np.random.normal(0.1, 0.01, size=(LAYER0 + 1, LAYER1)))
        self.weights2 = np.absolute(np.random.normal(0.1, 0.01, size=(LAYER1 + 1, LAYER2)))
        self.weights3 = np.absolute(np.random.normal(0.01, 0.001, size=(LAYER2 + 1, LAYER3)))

        self.normalize()

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

    def normalize(self):
        col_norm = np.average(self.weights1, axis = 0)
        col_norm = 1.0 / col_norm
        for j in range(LAYER1):
          self.weights1[:, j] *= col_norm[j]
        
        col_norm = np.average(self.weights2, axis = 0)
        col_norm = 1.0 / col_norm
        for j in range(LAYER2):
          self.weights2[:, j] *= col_norm[j]

    def run(self):
        scores = deque(maxlen=NUM_SCORES)
        lr = 0.01

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0

            elig1 = np.zeros(shape=(LAYER0 + 1, LAYER1))
            elig2 = np.zeros(shape=(LAYER1 + 1, LAYER2))
            elig3 = np.zeros(shape=(LAYER2 + 1, LAYER3))
            
            ALPHA = 1e-5
            prev_y2 = np.zeros(shape=(1, LAYER2 + 1))
            prev_y3 = np.zeros(shape=(1, LAYER3))
            
            reward = 0.0
            
            while not done:
                i += 1
                
                ################
                # feed forward
                ################
                y1 = np.dot(y1, self.weights1)
                y1 = y1 / np.max(y1)
                wy1 = np.dot(self.weights1, np.transpose(y1))
                d1 = x1 - np.transpose(wy1) / np.average(wy1) * np.average(x1)
                elig_grad1 = np.dot(np.transpose(y1), d1)
                elig1 = update_elig(elig1, np.transpose(elig_grad1))

                y2 = np.dot(y2, self.weights2)
                y2 = y2 / np.max(y2)
                wy2 = np.dot(self.weights2, np.transpose(y2))
                d2 = x2 - np.transpose(wy2) / np.average(wy2) * np.average(x2)
                elig_grad2 = np.dot(np.transpose(y2), d2)
                elig2 = update_elig(elig2, np.transpose(elig_grad2))

                y3 = np.dot(y3, self.weights3)
                # y3 = y3 / np.max(y3)
                wy3 = np.dot(self.weights3, np.transpose(y3))
                d3 = x3 - np.transpose(wy3) / np.average(wy3) * np.average(x3)
                elig_grad3 = np.dot(np.transpose(y3), d3)
                elig3 = update_elig(elig3, np.transpose(elig_grad3))
                
                ################
                # update weights
                ################
                grad = lr * elig1
                self.weights1 = np.clip(self.weights1 + grad, 0.001, 1.0)
                
                grad = lr * elig2
                self.weights2 = np.clip(self.weights2 + grad, 0.001, 1.0)
                
                #d = reward + y3 - prev_y3
                #dw = (ALPHA * (d + y3 - prev_y3) * elig3)
                #self.weights3 = np.clip(self.weights3 + dw, 0.0, 1.0)
                grad = lr * elig3 * reward
                self.weights3 = np.clip(self.weights3 + grad, 0.0001, 1.0)
                
                prev_y2 = x3  
                prev_y3 = y3    

                ################
                # chose action
                ################
                action = 0
                if (np.random.random() <= self.epsilon):
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(y3)
                    
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                state = next_state
                
            # normalize every 10 episodes
            if (((e+1) % 10 == 0) and e > 0):
                self.normalize()

            # update epsilon after episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # update scores
            scores.append(i)
            mean_score = np.mean(scores)
            
            # every 100 iterations print it out
            if (((e+1) % 100 == 0) and e > 0):
                print (e, mean_score, self.epsilon)

                
if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
