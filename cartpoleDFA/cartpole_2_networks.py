
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from nn_dfa import NNDFA

class DQNCartPoleSolver():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_max = 0.25
        self.epsilon_inc = 1.005
        self.alpha = 0.0005
        self.alpha_decay = 0.00
        self.n_episodes = 100000
        self.n_win_ticks = 195
        self.max_env_steps = 200

        EPSILON = 1.0
        
        LAYER1 = 4
        LAYER2 = 24
        LAYER3 = 1
        
        ################################################################################################################################
        weights1 = np.random.uniform(0.0, 1.0, size=(LAYER1 + 1, LAYER2)) * 2 * EPSILON - EPSILON
        weights2 = np.random.uniform(0.0, 1.0, size=(LAYER2 + 1, LAYER3)) * 2 * EPSILON - EPSILON
        
        b1 = np.random.uniform(0.25, 0.75, size=(LAYER1 + 1, LAYER3)) * 2 * EPSILON - EPSILON
        b2 = np.random.uniform(0.25, 0.75, size=(LAYER2 + 1, LAYER3)) * 2 * EPSILON - EPSILON
                        
        self.left = NNDFA(size=[LAYER1, LAYER2, LAYER3], weights=[weights1, weights2], fb_weights=[b1, b2], alpha=self.alpha, bias=True)
        ################################################################################################################################
        weights1 = np.random.uniform(0.0, 1.0, size=(LAYER1 + 1, LAYER2)) * 2 * EPSILON - EPSILON
        weights2 = np.random.uniform(0.0, 1.0, size=(LAYER2 + 1, LAYER3)) * 2 * EPSILON - EPSILON
        
        b1 = np.random.uniform(0.25, 0.75, size=(LAYER1 + 1, LAYER3)) * 2 * EPSILON - EPSILON
        b2 = np.random.uniform(0.25, 0.75, size=(LAYER2 + 1, LAYER3)) * 2 * EPSILON - EPSILON
                        
        self.right = NNDFA(size=[LAYER1, LAYER2, LAYER3], weights=[weights1, weights2], fb_weights=[b1, b2], alpha=self.alpha, bias=True)
        ################################################################################################################################

    def choose_action(self, state):
        if (np.random.random() <= self.epsilon):
            action = self.env.action_space.sample()
        else:
            l = self.left.predict(state)
            r = self.right.predict(state)
            if l > r:
                action = 0
            else:
                action = 1
                
        return action
        
    def predict(self, state):
        l = self.left.predict(state)
        r = self.right.predict(state)
        return max(l, r)
        
    def train(self, state, action, reward, next_state, done):
        if done:
            target = np.array(reward)
        else:
            target = np.array(reward + self.gamma * self.predict(next_state))
                       
        if action==0:
            self.left.train(state, target)
        else:
            self.right.train(state, target)

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            total_reward = 0
            state = self.env.reset()
            done = False
            i = 0
                           
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                                
                self.train(state, action, reward, next_state, done)
                state = next_state
                i += 1
                total_reward += reward

            scores.append(i)
            mean_score = np.mean(scores)
            
            if total_reward <= mean_score and self.epsilon < self.epsilon_max:
                self.epsilon *= self.epsilon_inc
            elif total_reward > mean_score and self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                            
            if e % 10 == 0:
                print(mean_score, self.epsilon)

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
    
    
    
    
    
