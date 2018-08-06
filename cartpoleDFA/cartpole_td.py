
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from nn_td import nn_td

def preprocess(state):
    new_state = np.zeros(8)
    new_state[0] = state[0]      if state[0] > 0 else 0
    new_state[1] = abs(state[0]) if state[0] < 0 else 0
    new_state[2] = state[1]      if state[1] > 0 else 0
    new_state[3] = abs(state[1]) if state[1] < 0 else 0
    new_state[4] = state[2]      if state[2] > 0 else 0
    new_state[5] = abs(state[2]) if state[2] < 0 else 0
    new_state[6] = state[3]      if state[3] > 0 else 0
    new_state[7] = abs(state[3]) if state[3] < 0 else 0
    return new_state

class DQNCartPoleSolver():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.gamma = 0.90
        self.lmda = 0.90
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.epsilon_max = 0.25
        self.epsilon_inc = 1.005
        self.alpha = 1e-4
        self.alpha_decay = 0.00
        self.n_episodes = 100000
        self.n_win_ticks = 195
        self.max_env_steps = 200
        
        self.num_layers = 3
        self.LAYER1 = 8
        self.LAYER2 = 24
        self.LAYER3 = 2
        ################################################################################################################################
        
        weights1 = np.random.uniform(0.0, 1.0, size=(self.LAYER1+1, self.LAYER2)) 
        weights2 = np.random.uniform(0.0, 1.0, size=(self.LAYER2+1, self.LAYER3)) 
        
        self.model = nn_td(size=[self.LAYER1, self.LAYER2, self.LAYER3], weights=[weights1, weights2], alpha=self.alpha, gamma=self.gamma, lmda=self.lmda, bias=True)
        ################################################################################################################################

    def choose_action(self, state):
        values = self.model.predict(state)

        if (np.random.random() <= self.epsilon):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(values)

        print(values)
        return action, values
                
    def train(self, state, prev_action, action, reward, prev_values, values):
        self.model.train(state, prev_action, action, reward, prev_values, values)

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            i = 0
            total_reward = 0
            
            done = False
            state = preprocess(self.env.reset())
            
            action, values = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            state = preprocess(next_state)
            prev_values = values
            prev_action = action
                
            while not done:            
                action, values = self.choose_action(state)
                self.train(state, prev_action, action, reward, prev_values, values)
                
                next_state, reward, done, _ = self.env.step(action)
                state = preprocess(next_state)
                prev_values = values
                prev_action = action

                i += 1
                total_reward += reward

            self.train(state, prev_action, action, reward, prev_values, np.zeros(self.LAYER3))

            scores.append(i)
            mean_score = np.mean(scores)
            # print (np.std(self.model.e[0]), np.min(self.model.e[0]), np.max(self.model.e[0]))
            self.model.clear()
            
            if total_reward <= mean_score and self.epsilon < self.epsilon_max:
                self.epsilon *= self.epsilon_inc
            elif total_reward > mean_score and self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
               
            if e % 10 == 0:
                print(mean_score, self.epsilon)
                # print(self.model.maxs(), self.model.avgs())

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
    
    
    
    
    
