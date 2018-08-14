
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from nn_sarsa import nn

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

class SarsaCartpole():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.gamma = 0.95
        self.lmda = 0.80
        self.epsilon = 1.0
        self.epsilon_min = 0.025
        self.epsilon_decay = 0.9975
        self.epsilon_max = 0.25
        self.epsilon_inc = 1.000
        self.alpha = 1e-6
        self.alpha_decay = 0.00
        self.n_episodes = 100000
        self.n_win_ticks = 195
        self.max_env_steps = 200

        EPSILON = 0.1
        
        LAYER1 = 8
        LAYER2 = 48
        LAYER3 = 24
        LAYER4 = 2
        
        weights1 = np.random.uniform(0.0, 1.0, size=(LAYER1 + 1, LAYER2)) * 2 * EPSILON
        weights2 = np.random.uniform(0.0, 1.0, size=(LAYER2 + 1, LAYER3)) * 2 * EPSILON
        weights3 = np.random.uniform(0.0, 1.0, size=(LAYER3 + 1, LAYER4)) * 2 * EPSILON

        self.model = nn(size=[LAYER1, LAYER2, LAYER3, LAYER4],  \
                        weights=[weights1, weights2, weights3], \
                        alpha=self.alpha,                       \
                        gamma=self.gamma,                       \
                        lmda=self.lmda,                         \
                        bias=True)

    def choose_action(self, state):
        values = self.model.predict(state)
    
        if (np.random.random() <= self.epsilon):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.model.predict(state))
            
        if (np.any(np.isnan(values)) or np.any(np.isinf(values))):
            assert(False)
        
        return action, values

    def train(self, state, action, reward, values, next_value, done):        
        if done:
            values[action] = reward
        else:
            values[action] = reward + self.gamma * next_value
                        
        self.model.train(state, action, values[action])

    def run(self):
        scores = deque(maxlen=100)
        prev = np.copy(self.model)

        for e in range(self.n_episodes):
            i = 0
            total_reward = 0

            done = False
            state = preprocess(self.env.reset())

            action, values = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = preprocess(next_state)

            while not done:
                next_action, next_values = self.choose_action(next_state)
                self.train(state, action, reward, values, next_values[next_action], done)
                
                state = next_state
                action = next_action
                values = next_values
                
                next_state, reward, done, _ = self.env.step(action)
                next_state = preprocess(next_state)
                
                #if e % 10 == 0:
                #    print (values, action)

                i += 1
                total_reward += reward
                
            if e % 100 == 0:
                print (np.max(self.model.e[0]), np.std(self.model.e[0]), np.max(self.model.e[1]), np.max(self.model.e[2]))
                
            self.train(state, action, 0.0, values, 0.0, done)
            self.model.clear()
            scores.append(i)
            mean_score = np.mean(scores)
            
            if total_reward <= mean_score and self.epsilon < self.epsilon_max:
                self.epsilon *= self.epsilon_inc
            elif total_reward > mean_score and self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if mean_score > 100:
                self.model.alpha = 1e-5
            else:
                self.model.alpha = 1e-4

            if e % 100 == 0:
                print(mean_score, self.epsilon)

if __name__ == '__main__':
    agent = SarsaCartpole()
    agent.run()
    
    
    
    
    
