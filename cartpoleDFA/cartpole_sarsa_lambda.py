
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from nn_sarsa import nn

class SarsaCartpole():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.gamma = 0.9
        self.lmda = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_max = 0.25
        self.epsilon_inc = 1.005
        self.alpha = 1e-4
        self.alpha_decay = 0.00
        self.n_episodes = 100000
        self.n_win_ticks = 195
        self.max_env_steps = 200

        EPSILON = 0.01
        
        LAYER1 = 4
        LAYER2 = 24
        LAYER3 = 2
        
        weights1 = np.random.uniform(0.0, 1.0, size=(LAYER1 + 1, LAYER2)) * 2 * EPSILON - EPSILON
        weights2 = np.random.uniform(0.0, 1.0, size=(LAYER2 + 1, LAYER3)) * 2 * EPSILON - EPSILON
                        
        self.model = nn(size=[LAYER1, LAYER2, LAYER3], \
                        weights=[weights1, weights2],  \
                        alpha=self.alpha,              \
                        gamma=self.gamma,              \
                        lmda=self.lmda,                \
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

        for e in range(self.n_episodes):
            i = 0
            total_reward = 0

            done = False
            state = self.env.reset()

            action, values = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)

            while not done:
                next_action, next_values = self.choose_action(next_state)
                self.train(state, action, reward, values, next_values[next_action], done)
                
                state = next_state
                action = next_action
                values = next_values
                
                next_state, reward, done, _ = self.env.step(action)
                
                if e % 10 == 0:
                    print (values, action)

                i += 1
                total_reward += reward

            self.model.clear()
            scores.append(i)
            mean_score = np.mean(scores)
            
            if total_reward <= mean_score and self.epsilon < self.epsilon_max:
                self.epsilon *= self.epsilon_inc
            elif total_reward > mean_score and self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                            
            if e % 10 == 0:
                print(mean_score, self.epsilon)

if __name__ == '__main__':
    agent = SarsaCartpole()
    agent.run()
    
    
    
    
    
