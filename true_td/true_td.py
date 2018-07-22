
import gym
import math
import numpy as np
from collections import deque

class Solver():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.alpha = 0.0001
        self.alpha_decay = 0.01
        self.lmda = 0.9
        self.n_episodes = 10000
        self.n_win_ticks = 195
        self.max_env_steps = 200

        self.w0 = np.random.normal(0.05, 0.01, size=(4, 24))
        self.w1 = np.random.normal(0.05, 0.01, size=(24, 48))
        self.w2 = np.random.normal(0.05, 0.01, size=(48, 2))
        
        self.e0 = np.zeros(shape=(4, 1))
        self.e1 = np.zeros(shape=(24, 1))
        self.e2 = np.zeros(shape=(48, 1))
        
    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])
        
    def choose_action(self, state):
        return self.env.action_space.sample() if (np.random.random() <= self.epsilon) else np.argmax(self.predict(state))

    def predict(self, state):
        y0 = state
        y1 = np.dot(y0, self.w0)
        y2 = np.dot(y1, self.w1)
        y3 = np.dot(y2, self.w2)
        return y3

    def train(self, current_state, next_state, reward):
    
        vc = np.max(self.predict(current_state))
        vn = np.max(self.predict(next_state))
        d = reward + (self.gamma * vn) - vc
        assert not math.isnan(d) and not math.isinf(d)
        
        y0 = current_state
        y1 = np.dot(y0, self.w0)
        y2 = np.dot(y1, self.w1)
        y3 = np.dot(y2, self.w2)
        
        self.e0 = self.gamma * self.lmda * self.e0 + (1 - self.alpha * self.gamma * self.lmda * self.e0 * np.transpose(y0)) * np.transpose(y0)
        self.e1 = self.gamma * self.lmda * self.e1 + (1 - self.alpha * self.gamma * self.lmda * self.e1 * np.transpose(y1)) * np.transpose(y1)
        self.e2 = self.gamma * self.lmda * self.e2 + (1 - self.alpha * self.gamma * self.lmda * self.e2 * np.transpose(y2)) * np.transpose(y2)

        self.w0 = self.w0 + (self.alpha * (d + vn - vc) * self.e0) - (self.alpha * (vn - vc) * np.transpose(y0))
        self.w1 = self.w1 + (self.alpha * (d + vn - vc) * self.e1) - (self.alpha * (vn - vc) * np.transpose(y1))
        self.w2 = self.w2 + (self.alpha * (d + vn - vc) * self.e2) - (self.alpha * (vn - vc) * np.transpose(y2))
        
    def reset(self):
        self.e0 = np.zeros(shape=(4, 1))
        self.e1 = np.zeros(shape=(24, 1))
        self.e2 = np.zeros(shape=(48, 1))
        
    def normalize(self):
        col_sum = np.sum(np.copy(self.w0), axis=0)
        col_factor = 1.0 / col_sum
        for i in range(24):
            self.w0[:, i] *= col_factor[i]
            
        col_sum = np.sum(np.copy(self.w1), axis=0)
        col_factor = 1.0 / col_sum
        for i in range(48):
            self.w1[:, i] *= col_factor[i]
            
        col_sum = np.sum(np.copy(self.w2), axis=0)
        col_factor = 1.0 / col_sum
        for i in range(2):
            self.w2[:, i] *= col_factor[i]

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
        
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.train(state, next_state, reward)
                state = next_state
                i += 1

            self.reset()
            self.normalize()
            self.epsilon = self.epsilon * self.epsilon_decay
            
            scores.append(i)
            mean_score = np.mean(scores)
            if e % 100 == 0:
                print(mean_score)
       

if __name__ == '__main__':
    agent = Solver()
    agent.run()
    
    
    
    
    
    
    
    
