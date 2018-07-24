
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
        self.alpha = 1e-4
        self.alpha_decay = self.alpha
        self.lmda = 0.9
        self.n_episodes = 100000
        self.n_win_ticks = 195
        self.max_env_steps = 200

        self.LAYER0 = 4
        self.LAYER1 = 24
        self.LAYER2 = 48
        self.LAYER3 = 2

        self.w0 = np.random.normal(0.05, 0.01, size=(self.LAYER0, self.LAYER1))
        self.w1 = np.random.normal(0.05, 0.01, size=(self.LAYER1, self.LAYER2))
        self.w2 = np.random.normal(0.05, 0.01, size=(self.LAYER2, self.LAYER3))
        
        self.prev_w0 = self.w0
        self.prev_w1 = self.w1
        self.prev_w2 = self.w2
        
        self.e0 = np.zeros(shape=(self.LAYER0, 1))
        self.e1 = np.zeros(shape=(self.LAYER1, 1))
        self.e2 = np.zeros(shape=(self.LAYER2, 1))
        
    def preprocess_state(self, state):
        return np.reshape(state, [1, self.LAYER0])
        
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

        self.e0 = np.clip(self.e0 - np.percentile(self.e0, 96), 0.0, 1.0)
        self.e1 = np.clip(self.e1 - np.percentile(self.e1, 96), 0.0, 1.0)
        self.e2 = np.clip(self.e2 - np.percentile(self.e2, 96), 0.0, 1.0)

        dw2 = (self.alpha * (d + vn - vc) * self.e2) - (self.alpha * (vn - vc) * np.transpose(y2))
        dw1 = (self.alpha * (d + vn - vc) * self.e1) - (self.alpha * (vn - vc) * np.transpose(y1))
        dw0 = (self.alpha * (d + vn - vc) * self.e0) - (self.alpha * (vn - vc) * np.transpose(y0))

        self.w0 = np.clip(self.w0 + dw0, 0.0, 1.0)
        self.w1 = np.clip(self.w1 + dw1, 0.0, 1.0)
        self.w2 = np.clip(self.w2 + dw2, 0.0, 1.0)
        
    def reset(self):
        self.e0 = np.zeros(shape=(self.LAYER0, 1))
        self.e1 = np.zeros(shape=(self.LAYER1, 1))
        self.e2 = np.zeros(shape=(self.LAYER2, 1))
        
    def normalize(self):
    
        col_sum = np.sum(np.copy(self.w0), axis=0)
        col_factor = 1.0 / col_sum
        for i in range(self.LAYER1):
            self.w0[:, i] *= col_factor[i]
            
        col_sum = np.sum(np.copy(self.w1), axis=0)
        col_factor = 1.0 / col_sum
        for i in range(self.LAYER2):
            self.w1[:, i] *= col_factor[i]
            
        col_sum = np.sum(np.copy(self.w2), axis=0)
        col_factor = 1.0 / col_sum
        for i in range(self.LAYER3):
            self.w2[:, i] *= col_factor[i]
            
        '''
        self.w0 = self.w0 * (1.0 / np.average(self.w0))
        self.w1 = self.w1 * (1.0 / np.average(self.w1))
        self.w2 = self.w2 * (1.0 / np.average(self.w2))
        '''
        
    def decay_epsilon(self, score):
        '''
        if self.epsilon_min < self.epsilon 
            self.epsilon = self.epsilon * self.epsilon_decay
        '''
        if score < 25:
            self.epsilon = 0.5
        elif score < 75:
            self.epsilon = 0.35
        elif score < 125:
            self.epsilon = 0.25
        else:
            self.epsilon = 0.10
        
        
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
                
            scores.append(i)
            mean_score = np.mean(scores)
            
            if e % 100 == 0:
            
                print(mean_score, self.epsilon)
                print (np.sum(np.absolute(self.prev_w0 - self.w0)), np.sum(self.w0), np.std(self.w0), np.std(self.e0))
                print (np.sum(np.absolute(self.prev_w1 - self.w1)), np.sum(self.w1), np.std(self.w1), np.std(self.e1))
                print (np.sum(np.absolute(self.prev_w2 - self.w2)), np.sum(self.w2), np.std(self.w2), np.std(self.e2))
                
                self.prev_w0 = self.w0
                self.prev_w1 = self.w1
                self.prev_w2 = self.w2

            self.reset()
            self.normalize()
            self.decay_epsilon(mean_score)

       

if __name__ == '__main__':
    agent = Solver()
    agent.run()
    
    
    
    
    
    
    
    
