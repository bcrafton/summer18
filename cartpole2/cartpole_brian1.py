
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# what was the policy gradient thing ??? 
# want to understand what they were doing ... 
# https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py

# http://kvfrans.com/simple-algoritms-for-solving-cartpole/
# https://keon.io/deep-q-learning/

class DQNCartPoleSolver():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_max = 0.5
        self.epsilon_inc = 1.000
        self.alpha = 0.001
        self.alpha_decay = 0.00
        self.n_episodes = 10000
        self.n_win_ticks = 195
        self.max_env_steps = 200

        self.model = Sequential()
        self.model.add(Dense(10, input_dim=4, activation='relu'))
        self.model.add(Dense(2, activation='relu'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

    def choose_action(self, state):
        return self.env.action_space.sample() if (np.random.random() <= self.epsilon) else np.argmax(self.model.predict(state))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def train(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        
        if done:
            target[0][action] = reward
            # target = target - self.model.predict(state)
        else:
            target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            # target = target - self.model.predict(state)
            
        self.model.fit(state, target, verbose=0)

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            total_reward = 0
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = self.preprocess_state(next_state)
                self.train(state, action, reward, next_state, done)
                state = next_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            
            if total_reward <= mean_score and self.epsilon < self.epsilon_max:
                # print("inc")
                self.epsilon *= self.epsilon_inc
            elif total_reward > mean_score and self.epsilon > self.epsilon_min:
                # print("decay")
                self.epsilon *= self.epsilon_decay
            
            if e % 10 == 0:
                print(mean_score, self.epsilon)
                # print(self.model.get_weights())

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
    
    
    
    
    
