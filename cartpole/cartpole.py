# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def get_reward(state, next_state, done):
    if done:
        return -0.5

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
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
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

        # Init model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=8, activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

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
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                reward = get_reward(state, next_state, done)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)
        
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
