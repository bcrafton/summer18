# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def get_reward(done, reward):
    """
    Return a more meaningful reward (i.e. depending on the outcome of every action and not just on winning)
    :param done: bool indicating if the episode if finished
    :param reward: reward returned by the env after taking an action
    :return:
    """
    # Failure
    if done and reward == 0:
        reward = - 100.0
    # Win
    elif done:
        reward = 100.0
    # Move to another case
    else:
        reward = - 1.0
    return reward

class DQNCartPoleSolver():
    def __init__(self, n_episodes=10000, max_env_steps=None, gamma=1.0, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.99, alpha=0.04, alpha_decay=0.01, batch_size=1, quiet=False):
        self.memory = deque(maxlen=1)
        self.env = gym.make('FrozenLake-v0')
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.quiet = quiet
        self.decay_step = 10
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential()
        self.model.add(Dense(4, input_dim=16, activation='linear'))
        # self.model.add(Dense(48, activation='linear'))
        # self.model.add(Dense(4, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            ret = self.env.action_space.sample()
        else:
            # argmax is different than max.
            ret = np.argmax(self.model.predict(state))

        # print (ret)
        # ret = self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))
        # print (ret)

        return ret

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        # return np.reshape(state, [1, 4])
        # return state
        # return np.reshape(state, [1, 1])

        ret = np.zeros(16)
        for i in range(16):
            if state == i:
                ret[i] = 1
        return np.reshape(ret, [1, 16])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state))
            x_batch.append(state[0])
            y_batch.append(y_target[0])

            # print (np.average(y_target[0]))
        
            # print (len(self.model.predict(next_state)))
            # print (np.max(self.model.predict(next_state)[0]))
            # print (np.max(self.model.predict(next_state)))

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.env.reset()
            state = self.preprocess_state(state)

            # print (self.env.spec.timestep_limit)
            for step in range(self.env.spec.timestep_limit):
                # print (state.shape())
                action = self.choose_action(state, self.get_epsilon(e))

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)

                reward = get_reward(done, reward)

                self.remember(state, action, reward, next_state, done)

                self.replay(self.batch_size)

                state = next_state

                if done:
                    # self.replay(self.batch_size)

                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay ** (e / self.decay_step)

                    scores.append(reward > 0)
                    mean_score = np.mean(scores)

                    print (mean_score)
                    # print (self.get_epsilon(e))
                    # print (self.model.get_weights())
                    np.save("weights", self.model.get_weights())

                    break
        

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
