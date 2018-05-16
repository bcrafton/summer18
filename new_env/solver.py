
import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import time
import tkinter as tk
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 4  # grid height
WIDTH = 4  # grid width


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # add img to canvas
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        self.triangle1 = canvas.create_image(250, 150, image=self.shapes[1])
        # self.triangle2 = canvas.create_image(150, 250, image=self.shapes[1])
        self.circle = canvas.create_image(250, 250, image=self.shapes[2])

        # pack all
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("./img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("./img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("./img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):

        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 5
        else:
            origin_x, origin_y = 42, 77

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()
        for i in range(HEIGHT):
            for j in range(WIDTH):
                for action in range(0, 4):
                    state = [i, j]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        self.text_value(j, i, round(temp, 2), action)

    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def state_to_coords(self, state):
        x = int(state[0] * 100 + 50)
        y = int(state[1] * 100 + 50)
        return [x, y]

    def reset(self):
        self.update()
        # time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        # self.render()
        # return observation
        return self.coords_to_state(self.canvas.coords(self.rectangle))


    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        # self.render()

        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        # move agent
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        # move rectangle to top level of canvas
        self.canvas.tag_raise(self.rectangle)
        next_state = self.canvas.coords(self.rectangle)

        # reward function
        if next_state == self.canvas.coords(self.circle):
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.triangle1)]: # [self.canvas.coords(self.triangle1), self.canvas.coords(self.triangle2)]:
            reward = -100
            done = True
        else:
            reward = -5
            done = False

        next_state = self.coords_to_state(next_state)
        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()

class Solver():
    def __init__(self, n_episodes=100000, max_env_steps=None, gamma=1.0, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.99, alpha=0.04, alpha_decay=0.04, batch_size=1, quiet=False):
        self.memory = deque(maxlen=1)
        self.env = Env()
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
            ret = random.randint(0, 3)
        else:
            ret = np.argmax(self.model.predict(state))

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
        real_scores = deque(maxlen=100)
        fluff_scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.env.reset()
            state = self.preprocess_state(state)

            # print (self.env.spec.timestep_limit)
            reward_sum = 0
            step = 0
            done = False
            while not done:
                step = step + 1
                action = self.choose_action(state, self.get_epsilon(e))
                # print (step, action)

                next_state, reward, done = self.env.step(action)
                next_state = self.preprocess_state(next_state)

                reward_sum = reward_sum + reward

                self.remember(state, action, reward, next_state, done)

                self.replay(self.batch_size)

                state = next_state

                if done:
                    # self.replay(self.batch_size)

                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay ** (e / self.decay_step)

                    real_scores.append(reward_sum > 0)
                    real_mean_score = np.mean(real_scores)

                    fluff_scores.append(reward > 0)
                    fluff_mean_score = np.mean(fluff_scores)

                    print (step, reward_sum, real_mean_score, fluff_mean_score)
                    # print (self.get_epsilon(e))
                    # print (self.model.get_weights())

        np.save("weights", self.model.get_weights())
        

if __name__ == '__main__':
    agent = Solver()
    agent.run()
