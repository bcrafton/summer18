
import numpy as np
import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 5  # grid height
WIDTH = 5  # grid width

def manhatten(coord, dest):
    return abs(dest[0] - coord[0]) + abs(dest[1] - coord[1])

class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.title('Q Learning')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))

        self.state = [0, 0]
        self.steps = 0

        self.nrows = 5
        self.ncols = 5

        self.wins = [[3,3]]
        self.fails = [[3,2], [2,3]]

        self.left = []
        self.right = []
        self.up = []
        self.down = []

        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []

        for ii in range(self.nrows):
            for jj in range(self.ncols):
                if ( ii == self.nrows-1 ):
                    self.up.append([ii, jj])

                if ( ii == 0 ):
                    self.down.append([ii, jj])

                if ( jj == self.ncols-1 ):
                    self.right.append([ii, jj])

                if ( jj == 0 ):
                    self.left.append([ii, jj])

        self.moves = np.zeros(shape=(self.nrows, self.ncols, 4))
        for ii in range(self.nrows):
            for jj in range(self.ncols):
                for kk in range(4):

                    next_state = self.next_state(kk, [ii, jj])
                    diff = manhatten([ii, jj], [4,4]) - manhatten(next_state, [4, 4])

                    # print (ii, jj, kk, diff)

                    if (diff > 0):
                        self.moves[ii][jj][kk] = 50
                    elif (diff < 0):
                        self.moves[ii][jj][kk] = 10
                    else:
                        self.moves[ii][jj][kk] = 20

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
        self.rectangle = canvas.create_image(50, 500 - 50, image=self.shapes[0])
        self.triangle1 = canvas.create_image(250, 500 - 350, image=self.shapes[1])
        self.triangle2 = canvas.create_image(350, 500 - 250, image=self.shapes[1])
        self.circle = canvas.create_image(350, 500 - 350, image=self.shapes[2])
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
        for i in range(WIDTH):
            for j in range(HEIGHT):
                for action in range(4):
                    temp = q_table[i][j][action]
                    self.text_value(i, j, round(temp, 2), action)

    def reset(self):
        self.update()
        time.sleep(0.5)

        self.state = [0, 0]
        self.steps = 0

        # self.canvas.move(self.rectangle, 50 - self.state[0] * 100, 450 - ((4 - self.state[1]) * 100))
        coords = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, 50 - coords[0], 450 - coords[1])
        self.render()

        return self.state_to_num(self.state)

    def state_to_num(self, state):
        return state[0] * self.ncols + state[1]

    def num_to_state(self, state):
        return [state/self.ncols, state%self.ncols]

    def coords_to_state(self, coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def state_to_coords(self, state):
        x = int(state[0] * 100 + 50)
        y = int(state[1] * 100 + 50)
        return [x, y]

    def image_move(self, action, current_state):
        if (action == 0) and (current_state not in self.up):
            return [0, -100]
        elif (action == 1) and (current_state not in self.down):
            return [0, 100]
        elif (action == 2) and (current_state not in self.left):
            return [-100, 0]
        elif (action == 3) and (current_state not in self.right):
            return [100, 0]
        else:
            return [0, 0]

    def next_state(self, action, current_state):
        next_state = current_state
        if (action == 0) and (current_state not in self.up):
            next_state = [current_state[0] + 1, current_state[1]]
        elif (action == 1) and (current_state not in self.down):
            next_state = [current_state[0] - 1, current_state[1]]
        # left = 2, right = 3 ...backwards but w.e.
        elif (action == 2) and (current_state not in self.left):
            next_state = [current_state[0], current_state[1] - 1]
        elif (action == 3) and (current_state not in self.right):
            next_state = [current_state[0], current_state[1] + 1]
        return next_state

    def step(self, action):
        self.steps = self.steps + 1

        next_state = self.next_state(action, self.state)
        image_move = self.image_move(action, self.state)

        if next_state in self.wins:
            reward = 1000
            done = True
        elif next_state in self.fails:
            reward = 0
            done = True
        else:
            reward = self.moves[self.state[0]][self.state[1]][action]
            if (self.steps >= 20):
                done = True
            else:
                done = False

        self.render()
        self.canvas.move(self.rectangle, image_move[0], image_move[1])
        self.canvas.tag_raise(self.rectangle)

        self.state = next_state
        return self.state_to_num(self.state), reward, done

    def render(self):
        time.sleep(0.03)
        self.update()
