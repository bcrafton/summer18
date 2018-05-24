
import numpy as np

def manhatten(coord, dest):
    return abs(dest[0] - coord[0]) + abs(dest[1] - coord[1])

class Env():
    def __init__(self):
        self.state = [0, 0]
        self.steps = 0

        self.nrows = 4
        self.ncols = 4

        self.wins = [[3,3]]
        self.fails = [[2,2]]

        self.left = []
        self.right = []
        self.up = []
        self.down = []

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
                    diff = manhatten([ii, jj], [3,3]) - manhatten(next_state, [3, 3])

                    # print (ii, jj, kk, diff)

                    if (diff > 0):
                        self.moves[ii][jj][kk] = 110
                    elif (diff < 0):
                        self.moves[ii][jj][kk] = 10
                    else:
                        self.moves[ii][jj][kk] = 10

    def reset(self):
        self.state = [0, 0]
        self.steps = 0
        return self.state_to_num(self.state)

    def state_to_num(self, state):
        return state[0] * self.ncols + state[1]

    def num_to_state(self, state):
        return [state/self.ncols, state%self.ncols]

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

        self.state = next_state
        return self.state_to_num(self.state), reward, done
