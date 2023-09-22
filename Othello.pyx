from __future__ import print_function
import numpy as np
cimport numpy as np

cdef class Othello:
    cpdef readonly int N
    cpdef readonly np.ndarray init_board
    cpdef readonly str black
    cpdef readonly str white
    cpdef readonly int action_space
    cpdef readonly int winNum
    cpdef readonly np.ndarray board
    cpdef readonly int turn
    cpdef readonly list valid_actions
    def __cinit__(self):
        self.N = 8
        self.init_board = np.array([['.'] * self.N] * self.N)
        self.black = 'O'
        self.white = 'X'
        self.action_space = self.N ** 2 + 1
        self.winNum = 3 # for neural network

        cdef int numOne
        cdef int numTwo

        numOne = int(self.N / 2) - 1
        numTwo = int(self.N / 2)

        cdef int i
        cdef int j
        for i in [numOne, numTwo]:
            for j in [numOne, numTwo]:
                if i == j:
                    self.init_board[i][j] = self.white
                else:
                    self.init_board[i][j] = self.black


        # Dynamic variables
        self.board = self.init_board.copy()
        self.turn = 0
        self.valid_actions = self.find_valid_actions()

    cpdef np.ndarray get_state(self):
        cdef np.ndarray state
        state = np.array([[[0,0,self.turn]] * self.N] * self.N, dtype=np.uint8)
        cdef int i
        cdef int j
        for i in range(self.N):
            for j in range(self.N):
                if self.board[i][j] == self.black:
                    state[i][j][0] = 1
                elif self.board[i][j] == self.white:
                    state[i][j][1] = 1
        return state

    cpdef np.ndarray reset(self):
        self.board = self.init_board.copy()
        self.turn = 0
        self.valid_actions = self.find_valid_actions()
        return self.get_state()

    cpdef void set_game(self, np.ndarray board, int turn):
        self.board = board.copy()
        self.turn = turn
        self.valid_actions = self.find_valid_actions()
        if len(self.valid_actions) == 0:
            self.valid_actions.append(self.action_space-1)

    cpdef tuple count_stones(self):
        cdef int b_count = 0
        cdef int w_count = 0
        cdef int i
        cdef int j
        for i in range(self.N):
            for j in range(self.N):
                if self.board[i][j] == self.black:
                    b_count += 1
                elif self.board[i][j] == self.white:
                    w_count += 1
        return b_count, w_count

    cpdef tuple step(self, int a):
        if a not in self.valid_actions:
            raise NameError('Invalid action')

        cdef int b_count
        cdef int w_count

        if a == self.action_space - 1:
            if self.turn == 0:
                self.turn = 1
            else:
                self.turn = 0
            self.valid_actions = self.find_valid_actions()
            if len(self.valid_actions) == 0:
                self.valid_actions.append(self.action_space-1)
                b_count, w_count = self.count_stones()
                if b_count == w_count:
                    return False, False, True
                if b_count > w_count:
                    return True, False, False
                if w_count > b_count:
                    return False, True, False
            return False, False, False

        cdef int x
        cdef int y
        cdef int dx
        cdef int dy
        cdef int num
        x, y = self.transform(a)
        if self.turn == 0:
            self.board[x][y] = self.black
            for dx, dy in [[1, 0], [0, 1], [1, 1], [1, -1]]:
                for num in [-1, 1]:
                    self.flip(x, y, dx * num, dy * num, self.black, [], False)
            self.turn = 1
        else:
            self.board[x][y] = self.white
            for dx, dy in [[1, 0], [0, 1], [1, 1], [1, -1]]:
                for num in [-1, 1]:
                    self.flip(x, y, dx * num, dy * num, self.white, [], False)
            self.turn = 0
        self.valid_actions = self.find_valid_actions()

        if len(self.valid_actions) == 0:
            self.valid_actions.append(self.action_space - 1)

        return False, False, False


    cpdef void flip(self, int x, int y, int dx, int dy, str stone, list flip_list, mode):
        if x + dx < 0 or y + dy < 0 or x + dx >= self.N or y + dy >= self.N or self.board[x + dx][y + dy] == '.':
            return

        if (not mode) and self.board[x + dx][y + dy] == stone:
            return

        if mode and self.board[x + dx][y + dy] == stone:
            for x, y in flip_list:
                self.board[x][y] = stone
            return

        flip_list.append([x + dx, y + dy])
        self.flip(x+dx, y+dy, dx, dy, stone, flip_list, True)

    cpdef list b_played(self):
        cdef list ls = []
        cdef int i
        cdef int j
        for i in range(self.N):
            for j in range(self.N):
                if self.board[i][j] == self.black:
                    ls.append([i,j])

        return ls

    cpdef list w_played(self):
        cdef list ls = []
        cdef int i
        cdef int j
        for i in range(self.N):
            for j in range(self.N):
                if self.board[i][j] == self.white:
                    ls.append([i, j])

        return ls

    cpdef list find_valid_actions(self):
        cdef list vas
        cdef int dx
        cdef int dy
        cdef int x
        cdef int y
        cdef int num
        cdef valid
        cdef int newX
        cdef int newY
        if self.turn == 0:
            vas = []
            for x, y in self.b_played():
                for dx, dy in [[1, 0], [0, 1], [1, 1], [1, -1]]:
                    for num in [-1, 1]:
                        valid, newX, newY = self.find_valid(x, y, dx * num, dy * num, 0)
                        if valid:
                            vas.append(self.revert(newX, newY))
            return vas

        else:
            vas = []
            for x, y in self.w_played():
                for dx, dy in [[1, 0], [0, 1], [1, 1], [1, -1]]:
                    for num in [-1, 1]:
                        valid, newX, newY = self.find_valid(x, y, dx * num, dy * num, 0)
                        if valid:
                            vas.append(self.revert(newX, newY))
            return vas



    cpdef tuple find_valid(self, int x, int y, int dx, int dy, int mode):
        x += dx
        y += dy
        if x < 0 or y < 0 or x >= self.N or y >= self.N:
            return False, x, y

        if self.turn == 0:
            if mode == 0:
                if self.board[x][y] != self.white:
                    return False, x, y
            else:
                if self.board[x][y] == self.black:
                    return False, x, y

                if self.board[x][y] == '.':
                    return True, x, y
            return self.find_valid(x, y, dx, dy, 1)
        else:
            if mode == 0:
                if self.board[x][y] != self.black:
                    return False, x, y
            else:
                if self.board[x][y] == self.white:
                    return False, x, y

                if self.board[x][y] == '.':
                    return True, x, y
            return self.find_valid(x, y, dx, dy, 1)

    cpdef void render(self):
        cdef int i
        cdef int j
        for i in range(self.N):
            for j in range(self.N):
                print(self.board[i][j], end = " ")
            print("")
        print("")

        for _ in range(5):
            print('')


    cpdef int revert(self, int x, int y):
        return x * self.N + y

    cpdef tuple transform(self, int action):
        cdef int x
        cdef int y
        x = int(action / self.N)
        y = action % self.N
        return x, y

    cpdef list getValidActions(self):
        return self.valid_actions

    cpdef list getSymmetries(self, np.ndarray state, np.ndarray pi, z):
        cdef list ss = []
        ss.append([state, pi, z])
        cdef double tempV = pi[self.N**2]

        cdef np.ndarray s
        cdef np.ndarray p
        cdef np.ndarray s1
        cdef np.ndarray p1
        s = state.copy()
        p = np.array(pi[:self.N**2]).copy()
        p = p.reshape(self.N, self.N)

        s1 = np.rot90(s, 2)
        p1 = np.rot90(p, 2)
        p1 = p1.reshape(self.N ** 2)
        p1 = np.append(p1, tempV)
        ss.append([s1, p1, z])

        s1 = np.rot90(s, 1)
        p1 = np.rot90(p, 1)
        s1 = self.reflect(s1)
        p1 = self.reflect(p1)
        p1 = p1.reshape(self.N ** 2)
        p1 = np.append(p1, tempV)
        ss.append([s1, p1, z])

        s1 = np.rot90(s, 3)
        p1 = np.rot90(p, 3)
        s1 = self.reflect(s1)
        p1 = self.reflect(p1)
        p1 = p1.reshape(self.N ** 2)
        p1 = np.append(p1, tempV)
        ss.append([s1, p1, z])
        return ss

    cpdef np.ndarray reflect(self, np.ndarray arr):
        cdef np.ndarray s = np.array(arr).copy()
        cdef int i
        for i in range(len(s)):
            s[i] = s[i][::-1]
        return s

    cpdef void draw_state(self, np.ndarray state):
        cdef int i
        cdef int j
        for i in range(self.N):
            for j in range(self.N):
                if state[i][j][0] == 1:
                    print(self.black, end = " ")
                elif state[i][j][1] == 1:
                    print(self.white, end = " ")
                else:
                    print('.', end = " ")
            print("")
        print("")
