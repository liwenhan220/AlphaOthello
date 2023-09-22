from Othello import Othello
import numpy as np
cimport numpy as np
import tensorflow as tf
import math
import time

cdef class MCTSNode:
    cpdef public np.ndarray link
    cpdef public np.ndarray pol
    cpdef public np.ndarray N
    cpdef public np.ndarray P
    cpdef public np.ndarray W
    cpdef public int virtual_loss
    cpdef public MCTSNode parent
    cpdef public int pre_move
    cpdef readonly isDummy
    def __cinit__(self, int action_space, np.ndarray pol, MCTSNode parent, int pre_move):
        self.link = np.array([None for _ in range(action_space)])
        self.P = pol
        self.N = np.zeros((action_space), dtype = np.int)
        self.W = np.zeros((action_space), dtype = np.float)
        self.virtual_loss = 0
        self.parent = parent
        self.pre_move = pre_move
        self.isDummy = False

cdef class DummyNode:
    cpdef public int virtual_loss
    cpdef readonly isDummy
    cpdef public pre_move
    cpdef public parent
    def __cinit__(self, pre_move, parent):
        self.virtual_loss = 0
        self.isDummy = True
        self.pre_move = pre_move
        self.parent = parent


cdef class MCTS:
    cpdef public float cpuct
    cpdef public float alpha
    cpdef public double temperature
    cpdef public game
    cpdef public sim_game
    cpdef public nn
    cpdef public MCTSNode root
    cpdef public debug
    cpdef public float noise_prop
    def __cinit__(self, nn, inject_noise=True, debug=False):
        self.cpuct = 1.5
        self.alpha = 0.3
        self.temperature = 1.2
        self.noise_prop = 0.25
        self.game = Othello()
        self.sim_game = Othello()
        self.nn = nn
        cdef np.ndarray state = self.game.reset()
        cdef np.ndarray p
        cdef float v
        p, v = self.pred(state)
        self.root = MCTSNode(self.game.action_space, p, None, -1)
        if inject_noise:
            self.inject_noise(self.root, self.game)
        self.debug = debug

    cpdef void backup(self, MCTSNode node, double val):
        if node.parent is None:
            return
        node.parent.W[node.pre_move] += val
        node.virtual_loss -= 1
        self.backup(node.parent, -val)

    cpdef tuple step(self, int a, inject_noise=True):
        cdef np.ndarray p
        cdef float v
        if self.root.link[a] is None:
            self.copy()
            black_win, white_win, draw = self.sim_game.step(a)
            if (black_win or white_win or draw):
                self.game.step(a)
                return (black_win, white_win, draw)
            else:
                p, v = self.pred(self.sim_game.get_state())
                self.root.link[a] = MCTSNode(self.game.action_space, p, self.root, a)
        self.root = self.root.link[a]
        if self.root is not None:
            self.root.parent = None
            self.root.pre_move = -1
        b_win, w_win, draw = self.game.step(a)
        if inject_noise:
            self.inject_noise(self.root, self.game)
        return (b_win, w_win, draw)

    cpdef void cleanup(self, node):
        if node.parent is None:
            return
        node.parent.N[node.pre_move] -= 1
        self.cleanup(node.parent)

    cpdef tuple expandToLeaf(self):
        self.copy()
        cdef MCTSNode curNode = self.root
        cdef int a
        cdef np.ndarray p
        cdef float v
        while True:
            a = self.find_best_action(curNode, self.sim_game)
            black_win, white_win, draw = self.sim_game.step(a)
            curNode.N[a] += 1
            if (self.sim_game.turn == 0 and black_win) or (self.sim_game.turn == 1 and white_win):
                curNode.W[a] -= 1
                self.backup(curNode, 1)
                return (None, None, None)
            if (self.sim_game.turn == 1 and black_win) or (self.sim_game.turn == 0 and white_win):
                curNode.W[a] += 1
                self.backup(curNode, -1)
                return (None, None, None)
            if draw:
                curNode.W[a] += 0
                self.backup(curNode, 0)
                return (None, None, None)
            if (curNode.link[a] is None):
                curNode.link[a] = DummyNode(a, curNode)
                curNode.link[a].virtual_loss += 1
                return (self.sim_game, a, curNode)
            if (curNode.link[a].isDummy):
                self.cleanup(curNode.link[a])
                return (None, None, None)
            curNode = curNode.link[a]
            curNode.virtual_loss += 1

    cpdef void parallel_sim(self, int numThreads = 1):
        cdef game
        cdef list states = []
        cdef list actions = []
        cdef list nodes = []
        cdef pols
        cdef vals

        cdef state
        cdef action
        cdef node
        cdef np.ndarray pol
        cdef float val

        for _ in range(numThreads):
            game, action, node = self.expandToLeaf()
            if game is not None:
                states.append(game.get_state())
                actions.append(action)
                nodes.append(node)
        if len(states) == 0:
            return
        pols, vals = self.nn(tf.constant(np.array(states).astype(np.float32)))
        pols = np.array(pols)
        vals = np.array(vals)

        for i in range(len(states)):
            pol = pols[i]
            val = vals[i][0]
            action = actions[i]
            node = nodes[i]
            if node.link[action].isDummy:
                node.link[action] = MCTSNode(self.game.action_space, pol, node, action)
                node.link[action].virtual_loss += 1
                self.backup(node.link[action], -val)

    cpdef void inject_noise(self, MCTSNode node, game):
        if node is None:
            return
        cdef int legal_moves = len(game.getValidActions())
        cdef np.ndarray a = np.array([self.alpha] * (game.action_space)) / legal_moves
        cdef np.ndarray dirichlet = np.random.dirichlet(a)
        node.P = node.P * (1 - self.noise_prop) + self.noise_prop * dirichlet

    cpdef int find_best_action(self, MCTSNode node, game):
        cdef int best_a = -1
        cdef float max_u =-float('inf')
        cdef int a
        cdef float u
        for a in game.getValidActions():
            if node.link[a] is not None:
                u = (node.W[a] - node.link[a].virtual_loss) / (node.N[a] + 1) + self.cpuct*node.P[a]*math.sqrt(sum(node.N))/(1+node.N[a])
            else:
                u = (node.W[a]) / (node.N[a] + 1) + self.cpuct*node.P[a]*math.sqrt(sum(node.N))/(1+node.N[a])
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a
        return a

    cpdef np.ndarray pi(self):
        N = self.root.N
        return N ** self.temperature / sum(N ** self.temperature)

    cpdef copy(self):
        self.sim_game.set_game(self.game.board, self.game.turn)

    cpdef tuple pred(self, np.ndarray state):
        p, v = self.nn(tf.constant(np.array([state]).astype(np.float32)))
        return np.array(p)[0], np.array(v)[0][0]

    cpdef void render(self):
        self.game.render()
    
    cpdef np.ndarray get_state(self):
        return self.game.get_state()

    cpdef float get_winrate(self):
        cdef int a
        a = np.argmax(self.root.N)
        return self.root.W[a] / (self.root.N[a] + 1) * 0.5 + 0.5

    cpdef void sim(self, int numSims, int numThreads = 1):
        cdef int curSims
        if self.root is None:
            curSims = 0
        else:
            curSims = sum(self.root.N)
        while self.root is None or sum(self.root.N) <= curSims + numSims:
            self.parallel_sim(numThreads)

    cpdef list getSymmetries(self, np.ndarray inS, np.ndarray inPi, z):
        return self.game.getSymmetries(inS, inPi, z)

    cpdef void renderS(self, state):
        self.game.renderS(state)

cpdef void testMCTS(numSims, numThreads):
    import time
    cdef double current_time
    model = tf.saved_model.load("trt_net")
    cdef MCTS mcts = MCTS(model)
    bw = False
    ww = False
    draw = False
    while not (bw or ww or draw):
        current_time = time.time()
        mcts.sim(numSims, numThreads)
        print(time.time()-current_time)
        print(sum(mcts.root.N))
        print('winrate: ' + str(mcts.get_winrate()))
        a = np.argmax(mcts.pi())
        print(mcts.pi())
        bw, ww, draw = mcts.step(a)
        mcts.render()
        if bw:
            print('black won')
        elif ww:
            print('white won')
        elif draw:
            print('draw')
