from mcts import *
import time
example_count = 46
name = 'data/example - ' + str(example_count) + '.npy'

numSims = 200
stochastic_plays = 12
numThreads = 32
def selfplay(model):
    examples = []
    count = 0
    game = Othello()
    mcts = MCTS(model)
    bw = False
    ww = False
    draw = False
    while not (bw or ww or draw):
        current_time = time.time()
        mcts.sim(numSims, numThreads)
        print(time.time() - current_time)
        pi = mcts.pi()
        if count <= stochastic_plays:
            a = np.argmax(np.random.multinomial(1, pi))
        else:
            a = np.argmax(pi)
            pi = np.zeros((game.action_space))
            pi[a] = 1
        examples += mcts.getSymmetries(mcts.get_state(), pi, None)
        bw, ww, draw = mcts.step(a)
        count += 1
        mcts.render()
        if (bw or ww or draw):
            assignRewards(examples, bw, ww, draw)
            success = False
            while not success:
                try:
                    origExample = list(np.load(name, allow_pickle=True))
                    success = True
                except:
                    pass
            examples = origExample + examples
            np.save(name, examples)
            mcts.render()

def assignRewards(examples, b_win, w_win, draw):
    if b_win:
        for i in range(len(examples)):
            s, _, _ = examples[i]
            if s[0][0][2] == 0:
                examples[i][2] = 1
            else:
                examples[i][2] = -1
    elif w_win:
        for i in range(len(examples)):
            s, _, _ = examples[i]
            if s[0][0][2] == 0:
                examples[i][2] = -1
            else:
                examples[i][2] = 1
    elif draw:
        for i in range(len(examples)):
            examples[i][2] = 0
    else:
        raise NameError('cannnot assign rewards')

model = tf.saved_model.load("trt_net")

for i in range(56):
    print("numEps: " + str(i))
    selfplay(model)
