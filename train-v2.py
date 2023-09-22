from cnn import *
from mcts import *
import random

minibatch = 1024
train_loops = 1000
min_c = 28
max_c = 46

#population param
IterGames = 301
GameMoves = 60
Syms = 4
lastIters = 40


def trainNNet(examples, net):
    if len(examples) < minibatch:
        return net

    net.save('oldnet')
    nnet = load_model('oldnet')

    for _ in range(train_loops):
        samples = random.sample(examples, minibatch)

        X = []
        y1 = []
        y2 = []
        for s, pi, z in samples:
            X.append(s)
            y1.append(np.array(pi))
            y2.append(np.array([z]))
        nnet.fit(np.array(X), [np.array(y1), np.array(y2)], batch_size=32)
    return nnet

def load_data(mi, ma):
    data = []
    print('loading data ...')
    for i in range(mi, ma + 1):
        success = False
        while not success:
            try:
                data += list(np.load('data/example - {}.npy'.format(i),allow_pickle=True))
                print('data-{} loaded!!!'.format(i))
                success = True
            except:
                pass
    return data

def pickdata(data):
    point = len(data) - IterGames * GameMoves * lastIters * Syms
    print(point)
    data = data[point:]
    return data


data = load_data(min_c, max_c)
data = pickdata(data)
nnet = load_model("nnet")

nnet.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=SGD(learning_rate=0.0002, momentum=0.9),metrics=['accuracy'])
nnet = trainNNet(data, nnet)
nnet.save("nnet")

