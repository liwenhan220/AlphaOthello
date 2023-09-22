import numpy as np
from cnn import *
from Othello import *

example = []
np.save('example.npy', example)

game = Othello()
model = create_model((game.N, game.N, 3), game.action_space)
model.save('nnet')
