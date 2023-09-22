from cnn import *
from Othello import *
game = Othello()
oldnet = load_model('nnet')
oldnet.save('oldnet')
model = create_model((game.N, game.N, 3), game.action_space)
model.save('nnet')
