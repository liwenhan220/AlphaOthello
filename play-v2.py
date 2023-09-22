import pygame, sys
import numpy as np
from mcts import *
import tensorflow as tf
import time

#tf.config.set_visible_devices([], 'GPU')

model = tf.saved_model.load('trt_net')
print('warming up ....')
for i in range(1, 33):
    model(tf.constant(np.random.uniform(0,1,[i,8,8,3]).astype(np.float32)))
print('model warmed up !!!!')
inject_noise = False
mcts = MCTS(model, inject_noise = inject_noise)
mcts.cpuct = 1.5
pygame.init()

SIZE = 600
YELLOW = [0, 200, 200]
BLACK = [0,0,0]
gap = 36

show_gap = 1.1
print('')
thinking_time = float(input('AI\'s thinking time in secs: '))
print('')
player = int(input('Pick your turn (Enter 0 for black, 1 for white): '))
pt = time.time()
screen = pygame.display.set_mode((SIZE, SIZE))
pygame.display.set_caption('Othello')

def draw_background():
    screen.fill(YELLOW)
    for i in range(8):
        pygame.draw.line(screen, BLACK, (75 * i, 0), (75 * i, 600), 5)
        pygame.draw.line(screen, BLACK, (0, 75 * i), (600, 75 * i), 5)

def draw_stones():
    for i in range(mcts.game.N):
        for j in range(mcts.game.N):
            if mcts.game.board[i][j] == 'O':
                pygame.draw.circle(screen, [0, 0, 0], (i * 75 + gap, j * 75 + gap), 30)
            elif mcts.game.board[i][j] == 'X':
                pygame.draw.circle(screen, [255, 255, 255], (i * 75 + gap, j * 75 + gap), 30)

def draw_valid():
    for a in mcts.game.valid_actions:
        x, y = mcts.game.transform(a)
        pygame.draw.circle(screen, [0, 255, 0], (x * 75 + gap, y * 75 + gap), 30)

def draw_a(a):
    x, y = mcts.game.transform(a)
    pygame.draw.circle(screen, [255, 0, 0], (x * 75 + gap, y * 75 + gap), 33)    

def draw_screen(b, a):
    draw_background()
    if b:
        draw_valid()
        draw_a(a)
    draw_stones()

def show_winrate(mcts):
    print('simulations: ' + str(sum(mcts.root.N)), end = "   ")
    if mcts.game.turn == 0:
        print ('black', end = ' ')
    else:
        print('white', end = ' ')
    winrate = mcts.get_winrate()
    winrate = int(winrate * 10000) / 100.0
    print('winrate: ' + str(winrate) + '%')
        
bw, ww, draw = False, False, False
last_show = time.time()
while not (bw or ww or draw):
    mcts.parallel_sim(32)
    if time.time() - last_show > show_gap:
        show_winrate(mcts)
        last_show = time.time()
    if player == mcts.game.turn:
        if len(mcts.game.valid_actions) == 1 and mcts.game.valid_actions[0] == mcts.game.action_space - 1:
            print('')
            print('You have no valid actions, passed')
            print('')
            bw, ww, draw = mcts.step(mcts.game.valid_actions[0], inject_noise = inject_noise)
            pt = time.time()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                try:
                    x, y = pygame.mouse.get_pos()                
                    x /= 75
                    y /= 75
                    x = int(x)
                    y = int(y)
                    for _ in range(5):
                        print('')
                    bw, ww, draw = mcts.step(mcts.game.revert(x,y), inject_noise = inject_noise)
                    pt = time.time()
                except:
                    print('Action not recognized, please try again')
    else:
        if len(mcts.game.valid_actions) == 1 and mcts.game.valid_actions[0] == mcts.game.action_space - 1:
            bw, ww, draw = mcts.step(mcts.game.valid_actions[0], inject_noise = inject_noise)
            print('AI was forced to pass')
        else:
            if time.time() - pt > thinking_time:
                a = np.argmax(mcts.pi())
                show_winrate(mcts)
                for _ in range(5):
                    print('')
                bw, ww, draw = mcts.step(a, inject_noise = inject_noise)
                if mcts.game.turn != player:
                    pt = time.time()
            else:
                pass
    try:
        draw_screen(player == mcts.game.turn, a)
    except:
        draw_background()
        if player == mcts.game.turn:
            draw_valid()
        draw_stones()   

    if bw:
        bcount, wcount = mcts.game.count_stones()
        print('black won by ' + str(bcount - wcount) + ' stones')
        break
    elif ww:
        bcount, wcount = mcts.game.count_stones()
        print('white won by ' + str(wcount - bcount) + ' stones')
        break
    elif draw:
        print('draw')
        break
    pygame.display.update()
input('press any key to end')
