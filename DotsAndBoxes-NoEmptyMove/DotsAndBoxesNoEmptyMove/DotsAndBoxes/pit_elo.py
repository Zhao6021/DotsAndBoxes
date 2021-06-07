import Arena_elo
from MCTS import MCTS
from MCTS_ori import MCTS_ori
from DotsAndBoxes.DotsAndBoxesGame import DotsAndBoxesGame
from DotsAndBoxes.DotsAndBoxesPlayers import *
from DotsAndBoxes.keras.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = False


g = DotsAndBoxesGame(3)

# all players
rp = RandomPlayer(g).play
hp = HumanPlayer(g).play

# iters = [11,16,25,34,43,50,60,65,75,85,90,98,101,105,110,116,126,131,140,148]
# iters = [11,25,30,45,66,115,125,130,152,173,180,194,200,203,210,218,283,309,355]
iters = [11,16,18,23,25,27,34,43,45,50,54,56,58,59,
         60,64,65,68,70,72,75,78,83,85,87,89,90,92,
         93,95,98,101,104,105,106,108,110,113,116,
         122,126,127,129,131,133,140,144,148,152]

for i in range( 1 , len(iters) ):
    # nnet players
    n1 = NNet(g)
    # weight1 = '1217-' + str(iters[i]) + '-exact_win.pth.tar'
    weight1 = 'checkpoint_' + str(iters[i]) + '.pth.tar'
    n1.load_checkpoint('./pretrained_models/exact_win/', weight1)
    args1 = dotdict({'numMCTSSims': 200, 'cpuct':3.75})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    if human_vs_cpu:
        player2 = hp
    else:
        n2 = NNet(g)
        # weight2 = '1217-' + str(iters[i-1]) + '-exact_win.pth.tar'
        weight2 = 'checkpoint_' + str(iters[i-1]) + '.pth.tar'
        n2.load_checkpoint('./pretrained_models/exact_win/',weight2)
        args2 = dotdict({'numMCTSSims': 200, 'cpuct': 3.75})
        mcts2 = MCTS(g, n2, args2)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
        player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    # p1_name = '1217-' + str(iters[i])
    # p2_name = '1217-' + str(iters[i-1])
    p1_name = str(iters[i]) + '-exact_win'
    p2_name = str(iters[i-1]) + '-exact_win'
    arena = Arena_elo.Arena_elo(n1p, player2, g, p1_name, p2_name, display=DotsAndBoxesGame.display)
    print(arena.playGames(100, verbose=False))

script_name = './log/script.txt'
f = open(script_name, 'a')
f.write('elo\n')
f.write('mm\n')
f.write('exactdist\n')
f.write('ratings\n\n')
f.close()
